"""The trainer behind `bake train`, also run as `python -m akeratos.train.entry`.

A run trains on synthetic Zipf ids (`akeratos/train/source.py`) or on a data-store split
(`akeratos/data/store.py`), packed into rows of T + 1 tokens. Each optimiser step sums the
dense gradients of K micro-batches of B rows, so the schedule counts optimiser steps of
B·T·K tokens. `run.json` records the recipe, the model, the compute dtype, the MTP switch, the
seeds, K under `accum` and the time limit under `minutes`.

Checkpoints (`akeratos/ckpt.py`, `akeratos/train/saver.py`) are saved at a fixed step
cadence and where the run stops, and hold the table under `table/` when it is on. Rerunning a
command with the same run directory resumes from its newest complete checkpoint, table
included, and refuses other configs. A run that stops because the data ran out saves nothing
new, since the source's cursor has moved past the last whole step.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, cast

from akeratos.clock import Clock, SystemClock
from akeratos.configs import ModelConfig
from akeratos.configs import from_keys as model_from_keys
from akeratos.data.packer import ExhaustedError, Packer
from akeratos.data.store import Split
from akeratos.env import tf32_off
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.rng import run_seeds, seed_hex
from akeratos.table.config import TableConfig
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.recipe import from_keys as recipe_from_keys
from akeratos.train.source import PackedSource, Source, SynthSource, preflight

if TYPE_CHECKING:
    from akeratos.train.loop import Run
    from akeratos.train.saver import Checkpointer

EXIT_OK, EXIT_STOPPED, EXIT_CONFIG, EXIT_DATA = 0, 3, 78, 22
EXIT_RESUMABLE = 75  # a signal stop with its step saved
TABLE_KEYS = {"format": "fmt"}  # a table switch key where it differs from the field's name


def _pairs(items: Sequence[str]) -> dict[str, Any]:
    """The `key=value` switches; a value is read as JSON where it parses, else as a string."""
    out: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"{item!r} is not key=value")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.train.entry",
        description="Train a model on synthetic ids or a data-store split.",
    )
    p.add_argument("--data", required=True, help="synthetic, or a data-store split directory")
    p.add_argument("--tokens", type=float, required=True, help="the token budget (tokens.budget)")
    p.add_argument("--out", type=Path, required=True, help="the run directory")
    p.add_argument("--rows", type=int, default=4, help="rows per step (B)")
    p.add_argument("--seq-len", type=int, default=2048, help="tokens per row (T)")
    p.add_argument(
        "--accum", type=int, default=1, help="micro-batches summed per optimiser step (K)"
    )
    p.add_argument(
        "--model", action="append", default=[], metavar="KEY=VALUE", help="a model switch"
    )
    p.add_argument(
        "--recipe", action="append", default=[], metavar="KEY=VALUE", help="a recipe switch"
    )
    p.add_argument(
        "--table", action="append", default=[], metavar="KEY=VALUE", help="a table switch"
    )
    p.add_argument("--seed", type=int, default=0, help="the master seed")
    p.add_argument("--init-seed", type=int, default=None, help="default: derived from --seed")
    p.add_argument("--data-seed", type=int, default=None, help="default: derived from --seed")
    p.add_argument("--buffer", type=int, default=1000, help="the packer's best-fit buffer")
    p.add_argument("--compute-dtype", default="bfloat16", choices=("bfloat16", "float32"))
    p.add_argument("--mtp", action="store_true", help="objective.mtp.mode same_head")
    p.add_argument(
        "--steps", type=int, default=None, help="stop after N steps (default: the budget's)"
    )
    p.add_argument("--minutes", type=float, default=None, help="stop after N minutes of wall time")
    p.add_argument("--print-every", type=int, default=10, help="a progress line every N steps")
    p.add_argument(
        "--ckpt-every", type=int, default=10_000, help="save every N steps (0: no checkpoints)"
    )
    p.add_argument("--keep", type=int, default=2, help="complete checkpoints to keep")
    p.add_argument(
        "--pin-decay",
        action="store_true",
        help="also keep the checkpoint at the decay's start, outside the rotation",
    )
    return p


def _say(line: str) -> None:
    sys.stderr.write(line + "\n")


def _source(
    a: argparse.Namespace, vocab: int, data_seed: int, saved: dict[str, Any] | None
) -> Source:
    """The batch source, at the saved cursor when resuming.

    A store split is first checked against the run's budget by `preflight`.
    """
    saved = dict(saved or {})
    kind = saved.pop("kind", None)
    if a.data == "synthetic":
        if kind not in (None, "synthetic") or saved.get("seed", seed_hex(data_seed)) != seed_hex(
            data_seed
        ):
            raise ConfigError("the checkpoint's source is not this synthetic source")
        return SynthSource(a.rows, a.seq_len, vocab, seed=data_seed, cursor=saved.get("i", 0))
    if kind not in (None, "packed"):
        raise ConfigError("the checkpoint's source is not a packed store")
    split = Split(Path(a.data).expanduser(), vocab=vocab)
    fit = preflight(
        split,
        a.rows * a.accum,
        a.seq_len,
        a.tokens,
        vocab=vocab,
        data_seed=data_seed,
        buffer=a.buffer,
    )
    _say(
        f"data: one epoch feeds {fit['steps_fed']:,} steps of {a.rows * a.accum} x {a.seq_len}; "
        f"the run needs {fit['steps']:,}"
    )
    try:
        packer = Packer(
            split,
            a.rows,
            a.seq_len,
            vocab=vocab,
            data_seed=data_seed,
            buffer=a.buffer,
            state=cast(Any, saved) if saved else None,
        )
    except ValueError as e:
        raise ConfigError(f"the checkpoint's data cursor does not fit: {e}") from e
    return PackedSource(packer)


def _table_config(a: argparse.Namespace, on: bool, width: int, seed: int) -> TableConfig | None:
    """The table's configuration from `--table` switches, or None with the table off.

    The table is on by default and then needs its size, `params`, the numbers it stores
    (`akeratos/table/config.py`). The other switches are `TableConfig`'s fields, `format` for
    `fmt`; the width is the model's d and the seed the master seed.
    """
    if not on:
        if a.table:
            raise ConfigError("--table switches with the model's table off (table=false)")
        return None
    keys = {TABLE_KEYS.get(k, k): v for k, v in _pairs(a.table).items()}
    known = {f.name for f in fields(TableConfig)} - {"width", "seed"}
    if unknown := sorted(set(keys) - known):
        raise ConfigError(f"unknown table switches {unknown}; one of {sorted(known)}")
    if "params" not in keys:
        raise ConfigError("the table is on: give its size as --table params=<numbers stored>")
    if "orders" in keys:
        keys["orders"] = tuple(keys["orders"])
    return TableConfig(width=width, seed=seed, **keys).validate()


@dataclass(frozen=True)
class Resolved:
    """A run's configuration as its flags give it, before any model is built or data is read.

    The model is `ModelConfig`'s defaults with the model switches applied. `names_path` says
    whether those switches name `attn.path`; when they do not, the run takes the path the
    checkpointer picks (`Checkpointer.attn_path`).
    """

    cfg: ModelConfig
    o: OptimConfig
    r: RecipeConfig
    seeds: dict[str, int]
    table: TableConfig | None
    compute_dtype: str
    mtp: bool
    accum: int
    names_path: bool


def resolve(a: argparse.Namespace) -> Resolved:
    """The model, optimiser, recipe, seeds, table, compute dtype and accumulation of the flags.

    It builds no model and reads no data. The master seed derives the init and data seeds unless
    they are given (`rng.run_seeds`).
    """
    recipe = _pairs(a.recipe) | {
        "tokens.budget": a.tokens,
        "batch.tokens_per_update": a.rows * a.seq_len * a.accum,
    }
    keys = _pairs(a.model)
    try:
        cfg = model_from_keys(keys)
        o, r = recipe_from_keys(recipe)
        seeds = run_seeds(a.seed, a.init_seed, a.data_seed)
    except (KeyError, ValueError, OverflowError) as e:  # an unknown key or an invalid config
        raise ConfigError(str(e)) from e
    tcfg = _table_config(a, cfg.table, cfg.d_model, seeds["seed"])
    named = "attn.path" in keys
    return Resolved(cfg, o, r, seeds, tcfg, a.compute_dtype, a.mtp, a.accum, named)


def _run(
    a: argparse.Namespace, run_cls: type[Run], saver_cls: type[Checkpointer], clock: Clock
) -> int:
    c = resolve(a)
    cfg, seeds = c.cfg, c.seeds
    saver = saver_cls(a.out, a.ckpt_every, a.keep, _say, pin_decay=a.pin_decay)
    if not c.names_path:
        path = saver.attn_path(cfg, c.compute_dtype, a.seq_len)
        cfg = replace(cfg, attn_path=path)
        _say(f"attn.path {path} ({'the checkpoint' if saver.resumed else 'default'})")
    source = _source(a, cfg.vocab_size, seeds["data_seed"], saver.source_state())
    table = None if c.table is None else saver.table(c.table)
    run = run_cls(
        cfg,
        c.o,
        c.r,
        source,
        a.out,
        seed=seeds["seed"],
        init_seed=seeds["init_seed"],
        data_seed=seeds["data_seed"],
        compute_dtype=c.compute_dtype,
        mtp=c.mtp,
        accum=c.accum,
        meta={"minutes": a.minutes},
        table=table,
        clock=clock,
    )
    saver.attach(run)
    _say(f"run {a.out}: {run.sched.steps:,} steps in the schedule, warm-up {run.sched.warmup:,}")
    return drive(run, saver, clock, until=a.steps, minutes=a.minutes, print_every=a.print_every)


def drive(
    run: Run,
    saver: Checkpointer,
    clock: Clock,
    *,
    until: int | None,
    minutes: float | None,
    print_every: int,
) -> int:
    """Train `run` to step `until` (default: its schedule's end) with `saver`; the exit code.

    It prints progress every `print_every` steps. Once `minutes` have passed on the monotonic
    clock, or on SIGINT or SIGTERM, it stops after the step in flight, so a probe runs for a set
    time while the schedule keeps the budget's horizon. A signal stop is resumable only when a
    checkpoint holds its step.
    """
    deadline = None if minutes is None else clock.monotonic_ns() + int(60e9 * minutes)

    def progress(_: Run, rec: dict[str, Any]) -> None:
        if print_every and (rec["step"] + 1) % print_every == 0:
            rate = rec["tokens_per_s"] or 0.0
            peak = rec.get("peak_bytes", 0) / 2**30
            sys.stderr.write(
                f"step {rec['step'] + 1:,} loss {rec['loss']:.4f} tok/s {rate:,.0f} "
                f"peak {peak:.2f} GiB\n"
            )
        if deadline is not None and clock.monotonic_ns() >= deadline and not run.stopping:
            sys.stderr.write(f"{minutes:g} minutes: stopping after this step\n")
            run.stop()

    def on_step(run: Run, rec: dict[str, Any]) -> None:
        progress(run, rec)
        saver.after_step(run)

    def on_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        signalled.append(signum)
        sys.stderr.write(f"signal {signum}: stopping after this step\n")
        run.stop()

    signalled: list[int] = []
    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        reached = run.train(until, on_step)
        saver.finish(run)
    except ExhaustedError:
        sys.stderr.write(f"the data ran out at step {run.step_no:,} (epochs exceeded)\n")
        return EXIT_CONFIG
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    sys.stderr.write(f"stopped at step {reached:,} of {run.sched.steps:,}\n")
    if reached == run.sched.steps:
        return EXIT_OK
    saved = bool(saver.every) and saver.saved_at == reached
    return EXIT_RESUMABLE if signalled and saved else EXIT_STOPPED


def main(
    argv: Sequence[str] | None,
    run_cls: type[Run],
    saver_cls: type[Checkpointer],
    clock: Clock | None = None,
    *,
    setup: Callable[[], None] | None = None,
) -> int:
    """Parse the flags and run `run_cls` (the loop's `Run`) with `saver_cls`; returns the exit code.

    The caller imports the loop and the `Checkpointer`, and with them MLX, only after
    `tf32_off()`, as this module's own entry below does; `main` calls it again, which refuses if
    MLX loaded with TF32 on. It builds the `SystemClock` that times the steps and the time limit,
    unless a test passes a fake (`akeratos/clock.py`). `setup` runs once per launch, resume
    included, before the first MLX allocation: the CLI caps MLX's cache there
    (`cli.run_trainer`).
    """
    a = _parser().parse_args(argv)
    try:
        tf32_off()
        if setup is not None:  # before the source, the table and the model are built
            setup()
        return _run(a, run_cls, saver_cls, SystemClock() if clock is None else clock)
    except ConfigError as e:
        sys.stderr.write(f"configuration error: {e}\n")
        return EXIT_CONFIG
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA


if __name__ == "__main__":
    # The loop imports MLX, which reads MLX_ENABLE_TF32 once, so TF32 goes off first.
    tf32_off()
    from akeratos.train import loop, saver

    raise SystemExit(main(sys.argv[1:], loop.Run, saver.Checkpointer))
