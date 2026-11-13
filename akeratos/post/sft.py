"""Supervised fine-tuning, the first two post-training stages, on a frozen table.

Stage 1 trains the chat format: whole conversations packed into rows, loss on answer tokens
only, ids 1 and 2 unmasked and every other reserved id masked (`akeratos/post/run.py`). Stage
2 is safety fine-tuning, the same supervised stage from a stage-1 checkpoint on the safety set:
refusals, deep-alignment recoveries, and benign and clinical over-refusal examples, in the same
chat format with the store's masks. Every recipe switch starts at the base run's value, and a
stage sets only what `_recipe` lists.

Each checkpoint holds the frozen table, so it is whole for `eval.heldout`. Rerunning with the
same run directory resumes, and a resume with another base or setting is refused. With held-out
data the end of a stage writes `assistant_loss.json`; a sweep runs one stage per rate factor
into `rate-<factor>/` and writes `sweep.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from akeratos.clock import Clock, SystemClock
from akeratos.configs import from_keys as model_from_keys
from akeratos.data.packer import ExhaustedError
from akeratos.env import tf32_off
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.post.data import (
    STAGE1_IDS,
    ChatSource,
    documents,
    epochs_of,
    pack_rows,
    stage_config,
)
from akeratos.rng import run_seeds
from akeratos.train.recipe import KEYS as RECIPE_KEYS
from akeratos.train.recipe import from_keys as recipe_from_keys
from akeratos.train.recipe import steps_for

if TYPE_CHECKING:
    from akeratos.post.run import SFTRun
    from akeratos.train.loop import Run
    from akeratos.train.saver import Checkpointer

EXIT_OK, EXIT_STOPPED, EXIT_CONFIG, EXIT_DATA, EXIT_RESUMABLE = 0, 3, 78, 22, 75
STAGES = (1, 2)  # supervised fine-tuning and the safety fork
WARMUP_SHARE = 0.03  # 3% of the steps
MIN_STEPS = 2  # warm-up and the decay each take at least one


def _pairs(items: Sequence[str]) -> dict[str, Any]:
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
        prog="python -m akeratos.post.sft",
        description="Stage 1 or 2 supervised fine-tuning on a frozen table.",
    )
    p.add_argument("--stage", type=int, choices=STAGES, default=1, help="post.stage")
    p.add_argument("--base", type=Path, required=True, help="a checkpoint, run directory or export")
    p.add_argument("--weights", choices=("raw", "ema"), default="raw", help="the base's weights")
    p.add_argument("--data", type=Path, required=True, help="the training chat directory")
    p.add_argument("--heldout", type=Path, default=None, help="the held-out chat directory")
    p.add_argument("--tokens", type=float, required=True, help="the token budget of each stage")
    p.add_argument("--out", type=Path, required=True, help="the run directory")
    rate = p.add_mutually_exclusive_group(required=True)
    rate.add_argument("--rate", type=float, help="the peak as a factor of the base run's")
    rate.add_argument("--sweep", help="comma-separated factors, one short stage each")
    p.add_argument("--rows", type=int, default=2, help="rows per micro-batch (B)")
    p.add_argument("--seq-len", type=int, default=2048, help="tokens per row (T)")
    p.add_argument("--accum", type=int, default=32, help="micro-batches per optimiser step (K)")
    p.add_argument("--epochs", type=int, default=None, help="default: the store's epochs")
    p.add_argument("--filter", choices=("on", "off"), default="on", help="post.sft.filter")
    p.add_argument(
        "--recipe", action="append", default=[], metavar="KEY=VALUE", help="a recipe switch"
    )
    p.add_argument("--seed", type=int, default=0, help="the master seed")
    p.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default=None)
    p.add_argument("--steps", type=int, default=None, help="stop after N steps")
    p.add_argument("--minutes", type=float, default=None, help="stop after N minutes")
    p.add_argument("--print-every", type=int, default=10, help="a progress line every N steps")
    p.add_argument("--ckpt-every", type=int, default=50, help="save every N steps (0: none)")
    p.add_argument("--keep", type=int, default=1, help="complete checkpoints to keep")
    return p


def _say(line: str) -> None:
    sys.stderr.write(line + "\n")


def _recipe(a: argparse.Namespace, base: dict[str, Any], factor: float) -> dict[str, Any]:
    """The base run's recipe keys with the stage's settings over them, then `--recipe`'s.

    The optimiser is AdamW with weight decay 0 (independent, even on a coupled base) and no EMA.
    The first 3% of the steps warm up, then the rate decays linearly to 0. Each group's peak is
    `factor` times the base run's: its `lr.peak`, and its `lr.emb` when it set one. At stage 2
    the base is the stage-1 run, whose peak is already a factor of pretraining's. An update holds
    rows x seq_len x accum tokens, 131,072 at 2 x 2,048 x 32.
    """
    tpu = a.rows * a.seq_len * a.accum
    steps = steps_for(a.tokens, tpu)
    if steps < MIN_STEPS:
        raise ConfigError(
            f"--tokens {a.tokens:g} is {steps} step(s) of {tpu:,}; a stage needs {MIN_STEPS}"
        )
    keys: dict[str, Any] = {k: base[part][f] for k, (part, f) in RECIPE_KEYS.items()}
    keys |= {
        "tokens.budget": a.tokens,
        "batch.tokens_per_update": tpu,
        "lr.peak": factor * base["optim"]["lr_peak"],
        "lr.schedule": "d2z",
        "lr.final_frac": 0.0,
        "lr.warmup": max(1, math.ceil(round(WARMUP_SHARE * steps, 9))),
        "opt.weight_decay_mode": "independent",  # coupled ignores decay_ind
        "opt.decay_ind": 0.0,
        "ema.enabled": False,
    }
    if base["optim"]["eta_emb"] is not None:
        keys["lr.emb"] = factor * base["optim"]["eta_emb"]
    return keys | _pairs(a.recipe)


def _check_base(stage: int, saved: dict[str, Any], where: Path) -> None:
    """Stage 1 starts from pretraining and stage 2 from stage 1, as the base's `post` shows."""
    prior = saved.get("post")
    had = None if prior is None else prior.get("stage")
    want = None if stage == 1 else stage - 1
    if had != want:
        names = {None: "a pretraining checkpoint", 1: "a stage-1 checkpoint"}
        found = names.get(had, f"a stage-{had} checkpoint")
        raise ConfigError(f"{where}: stage {stage} starts from {names[want]}; this is {found}")


def _stage(
    a: argparse.Namespace,
    run_cls: type[SFTRun],
    saver_cls: type[Checkpointer],
    clock: Clock,
    factor: float,
    out: Path,
) -> int:
    """One stage at `factor` into `out`; returns its exit code."""
    base = run_cls.base_checkpoint(a.base)
    saved = base.state["config"]
    _check_base(a.stage, saved, base.path)
    try:
        cfg = stage_config(model_from_keys(saved["model"]), a.stage)
        o, r = recipe_from_keys(_recipe(a, saved, factor))
        seeds = run_seeds(a.seed)
    except (KeyError, ValueError, OverflowError) as e:
        raise ConfigError(str(e)) from e
    if not cfg.doc_mask:  # packed conversations would attend to each other
        raise ConfigError(f"{base.path}: mask.packing is causal; a stage packs on document masks")
    rendered = documents(a.data, cfg.vocab_size)
    epochs = epochs_of(a.data) if a.epochs is None else a.epochs
    saver = saver_cls(out, a.ckpt_every, a.keep, _say)
    state = saver.source_state()
    if state is not None and state.get("kind") != "chat":
        raise ConfigError(f"{out}: the checkpoint's source is not chat data")
    ds, ep = seeds["data_seed"], epochs
    source = ChatSource(rendered, a.rows, a.seq_len, data_seed=ds, epochs=ep, state=state)
    need = steps_for(a.tokens, o.tokens_per_update) * a.accum * a.rows
    have = ChatSource(rendered, a.rows, a.seq_len, data_seed=ds, epochs=ep).rows_available()
    _say(f"data: {len(rendered) - source.dropped} documents, {source.dropped} longer than a row")
    if need > have:
        msg = f"the budget needs {need:,} rows; {epochs} epoch(s) of the data fill {have:,}"
        raise ConfigError(msg)
    table = run_cls.table_of(saver.resumed or base, cfg)
    post = {
        "stage": a.stage,
        "base": str(base.path),
        "base_step": base.step,
        "weights": a.weights,
        "rate_factor": factor,
        "unmasked": list(STAGE1_IDS),
        "filter": a.filter == "on",  # recorded only: the chat data marks no factual item
        "epochs": epochs,
    }
    run = run_cls(
        cfg,
        o,
        r,
        source,
        out,
        base=base,
        weights=a.weights,
        post=post,
        table=table,
        clock=clock,
        seed=seeds["seed"],
        compute_dtype=a.compute_dtype or saved["compute_dtype"],
        accum=a.accum,
        meta={"minutes": a.minutes},
    )
    saver.attach(run)
    _say(f"stage {a.stage} at {factor:g} x the base peak: {run.sched.steps:,} steps into {out}")
    code = _drive(a, run, saver, clock)
    if a.heldout is not None and code in (EXIT_OK, EXIT_STOPPED):
        loss = _heldout_loss(a, run, cfg.vocab_size)
        rec = {"rate_factor": factor, "step": run.step_no, "assistant_loss": loss}
        (out / "assistant_loss.json").write_text(json.dumps(rec) + "\n")
        _say(f"held-out assistant loss {loss:.4f}")
    return code


def _heldout_loss(a: argparse.Namespace, run: SFTRun, vocab: int) -> float:
    rows = pack_rows(documents(a.heldout, vocab), a.seq_len)
    return run.assistant_loss(rows, a.rows)


def _drive(a: argparse.Namespace, run: SFTRun, saver: Checkpointer, clock: Clock) -> int:
    """`akeratos/train/entry.py`'s loop: progress, the wall-time stop, signals, saves."""
    deadline = None if a.minutes is None else clock.monotonic_ns() + int(60e9 * a.minutes)
    signalled: list[int] = []

    def on_step(run: Run, rec: dict[str, Any]) -> None:
        if a.print_every and (rec["step"] + 1) % a.print_every == 0:
            _say(f"step {rec['step'] + 1:,} loss {rec['loss']:.4f}")
        if deadline is not None and clock.monotonic_ns() >= deadline and not run.stopping:
            _say(f"{a.minutes:g} minutes: stopping after this step")
            run.stop()
        saver.after_step(run)

    def on_signal(signum: int, frame: FrameType | None) -> None:
        del frame
        signalled.append(signum)
        _say(f"signal {signum}: stopping after this step")
        run.stop()

    previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        reached = run.train(a.steps, on_step)
        saver.finish(run)
    except ExhaustedError:
        _say(f"the data ran out at step {run.step_no:,}")
        return EXIT_CONFIG
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    _say(f"stopped at step {reached:,} of {run.sched.steps:,}")
    if reached == run.sched.steps:
        return EXIT_OK
    saved = bool(saver.every) and saver.saved_at == reached
    return EXIT_RESUMABLE if signalled and saved else EXIT_STOPPED


def _sweep(
    a: argparse.Namespace, run_cls: type[SFTRun], saver_cls: type[Checkpointer], clock: Clock
) -> int:
    """One stage per factor, each scored on the held-out answer tokens, then `sweep.json`.

    The pick is the factor of the lowest loss; ties go to the lower rate.
    """
    if a.heldout is None:
        raise ConfigError("--sweep scores each rate on --heldout")
    try:
        factors = sorted({float(f) for f in a.sweep.split(",")})
    except ValueError as e:
        raise ConfigError(f"--sweep {a.sweep!r}: {e}") from e
    losses: dict[str, float] = {}
    for f in factors:
        out = a.out / f"rate-{f:g}"
        code = _stage(a, run_cls, saver_cls, clock, f, out)
        if code != EXIT_OK:
            return code
        losses[f"{f:g}"] = json.loads((out / "assistant_loss.json").read_text())["assistant_loss"]
    pick = min(factors, key=lambda f: (losses[f"{f:g}"], f))
    rec = {"tokens": a.tokens, "loss": losses, "pick": pick}
    (a.out / "sweep.json").write_text(json.dumps(rec, indent=1) + "\n")
    _say(f"sweep: {losses}; pick {pick:g} x the base peak")
    return EXIT_OK


def main(
    argv: Sequence[str] | None,
    run_cls: type[SFTRun],
    saver_cls: type[Checkpointer],
    clock: Clock | None = None,
) -> int:
    """Parse the flags and run the stage or the sweep; returns the exit code.

    `run_cls` is `post.run.SFTRun` and `saver_cls` the `Checkpointer`; the caller imports them,
    and with them MLX, only after `tf32_off()`, as this module's own entry does. `clock` is the
    `SystemClock` unless a test passes a fake.
    """
    a = _parser().parse_args(argv)
    try:
        tf32_off()
        clock = SystemClock() if clock is None else clock
        if a.sweep is not None:
            return _sweep(a, run_cls, saver_cls, clock)
        return _stage(a, run_cls, saver_cls, clock, a.rate, a.out)
    except ConfigError as e:
        _say(f"configuration error: {e}")
        return EXIT_CONFIG
    except DataMismatchError as e:
        _say(f"data error: {e}")
        return EXIT_DATA


if __name__ == "__main__":
    # The run imports MLX, which reads MLX_ENABLE_TF32 once, so TF32 goes off first.
    tf32_off()
    from akeratos.post import run
    from akeratos.train import saver

    raise SystemExit(main(sys.argv[1:], run.SFTRun, saver.Checkpointer))
