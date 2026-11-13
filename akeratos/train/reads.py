"""Rebuild how often each table row was read in a run: `python -m akeratos.train.reads`.

A checkpoint keeps each table row's read count. A run started before the in-run counter
(`Table.track_reads`) has none, so this replays its data from the start and counts again. The
counts depend only on the ids, the table's hash seed and the data order, so no model or table
is built. It imports the checkpoint reader and the table's module, and with them MLX, but runs
no MLX computation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from akeratos import ckpt
from akeratos.data.packer import Packer
from akeratos.data.store import Split
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.rng import seed_hex
from akeratos.table.config import TableConfig
from akeratos.table.hasher import keys, multipliers
from akeratos.table.table import READS, load_reads
from akeratos.train.source import PackedSource, Source, SynthSource

EXIT_OK, EXIT_CONFIG, EXIT_DATA = 0, 78, 22  # as `akeratos/train/entry.py` spells them
TABLE = "table"  # the table's subdirectory of a checkpoint (`akeratos/train/saver.py`)
CHUNK = 1 << 24  # lookups counted per `np.bincount`


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.train.reads",
        description="Rebuild a run's per-row table read counts.",
    )
    p.add_argument("--ckpt", type=Path, required=True, help="a checkpoint or a run directory")
    p.add_argument("--data", required=True, help="the run's --data: synthetic or its split")
    p.add_argument("--rows", type=int, default=4, help="the run's --rows")
    p.add_argument("--seq-len", type=int, default=2048, help="the run's --seq-len")
    p.add_argument("--buffer", type=int, default=1000, help="the run's --buffer")
    p.add_argument("--out", type=Path, required=True, help="where to write the counts (.npy)")
    return p


def _table_config(config: dict[str, Any]) -> TableConfig:
    saved: object = config.get("table")
    if not isinstance(saved, dict):
        raise ConfigError("the checkpoint's run had no table")
    known = {f.name for f in fields(TableConfig)}
    kw: dict[str, Any] = {k: v for k, v in cast(dict[str, Any], saved).items() if k in known}
    kw["orders"] = tuple(kw["orders"])
    return TableConfig(**kw).validate()


def _source(a: argparse.Namespace, vocab: int, data_seed: int, saved: dict[str, Any]) -> Source:
    """The run's source at its start, checked against the checkpoint's where it can be.

    The flags must be the run's own. For a store run the packer's fingerprint, which covers the
    documents, the data seed, the buffer and the batch shape, must equal the checkpoint's, so a
    wrong flag stops before the replay.
    """
    kind = saved.get("kind")
    if a.data == "synthetic":
        if kind != "synthetic" or saved.get("seed") != seed_hex(data_seed):
            raise ConfigError("the checkpoint's source is not this synthetic source")
        return SynthSource(a.rows, a.seq_len, vocab, seed=data_seed)
    if kind != "packed":
        raise ConfigError("the checkpoint's source is not a packed store")
    split = Split(Path(a.data).expanduser(), vocab=vocab)
    packer = Packer(split, a.rows, a.seq_len, vocab=vocab, data_seed=data_seed, buffer=a.buffer)
    if packer.fingerprint != saved.get("fingerprint"):
        raise ConfigError(
            "the data, --rows, --seq-len or --buffer differ from the run's (packer fingerprint)"
        )
    return PackedSource(packer)


def replay(
    source: Source, cfg: TableConfig, batches: int, say: Callable[[str], None] | None = None
) -> npt.NDArray[np.int64]:
    """The read counts of `batches` batches from `source`: k·h lookups per token, per row.

    Each batch's ids are hashed as the run hashed them (`akeratos/table/hasher.py`), skipped
    steps included, as the in-run counter does.
    """
    mults = multipliers(cfg.seed, cfg.orders, cfg.heads)
    rows = cfg.rows()
    reads = np.zeros(rows, np.int64)
    pending: list[npt.NDArray[np.int64]] = []
    held = 0
    for i in range(batches):
        got = keys(source.next().ids, cfg.orders, mults, cfg.rows_per_slot(), cfg.heads)
        pending.append(got.reshape(-1))
        held += got.size
        if held >= CHUNK or i == batches - 1:
            reads += np.bincount(np.concatenate(pending), minlength=rows)
            pending, held = [], 0
        if say is not None and (i + 1) % 10_000 == 0:
            say(f"{i + 1:,} of {batches:,} batches")
    return reads


def run(a: argparse.Namespace) -> int:
    """Replay the checkpoint's data, write the counts and compare them with any in-run ones.

    `--ckpt` is a checkpoint, or a run directory for its newest complete one; its `state.json`
    gives the configs and the step. The replay reads step x accum batches, after which the source
    must stand at the checkpoint's saved cursor. The counts are written with `np.save`, one int64
    per row; in-run counts in the checkpoint that differ raise `DataMismatchError`.
    """
    path = ckpt.latest(a.ckpt) or a.ckpt
    try:
        c = ckpt.load(path)
    except ValueError as e:
        raise ConfigError(f"{a.ckpt}: no complete checkpoint ({e})") from e
    config = c.state["config"]
    cfg = _table_config(config)
    vocab = int(config["model"]["vocab_size"])
    data_seed = int(config["data_seed"], 16)
    accum = int(config["accum"])
    saved = cast(dict[str, Any], c.state["source"])
    source = _source(a, vocab, data_seed, saved)
    _say(f"replaying {c.step:,} steps x {accum} batches of {c.path}")
    reads = replay(source, cfg, c.step * accum, _say)
    if source.state() != saved:
        raise ConfigError(f"the replay did not end at {c.path}'s data cursor")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        np.save(f, reads, allow_pickle=False)
    tokens = c.step * accum * a.rows * a.seq_len
    _say(
        f"wrote {a.out}: {tokens:,} tokens, {int(reads.sum()):,} reads over {cfg.rows():,} rows "
        f"({int((reads > 0).sum()):,} read at least once, at most {int(reads.max()):,})"
    )
    inrun = c.path / TABLE / READS
    if inrun.exists():
        if not np.array_equal(load_reads(inrun, cfg.rows()), reads):
            raise DataMismatchError(f"the replayed counts differ from {inrun}")
        _say(f"the replayed counts equal the in-run counts in {inrun}")
    return EXIT_OK


def _say(line: str) -> None:
    sys.stderr.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the flags and rebuild the counts; returns the exit code."""
    a = _parser().parse_args(argv)
    try:
        return run(a)
    except ConfigError as e:
        _say(f"configuration error: {e}")
        return EXIT_CONFIG
    except DataMismatchError as e:
        _say(f"data error: {e}")
        return EXIT_DATA


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
