"""Held-out bits per byte, document by document, over a saved checkpoint.

The data store's held-out split holds whole documents, each `[0] + ids`. Every target of every
document read is scored once, and

    bits per byte = Σ nats / (ln 2 · Σ bytes of the scored targets)

with both sums in float64 on the host. The raw weights and, when the run kept one, the EMA are
scored; a model with the table on is scored with its table term and without it. Nothing here
depends on the model's size.

`main` prints one JSON record. It can also write the record to a file, and the `RECORDS` files:
the per-document and per-target numbers each bits per byte is summed from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from akeratos import ckpt, export
from akeratos.configs import MASKED_IDS, ModelConfig, from_keys
from akeratos.data.packer import RESERVED, SEP_ID, segment_ids
from akeratos.data.store import U16, U64, Split
from akeratos.errors import AkeratosError, ConfigError, DataMismatchError
from akeratos.eval.heldout_flags import DESCRIPTION, add_flags
from akeratos.model import Params, head_logits, hidden, masked_id_mask, table_term
from akeratos.table.config import TableConfig
from akeratos.table.table import Table

LN2 = math.log(2.0)
EXIT_CONFIG, EXIT_DATA = 78, 22
DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}
TABLE_DIR = "table"  # a checkpoint's table subdirectory
RECORDS = ("documents.npy", "targets.npy")  # what --records writes
Term = Callable[[NDArray[np.int64]], mx.array]


def token_bytes(store: Path, vocab: int) -> NDArray[np.int64]:
    """Each id's raw byte length from `<store>/token_bytes.u16`, 0 for ids 0-16, checked.

    A wrong file is refused rather than left to skew bits per byte.
    """
    path = store / "token_bytes.u16"
    if not path.is_file():
        raise DataMismatchError(f"{path}: missing")
    raw = path.read_bytes()
    tb = np.frombuffer(raw, dtype=U16)
    if len(tb) != vocab or tb[: RESERVED[1] + 1].any():
        msg = f"{path}: {len(tb)} entries for a vocabulary of {vocab}, or bytes on ids 0-16"
        raise DataMismatchError(msg)
    root = store / "manifest.json"
    if root.exists():
        want = json.loads(root.read_text()).get("outputs", {}).get("token_bytes_sha256")
        if want is not None and want != hashlib.sha256(raw).hexdigest():
            raise DataMismatchError(f"{path}: SHA-256 differs from the root manifest's")
    return tb.astype(np.int64)


def _sidecars(split: Split, n_docs: int) -> NDArray[np.uint64]:
    m = json.loads((split.path / "manifest.json").read_text())
    names, shards = m.get("bytes"), m["shards"]
    if (
        names is None
        or [n[:-6] for n in names] != [s[:-4] for s in shards]
        or not all(re.fullmatch(r"[0-9]{6}\.bytes", n) for n in names)
        or not all((split.path / n).is_file() for n in names)
    ):
        raise DataMismatchError(f"{split.path}: no `.bytes` sidecar beside each shard")
    # Each sidecar holds its shard's documents' canonical UTF-8 byte counts, uint64 little-endian.
    nbytes = np.concatenate([np.fromfile(split.path / n, U64) for n in names])
    if len(nbytes) != n_docs:
        msg = f"{split.path}: {len(nbytes)} byte counts for {n_docs} documents"
        raise DataMismatchError(msg)
    return nbytes


class HeldOut:
    """The held-out split's scored rows: ids [R, T + 1] and the target mask [R, T].

    Documents are read in the split's order until `tokens` targets are held, whole documents only
    (all of them when `tokens` is None). A row's inputs are row[:-1] and its targets row[1:].
    - With `doc_mask` (`mask.packing` document), documents that fit are packed in order and one
      that does not opens a new row. The target that crosses into the next document (its id 0)
      and the zero padding at a row's end are masked. The segment ids keep each document to
      itself, so packing changes only its absolute RoPE positions, which cancel in attention up
      to rounding.
    - A document longer than T + 1 is cut into windows of T + 1 that overlap by one token, each in
      a row of its own, so every target is scored once, with its context cut at its window's
      start.
    - Without `doc_mask` (`mask.packing` causal) every document gets rows of its own.

    Before anything is scored, each document's token bytes must sum to its `.bytes` entry, so the
    bytes of the scored targets sum to the documents' bytes. `doc_targets` and `doc_bytes` [docs]
    are each document's targets and bytes. A split whose vocabulary is not `vocab` (the model's)
    raises `DataMismatchError`.
    """

    def __init__(
        self,
        split: str | Path,
        T: int,
        tokens: int | None = None,
        doc_mask: bool = True,
        vocab: int | None = None,
    ) -> None:
        self.T = T
        sp = Split(Path(split), vocab=vocab)
        tb = token_bytes(sp.path.parent, sp.vocab)
        n_docs = len(sp.lengths)
        nbytes = _sidecars(sp, n_docs)
        rows: list[list[NDArray[np.int64]]] = []
        cur: list[NDArray[np.int64]] = []
        self.docs = self.targets = self.bytes = 0
        for i in range(n_docs):
            if tokens is not None and self.targets >= tokens:
                break
            d = sp.read(i, int(sp.lengths[i]))
            self._check(sp, i, d, tb, int(nbytes[i]))
            self.docs, self.targets = self.docs + 1, self.targets + len(d) - 1
            self.bytes += int(nbytes[i])
            if len(d) <= T + 1 and doc_mask and sum(map(len, cur)) + len(d) <= T + 1:
                cur.append(d)
                continue
            if cur:
                rows.append(cur)
            cur = [d] if len(d) <= T + 1 else []
            if not cur:
                rows.extend([d[s : s + T + 1]] for s in range(0, len(d) - 1, T))
        if cur:
            rows.append(cur)
        self.doc_targets = sp.lengths[: self.docs].astype(np.int64) - 1
        self.doc_bytes = nbytes[: self.docs].astype(np.int64)
        self.ids = np.zeros((len(rows), T + 1), np.int64)
        self.mask = np.zeros((len(rows), T), np.float32)
        for r, pieces in enumerate(rows):
            o = 0
            for p in pieces:
                self.ids[r, o : o + len(p)] = p
                self.mask[r, o : o + len(p) - 1] = 1
                o += len(p)
        live = tb[self.ids[:, 1:][self.mask > 0]]
        if len(live) != self.targets or int(live.sum()) != self.bytes:
            raise AssertionError("packing lost or repeated a target")

    @staticmethod
    def _check(sp: Split, i: int, d: NDArray[np.int64], tb: NDArray[np.int64], n: int) -> None:
        body = d[1:]
        if d[0] != SEP_ID or (body == SEP_ID).any() or (body >= sp.vocab).any():
            raise DataMismatchError(f"{sp.path}: document {i} must hold id 0 at its start only")
        if ((body >= RESERVED[0]) & (body <= RESERVED[1])).any():
            raise DataMismatchError(f"{sp.path}: document {i} holds a reserved id 1-16")
        got = int(tb[body].sum())
        if got != n:
            msg = f"{sp.path}: document {i}'s token bytes sum to {got}, its `.bytes` entry is {n}"
            raise DataMismatchError(msg)

    def batches(
        self, B: int
    ) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float32]]]:
        """(ids, targets, mask) [B, T] each; the last batch is padded with masked rows of zeros."""
        for a in range(0, len(self.ids), B):
            ids = np.zeros((B, self.T + 1), np.int64)
            m = np.zeros((B, self.T), np.float32)
            n = len(self.ids[a : a + B])
            ids[:n], m[:n] = self.ids[a : a + B], self.mask[a : a + B]
            yield ids[:, :-1], ids[:, 1:], m

    def bpb(self, nats: NDArray[np.float64]) -> float:
        """Bits per byte of per-target nats [targets]."""
        return float(np.sum(nats, dtype=np.float64) / (LN2 * self.bytes))

    def windows(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Each target's document and window [targets], in reading order (int32).

        Window k of a document holds its targets kT to kT + T - 1, counted from 0.
        """
        doc = np.repeat(np.arange(self.docs, dtype=np.int32), self.doc_targets)
        first = np.repeat(np.cumsum(self.doc_targets) - self.doc_targets, self.doc_targets)
        return doc, ((np.arange(self.targets) - first) // self.T).astype(np.int32)


def row_nats(
    params: Params,
    cfg: ModelConfig,
    ids: NDArray[np.int64],
    targets: NDArray[np.int64],
    dtype: mx.Dtype = mx.float32,
    term: Term | None = None,
) -> NDArray[np.float64]:
    """Nats [B, T] of every target of the rows ids, targets [B, T] (float64).

    `term(ids)` gives the table term t [B, T, d]; None scores without it, which equals every
    tl = 0.
    """
    x = mx.array(ids.astype(np.int32))
    seg = mx.array(segment_ids(ids.astype(np.int32)))
    h = hidden(params, cfg, x, seg, dtype, None if term is None else term(ids))
    z = head_logits(params, cfg, h, masked_id_mask(cfg))
    y = mx.array(targets.astype(np.int32))
    # By hand over the head's FP32 logits, since `eval` imports nothing from `train`
    # (docs/architecture.md); a unit test holds this equal to `akeratos/train/loop.py`'s
    # `token_ce`.
    n = mx.logsumexp(z, axis=-1) - mx.take_along_axis(z, y[..., None], axis=-1)[..., 0]
    return np.array(n).astype(np.float64)


def score(
    params: Params,
    cfg: ModelConfig,
    held: HeldOut,
    B: int,
    dtype: mx.Dtype = mx.float32,
    term: Term | None = None,
) -> NDArray[np.float64]:
    """Per-target nats [held.targets] in reading order (float64), as `row_nats` gives them."""
    nats = [row_nats(params, cfg, ids, tg, dtype, term)[m > 0] for ids, tg, m in held.batches(B)]
    return np.concatenate(nats)


def table_term_of(table: Table, d: int, dtype: mx.Dtype) -> Term:
    """The table term t [B, T, d] of input ids [B, T]: keys, working set, term, as in training."""

    def term(ids: NDArray[np.int64]) -> mx.array:
        ws = table.gather(table.keys(ids))
        return table_term(ws.W, ws.idx, d, dtype)

    return term


def no_write_back() -> int:
    """The store's memory counter, read only around a write-back, which a readout never makes.

    `eval` calls none of MLX's memory counters, which belong to `akeratos/_compat/mlx_seam.py`.
    """
    raise AkeratosError("the held-out readout never writes the table back")


def load_table(c: ckpt.Checkpoint, cfg: ModelConfig) -> Table:
    """The run's table: its configuration from the run's state, its rows from `TABLE_DIR`.

    `Table.load` checks `table.json` against that configuration; a table with no update event
    yet holds only the record and loads as zero rows.
    """
    saved: dict[str, Any] | None = c.state["config"].get("table")
    if not saved:
        raise ConfigError(f"{c.path} holds no table; pass --table off for the table-off readout")
    try:
        keys: dict[str, Any] = dict(saved)
        keys["orders"] = tuple(keys["orders"])
        tcfg = TableConfig(**keys).validate()
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{c.path}: the run's table configuration: {e}") from e
    if tcfg.width != cfg.d_model:
        raise ConfigError(f"{c.path}: table width {tcfg.width}, the model's d {cfg.d_model}")
    path = c.path / TABLE_DIR
    try:
        return Table.load(path, tcfg, no_write_back)
    except (AkeratosError, ValueError) as e:  # ValueError: a table.json that is not JSON
        raise ConfigError(f"{path}: {e}") from e


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.eval.heldout",
        description=DESCRIPTION,
    )
    add_flags(p)
    return p


class Opened(NamedTuple):
    """A checkpoint opened for a readout: its weights by name (raw, and ema when kept)."""

    c: ckpt.Checkpoint
    cfg: ModelConfig
    dtype: mx.Dtype
    table: Table | None
    term: Term | None
    weights: dict[str, Params]


def scoring_config(cfg: ModelConfig) -> ModelConfig:
    """The configuration a readout scores with: `attn.path` fused becomes banded.

    The fused kernels are a training path with the same mask, so runs differ only in their
    weights, never in how they are scored.
    """
    return replace(cfg, attn_path="banded") if cfg.attn_path == "fused" else cfg


def open_ckpt(path: Path, table: str = "on", compute_dtype: str | None = None) -> Opened:
    """The checkpoint `export.checkpoint` opens at `path`, with its table unless `table` off.

    `path` is a checkpoint, a run directory, whose newest complete checkpoint is used, or an
    export. No complete checkpoint or export, or a table that cannot be read, raises
    `ConfigError`; an export whose files are not the ones its `export.SUMS` lists raises
    `DataMismatchError`.
    """
    c = export.checkpoint(path)
    config = c.state["config"]
    cfg = scoring_config(from_keys(config["model"]))
    tab = load_table(c, cfg) if cfg.table and table == "on" else None
    dtype = DTYPES[compute_dtype or config["compute_dtype"]]
    term = None if tab is None else table_term_of(tab, cfg.d_model, dtype)
    weights = {"raw": c.arrays("dense")}
    if (c.path / "ema.safetensors").is_file():
        weights["ema"] = c.arrays("ema")
    return Opened(c, cfg, dtype, tab, term, weights)


def opened_record(o: Opened) -> dict[str, Any]:
    """The record's checkpoint fields: path, step, table and dtype."""
    rec: dict[str, Any] = {"ckpt": str(o.c.path), "step": o.c.step}
    rec |= {"table": "off" if o.table is None else "on", "dtype": str(o.dtype)}
    if o.table is not None:
        rec["table_event"] = o.table.event
    return rec


def make_parent_dirs(*paths: Path | None) -> None:
    """Make the folder of each output file before any work, so a bad path costs no scoring.

    A None path, an output not asked for, is skipped.
    """
    for p in filter(None, paths):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            folder = p.is_dir()
        except OSError as e:  # a file in the way, or a place this process cannot write
            raise ConfigError(f"{p}: its folder cannot be made ({e})") from e
        if folder:
            raise ConfigError(f"{p} is a folder, not a file")


def _base_mask(cfg: ModelConfig) -> ModelConfig:
    """`cfg` with the reserved ids 1-16 added to its masked ids, as pretraining masks them.

    A fine-tune that unmasked ids 1 and 2 is then scored over the same softmax as its base model.
    """
    return replace(cfg, masked_ids=tuple(sorted({*cfg.masked_ids, *MASKED_IDS})))


def _write_records(
    out: Path, held: HeldOut, scored: dict[str, NDArray[np.float64]]
) -> dict[str, str]:
    """Write the `RECORDS` files into `out`; returns each file's SHA-256 by name.

    Each scored set (`raw`, `ema`, and with the table on `raw.table_off` and `ema.table_off`) is
    a field of both files. The documents file has a row per document read: its targets, its bytes
    and its nats under each set, summed in float64. The targets file has a row per target in
    reading order: its document, its window (`HeldOut.windows`) and its nats under each set.
    """
    doc, window = held.windows()
    per_doc: dict[str, NDArray[Any]] = {"targets": held.doc_targets, "bytes": held.doc_bytes}
    per_target: dict[str, NDArray[Any]] = {"document": doc, "window": window}
    for name, nats in scored.items():
        per_doc[name] = np.bincount(doc, weights=nats, minlength=held.docs)
        per_target[name] = nats.astype(np.float32)  # exact: the head's nats are float32
    sha: dict[str, str] = {}
    for file, cols in zip(RECORDS, (per_doc, per_target), strict=True):
        np.save(out / file, np.rec.fromarrays(list(cols.values()), names=list(cols)))
        sha[file] = hashlib.sha256((out / file).read_bytes()).hexdigest()
    return sha


def _readout(a: argparse.Namespace) -> dict[str, Any]:
    make_parent_dirs(a.out)
    if a.records is not None:
        if any((a.records / f).exists() for f in RECORDS):
            raise ConfigError(f"{a.records} holds records already; pass another --records")
        try:
            a.records.mkdir(parents=True, exist_ok=True)
        except OSError as e:  # a file, or a place this process cannot write
            raise ConfigError(f"{a.records}: {e}; pass another --records") from e
    o = open_ckpt(a.ckpt, a.table, a.compute_dtype)
    cfg = _base_mask(o.cfg) if a.mask == "base" else o.cfg
    dtype, table, term, weights = o.dtype, o.table, o.term, o.weights
    tokens = None if a.tokens is None else int(a.tokens)
    held = HeldOut(a.data, a.seq_len, tokens, cfg.doc_mask, vocab=cfg.vocab_size)
    rec = opened_record(o)
    rec |= {"data": str(a.data), "seq_len": a.seq_len}
    rec |= {"docs": held.docs, "targets": held.targets, "bytes": held.bytes}
    scored: dict[str, NDArray[np.float64]] = {}
    # With the table on, each set of weights is scored again without its table term (`table_off`:
    # no table term at any layer).
    for name, p in weights.items():
        nats = scored[name] = score(p, cfg, held, a.rows, dtype, term)
        rec[name] = {"bpb": held.bpb(nats), "nats_per_target": float(nats.mean(dtype=np.float64))}
        if table is not None:
            off = scored[f"{name}.table_off"] = score(p, cfg, held, a.rows, dtype)
            rec[name]["table_off"] = {"bpb": held.bpb(off)}
    if a.mask == "base":
        rec["mask"] = "base"
    if a.records is not None:
        rec["records"] = {"dir": str(a.records), "sha256": _write_records(a.records, held, scored)}
    return rec


def main(argv: Sequence[str] | None = None) -> int:
    """Score the checkpoint and print one JSON record; returns the exit code."""
    return report(_parser().parse_args(argv))


def report(a: argparse.Namespace) -> int:
    """Score the checkpoint the parsed flags name and print one JSON record; the exit code."""
    try:
        rec = _readout(a)
    except ConfigError as e:
        sys.stderr.write(f"configuration error: {e}\n")
        return EXIT_CONFIG
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA
    line = json.dumps(rec, sort_keys=True)
    if a.out is not None:
        a.out.write_text(line + "\n")
    sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
