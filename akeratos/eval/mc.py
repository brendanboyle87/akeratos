"""Zero-shot multiple-choice scores over a saved checkpoint, on Akeratos's own forward pass.

The five tasks, their strings, the token split and the metrics come from
`akeratos/eval/tasks.py`, which the peer baselines share, so both score the same items. The
eval set is prepared by `scripts/data/prepare_evals.py`; nothing here downloads. The text is
encoded with the tokenizer the data store was encoded with.

The raw weights and, when the run kept one, the EMA are scored. A model with the table on is
scored with its table term and without it (`table_off`), as the held-out readout does. `main`
prints one JSON record, and can also write it to a file and write per-item results as JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from akeratos import export
from akeratos.configs import ModelConfig, from_keys
from akeratos.data.packer import RESERVED, SEP_ID, segment_ids
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval import tasks
from akeratos.eval.heldout import (
    DTYPES,
    EXIT_CONFIG,
    EXIT_DATA,
    Term,
    load_table,
    make_parent_dirs,
    scoring_config,
    table_term_of,
)
from akeratos.eval.tasks import Item
from akeratos.model import Params, head_logits, hidden, masked_id_mask

PAD_TO = 64
Scored = tuple[float, bool]  # (Σ log p over the continuation, every target greedy)
MAX_TOKENIZER_BYTES = 32 << 20


def encoder(store: str | Path) -> tuple[Callable[[str], list[int]], str]:
    """(encode, SHA-256): the store's own tokenizer, checked against its root manifest.

    The tokenizer is the one under `<store>/tokenizer/`, the merge-prefix cut when the store has
    one. It encodes with `encode_special_tokens` on and adds no special tokens.
    """
    root = Path(store)
    try:
        t = json.loads((root / "manifest.json").read_text())["tokenizer"]
        cut = t.get("merge_prefix")
        name = "tokenizer.json" if cut is None else f"tokenizer.prefix-{cut}.json"
        want = {e["path"]: e["sha256"] for e in t["files"]}[name]
        raw = (root / "tokenizer" / name).read_bytes()
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise DataMismatchError(f"{root}: no store tokenizer to encode with ({e!r})") from None
    sha = hashlib.sha256(raw).hexdigest()
    if len(raw) > MAX_TOKENIZER_BYTES or sha != want:
        raise DataMismatchError(
            f"{root}/tokenizer/{name}: SHA-256 {sha} differs from the manifest's"
        )
    tok = Tokenizer.from_buffer(raw)
    tok.encode_special_tokens = True  # `from_buffer` leaves it False (tokenizers 0.23.3)
    if not tok.encode_special_tokens:
        raise DataMismatchError(f"{root}: the tokenizer would match special-token text")

    def encode(s: str) -> list[int]:
        return list(tok.encode(s, add_special_tokens=False).ids)

    return encode, sha


def requests_of(
    items: Sequence[Item], encode: Callable[[str], Sequence[int]], max_len: int, vocab: int
) -> list[tuple[list[int], int]]:
    """Every item's requests in order, as (ids, n) with the prefix [0] (`tasks.encode_pair`).

    Each request is one sequence [0] + ctx + cont, with the id 0 that opens every training
    document first, cut from the left to `max_len` + 1 tokens. A continuation that does not fit in
    `max_len` raises `DataMismatchError`.
    """
    out: list[tuple[list[int], int]] = []
    for it in items:
        for q in it["requests"]:
            try:
                ids, n = tasks.encode_pair(
                    encode, q["context"], q["continuation"], [SEP_ID], max_len
                )
            except ValueError as e:
                raise DataMismatchError(f"{it['task']} item {it['doc_id']}: {e}") from None
            body = np.asarray(ids[1:] if ids[0] == SEP_ID else ids, np.int64)
            if (body <= RESERVED[1]).any() or (body >= vocab).any():
                msg = f"{it['task']} item {it['doc_id']}: an id 0-16 or >= {vocab} in a request"
                raise DataMismatchError(msg)
            out.append((ids, n))
    return out


def loglikelihoods(
    params: Params,
    cfg: ModelConfig,
    seqs: Sequence[tuple[Sequence[int], int]],
    B: int,
    dtype: mx.Dtype = mx.float32,
    term: Term | None = None,
) -> list[Scored]:
    """(Σ log p, greedy) of each sequence's last n targets, in the order given.

    A sequence is greedy when every one of those targets is the logits' argmax. The logits are the
    head's FP32 ones with ids 1-16 masked, and log p = z[y] - logsumexp(z), as the held-out readout
    scores, so `eval` imports nothing from `train`. Sequences run in batches of `B`, longest
    first, padded with zeros to a multiple of `PAD_TO`. The padding sits after each sequence's last
    token and opens segments of its own, so it changes no scored position.
    """
    masked = masked_id_mask(cfg)
    oracle = replace(cfg, attn_path="masked")
    out: list[Scored | None] = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][0]))
    for a in range(0, len(order), B):
        rows = order[a : a + B]
        L = max(len(seqs[i][0]) - 1 for i in rows)
        P = -(-L // PAD_TO) * PAD_TO
        ids = np.zeros((B, P + 1), np.int64)
        for r, i in enumerate(rows):
            ids[r, : len(seqs[i][0])] = seqs[i][0]
        x, tg = ids[:, :-1], ids[:, 1:]
        # A padded length that is not a multiple of the window runs on the full-mask path
        # (`attn.path` masked), which has the banded path's semantics at any length.
        run = oracle if cfg.window is not None and P % cfg.window else cfg
        h = hidden(
            params,
            run,
            mx.array(x.astype(np.int32)),
            mx.array(segment_ids(x.astype(np.int32))),
            dtype,
            # The gather sees the padding too; padding opens its own segments, so no scored
            # position reads it, and the table is never written here.
            None if term is None else term(x),
        )
        z = head_logits(params, run, h, masked)
        y = mx.array(tg.astype(np.int32))
        logp = mx.take_along_axis(z, y[..., None], axis=-1)[..., 0] - mx.logsumexp(z, axis=-1)
        hit = mx.argmax(z, axis=-1) == y
        lp, ok = np.array(logp).astype(np.float64), np.array(hit).astype(np.bool_)
        for r, i in enumerate(rows):
            end = len(seqs[i][0]) - 1
            s = slice(end - seqs[i][1], end)
            out[i] = (float(lp[r, s].sum()), bool(ok[r, s].all()))
    return [o for o in out if o is not None]


def score_set(
    params: Params,
    cfg: ModelConfig,
    items: dict[str, list[Item]],
    reqs: dict[str, list[tuple[list[int], int]]],
    B: int,
    dtype: mx.Dtype,
    term: Term | None = None,
) -> tuple[dict[str, Any], dict[str, list[list[Scored]]]]:
    """Each task's metrics and its per-item, per-request results."""
    metrics: dict[str, Any] = {}
    per: dict[str, list[list[Scored]]] = {}
    for t, its in items.items():
        flat = loglikelihoods(params, cfg, reqs[t], B, dtype, term)
        res: list[list[Scored]] = []
        k = 0
        for it in its:
            res.append(flat[k : k + len(it["requests"])])
            k += len(it["requests"])
        metrics[t], per[t] = tasks.aggregate(t, its, res), res
    return metrics, per


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.eval.mc",
        description="Zero-shot multiple-choice scores over a checkpoint.",
    )
    p.add_argument("--ckpt", type=Path, required=True, help="a checkpoint, run directory or export")
    p.add_argument("--set", type=Path, required=True, help="a prepared set: <root>/<set_id>")
    p.add_argument(
        "--store", type=Path, required=True, help="the data store whose tokenizer the run used"
    )
    p.add_argument(
        "--tasks", nargs="+", choices=sorted(tasks.TASKS), default=None, help="default: all"
    )
    p.add_argument("--seq-len", type=int, default=2048, help="the most tokens of context (T)")
    p.add_argument("--rows", type=int, default=8, help="sequences per forward pass (B)")
    p.add_argument(
        "--decontam", type=Path, default=None, help="leave out what <task>.decontam.json drop"
    )
    p.add_argument("--limit", type=int, default=None, help="score the first N items of each task")
    p.add_argument(
        "--compute-dtype", choices=tuple(DTYPES), default=None, help="default: the run's"
    )
    p.add_argument("--table", choices=("on", "off"), default="on", help="off: no table term")
    p.add_argument("--out", type=Path, default=None, help="also write the record here")
    p.add_argument("--samples", type=Path, default=None, help="write per-item results (JSONL)")
    return p


def _set_items(
    a: argparse.Namespace, tok_sha: str
) -> tuple[dict[str, object], dict[str, list[Item]], dict[str, object] | None]:
    """The manifest, the items to score after `--decontam` and `--limit`, and the decontam record.

    `--decontam` leaves items out before `--limit` cuts each task to its first N. The record holds
    both, so a record made with `--limit`, or without `--decontam`, cannot back a published score.
    """
    manifest, items = tasks.read_set(a.set, a.tasks)
    decontam = None
    if a.decontam is not None:
        fp = a.store / "fingerprint.json"
        try:
            fingerprint = json.loads(fp.read_text())["fingerprint"] if fp.is_file() else None
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise DataMismatchError(f"{fp}: no readable fingerprint ({e})") from None
        items, decontam = tasks.read_decontam(a.decontam, manifest, items, tok_sha, fingerprint)
    if a.limit is not None:
        items = {t: its[: a.limit] for t, its in items.items()}
    return manifest, items, decontam


def _readout(a: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    make_parent_dirs(a.out, a.samples)
    c = export.checkpoint(a.ckpt)
    config = c.state["config"]
    cfg = scoring_config(from_keys(config["model"]))
    encode, tok_sha = encoder(a.store)
    manifest, items, decontam = _set_items(a, tok_sha)
    reqs = {t: requests_of(its, encode, a.seq_len, cfg.vocab_size) for t, its in items.items()}
    table = load_table(c, cfg) if cfg.table and a.table == "on" else None
    dtype = DTYPES[a.compute_dtype or config["compute_dtype"]]
    term = None if table is None else table_term_of(table, cfg.d_model, dtype)
    weights = {"raw": c.arrays("dense")}
    if (c.path / "ema.safetensors").is_file():
        weights["ema"] = c.arrays("ema")
    rec: dict[str, Any] = {"ckpt": str(c.path), "step": c.step, "set": str(a.set)}
    rec |= {"set_id": manifest.get("set_id"), "store": str(a.store), "tokenizer_sha256": tok_sha}
    rec |= {"seq_len": a.seq_len, "limit": a.limit, "decontam": decontam, "dtype": str(dtype)}
    rec |= {"table": "off" if table is None else "on", "shots": 0, "prefix": [SEP_ID]}
    if table is not None:
        rec["table_event"] = table.event
    samples: list[dict[str, Any]] = []
    runs = [("table" if table is not None else "off", term)] + (
        [("table_off", None)] if table else []
    )
    for name, p in weights.items():
        rec[name] = {}
        for kind, tm in runs:
            m, per = score_set(p, cfg, items, reqs, a.rows, dtype, tm)
            if kind == "table_off":
                rec[name]["table_off"] = m
            else:
                rec[name] |= m
            for t, its in items.items():
                for it, res in zip(its, per[t], strict=True):
                    row = {"task": t, "doc_id": it["doc_id"], "weights": name, "term": kind}
                    samples.append(row | {"ll": [r[0] for r in res], "greedy": [r[1] for r in res]})
    return rec, samples


def main(argv: Sequence[str] | None = None) -> int:
    """Score the checkpoint and print one JSON record; returns the exit code."""
    a = _parser().parse_args(argv)
    try:
        rec, samples = _readout(a)
    except ConfigError as e:
        sys.stderr.write(f"configuration error: {e}\n")
        return EXIT_CONFIG
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA
    line = json.dumps(rec, sort_keys=True)
    if a.out is not None:
        a.out.write_text(line + "\n")
    if a.samples is not None:
        a.samples.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in samples))
    sys.stdout.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
