"""Decontamination: token 13-gram overlap of eval and SFT items with the training store.

The check runs in token ids at the store's own tokenizer, so it compares exactly what the model
was trained on. An n-gram is n consecutive ids of one item segment, or of one shard of the
store's `train/` split. A window that holds an id from 0 to 16 is skipped on both sides; in a
shard only id 0 occurs, at each document's start, so no counted n-gram spans a document
boundary. An item is overlapping when any of its n-grams occurs in a shard, and it is dropped
whole. An SFT conversation is also checked against every eval item's n-grams.

Each set gets one record, `<out>/<set>.decontam.json`, laid out as `records` says. This module
imports NumPy, the standard library and `akeratos.data` only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from akeratos.data.packer import RESERVED
from akeratos.data.store import Split
from akeratos.errors import DataMismatchError

N = 13
ITEMS_SCHEMA = "metalforge.decontam-items/v1"
SCHEMAS = {
    "sft": ("metalforge.sft-decontam/v1", "conversations_sha256"),
    "eval": ("metalforge.eval-decontam/v1", "task_sha256"),
}
MATCHES_SHOWN = 3
CHUNK = 1 << 24
LOW = RESERVED[1] + 1  # ids below this (0 and the reserved 1-16) never sit in a counted window
SET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
EXIT_CONFIG, EXIT_DATA = 78, 22

_P = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier of the polynomial hash
_M1 = np.uint64(0xBF58476D1CE4E5B9)  # splitmix64's finaliser, so the top bits are well mixed
_M2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True)
class ItemSet:
    """One tokenised set: its header fields and, per item, its id and segments."""

    name: str
    kind: str
    source: str
    source_sha256: str
    tokenizer_sha256: str
    ids: list[str | int]
    segments: list[list[NDArray[np.uint16]]]


def _hash(windows: NDArray[np.uint16]) -> NDArray[np.uint64]:
    """64-bit hashes of the rows of an (m, n) window matrix; arithmetic wraps mod 2**64."""
    h = np.zeros(windows.shape[0], np.uint64)
    for j in range(windows.shape[1]):
        h = h * _P + windows[:, j].astype(np.uint64)
    h ^= h >> np.uint64(30)
    h *= _M1
    h ^= h >> np.uint64(27)
    h *= _M2
    return h ^ (h >> np.uint64(31))


def _windows(seq: NDArray[np.uint16], n: int) -> NDArray[np.uint16]:
    """The (len - n + 1, n) windows of `seq` that hold no id below LOW."""
    if len(seq) < n:
        return np.zeros((0, n), np.uint16)
    w = np.lib.stride_tricks.sliding_window_view(seq, n)
    return w[w.min(axis=1) >= LOW]


@dataclass(frozen=True)
class Index:
    """The items' distinct n-grams, sorted by hash, and which item owns each.

    `keys[u]` is the n-gram with hash `hashes[u]`; `pair_key` and `pair_item` list each
    (n-gram, item) pair once, items numbered across the sets in order, and each item's n-grams in
    the order they first occur in it.
    """

    n: int
    hashes: NDArray[np.uint64]
    keys: NDArray[np.uint16]
    pair_key: NDArray[np.int64]
    pair_item: NDArray[np.int64]
    checked: NDArray[np.bool_]
    bits: int
    bitmap: NDArray[np.uint8]


def build_index(sets: Sequence[ItemSet], n: int = N) -> Index:
    """Index every n-gram of every item of `sets`."""
    windows: list[NDArray[np.uint16]] = []
    owners: list[NDArray[np.int64]] = []
    item = 0
    for s in sets:
        for segs in s.segments:
            for seg in segs:
                w = _windows(seg, n)
                windows.append(w)
                owners.append(np.full(len(w), item, np.int64))
            item += 1
    allw = np.concatenate(windows) if windows else np.zeros((0, n), np.uint16)
    owner = np.concatenate(owners) if owners else np.zeros(0, np.int64)
    h = _hash(allw)
    order = np.argsort(h, kind="stable")
    h, allw, owner = h[order], allw[order], owner[order]
    first = np.ones(len(h), np.bool_)
    first[1:] = h[1:] != h[:-1]
    if (~first).any():
        dup = np.flatnonzero(~first)
        if (allw[dup] != allw[dup - 1]).any():
            raise DataMismatchError("two different item n-grams share a 64-bit hash")
    key_of = np.empty(len(h), np.int64)
    key_of[order] = np.cumsum(first) - 1  # back in item order
    pair_item = owner[np.argsort(order)]
    _, keep = np.unique(key_of * max(item, 1) + pair_item, return_index=True)
    keep = np.sort(keep)  # each (n-gram, item) once, by first position in the item
    checked = np.zeros(item, np.bool_)
    checked[owner] = True
    bits = min(32, max(16, math.ceil(math.log2(max(1, int(first.sum())))) + 5))
    bitmap = np.zeros(1 << (bits - 3), np.uint8)
    hashes = h[first]
    slot = (hashes >> np.uint64(64 - bits)).astype(np.int64)
    np.bitwise_or.at(bitmap, slot >> 3, (1 << (slot & 7)).astype(np.uint8))
    return Index(
        n,
        hashes,
        allw[first],
        key_of[keep],
        pair_item[keep],
        checked,
        bits,
        bitmap,
    )


@dataclass
class Scan:
    """Per distinct n-gram of an index: its hits in the shards and its first (shard, document)."""

    hits: NDArray[np.int64]
    shard: NDArray[np.int64]
    document: NDArray[np.int64]
    shards: int


def _chunks(tokens: int, n: int, chunk: int) -> Iterator[tuple[int, int]]:
    """(start, end) token ranges whose windows start in [start, end - n + 1) and cover a shard."""
    for a in range(0, max(tokens - n + 1, 0), chunk):
        yield a, min(a + chunk + n - 1, tokens)


def _match(idx: Index, ids: NDArray[np.uint16]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The positions in `ids` where a counted window equals an index key, and that key.

    Each window is hashed to 64 bits; a bitmap of the items' hashes filters the windows, a binary
    search finds the candidates' keys, and every hit is confirmed by comparing the n ids
    themselves, so a hash collision cannot drop an item.
    """
    n = idx.n
    w = np.lib.stride_tricks.sliding_window_view(ids, n)
    h = _hash(w)
    slot = (h >> np.uint64(64 - idx.bits)).astype(np.int64)
    pos = np.flatnonzero((idx.bitmap[slot >> 3] >> (slot & 7)) & 1)
    if len(pos) == 0 or len(idx.hashes) == 0:
        return pos, pos
    low = np.concatenate([[0], np.cumsum(ids < LOW)])
    pos = pos[low[pos + n] == low[pos]]
    u = np.minimum(np.searchsorted(idx.hashes, h[pos]), len(idx.hashes) - 1)
    keep = idx.hashes[u] == h[pos]
    pos, u = pos[keep], u[keep]
    same = (w[pos] == idx.keys[u]).all(axis=1)
    return pos[same], u[same].astype(np.int64)


def scan(idx: Index, split: Split, shards: int | None = None, chunk: int = CHUNK) -> Scan:
    """Stream the first `shards` shards of `split` (all when None) against `idx`, in chunks."""
    k = len(split.shards) if shards is None else min(shards, len(split.shards))
    out = Scan(
        np.zeros(len(idx.hashes), np.int64),
        np.full(len(idx.hashes), -1, np.int64),
        np.full(len(idx.hashes), -1, np.int64),
        k,
    )
    for s in range(k):
        data = split.shards[s]
        starts = split.doc_start[split.doc_shard == s]
        for a, b in _chunks(len(data), idx.n, chunk):
            pos, u = _match(idx, np.asarray(data[a:b]))
            if len(u) == 0:
                continue
            out.hits += np.bincount(u, minlength=len(idx.hashes))
            ukeys, first = np.unique(u, return_index=True)
            new = out.shard[ukeys] < 0
            ukeys, first = ukeys[new], first[new]
            out.shard[ukeys] = s
            out.document[ukeys] = np.searchsorted(starts, a + pos[first], side="right") - 1
    return out


def store_meta(store: Path, split: Split, sc: Scan) -> dict[str, Any]:
    """The store as a record names it: path, fingerprint, split, shards scanned and sizes."""
    fp = store / "fingerprint.json"
    names: list[str] = json.loads((split.path / "manifest.json").read_text())["shards"]
    return {
        "path": str(store),
        "fingerprint": json.loads(fp.read_text())["fingerprint"] if fp.is_file() else None,
        "split": split.path.name,
        "shards": len(split.shards),
        "shards_scanned": names[: sc.shards],
        "tokens": split.tokens,
        "documents": len(split.doc_start),
    }


def _flat(s: ItemSet) -> tuple[NDArray[np.uint16], NDArray[np.int64], NDArray[np.int64]]:
    """The set's segments joined, each after an id 0, with each segment's start and its item."""
    segs = [seg for item in s.segments for seg in item]
    owner = np.array([i for i, item in enumerate(s.segments) for _ in item], np.int64)
    lengths = np.array([len(seg) + 1 for seg in segs], np.int64)
    starts = np.cumsum(lengths) - lengths
    sep = np.zeros(1, np.uint16)
    flat = np.concatenate([np.concatenate([sep, seg]) for seg in segs] or [sep])
    return flat, starts, owner


def sft_against_eval(sets: Sequence[ItemSet], n: int = N) -> dict[str, dict[int, list[Any]]]:
    """Per SFT set, its conversations holding an eval item's n-gram, and the eval items matched.

    The SFT text is training data for the post-trained model, so a conversation that quotes an
    eval item is dropped too; the eval set stays the same for the base and the SFT model.
    """
    evals = [s for s in sets if s.kind == "eval"]
    out: dict[str, dict[int, list[Any]]] = {}
    if not evals:
        return out
    idx = build_index(evals, n)
    names = [(s.name, item_id) for s in evals for item_id in s.ids]
    owners: dict[int, list[int]] = {}
    for key, item in zip(idx.pair_key.tolist(), idx.pair_item.tolist(), strict=True):
        owners.setdefault(key, []).append(item)
    for s in (s for s in sets if s.kind == "sft"):
        flat, starts, owner = _flat(s)
        found: dict[int, list[Any]] = {}
        for a, b in _chunks(len(flat), n, CHUNK):
            pos, u = _match(idx, flat[a:b])
            items = owner[np.searchsorted(starts, a + pos, side="right") - 1]
            for item, key in zip(items.tolist(), u.tolist(), strict=True):  # in text order
                matched = found.setdefault(item, [])
                for e in owners[key]:
                    hit = {"set": names[e][0], "id": names[e][1], "ngram": idx.keys[key].tolist()}
                    if all((m["set"], m["id"]) != (hit["set"], hit["id"]) for m in matched):
                        matched.append(hit)
        out[s.name] = found
    return out


def records(
    sets: Sequence[ItemSet],
    idx: Index,
    sc: Scan,
    store: dict[str, Any],
    against_eval: dict[str, dict[int, list[Any]]] | None = None,
) -> list[dict[str, Any]]:
    """One drop record per set, in the order of `sets`.

    A record's schema and the hash it is bound by are its kind's entry in `SCHEMAS`; the SFT
    record is the drop list post-training reads. Both kinds carry `drop` (the dropped ids in the
    items file's order) and `report`: the n, the matching rule, the store, the tokenizer, the
    counts, and per overlapping item its hit count and up to `MATCHES_SHOWN` matched n-grams, each
    with its shard and document (the document's index within its shard, from 0). An item with no
    n-gram (every segment shorter than n) cannot be checked; it is kept and counted as
    `unchecked`. A scan of only some shards writes `"complete": false`. `against_eval` (from
    `sft_against_eval`) adds each SFT set's conversations that quote an eval item to its `drop`,
    listed under `report.eval_overlaps`.
    """
    total = sum(len(s.ids) for s in sets)
    item_hits = np.bincount(idx.pair_item, weights=sc.hits[idx.pair_key], minlength=total)
    shown: dict[int, list[int]] = {}
    hit = sc.hits[idx.pair_key] > 0
    for key, item in zip(idx.pair_key[hit].tolist(), idx.pair_item[hit].tolist(), strict=True):
        if len(shown.setdefault(item, [])) < MATCHES_SHOWN:
            shown[item].append(key)
    names: list[str] = store["shards_scanned"]
    out: list[dict[str, Any]] = []
    base = 0
    for s in sets:
        schema, bind = SCHEMAS[s.kind]
        overlaps: list[dict[str, Any]] = []
        for i, item_id in enumerate(s.ids):
            if item_hits[base + i] > 0:
                matches = [
                    {
                        "ngram": idx.keys[k].tolist(),
                        "shard": names[int(sc.shard[k])],
                        "document": int(sc.document[k]),
                    }
                    for k in shown[base + i]
                ]
                overlaps.append(
                    {"id": item_id, "hits": int(item_hits[base + i]), "matches": matches}
                )
        report: dict[str, Any] = {
            "set": s.name,
            "kind": s.kind,
            "source": s.source,
            "n": idx.n,
            "match": (
                f"exact {idx.n}-grams of token ids at the store's tokenizer, within one segment"
                f" or one shard; windows holding an id below {LOW} are skipped, so none spans a"
                " document boundary; an item with any hit is dropped whole; an eval item is"
                " checked as its gold request's context plus continuation; an SFT conversation"
                " is checked against the store and against every eval item"
            ),
            "tokenizer_sha256": s.tokenizer_sha256,
            "store": store,
            "complete": len(names) == store["shards"],
            "items": len(s.ids),
            "unchecked": int((~idx.checked[base : base + len(s.ids)]).sum()),
            "overlaps": overlaps,
        }
        cross = (against_eval or {}).get(s.name, {}) if s.kind == "sft" else {}
        if s.kind == "sft":
            report["eval_overlaps"] = [{"id": s.ids[i], "eval": cross[i]} for i in sorted(cross)]
        dropped = {o["id"] for o in overlaps} | {s.ids[i] for i in cross}
        drop = [item_id for item_id in s.ids if item_id in dropped]
        report["dropped"] = len(drop)
        out.append({"schema": schema, bind: s.source_sha256, "drop": drop, "report": report})
        base += len(s.ids)
    return out


def _header(head: object, path: Path) -> dict[str, Any]:
    if not isinstance(head, dict):
        raise DataMismatchError(f"{path}: line 1 is not a {ITEMS_SCHEMA} header")
    h = cast("dict[str, Any]", head)
    if h.get("schema") != ITEMS_SCHEMA:
        raise DataMismatchError(f"{path}: line 1 is not a {ITEMS_SCHEMA} header")
    if h.get("kind") not in SCHEMAS or not SET_NAME.fullmatch(str(h.get("set", ""))):
        raise DataMismatchError(f"{path}: bad kind {h.get('kind')!r} or set {h.get('set')!r}")
    for k in ("source_sha256", "tokenizer_sha256"):
        if not SHA256.fullmatch(str(h.get(k, ""))):
            raise DataMismatchError(f"{path}: {k} is not a SHA-256")
    return h


def _item(line: object, vocab: int, where: str) -> tuple[str | int, list[NDArray[np.uint16]]]:
    rec = cast("dict[str, Any]", line) if isinstance(line, dict) else {}
    segments: object = rec.get("segments")
    if not isinstance(segments, list):
        raise DataMismatchError(f"{where}: not an item line")
    item_id = rec.get("id")
    if not isinstance(item_id, str | int) or isinstance(item_id, bool):
        raise DataMismatchError(f"{where}: the id is not a string or an integer")
    segs: list[NDArray[np.uint16]] = []
    for seg in cast("list[object]", segments):
        try:
            a = np.asarray(seg, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError) as e:
            raise DataMismatchError(f"{where}: a segment that is not a list of ids") from e
        if len(a) and (a.min() < 0 or a.max() >= vocab):
            raise DataMismatchError(f"{where}: an id outside [0, {vocab})")
        segs.append(a.astype(np.uint16))
    return item_id, segs


def read_items(path: Path, vocab: int, fingerprint: str | None) -> ItemSet:
    """Read one items file, checking its header, its ids and that it names `fingerprint`.

    An items file (`ITEMS_SCHEMA`, JSON Lines) holds one tokenised set, as
    `scripts/data/decontam_items.py` writes it with the store's tokenizer. Line 1 is the header:
    `schema`, `set` (a file-name-safe name), `kind` ("sft" or "eval"), `source` (the file it
    read), `source_sha256`, `tokenizer_sha256`, `store_fingerprint` and `items` (the count). Each
    later line is one item: `id` (a string or an integer, unique in the set) and `segments`, lists
    of token ids. An n-gram never spans two segments: an SFT conversation has one segment per
    message content, and an eval item one for its gold request's context plus continuation, so
    wrong choices in the training text do not leak the answer.
    """
    try:
        text = path.read_text(encoding="utf-8")
        # Split on "\n" alone: str.splitlines also breaks at U+2028 and other separators that
        # a JSON string may hold unescaped.
        lines = [json.loads(line) for line in text.removesuffix("\n").split("\n")] if text else []
    except ValueError as e:  # not UTF-8, or a line that is not JSON
        raise DataMismatchError(f"{path}: {e}") from e
    if not lines:
        raise DataMismatchError(f"{path}: empty")
    h = _header(lines[0], path)
    if fingerprint is not None and h.get("store_fingerprint") != fingerprint:
        msg = f"{path}: tokenised for store {h.get('store_fingerprint')!r}, not {fingerprint!r}"
        raise DataMismatchError(msg)
    ids: list[str | int] = []
    segments: list[list[NDArray[np.uint16]]] = []
    for i, line in enumerate(lines[1:], start=2):
        item_id, segs = _item(line, vocab, f"{path}:{i}")
        ids.append(item_id)
        segments.append(segs)
    if len(set(ids)) != len(ids) or len(ids) != h.get("items"):
        raise DataMismatchError(f"{path}: repeated ids, or not the header's {h.get('items')} items")
    return ItemSet(
        str(h["set"]),
        str(h["kind"]),
        str(h.get("source", "")),
        str(h["source_sha256"]),
        str(h["tokenizer_sha256"]),
        ids,
        segments,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.data.decontam",
        description="Token n-gram overlap of eval and SFT items with a store's train split.",
    )
    p.add_argument("--store", type=Path, required=True, help="the data store (its train/ is read)")
    p.add_argument("--items", type=Path, nargs="+", required=True, help="items files, one per set")
    p.add_argument("--out", type=Path, required=True, help="the folder for the records")
    p.add_argument("--n", type=int, default=N, help="n-gram length in tokens")
    p.add_argument("--shards", type=int, default=None, help="scan only the first K shards")
    return p


def _run(a: argparse.Namespace) -> dict[str, Any]:
    split = Split(a.store / "train")
    fp = a.store / "fingerprint.json"
    fingerprint = json.loads(fp.read_text())["fingerprint"] if fp.is_file() else None
    sets = [read_items(p, split.vocab, fingerprint) for p in a.items]
    if len({s.name for s in sets}) != len(sets):
        raise DataMismatchError("two items files name the same set")
    idx = build_index(sets, a.n)
    sc = scan(idx, split, a.shards)
    recs = records(sets, idx, sc, store_meta(a.store, split, sc), sft_against_eval(sets, a.n))
    a.out.mkdir(parents=True, exist_ok=True)
    for s, rec in zip(sets, recs, strict=True):
        (a.out / f"{s.name}.decontam.json").write_text(json.dumps(rec, indent=1) + "\n")
    return {
        "complete": recs[0]["report"]["complete"],
        "shards_scanned": sc.shards,
        "sets": {
            s.name: [r["report"]["items"], r["report"]["dropped"]]
            for s, r in zip(sets, recs, strict=True)
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Check the items against the store, write one record per set; returns the exit code."""
    a = _parser().parse_args(argv)
    if a.n < 1 or (a.shards is not None and a.shards < 1):
        sys.stderr.write("configuration error: --n and --shards must be at least 1\n")
        return EXIT_CONFIG
    try:
        summary = _run(a)
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
