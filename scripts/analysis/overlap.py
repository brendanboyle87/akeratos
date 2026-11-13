"""M8's scan: the 13-grams each held-out document shares with the store's training split.

    .venv/bin/python -I -B docs/research/billet/overlap.py --store ~/mf-data/9545f2d205efd574
        --out DIR

(one line, from a checkout with its own venv). M1, the analysis plan posted on #235 on
2026-11-01, splits the primary Δ into held-out documents with and without a 13-gram in the
store's training split, the former also by quartile of overlap share (the share of their
distinct 13-grams found there), set from this scan alone. The training split is every shard of
the store's train/, not only the text a run trained on. The matching is
`metalforge.data.decontam`'s, unchanged: exact 13-grams of token ids; a window holding an id
below 17 is skipped, so none spans a document boundary; every hit is confirmed id by id. Each
held-out document is one item of one segment: its ids after the leading 0, the text the scorer
scores.

It refuses a DIR that already holds overlap.npy or overlap.json, so a sample cannot replace a
full scan. Before it scans, it checks:
- that the store is Billet's: fingerprint.json's fingerprint starts 9545f2d205efd574, as
  analysis.py requires, and the store's manifest.json names that fingerprint and the SHA-256 of
  each split manifest;
- every shard file it will read (.bin and .idx) against its split manifest's SHA-256;
- that the two splits have the same vocab, and that id 0 opens each held-out document and
  occurs nowhere else.
A failed check exits 2 and writes nothing. It reads no score, model or run folder.

DIR gets two files:
- overlap.npy: one row per held-out document, in the split's order (the scorer's), with int64
  fields `tokens` (its ids after the leading 0), `distinct` (its distinct 13-grams), `found`
  (how many of those occur in the training split) and `hits` (their occurrences there,
  summed). docs/research/billet/analysis.py reads `distinct` and `found`.
- overlap.json: the command; the code (the checkout's commit; `uncommitted`, the lines
  `git status --porcelain` prints, an empty list when the checkout is clean; and the SHA-256s
  of this file, decontam.py, store.py and packer.py, whose RESERVED sets the 17); the store's
  fingerprint and the SHA-256s of fingerprint.json, manifest.json and both split manifests; the
  times; the counts; and M1's quartile edges of the share found ÷ distinct over the documents
  with any 13-gram found. Edge j of 4 is the ⌈jn/4⌉-th smallest share, and a share equal to an
  edge falls in the quartile below, as in analysis.py.

`--shards K` scans only the first K training shards, to time a sample; overlap.json then says
`"complete": false`. It imports NumPy, the standard library, `metalforge.data` and
`metalforge.errors` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from metalforge.data import decontam
from metalforge.data.decontam import ItemSet, build_index, scan
from metalforge.data.store import Split
from metalforge.errors import DataMismatchError

FIELDS = np.dtype([("tokens", "<i8"), ("distinct", "<i8"), ("found", "<i8"), ("hits", "<i8")])
EXIT_REFUSED = 2
STORE = "9545f2d205efd574"  # Billet's store: how its fingerprint starts (analysis.py's STORE)
STORE_FILES = ("fingerprint.json", "manifest.json", "heldout/manifest.json", "train/manifest.json")


class RefusedError(Exception):
    """An input failed a check."""


def sha256(path: Path) -> str:
    """The file's SHA-256."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked(path: Path, named: str, shards: int | None = None) -> Split:
    """The split, once its manifest's SHA-256 is `named` and its first `shards` shards match it.

    `named` comes from the store's manifest.json; `shards` None checks every shard.
    """
    if sha256(path / "manifest.json") != named:
        raise RefusedError(f"{path / 'manifest.json'}: not the one the store's manifest.json names")
    split = Split(path)
    m = json.loads((path / "manifest.json").read_text())
    names, bins, idxs = m["shards"], m.get("sha256_bin") or [], m.get("sha256_idx") or []
    if not len(bins) == len(idxs) == len(names):
        raise RefusedError(
            f"{path / 'manifest.json'}: no sha256_bin and sha256_idx for every shard"
        )
    for name, b, x in list(zip(names, bins, idxs, strict=True))[:shards]:
        if sha256(path / name) != b or sha256(path / f"{name[:-4]}.idx") != x:
            raise RefusedError(f"{path / name}: a shard file differs from its manifest's SHA-256")
    return split


def documents(split: Split) -> list[np.ndarray]:
    """Each document's ids after its leading 0, once id 0 opens every document and only there."""
    docs: list[np.ndarray] = []
    for s, data in enumerate(split.shards):
        starts = split.doc_start[split.doc_shard == s]
        ids = np.asarray(data)
        if (ids[starts] != 0).any() or np.count_nonzero(ids == 0) != len(starts):
            raise RefusedError(f"{split.path}: id 0 does not open a document, or occurs inside one")
        ends = [*starts[1:].tolist(), len(ids)]
        docs += [ids[a + 1 : b] for a, b in zip(starts.tolist(), ends, strict=True)]
    return docs


def count(docs: list[np.ndarray], train: Split, shards: int | None) -> tuple[np.ndarray, int]:
    """Each document's row of FIELDS, and how many training shards were scanned."""
    n = len(docs)
    # build_index reads only the segments; the other ItemSet fields belong to items files.
    idx = build_index([ItemSet("heldout", "eval", "", "", "", list(range(n)), [[d] for d in docs])])
    sc = scan(idx, train, shards)
    seen = sc.hits[idx.pair_key]  # per (13-gram, document) pair: the 13-gram's training hits
    rows = np.zeros(n, FIELDS)
    rows["tokens"] = [len(d) for d in docs]
    rows["distinct"] = np.bincount(idx.pair_item, minlength=n)
    rows["found"] = np.bincount(idx.pair_item[seen > 0], minlength=n)
    hits = np.zeros(n, np.int64)
    np.add.at(hits, idx.pair_item, seen)
    rows["hits"] = hits
    return rows, sc.shards


def quartiles(share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """M1's three quartile edges of `share`, and how many values fall in each quartile."""
    n = len(share)
    if n == 0:
        return np.zeros(0), np.zeros(4, np.int64)
    edges = np.sort(share)[[-(-j * n // 4) - 1 for j in (1, 2, 3)]]
    return edges, np.bincount(np.searchsorted(edges, share, side="left"), minlength=4)


def result(rows: np.ndarray) -> dict[str, Any]:
    """The counts, the share of documents with any 13-gram found, and the quartile edges."""
    hit = rows["found"] > 0
    share = rows["found"][hit] / rows["distinct"][hit]
    edges, per = quartiles(share)
    return {
        "documents": len(rows),
        "no_13gram": int((rows["distinct"] == 0).sum()),
        "none_found": int((~hit).sum()),
        "any_found": int(hit.sum()),
        "any_found_share": float(hit.mean()),
        "distinct": int(rows["distinct"].sum()),
        "found": int(rows["found"].sum()),
        "all_found": int((hit & (rows["found"] == rows["distinct"])).sum()),
        "share_min_max": [float(share.min()), float(share.max())] if len(share) else [],
        "quartile_edges": edges.tolist(),
        "quartile_documents": per.tolist(),
    }


def code() -> dict[str, Any]:
    """The checkout's commit and changes, and the SHA-256s of the files that define the match.

    Outside a git checkout, the commit and changes are None.
    """
    here = Path(__file__).resolve()

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(here.parent), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changes = git("status", "--porcelain").splitlines()
    except (OSError, subprocess.CalledProcessError):
        commit, changes = None, None
    d = Path(decontam.__file__)
    files = (here, d, d.with_name("store.py"), d.with_name("packer.py"))
    return {"commit": commit, "uncommitted": changes} | {f.name: sha256(f) for f in files}


def main(argv: Sequence[str] | None = None) -> int:
    """Scan, then write overlap.npy and overlap.json; returns the exit code."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", type=Path, required=True, help=f"the data store, ~/mf-data/{STORE}")
    p.add_argument("--out", type=Path, required=True, help="the folder for the two files")
    p.add_argument("--shards", type=int, help="scan only the first K training shards")
    a = p.parse_args(argv)
    if a.shards is not None and a.shards < 1:
        p.error("--shards must be at least 1")
    started, t0 = datetime.now(UTC), time.monotonic()
    try:
        if any((a.out / f).exists() for f in ("overlap.npy", "overlap.json")):
            raise RefusedError(f"{a.out}: already holds overlap.npy or overlap.json")
        fingerprint = json.loads((a.store / "fingerprint.json").read_text())["fingerprint"]
        outputs = json.loads((a.store / "manifest.json").read_text())["outputs"]
        if not str(fingerprint).startswith(STORE) or outputs["fingerprint"] != fingerprint:
            raise RefusedError(
                f"{a.store}: not store {STORE} by fingerprint.json and manifest.json"
            )
        held = checked(a.store / "heldout", outputs["heldout_manifest_sha256"])
        train = checked(a.store / "train", outputs["train_manifest_sha256"], a.shards)
        if held.vocab != train.vocab:
            raise RefusedError(f"{a.store}: held-out vocab {held.vocab}, training {train.vocab}")
        docs = documents(held)
        t1 = time.monotonic()
        rows, scanned = count(docs, train, a.shards)
    except (RefusedError, DataMismatchError, OSError, KeyError, TypeError, ValueError) as e:
        sys.stderr.write(f"overlap: refused: {e}\n")
        return EXIT_REFUSED
    t2 = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "overlap.npy", rows)
    record = {
        "what": "M8: each held-out document's distinct 13-grams found in the training split",
        "command": sys.orig_argv,
        "cwd": str(Path.cwd()),
        "started": started.isoformat(timespec="seconds"),
        "seconds": {"checks": round(t1 - t0, 1), "index_and_scan": round(t2 - t1, 1)},
        "code": code(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n": decontam.N,
        "store": {
            "path": str(a.store),
            "fingerprint": fingerprint,
            **{f: sha256(a.store / f) for f in STORE_FILES},
        },
        "heldout": {"shards": len(held.shards), "documents": len(docs), "tokens": held.tokens},
        "train": {"shards": len(train.shards), "shards_scanned": scanned, "tokens": train.tokens},
        "complete": scanned == len(train.shards),
        "result": result(rows),
        "outputs": {"overlap.npy": sha256(a.out / "overlap.npy")},
    }
    (a.out / "overlap.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    sys.stdout.write(json.dumps(record["result"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
