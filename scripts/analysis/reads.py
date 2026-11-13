"""S3's read counts: how often training read the table rows each held-out target is predicted from.

A target's read count is the training read count of the bigram row plus that of the trigram row that
the table reads at the input position predicting the target, from Billet's final `table/reads.npy`,
which counts every lookup in every micro-batch of the run's 24,415 steps (`Table.gather`,
akeratos/table/table.py). scripts/analysis/analysis.py splits Δ = bpb(Billet) − bpb(twin) into
deciles of held-out targets by these counts.

The rows are the ones akeratos/eval/heldout.py's `HeldOut` builds when it scores Billet, keyed as
the scorer keys them (`Table.keys`), so no n-gram reaches back past its document's start or its
window's. The counts come in the scorer's reading order: each row's unmasked targets, row by row.

It writes reads.npy, int64 [targets], and reads.json: the command, the code, the library versions,
each input's SHA-256, the checkpoint and its table, the held-out counts and reads.npy's SHA-256. An
input that fails a check is refused before anything is written. It reads no score, and imports
NumPy, the standard library and akeratos, which imports MLX.
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

from akeratos import ckpt
from akeratos.errors import AkeratosError
from akeratos.eval.heldout import TABLE_DIR, HeldOut
from akeratos.configs import from_keys
from akeratos.table.config import TableConfig
from akeratos.table.hasher import keys, multipliers
from akeratos.table.table import READS, Table, load_reads

EXIT_REFUSED = 2
STORE = "9545f2d205efd574"  # Billet's store: how its fingerprint starts (analysis.py's STORE)
STEPS = 24_415  # Billet's steps; its final checkpoint is ckpt-00024415
TOKENS_PER_STEP = 131_072  # Billet's --rows 2 x --seq-len 2048 x --accum 32
ORDERS = (2, 3)  # Billet's table: a bigram and a trigram row per token, one head
SEQ_LEN = 2_048  # the scorer's T, as analysis.py requires of the records
CHUNK = 256  # rows keyed at a time
REUSED = (
    "ckpt.py",
    "rng.py",
    "data/packer.py",
    "data/store.py",
    "eval/heldout.py",
    "configs.py",
    "table/config.py",
    "table/hasher.py",
    "table/table.py",
)


class RefusedError(Exception):
    """An input failed a check."""


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def store_files(split: Path) -> dict[str, str]:
    """Each file of the held-out split and store root it reads, by SHA-256, once checked.

    The store must be Billet's by its fingerprint and manifests, and every held-out shard (.bin and
    .idx) must have its manifest's SHA-256.
    """
    root = split.parent
    fingerprint = json.loads((root / "fingerprint.json").read_text())["fingerprint"]
    outputs = json.loads((root / "manifest.json").read_text())["outputs"]
    if not str(fingerprint).startswith(STORE) or outputs["fingerprint"] != fingerprint:
        raise RefusedError(f"{root}: not store {STORE} by fingerprint.json and manifest.json")
    named = (root / "fingerprint.json", root / "manifest.json", root / "token_bytes.u16")
    files = {str(p): sha256(p) for p in (*named, split / "manifest.json")}
    if files[str(split / "manifest.json")] != outputs["heldout_manifest_sha256"]:
        raise RefusedError(
            f"{split / 'manifest.json'}: not the one the store's manifest.json names"
        )
    m = json.loads((split / "manifest.json").read_text())
    names, bins, idxs = m["shards"], m.get("sha256_bin") or [], m.get("sha256_idx") or []
    if not len(bins) == len(idxs) == len(names):
        raise RefusedError(
            f"{split / 'manifest.json'}: no sha256_bin and sha256_idx for every shard"
        )
    for name, b, x in zip(names, bins, idxs, strict=True):
        for f, want in ((split / name, b), (split / f"{name[:-4]}.idx", x)):
            files[str(f)] = sha256(f)
            if files[str(f)] != want:
                raise RefusedError(f"{f}: differs from its manifest's SHA-256")
    files |= {str(split / n): sha256(split / n) for n in m.get("bytes") or []}
    return files


def billet_table(path: Path) -> tuple[ckpt.Checkpoint, TableConfig, dict[str, Any], np.ndarray]:
    """Billet's final checkpoint with its table configuration, record and read counts, checked.

    The table's configuration, read as the scorer reads it, must give its `table.json`, and the read
    counts must be non-negative and sum to every lookup of every step.
    """
    c = ckpt.load(path)
    if c.step != STEPS:
        raise RefusedError(f"{path}: step {c.step}, not Billet's final step {STEPS}")
    saved = c.state["config"]["table"]
    if not isinstance(saved, dict):
        raise RefusedError(f"{path}: its run had no table")
    kw = dict(saved)  # as akeratos/eval/heldout.py's load_table reads it
    kw["orders"] = tuple(kw["orders"])
    cfg = TableConfig(**kw).validate()
    if cfg.orders != ORDERS or cfg.heads != 1:
        raise RefusedError(f"{path}: orders {cfg.orders}, {cfg.heads} heads; Billet's: {ORDERS}, 1")
    meta = json.loads((path / TABLE_DIR / "table.json").read_text())
    event = meta.get("event")
    if not isinstance(event, int) or isinstance(event, bool) or meta != Table.record(cfg, event):
        raise RefusedError(
            f"{path / TABLE_DIR / 'table.json'}: not the table its configuration gives"
        )
    counts = load_reads(path / TABLE_DIR / READS, cfg.rows())
    want = STEPS * TOKENS_PER_STEP * cfg.lookups()
    if (counts < 0).any() or int(counts.sum()) != want:
        raise RefusedError(f"{path / TABLE_DIR / READS}: not {want:,} non-negative reads in all")
    return c, cfg, meta, counts


def per_target(held: HeldOut, cfg: TableConfig, counts: np.ndarray) -> np.ndarray:
    """Each target's summed read count, in the scorer's reading order: row by row, unmasked only."""
    mults = multipliers(cfg.seed, cfg.orders, cfg.heads)
    out = []
    for a in range(0, len(held.ids), CHUNK):
        rows = keys(held.ids[a : a + CHUNK, :-1], cfg.orders, mults, cfg.rows_per_slot(), cfg.heads)
        out.append(counts[rows].sum(axis=-1)[held.mask[a : a + CHUNK] > 0])
    return np.concatenate(out)


def code() -> dict[str, Any]:
    """The checkout's commit and changes, and the SHA-256s of this file and the modules it reuses.

    The changes are the lines `git status --porcelain` prints. Outside a git checkout, the commit
    and changes are None.
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
    pkg = Path(ckpt.__file__).resolve().parent
    mods = {f"akeratos/{m}": sha256(pkg / m) for m in REUSED}
    return {"commit": commit, "uncommitted": changes, here.name: sha256(here)} | mods


def main(argv: Sequence[str] | None = None) -> int:
    """Check the inputs, then write reads.npy and reads.json; returns the exit code."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True, help="Billet's final checkpoint folder")
    p.add_argument("--split", type=Path, required=True, help=f"~/mf-data/{STORE}/heldout")
    p.add_argument("--out", type=Path, required=True, help="the folder for the two files")
    a = p.parse_args(argv)
    started, t0 = datetime.now(UTC), time.monotonic()
    try:
        if a.out.is_file() or any((a.out / f).exists() for f in ("reads.npy", "reads.json")):
            raise RefusedError(f"{a.out}: a file, or already holds reads.npy or reads.json")
        files = store_files(a.split)
        c, cfg, meta, counts = billet_table(a.ckpt)
        for f in (
            a.ckpt / ckpt.STATE,
            a.ckpt / TABLE_DIR / "table.json",
            a.ckpt / TABLE_DIR / READS,
        ):
            files[str(f)] = sha256(f)
        model = from_keys(c.state["config"]["model"])
        held = HeldOut(a.split, SEQ_LEN, None, model.doc_mask, vocab=model.vocab_size)
        reads = per_target(held, cfg, counts)
    except (
        RefusedError,
        AkeratosError,
        OSError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        sys.stderr.write(f"reads: refused: {e}\n")
        return EXIT_REFUSED
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "reads.npy", reads)
    record = {
        "what": "S3: each held-out target's training reads of its bigram and trigram rows",
        "command": sys.orig_argv,
        "cwd": str(Path.cwd()),
        "started": started.isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t0, 1),
        "code": code(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs": files,
        "checkpoint": {
            "path": str(a.ckpt),
            "step": c.step,
            "table_event": meta["event"],
            "rows": cfg.rows(),
            "orders": list(cfg.orders),
            "heads": cfg.heads,
            "mult_seed": meta["mult_seed"],
            "reads": int(counts.sum()),
        },
        "heldout": {
            "split": str(a.split),
            "seq_len": SEQ_LEN,
            "doc_mask": model.doc_mask,
            "documents": held.docs,
            "targets": held.targets,
            "bytes": held.bytes,
            "rows": len(held.ids),
        },
        "result": {"targets": len(reads), "min": int(reads.min()), "max": int(reads.max())},
        "outputs": {"reads.npy": sha256(a.out / "reads.npy")},
    }
    (a.out / "reads.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    sys.stdout.write(json.dumps(record["result"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
