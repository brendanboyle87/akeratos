"""S6's memorisation split: training documents Billet read in each third of training.

It replays Billet's training order with the packer itself, picks `PER_THIRD` documents from each
third, and writes them as a held-out split that akeratos/eval/heldout.py scores unchanged, each
arm's final checkpoint in turn. scripts/analysis/analysis.py reads those records and thirds.npy and
reports Δ on these documents beside the held-out Δ, and each arm's training − held-out gap by third.

The --out folder, which must not exist, gets `OUTPUTS`: heldout/, one shard in the store's split
format (akeratos/data/store.py) with its manifest; a copy of the store's token_bytes.u16 with the
manifest.json the scorer checks it against; and thirds.npy, int64 [documents], 0, 1 or 2 in the
split's order. split.json, written last, records the command, the code, the library versions, each
input's SHA-256, the order, the replay's counts, the selection and each output's SHA-256. An input
that fails a check is refused before anything is written. It reads no score, and imports NumPy, the
standard library and akeratos, which imports MLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from array import array
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from akeratos import ckpt
from akeratos.data.packer import ExhaustedError, Packer
from akeratos.data.store import Split
from akeratos.errors import AkeratosError
from akeratos.eval.heldout import token_bytes
from akeratos.rng import derive, seed_hex
from akeratos.train.source import PackedSource

EXIT_REFUSED = 2
STORE = "9545f2d205efd574"  # Billet's store: how its fingerprint starts (analysis.py's STORE)
SEED = 20_261_101  # analysis.py's SEED; the selection's keys derive from it
# Billet's training order: its command's --rows, --seq-len and --accum (docs/reproduce.md),
# akeratos/train/entry.py's default --buffer and one epoch, its 24,415 steps, and the data seed
# that the default --seed 0 derives (`run_seeds` in akeratos/rng.py: derive(0, "data")).
ORDER = {
    "rows": 2,
    "seq_len": 2_048,
    "accum": 32,
    "buffer": 1_000,
    "steps": 24_415,
    "data_seed": 0x84D17AD2E2342315,
}
PER_THIRD = 2_438  # 7,314 documents in all, as in the held-out split
REUSED = (
    "ckpt.py",
    "rng.py",
    "data/packer.py",
    "data/store.py",
    "eval/heldout.py",
    "train/source.py",
)
OUTPUTS = (
    "heldout/000000.bin",
    "heldout/000000.idx",
    "heldout/000000.bytes",
    "heldout/manifest.json",
    "token_bytes.u16",
    "manifest.json",
    "thirds.npy",
)


class RefusedError(Exception):
    """An input failed a check."""


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def billet_ckpt(path: Path) -> tuple[ckpt.Checkpoint, dict[str, Any]]:
    """Billet's final checkpoint and its saved data cursor, once checked."""
    c = ckpt.load(path)
    config, saved = c.state["config"], dict(c.state["source"])
    if c.step != ORDER["steps"] or saved.get("kind") != "packed":
        raise RefusedError(
            f"{path}: step {c.step} of a {saved.get('kind')!r} source, not Billet's final"
        )
    if (
        config["data_seed"] != seed_hex(ORDER["data_seed"])
        or int(config["accum"]) != ORDER["accum"]
        or not isinstance(config["table"], dict)
        or config["model"]["doc_mask"] is not True
    ):
        raise RefusedError(
            f"{path}: data seed {config['data_seed']}, accum {config['accum']}, table "
            f"{'on' if isinstance(config['table'], dict) else 'off'}, document mask "
            f"{config['model']['doc_mask']}: not Billet's"
        )
    return c, saved


def store(root: Path, vocab: int) -> tuple[Split, np.ndarray, dict[str, str]]:
    """Billet's training split and token bytes, once checked, and the SHA-256 of each file read.

    The store must be Billet's by its fingerprint and manifests, every training shard (.bin and
    .idx) must have its manifest's SHA-256, and token_bytes.u16 must pass the scorer's `token_bytes`
    check.
    """
    fingerprint = json.loads((root / "fingerprint.json").read_text())["fingerprint"]
    outputs = json.loads((root / "manifest.json").read_text())["outputs"]
    if not str(fingerprint).startswith(STORE) or outputs["fingerprint"] != fingerprint:
        raise RefusedError(f"{root}: not store {STORE} by fingerprint.json and manifest.json")
    train = root / "train"
    named = (root / "fingerprint.json", root / "manifest.json", train / "manifest.json")
    files = {str(p): sha256(p) for p in named}
    if files[str(train / "manifest.json")] != outputs["train_manifest_sha256"]:
        raise RefusedError(
            f"{train / 'manifest.json'}: not the one the store's manifest.json names"
        )
    m = json.loads((train / "manifest.json").read_text())
    names, bins, idxs = m["shards"], m.get("sha256_bin") or [], m.get("sha256_idx") or []
    if not len(bins) == len(idxs) == len(names):
        raise RefusedError(
            f"{train / 'manifest.json'}: no sha256_bin and sha256_idx for every shard"
        )
    for name, b, x in zip(names, bins, idxs, strict=True):
        for f, want in ((train / name, b), (train / f"{name[:-4]}.idx", x)):
            files[str(f)] = sha256(f)
            if files[str(f)] != want:
                raise RefusedError(f"{f}: differs from its manifest's SHA-256")
    tb = token_bytes(root, vocab)
    files[str(root / "token_bytes.u16")] = sha256(root / "token_bytes.u16")
    return Split(train, vocab=vocab), tb, files


class Logged:
    """The training split as the packer's documents, logging each read: its document and tokens."""

    def __init__(self, split: Split) -> None:
        self.split, self.lengths = split, split.lengths
        self.doc, self.taken = array("q"), array("q")

    def read(self, index: int, count: int) -> np.ndarray:
        """The split's first `count` tokens of document `index`, logged."""
        self.doc.append(index)
        self.taken.append(count)
        return self.split.read(index, count)


def replay(split: Split, vocab: int, saved: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Each document Billet's training read, in order: its index, its tokens taken and its third.

    The packer (akeratos/data/packer.py) rebuilds the order over the store's training split with
    the run's settings and runs as the run ran: every micro-batch of every step, each row read from
    the split and checked, while `Logged` keeps each document read and how many of its tokens its
    row took. The packer's fingerprint (the documents' lengths, the data seed, the buffer and the
    batch shape) must equal the checkpoint's, and its end state the checkpoint's data cursor (epoch,
    cursor, buffer, tokens read and used), as akeratos/train/reads.py requires of its replay.
    """
    o, log = ORDER, Logged(split)
    packer = Packer(
        log, o["rows"], o["seq_len"], vocab=vocab, data_seed=o["data_seed"], buffer=o["buffer"]
    )
    if packer.fingerprint != saved.get("fingerprint"):
        raise RefusedError(
            "the packer's fingerprint (lengths, data seed, buffer, rows, T) is not the checkpoint's"
        )
    source = PackedSource(packer)
    ends = np.empty(o["steps"] * o["accum"], np.int64)  # documents logged by each micro-batch's end
    for m in range(len(ends)):
        source.next()
        ends[m] = len(log.doc)
    if source.state() != saved:
        raise RefusedError("the replay does not end at the checkpoint's data cursor")
    doc, taken = np.frombuffer(log.doc, np.int64), np.frombuffer(log.taken, np.int64)
    step = np.searchsorted(ends, np.arange(len(doc)), side="right") // o["accum"]
    # Step s, counted from 0, is in third ⌊3s/24,415⌋: steps 0-8,138, 8,139-16,276 and
    # 16,277-24,414. A document's third is that of the step whose micro-batches hold its row.
    return doc, taken, 3 * step // o["steps"]


def select(taken: np.ndarray, third: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The split's documents as log positions, in training order, and each third's candidates.

    From each third, `PER_THIRD` documents are drawn without replacement and with equal chances from
    those whose row took at least 2 tokens (one target), by `Generator.choice` on a Philox generator
    keyed by `derive(SEED, "s6", third)` (akeratos/rng.py). Sorted, the picks keep training order,
    so the thirds are contiguous.
    """
    picked, candidates = [], []
    for j in range(3):
        cand = np.flatnonzero((third == j) & (taken >= 2))
        candidates.append(len(cand))
        if len(cand) < PER_THIRD:
            raise RefusedError(
                f"third {j}: {len(cand)} documents with a target, fewer than {PER_THIRD}"
            )
        gen = np.random.Generator(np.random.Philox(key=derive(SEED, "s6", j)))
        picked.append(np.sort(gen.choice(cand, size=PER_THIRD, replace=False)))
    return np.concatenate(picked), candidates


def write(
    out: Path, root: Path, vocab: int, tb: np.ndarray, parts: list[np.ndarray], thirds: np.ndarray
) -> dict[str, str]:
    """Write the split, the store files the scorer reads with it and thirds.npy; their SHA-256s."""
    held = out / "heldout"
    held.mkdir(parents=True)
    body = np.concatenate(parts).astype("<u2")
    body.tofile(held / "000000.bin")
    np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype("<u8").tofile(
        held / "000000.idx"
    )
    # A part's bytes: token_bytes.u16 summed over its ids after the 0, the sum the scorer checks.
    np.array([int(tb[p[1:]].sum()) for p in parts], "<u8").tofile(held / "000000.bytes")
    m: dict[str, Any] = {"dtype": "uint16", "eos": 0, "sep_position": "start", "vocab": vocab}
    m |= {
        "shards": ["000000.bin"],
        "tokens": [len(body)],
        "documents": [len(parts)],
        "bytes": ["000000.bytes"],
    }
    m |= {f"sha256_{k}": [sha256(held / f"000000.{k}")] for k in ("bin", "idx", "bytes")}
    (held / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    shutil.copyfile(root / "token_bytes.u16", out / "token_bytes.u16")
    named = {
        "token_bytes_sha256": sha256(out / "token_bytes.u16"),
        "heldout_manifest_sha256": sha256(held / "manifest.json"),
    }
    root_manifest = {"what": "S6's memorisation split of store " + STORE, "outputs": named}
    (out / "manifest.json").write_text(json.dumps(root_manifest, indent=1) + "\n")
    np.save(out / "thirds.npy", thirds.astype(np.int64))
    return {n: sha256(out / n) for n in OUTPUTS}


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
    """Check the inputs, replay the order, then write the split; returns the exit code."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", type=Path, required=True, help=f"the data store, ~/mf-data/{STORE}")
    p.add_argument("--ckpt", type=Path, required=True, help="Billet's final checkpoint folder")
    p.add_argument("--out", type=Path, required=True, help="the new folder for the split")
    a = p.parse_args(argv)
    started, t0 = datetime.now(UTC), time.monotonic()
    try:
        if a.out.exists():
            raise RefusedError(f"{a.out}: already exists")
        c, saved = billet_ckpt(a.ckpt)
        vocab = int(c.state["config"]["model"]["vocab_size"])
        split, tb, files = store(a.store, vocab)
        files[str(a.ckpt / ckpt.STATE)] = sha256(a.ckpt / ckpt.STATE)
        t1 = time.monotonic()
        doc, taken, third = replay(split, vocab, saved)
        t2 = time.monotonic()
        picked, candidates = select(taken, third)
    except (
        RefusedError,
        AkeratosError,
        ExhaustedError,
        OSError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as e:
        sys.stderr.write(f"s6: refused: {e}\n")
        return EXIT_REFUSED
    # A part is a document as its row took it: its first `taken` tokens, the leading id 0 included.
    # The packer crops a document to fill a row and discards the rest, so a cropped document was
    # trained on its head only, and only its head is scored. A part holds at most 2,049 tokens, so
    # the scorer reads it in one row as window 0, with the context training gave it: packing moves
    # only its absolute positions, which cancel in attention up to rounding.
    parts = [split.read(int(doc[i]), int(taken[i])) for i in picked]
    outputs = write(a.out, a.store, vocab, tb, parts, third[picked])
    n, cropped = ORDER["steps"], taken < split.lengths[doc]
    record = {
        "what": "S6: training documents Billet read in each third of training, as a held-out split",
        "command": sys.orig_argv,
        "cwd": str(Path.cwd()),
        "started": started.isoformat(timespec="seconds"),
        "seconds": {
            "checks": round(t1 - t0, 1),
            "replay": round(t2 - t1, 1),
            "write": round(time.monotonic() - t2, 1),
        },
        "code": code(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "inputs": files,
        "store": {"path": str(a.store), "vocab": vocab},
        "checkpoint": {"path": str(a.ckpt), "step": c.step},
        "order": ORDER
        | {
            "data_seed": seed_hex(ORDER["data_seed"]),
            "epochs": 1,
            "fingerprint": saved["fingerprint"],
        },
        "replay": {
            "micro_batches": n * ORDER["accum"],
            "documents": len(doc),
            "cropped": int(cropped.sum()),
            "end": {k: saved[k] for k in ("epoch", "cursor", "read", "used")},
        },
        "selection": {
            "seed": SEED,
            "stream": "s6",
            "steps": [[-(-j * n // 3), -(-(j + 1) * n // 3) - 1] for j in range(3)],
            "candidates": candidates,
            "per_third": PER_THIRD,
        },
        "split": {
            "documents": len(parts),
            "targets": sum(len(q) - 1 for q in parts),
            "bytes": int(sum(int(tb[q[1:]].sum()) for q in parts)),
            "cropped": int(cropped[picked].sum()),
        },
        "outputs": outputs,
    }
    (a.out / "split.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    sys.stdout.write(json.dumps(record["split"]) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
