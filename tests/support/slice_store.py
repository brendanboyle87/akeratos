"""A small data store in `scripts/data/prepare_store.py`'s layout, for the speedrun's tests.

`write_store` writes what `speedrun/slices.py` reads from the real store: two training shards,
a one-shard held-out split with its `.bytes` sidecar, `token_bytes.u16`, the split manifests
with each shard's SHA-256, `SHA256SUMS`, and a root manifest that records the split manifests'
and `token_bytes.u16`'s SHA-256s, the dataset and tokenizer revisions and the licence. Two of
the store's other files are there too, which a cut must leave out. The ids are SmolLM2's
49,152 by default, so the product's readers open the store with that vocabulary; a
smaller `vocab` suits a tiny model. The driver's tests need more training documents than the
packer's buffer of 1,000, so `train_docs` sets each training shard's count.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

VOCAB = 49_152
DATASET = ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9")
TOKENIZER = ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
SPDX = "ODC-By-1.0 AND LicenseRef-CommonCrawl-ToU"
ATTRIBUTION = "Contains information from FineWeb-Edu (the fixture's notice)."
FINGERPRINT = "f1" * 32
TRAIN_SHARDS = 2


class Store(NamedTuple):
    """A fixture store: its folder, its pin, its held-out documents and its token bytes."""

    path: Path
    pin: str  # the SHA-256 of its manifest.json
    heldout: list[NDArray[np.int64]]
    tb: NDArray[np.uint16]


def sha256(data: bytes) -> str:
    """The SHA-256 of `data`, in hex."""
    return hashlib.sha256(data).hexdigest()


def canon(value: object) -> bytes:
    """A manifest as `scripts/data/prepare_store.py` writes it: sorted keys, no spaces, ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def token_table(vocab: int = VOCAB) -> NDArray[np.uint16]:
    """Each id's length in bytes: 1 to 4, and 0 for ids 0-16."""
    tb = np.random.default_rng(0).integers(1, 5, size=vocab).astype("<u2")
    tb[:17] = 0
    return tb


def documents(seed: int, n: int, long: int = 0, vocab: int = VOCAB) -> list[NDArray[np.int64]]:
    """`n` documents of 3 to 40 ids from 17 up, then `long` documents of 2,100 ids."""
    rng = np.random.default_rng(seed)
    docs = [rng.integers(17, vocab, size=int(rng.integers(3, 41))) for _ in range(n)]
    return docs + [rng.integers(17, vocab, size=2_100) for _ in range(long)]


def _shard(
    folder: Path, name: str, docs: list[NDArray[np.int64]], tb: NDArray[np.uint16], sidecar: bool
) -> dict[str, Any]:
    """Write shard `name`'s `.bin` and `.idx` (and `.bytes`); returns its manifest entries."""
    body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
    idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
    nbytes = np.array([int(tb[d].sum()) for d in docs], "<u8")
    (folder / f"{name}.bin").write_bytes(body.tobytes())
    (folder / f"{name}.idx").write_bytes(idx.tobytes())
    if sidecar:
        (folder / f"{name}.bytes").write_bytes(nbytes.tobytes())
    return {
        "tokens": len(body),
        "documents": len(docs),
        "utf8_bytes": int(nbytes.sum()),
        "max_id": int(body.max()),
        "sha256_bin": sha256(body.tobytes()),
        "sha256_idx": sha256(idx.tobytes()),
        "end_cursor": None if sidecar else {"file": "f.parquet", "row": 0, "row_group": int(name)},
    }


def _split(folder: Path, shards: list[list[NDArray[np.int64]]], tb: NDArray[np.uint16]) -> bytes:
    """Write a split and its manifest, with the held-out fields when it is `heldout`."""
    folder.mkdir(parents=True)
    held = folder.name == "heldout"
    names = [f"{i:06d}" for i in range(len(shards))]
    entries = [_shard(folder, n, docs, tb, held) for n, docs in zip(names, shards, strict=True)]
    m: dict[str, Any] = {"dtype": "uint16", "eos": 0, "vocab": len(tb), "split": folder.name}
    m |= {"format": "metalforge-a21-standin.split/v0", "sep_id": 0, "sep_position": "start"}
    m |= {"reserved_ids": [1, 16], "shards": [f"{n}.bin" for n in names]}
    m["idx"] = [f"{n}.idx" for n in names]
    for key in entries[0]:
        m[key] = [e[key] for e in entries]
    if held:
        docs, toks = sum(m["documents"]), sum(m["tokens"])
        m |= {"bytes": [f"{n}.bytes" for n in names], "documents_total": docs}
        m |= {"tokens_total": toks, "target_tokens_total": toks - docs}
        m["utf8_bytes_total"] = sum(m["utf8_bytes"])
    raw = canon(m)
    (folder / "manifest.json").write_bytes(raw)
    return raw


def write_store(root: Path, train_docs: int = 60, vocab: int = VOCAB) -> Store:
    """Write the fixture store into the new folder `root`."""
    tb = token_table(vocab)
    heldout = documents(1, 40, long=2, vocab=vocab)
    shards = [documents(10 + i, train_docs, vocab=vocab) for i in range(TRAIN_SHARDS)]
    train = _split(root / "train", shards, tb)
    held = _split(root / "heldout", [heldout], tb)
    (root / "token_bytes.u16").write_bytes(tb.tobytes())
    (root / "fingerprint.json").write_bytes(canon({"fingerprint": FINGERPRINT}))
    (root / "train" / "000000.json").write_bytes(canon({"index": 0, "path": "train/000000.bin"}))
    outputs = {"token_bytes_sha256": sha256(tb.tobytes()), "fingerprint": FINGERPRINT}
    outputs |= {"train_manifest_sha256": sha256(train), "heldout_manifest_sha256": sha256(held)}
    manifest = canon(
        {
            "source": {"repo_id": DATASET[0], "revision": DATASET[1], "sample": "sample/10BT"},
            "tokenizer": {"repo_id": TOKENIZER[0], "revision": TOKENIZER[1]},
            "licence": {"spdx_expected": SPDX, "attribution_notice": ATTRIBUTION},
            "outputs": outputs,
        }
    )
    (root / "manifest.json").write_bytes(manifest)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    sums = "".join(f"{sha256(p.read_bytes())}  {p.relative_to(root).as_posix()}\n" for p in files)
    (root / "SHA256SUMS").write_text(sums)
    return Store(root, sha256(manifest), heldout, tb)
