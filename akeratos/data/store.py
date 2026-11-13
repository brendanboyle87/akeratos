"""The read side of the data store's split format.

A split directory (`<store>/train/`, `<store>/heldout/`), as the preparation script writes it:
- `NNNNNN.bin`: raw, headerless little-endian uint16 ids. Each document is `[0] + ids`.
- `NNNNNN.idx`: little-endian uint64 document start offsets, N + 1 entries for N documents.
- `manifest.json`: `dtype` "uint16", `eos` 0, `vocab`, `shards`, `tokens`, and per-shard `idx`,
  `documents` and `sha256_bin`/`sha256_idx` where the script records them.

`Split` checks the layout before it maps a byte: every shard name fully matches
`[0-9]{6}.bin`, every file is exactly tokens x 2 bytes, and every `.idx` holds documents + 1
offsets with idx[0] = 0, strictly increasing, idx[-1] = tokens. It does not read the shards: the
packer checks each document it reads (id 0 at its start and nowhere else, ids in range), and the
preparation script checks the SHA-256s. `Split` is the packer's `Documents`. This module imports
NumPy and the standard library only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from akeratos.data.packer import SEP_ID
from akeratos.errors import DataMismatchError

U16 = np.dtype("<u2")
U64 = np.dtype("<u8")
SHARD_NAME = re.compile(r"[0-9]{6}\.bin")


def _check_idx(idx: NDArray[np.uint64], tokens: int, where: Path) -> None:
    starts = idx.astype(np.int64)
    if len(starts) <= 1 or starts[0] != 0 or starts[-1] != tokens or (np.diff(starts) <= 0).any():
        msg = f"{where}: offsets break the rule (idx[0] = 0, strictly increasing, idx[-1] = tokens)"
        raise DataMismatchError(msg)


class Split:
    """A split opened read-only: one memory map per shard and a table of every document."""

    def __init__(self, path: str | Path, vocab: int | None = None) -> None:
        self.path = Path(path)
        m = json.loads((self.path / "manifest.json").read_text())
        if (m.get("dtype"), m.get("eos"), m.get("sep_position", "start")) != (
            "uint16",
            SEP_ID,
            "start",
        ):
            raise DataMismatchError(f"{self.path}: not an id-0-first uint16 split")
        self.vocab = int(m["vocab"])
        if vocab is not None and vocab != self.vocab:
            msg = f"{self.path}: the split's vocabulary is {self.vocab}, the model's {vocab}"
            raise DataMismatchError(msg)
        names: list[str] = m["shards"]
        counts = [int(c) for c in m["tokens"]]
        docs: list[int] | None = m.get("documents")
        self.shards: list[NDArray[np.uint16]] = []
        starts: list[NDArray[np.int64]] = []
        owner: list[NDArray[np.int32]] = []
        for s, (name, count) in enumerate(zip(names, counts, strict=True)):
            idx = self._open_shard(name, count, None if docs is None else int(docs[s]))
            starts.append(idx[:-1].astype(np.int64))
            owner.append(np.full(len(idx) - 1, s, np.int32))
        if not starts:
            raise DataMismatchError(f"{self.path}: no shards")
        self.doc_start = np.concatenate(starts)
        self.doc_shard = np.concatenate(owner)
        ends = np.concatenate([np.append(a[1:], c) for a, c in zip(starts, counts, strict=True)])
        self._lengths = ends - self.doc_start
        self.tokens = sum(counts)

    def _open_shard(self, name: str, count: int, docs: int | None) -> NDArray[np.uint64]:
        bin_path, idx_path = self.path / name, self.path / f"{name[:-4]}.idx"
        if not SHARD_NAME.fullmatch(name):
            raise DataMismatchError(f"{self.path}: shard name {name!r} is not [0-9]{{6}}.bin")
        size = bin_path.stat().st_size
        if size != count * U16.itemsize:
            raise DataMismatchError(f"{bin_path}: {size} bytes, expected {count} x 2")
        idx = np.fromfile(idx_path, dtype=U64)
        if docs is not None and len(idx) != docs + 1:
            raise DataMismatchError(f"{idx_path}: {len(idx)} offsets for {docs} documents")
        _check_idx(idx, count, idx_path)
        self.shards.append(np.memmap(bin_path, dtype=U16, mode="r", shape=(count,)))
        return idx

    @property
    def lengths(self) -> NDArray[np.int64]:
        """Each document's length in tokens, its separator included."""
        return self._lengths

    def read(self, index: int, count: int) -> NDArray[np.int64]:
        """The first `count` tokens of document `index`, or all of a shorter one, as int64."""
        shard, start = int(self.doc_shard[index]), int(self.doc_start[index])
        n = min(count, int(self._lengths[index]))
        return np.asarray(self.shards[shard][start : start + n], dtype=np.int64)
