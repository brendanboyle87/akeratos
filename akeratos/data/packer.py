"""Best-fit, document-aligned packing of documents into training rows.

Rows are T + 1 tokens: the inputs are row[:-1] and the targets row[1:]. Every row starts at a
document start and no token is padding; the cost is the cropped tails. Every document begins
with the separator id 0, which opens a segment; attention stays inside a segment and positions
run on across the row, so the model needs only the segment ids.

The documents come through `Documents`, so the packer does not fix the shard format. This
module imports NumPy, the standard library and MLX-free product modules only.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from akeratos.errors import DataMismatchError
from akeratos.rng import derive

SEP_ID = 0  # `<|endoftext|>`: separates documents and opens a segment
RESERVED = (1, 16)  # reserved ids: never emitted, masked out of the softmax

Ids = NDArray[np.int64]


class ExhaustedError(RuntimeError):
    """Every document has been read as many times as the run's epochs allow."""


class Documents(Protocol):
    """A read-only store of documents, each `[0] + ids`."""

    @property
    def lengths(self) -> Ids:
        """Each document's length in tokens, its separator included."""
        ...

    def read(self, index: int, count: int) -> Ids:
        """The first `count` tokens of document `index`."""
        ...


@dataclass(frozen=True)
class PackedRows:
    """One batch of packed rows, each field int32 [B, T]."""

    inputs: NDArray[np.int32]
    targets: NDArray[np.int32]
    segment_ids: NDArray[np.int32]


class PackerState(TypedDict):
    """The packer's checkpointed state; JSON round trips keep it whole."""

    fingerprint: str  # the documents' lengths, the data seed, the buffer and the batch shape
    epoch: int
    cursor: int
    buffer: list[int]
    read: int
    used: int


Pull = Callable[[], tuple[int, int]]


class _BestFit:
    """The buffer and the row filling, over (document, length) pairs only.

    A row is filled the way nanochat's best-fit loader fills one: keep `size` documents
    buffered; place the longest buffered document that fits the space left in the row, the
    earliest buffered among equal lengths; when none fits, take the shortest buffered document
    (again the earliest among ties) and crop it to fill the row exactly, discarding its tail.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self.keys: list[tuple[int, int, int]] = []  # sorted (length, arrival, document)
        self.arrival = 0

    def add(self, doc: int, length: int) -> None:
        bisect.insort(self.keys, (length, self.arrival, doc))
        self.arrival += 1

    def row(self, cap: int, pull: Pull) -> list[tuple[int, int, int]]:
        """Fill one row of `cap` tokens: [(document, length, taken)] in row order."""
        out: list[tuple[int, int, int]] = []
        left = cap
        while left > 0:
            while len(self.keys) < self.size:
                self.add(*pull())
            i = bisect.bisect_right(self.keys, (left, math.inf)) - 1
            if i >= 0:  # the longest that fits; the earliest buffered among that length
                i = bisect.bisect_left(self.keys, (self.keys[i][0], -1))
                length, _, doc = self.keys.pop(i)
                out.append((doc, length, length))
                left -= length
            else:  # nothing fits: crop the shortest
                length, _, doc = self.keys.pop(0)
                out.append((doc, length, left))
                left = 0
        return out

    def docs(self) -> list[int]:
        """The buffered documents in arrival order."""
        return [doc for _, _, doc in sorted(self.keys, key=lambda key: key[1])]


class Packer:
    """A resumable source of packed batches of `rows` x `seq_len` tokens.

    Epoch e reads every document once, in the order of a permutation drawn from the Philox
    generator keyed by `derive(data_seed, "data", e)`. The whole state is (epoch, cursor, the
    buffered documents in arrival order) and its counters, so a resumed run reads the same
    batches; take it only at an optimiser-step boundary. Reading past the last epoch raises
    `ExhaustedError`; a second epoch must be asked for, and a warning marks the wrap. As in
    nanochat, the buffer refills before each pick, so the last `buffer` - 1 or fewer documents of
    the final epoch are never emitted, and a state taken after `ExhaustedError` matches no batch
    boundary, so it is never checkpointed.

    Ids 1-16 are reserved. A plain pretraining run admits none of them in a document; a run that
    uses some, such as post-training with its chat markers, names them in `allow_reserved`. A
    document that breaks the id rules raises `DataMismatchError`.
    """

    def __init__(
        self,
        docs: Documents,
        rows: int,
        seq_len: int,
        *,
        vocab: int,
        data_seed: int = 0,
        buffer: int = 1000,
        epochs: int = 1,
        allow_reserved: frozenset[int] = frozenset(),
        state: PackerState | None = None,
    ) -> None:
        self.docs = docs
        self.lengths = np.asarray(docs.lengths, dtype=np.int64)
        if len(self.lengths) == 0:
            raise ValueError("no documents to pack")
        self.rows, self.seq_len, self.vocab = rows, seq_len, vocab
        self.data_seed, self.epochs = data_seed, epochs
        low, high = RESERVED
        if not allow_reserved <= frozenset(range(low, high + 1)):
            raise ValueError(f"allow_reserved names ids outside {low}-{high}")
        self.refused = np.array(sorted(set(range(low, high + 1)) - allow_reserved), dtype=np.int64)
        key = hashlib.sha256(self.lengths.tobytes())
        key.update(f"data_seed {data_seed} buffer {buffer} rows {rows} seq_len {seq_len}".encode())
        self.fingerprint = key.hexdigest()
        self.fit = _BestFit(buffer)
        self.epoch = self.cursor = self.read = self.used = 0
        if state is not None:
            self._resume(state)
        self.order = self._permutation(self.epoch)

    def _permutation(self, epoch: int) -> Ids:
        return _order(len(self.lengths), self.data_seed, epoch)

    def _pull(self) -> tuple[int, int]:
        if self.cursor == len(self.lengths):
            if self.epoch + 1 >= self.epochs:
                raise ExhaustedError(f"every document has been read {self.epochs} time(s)")
            self.epoch, self.cursor = self.epoch + 1, 0
            self.order = self._permutation(self.epoch)
            warnings.warn(f"data epoch {self.epoch}: re-reading documents", stacklevel=2)
        doc = int(self.order[self.cursor])
        self.cursor += 1
        return doc, int(self.lengths[doc])

    def _row(self) -> Ids:
        cap = self.seq_len + 1
        parts: list[Ids] = []
        for doc, length, taken in self.fit.row(cap, self._pull):
            part = self.docs.read(doc, taken)
            self._check(doc, part)
            parts.append(part)
            self.read += length
            self.used += taken
        return np.concatenate(parts)

    def _check(self, doc: int, part: Ids) -> None:
        """The id rules, checked in int64 before any cast."""
        broken = ""
        if int(part.min()) < 0 or int(part.max()) >= self.vocab:
            broken = f"an id outside 0-{self.vocab - 1}"
        elif part[0] != SEP_ID or (part[1:] == SEP_ID).any():
            broken = f"id {SEP_ID} not exactly at its start"
        elif np.isin(part, self.refused).any():
            broken = "a reserved id not in allow_reserved"
        if broken:
            raise DataMismatchError(f"document {doc} breaks the format: {broken}")

    def __iter__(self) -> Packer:
        return self

    def __next__(self) -> PackedRows:
        """The next batch; past the last epoch, `ExhaustedError` rather than StopIteration."""
        packed = np.stack([self._row() for _ in range(self.rows)]).astype(np.int32)
        inputs = packed[:, :-1]
        return PackedRows(inputs=inputs, targets=packed[:, 1:], segment_ids=segment_ids(inputs))

    def state(self) -> PackerState:
        return {
            "fingerprint": self.fingerprint,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "buffer": self.fit.docs(),
            "read": self.read,
            "used": self.used,
        }

    def _resume(self, state: PackerState) -> None:
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("packer state from other documents, data seed, buffer or batch shape")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        for doc in state["buffer"]:
            self.fit.add(doc, int(self.lengths[doc]))
        self.read, self.used = state["read"], state["used"]


def _order(count: int, data_seed: int, epoch: int) -> Ids:
    """The order in which epoch `epoch` reads `count` documents."""
    philox = np.random.Philox(key=derive(data_seed, "data", epoch))
    return np.random.Generator(philox).permutation(count)


def epoch_rows(lengths: Ids, seq_len: int, *, data_seed: int = 0, buffer: int = 1000) -> int:
    """How many rows of `seq_len` inputs `Packer` gives in one epoch of documents this long.

    Which documents fill a row depends on their lengths and their order alone, so this fills
    the epoch's rows without reading a document. Count the whole epoch, never a sample: the
    buffer holds back the documents longer than a row and crops them later, so on FineWeb-Edu
    at T 2,048 the first thousand rows lose about 4% of their tokens and a whole epoch over 30%.
    """
    order = _order(len(lengths), data_seed, 0)

    def sizes() -> Iterator[tuple[int, int]]:
        for start in range(0, len(order), 65536):
            part: list[int] = lengths[order[start : start + 65536]].tolist()
            yield from enumerate(part, start)

    feed = sizes()

    def pull() -> tuple[int, int]:
        try:
            return next(feed)
        except StopIteration:
            raise ExhaustedError("every document has been read") from None

    fit, rows = _BestFit(buffer), 0
    try:
        while True:
            fit.row(seq_len + 1, pull)
            rows += 1
    except ExhaustedError:
        return rows


def segment_ids(inputs: NDArray[np.int32]) -> NDArray[np.int32]:
    """Segment ids per row: 0 for the first document, rising by one at each separator after it."""
    opens = (inputs == SEP_ID).astype(np.int32)
    opens[:, 0] = 0
    return np.cumsum(opens, axis=1, dtype=np.int32)
