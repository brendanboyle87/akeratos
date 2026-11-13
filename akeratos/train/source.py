"""The training loop's batch sources, and the check that the data can feed the whole run.

A source gives one batch, a `StepRows` of NumPy arrays, per optimiser step from `next()`, and
`state()` for the checkpoint's cursor. The loop converts the arrays to `mx.array`, so neither
this module nor `akeratos.data` imports MLX. `preflight` refuses, before the model is built,
a split that cannot feed the run in one epoch, so a long run cannot stop near its end with "the
data ran out". It counts the steps one epoch feeds from the documents' lengths.

Shape legend: N steps, B rows, T inputs per row.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from akeratos.data.packer import (
    SEP_ID,
    Documents,
    ExhaustedError,
    Packer,
    epoch_rows,
    segment_ids,
)
from akeratos.errors import ConfigError
from akeratos.rng import seed_hex
from akeratos.train.recipe import steps_for

# `SynthSource` draws Zipf content ids (p ∝ rank^-1.1) in 17..V-1 with geometric document
# lengths, every row opening with the separator 0, so every row starts at a document start. Batch
# i draws from its own generator, seeded by (seed, i), so the cursor is just i. Steps in `poison`
# get an all-zero loss mask, whose loss is 0/0: tests use them to force a skipped step.
# `PackedSource` wraps the best-fit `Packer` over any `Documents` (the data store's `Split`).
FIRST_CONTENT_ID = 17  # ids 1-16 are reserved and never emitted


@dataclass(frozen=True)
class StepRows:
    """One step's rows, int32 [B, T]; `lmask` None reads as all ones."""

    ids: NDArray[np.int32]
    targets: NDArray[np.int32]
    segment_ids: NDArray[np.int32]
    lmask: NDArray[np.float32] | None = None


class Source(Protocol):
    """What the loop reads a batch from."""

    def next(self) -> StepRows:
        """The next step's batch."""
        ...

    def state(self) -> dict[str, Any]:
        """The cursor a checkpoint records."""
        ...


def zipf_probs(vocab: int, alpha: float = 1.1) -> NDArray[np.float64]:
    """The Zipf weights, p ∝ rank^-alpha, over the content ids 17..V-1."""
    p = np.arange(1, vocab - FIRST_CONTENT_ID + 1, dtype=np.float64) ** -alpha
    return p / p.sum()


class SynthSource:
    """Synthetic Zipf documents laid out as separator-aligned rows of `rows` x `seq_len`."""

    def __init__(
        self,
        rows: int,
        seq_len: int,
        vocab: int,
        *,
        seed: int = 0,
        mean_doc: int = 200,
        poison: Iterable[int] = (),
        cursor: int = 0,
    ) -> None:
        self.rows, self.seq_len, self.vocab = rows, seq_len, vocab
        self.seed, self.mean_doc = seed, mean_doc
        self.poison = frozenset(poison)
        self.i = cursor
        self.probs = zipf_probs(vocab)

    def next(self) -> StepRows:
        """Batch i, from its own generator seeded by (seed, i)."""
        rng = np.random.default_rng([self.seed, self.i])
        toks = rng.choice(
            self.vocab - FIRST_CONTENT_ID, size=(self.rows, self.seq_len + 1), p=self.probs
        )
        toks += FIRST_CONTENT_ID
        for row in toks:
            pos = 0
            while pos < self.seq_len + 1:
                row[pos] = SEP_ID
                pos += 1 + int(rng.geometric(1.0 / self.mean_doc))
        ids = toks[:, :-1].astype(np.int32)
        lmask = np.zeros(ids.shape, np.float32) if self.i in self.poison else None
        self.i += 1
        return StepRows(ids, toks[:, 1:].astype(np.int32), segment_ids(ids), lmask)

    def state(self) -> dict[str, Any]:
        """The cursor: the next batch's index and the seed, persisted as hex."""
        return {"kind": "synthetic", "i": self.i, "seed": seed_hex(self.seed)}


class PackedSource:
    """Batches from the `Packer`; its state is the packer's own."""

    def __init__(self, packer: Packer) -> None:
        self.packer = packer

    def next(self) -> StepRows:
        """The packer's next batch; raises `ExhaustedError` past the last epoch."""
        rows = next(self.packer)
        return StepRows(rows.inputs, rows.targets, rows.segment_ids)

    def state(self) -> dict[str, Any]:
        """The packer's checkpointed state."""
        return {"kind": "packed", **self.packer.state()}


# Each row of T inputs takes T + 1 tokens, and best-fit packing discards the tails of the
# documents it crops. This check counts the rows one epoch packs into, with the run's own seed
# and buffer (`epoch_rows`), and divides by a step's rows. It also reads `sample_rows` rows,
# which checks the sampled documents' ids, so a broken split stops here with
# `DataMismatchError`.
def preflight(
    docs: Documents,
    rows: int,
    seq_len: int,
    tokens: float,
    *,
    vocab: int,
    data_seed: int = 0,
    buffer: int = 1000,
    sample_rows: int = 1000,
    allow_reserved: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """The steps a run of `tokens` trained tokens needs and the steps one epoch of the split feeds.

    `rows` is a step's rows, every micro-batch counted. `allow_reserved` is the packer's: the
    reserved ids a document may hold (a cool-down mix's). A split that cannot feed the run in
    one epoch raises `ConfigError` with the shortfall.
    """
    probe = Packer(
        docs,
        1,
        seq_len,
        vocab=vocab,
        data_seed=data_seed,
        buffer=buffer,
        allow_reserved=allow_reserved,
    )
    sampled = 0
    try:
        for _ in range(sample_rows):
            next(probe)
            sampled += 1
    except ExhaustedError:
        pass
    steps = steps_for(tokens, rows * seq_len)
    lengths = np.asarray(docs.lengths, dtype=np.int64)
    fed = epoch_rows(lengths, seq_len, data_seed=data_seed, buffer=buffer) // rows
    if fed < steps:
        msg = (
            f"the split cannot feed {steps:,} steps of {rows} x {seq_len}: one epoch of it "
            f"feeds {fed:,}; short by {steps - fed:,}"
        )
        raise ConfigError(msg)
    return {"sampled_rows": sampled, "steps": steps, "steps_fed": fed}
