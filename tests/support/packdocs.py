"""The packer tests' documents, shared by the unit and repro suites.

`store` packs documents the way the data store holds them, each opened by id 0, and
`zipf_docs` draws documents with Zipf-distributed ids and geometric lengths.
"""

import numpy as np
from numpy.typing import NDArray

from akeratos.data.packer import Ids

V = 512


class TokenDocs:
    """Documents held as one token array and N + 1 start offsets (in memory or memory-mapped)."""

    def __init__(self, tokens: NDArray[np.integer], offsets: NDArray[np.integer]) -> None:
        starts = np.asarray(offsets, dtype=np.int64)
        steps = np.diff(starts)
        if steps.size == 0 or starts[0] != 0 or starts[-1] != len(tokens) or (steps <= 0).any():
            raise ValueError("offsets must run from 0 to len(tokens) and strictly increase")
        self._tokens = tokens
        self._starts = starts
        self._lengths = steps

    @property
    def lengths(self) -> Ids:
        """Each document's length in tokens, its separator included."""
        return self._lengths

    def read(self, index: int, count: int) -> Ids:
        """The first `count` tokens of document `index`."""
        start = int(self._starts[index])
        return np.asarray(self._tokens[start : start + count], dtype=np.int64)


def document_mask(segments: NDArray[np.int32]) -> NDArray[np.bool_]:
    """The intra-document causal mask [B, T, T]: query i sees key j iff j <= i in its segment.

    A dense reference for tests: at B 100 and T 2,048 it holds about 420 MB, so the model builds
    its own masks from the segment ids.
    """
    T = segments.shape[1]
    causal = np.tril(np.ones((T, T), dtype=bool))
    return (segments[:, :, None] == segments[:, None, :]) & causal


def store(docs: list[NDArray[np.int64]]) -> TokenDocs:
    """Documents given without their separator, stored as `[0] + ids` each."""
    body = [np.concatenate([[0], d]) for d in docs]
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in body])])
    return TokenDocs(np.concatenate(body).astype(np.uint16), offsets)


def zipf_docs(seed: int, n: int, mean: int = 20, long_every: int = 37, long_len: int = 150):
    """Zipf-distributed content ids in 17..V-1, geometric lengths, every `long_every`-th long."""
    rng = np.random.default_rng(seed)
    docs = []
    for i in range(n):
        size = (
            long_len if long_every and i % long_every == long_every - 1 else rng.geometric(1 / mean)
        )
        docs.append(17 + (rng.zipf(1.3, size) - 1) % (V - 17))
    return docs
