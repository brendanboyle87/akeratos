"""Turning token ids into table row numbers, on the CPU, before the batch reaches the GPU.

For order n at position t, take the n ids ending at t. A position before t's segment start
(the last separator id 0 at or before t, or the row start) becomes the pad 65,536, so no key
crosses a document. Then for each of the n tokens, in uint64 with wrap-around:

    x = tok + 1;  acc = (acc ^ (x·mult[i]))·0x9E3779B97F4A7C15

followed by acc ^= acc >> 29 and row = acc mod rows_per_slot. Each (order, head) owns a
contiguous slot of one global row space, order-major: slot j = order·heads + head starts at
j·rows_per_slot. The odd multipliers come from the `table.mult` stream, one array per (order,
head), orders outer, and are stored with the table.
"""

import numpy as np
import numpy.typing as npt

from akeratos.rng import derive

SEP_ID = 0  # <|endoftext|> opens a segment; the model's config holds the same id
PAD_ID = 65_536  # outside every uint16 id
M64 = np.uint64(0x9E3779B97F4A7C15)

U64 = npt.NDArray[np.uint64]
I64 = npt.NDArray[np.int64]


def multipliers(seed: int, orders: tuple[int, ...], heads: int = 1) -> list[U64]:
    """Odd uint64 multipliers in [1, 2⁶³), one array of n per (order, head), orders outer."""
    gen = np.random.Generator(np.random.Philox(key=derive(seed, "table.mult")))
    return [
        gen.integers(1, 2**63, size=n, dtype=np.uint64) | np.uint64(1)
        for n in orders
        for _ in range(heads)
    ]


def segment_starts(ids: I64) -> I64:
    """[B, T] int64: the position of each position's segment start."""
    B, T = ids.shape
    pos = np.broadcast_to(np.arange(T, dtype=np.int64), (B, T))
    return np.maximum.accumulate(np.where(ids == SEP_ID, pos, 0), axis=1)


def keys(
    ids: npt.ArrayLike,
    orders: tuple[int, ...],
    mults: list[U64],
    rows_per_slot: int,
    heads: int = 1,
) -> I64:
    """Global row ids [B, T, k·heads] (int64) of the n-grams ending at each position."""
    a: I64 = np.asarray(ids, dtype=np.int64)
    if a.min(initial=0) < 0 or a.max(initial=0) >= PAD_ID:
        raise ValueError("ids must lie in [0, 65,536): the pad sits outside every uint16 id")
    B, T = a.shape
    start = segment_starts(a)
    pos = np.arange(T, dtype=np.int64)
    lagged: list[U64] = []  # lagged[j][b, t]: the id at t - j, or the pad before the segment
    for j in range(max(orders)):
        pads = np.full((B, min(j, T)), PAD_ID, np.int64)
        shifted = np.concatenate([pads, a[:, : max(T - j, 0)]], axis=1)
        lagged.append(np.where(pos - j >= start, shifted, PAD_ID).astype(np.uint64))
    slots = [n for n in orders for _ in range(heads)]
    out = np.empty((B, T, len(slots)), np.int64)
    rps = np.uint64(rows_per_slot)
    with np.errstate(over="ignore"):
        for j, (n, mult) in enumerate(zip(slots, mults, strict=True)):
            acc = np.zeros((B, T), np.uint64)
            for i in range(n):  # token i of the n-gram sits at lag n - 1 - i
                x = lagged[n - 1 - i] + np.uint64(1)
                acc = (acc ^ (x * mult[i])) * M64
            acc ^= acc >> np.uint64(29)
            out[:, :, j] = (acc % rps).astype(np.int64) + j * rows_per_slot
    return out
