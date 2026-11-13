"""Token ids and split-format helpers shared by the data scripts.

The split format is the one `akeratos/data/store.py` reads.
"""

import numpy as np

SEP_ID = 0              # <|endoftext|>: separates documents and opens a segment
RESERVED = (1, 16)      # ids 1-16: reserved, never emitted, masked out of the softmax
FORMAT = "metalforge-a21-standin.split/v0"
FIRST_ID = RESERVED[1] + 1                   # 17: the lowest id a document may hold after its separator
U16 = np.dtype("<u2")
U64 = np.dtype("<u8")


def content_ids(ids, vocab):
    """A document's ids as '<u2', after checking 17 <= id <= V - 1, which keeps ids 0-16 out."""
    a = np.asarray(ids)
    if a.dtype.kind not in "iu":
        raise TypeError(f"token ids of dtype {a.dtype}: integers only")
    a = a.astype(np.int64)  # checked in int64: NumPy's cast wraps silently (65,536 becomes 0, a false start)
    if a.size and (int(a.min()) < FIRST_ID or int(a.max()) > vocab - 1):
        raise ValueError(f"token ids in [{int(a.min())}, {int(a.max())}], outside [{FIRST_ID}, {vocab - 1}]")
    return a.astype(U16)


def check_idx(idx, tokens, where):
    if len(idx) < 2 or idx[0] != 0 or idx[-1] != tokens or not np.all(np.diff(idx.astype(np.int64)) > 0):
        raise ValueError(f"{where}: offsets break the index rule (idx[0] = 0, strictly increasing, idx[-1] = tokens)")


def close_points(lengths, shard_tokens):
    """The document indices where shards start, then the document count.

    A shard closes before the first document that would take it past `shard_tokens`; a document longer than
    that gets a shard of its own.
    """
    cuts, size = [0], 0
    for i, L in enumerate(lengths):
        if size and size + L > shard_tokens:
            cuts.append(i)
            size = 0
        size += L
    return cuts + [len(lengths)]
