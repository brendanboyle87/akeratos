"""The greedy decoder for uncurated samples and IFEval.

`greedy` appends the arg-max token of the head's masked FP32 logits until it emits the stop id
(`STOP_ID`) or `max_new` tokens are out; an emitted id 0 would open a new document, so it ends
the sample as not stopped. Each step runs the whole sequence again, with no key-value cache: a
sample is a few hundred tokens. The prompt is one segment, so it opens with id 0 and holds no
other.

With the table on, the table term of each step's ids is gathered from `table` as in training
(`Table.keys`, `Table.gather`, `model.table_term`), and the decoder never updates the table.
The ids the configuration masks (`head.masked_ids`) can never be emitted; a post-trained
configuration unmasks ids 1 and 2 (`akeratos/post/data.py`'s `stage_config`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from akeratos.configs import SEP_ID, ModelConfig
from akeratos.model import Params, head_logits, hidden, masked_id_mask, table_term
from akeratos.table.table import Table

STOP_ID = 2  # `<|im_end|>`: every turn ends there, and generation stops there


@dataclass(frozen=True)
class Sample:
    """The generated ids, the stop id excluded, and whether the decoder emitted the stop id."""

    ids: tuple[int, ...]
    stopped: bool


def _padded(ids: list[int], window: int | None) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """The ids as [1, n], padded with id 0 to a multiple of the window, and the segment ids.

    Banded windows need the length to be a multiple of the window (`akeratos/model.py`). The
    padding comes after the last real position and is a segment of its own, so causal attention
    keeps it out of every real position's output.
    """
    n = len(ids)
    m = n if window is None else -(-n // window) * window
    x = np.zeros((1, m), np.int32)
    x[0, :n] = ids
    seg = np.zeros((1, m), np.int32)
    seg[0, n:] = 1
    return x, seg


def greedy(
    params: Params,
    cfg: ModelConfig,
    context: Sequence[int],
    max_new: int,
    *,
    table: Table | None = None,
    dtype: mx.Dtype = mx.float32,
    stop: int = STOP_ID,
) -> Sample:
    """Decode greedily from the prompt `context` until `stop` or `max_new` new tokens."""
    ids = [int(i) for i in context]
    if not ids or ids[0] != SEP_ID or SEP_ID in ids[1:]:
        raise ValueError("a prompt is one segment: id 0 first and nowhere else")
    if max(ids) >= cfg.vocab_size or min(ids) < 0:
        raise ValueError("a prompt id outside [0, vocab_size)")
    if cfg.table != (table is not None):
        raise ValueError("the model's table term and the decoder's table must be on together")
    masked = masked_id_mask(cfg)
    out: list[int] = []
    for _ in range(max_new):
        n = len(ids)
        x, seg = _padded(ids, cfg.window)
        t = None
        if table is not None:
            ws = table.gather(table.keys(x.astype(np.int64)))
            t = table_term(ws.W, ws.idx, cfg.d_model, dtype)
        h = hidden(params, cfg, mx.array(x), mx.array(seg), dtype, t)
        z = head_logits(params, cfg, h[:, n - 1 : n, :], masked)
        nxt = int(np.asarray(mx.argmax(z[0, -1]), np.int64))
        if nxt in (stop, SEP_ID):  # id 0 would open a new document: the sample ends unstopped
            return Sample(tuple(out), nxt == stop)
        out.append(nxt)
        ids.append(nxt)
    return Sample(tuple(out), False)
