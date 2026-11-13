"""The fused cross-entropy sees row-contiguous logits.

`mx.fast.cross_entropy` returns wrong gradients for large non-contiguous inputs on Metal
(docs/mlx-notes.md), so `token_ce` makes its logits contiguous first. Both checks compare
bytes, so the guard is bitwise-neutral: a transposed view scores as its contiguous copy, and
row-contiguous logits score exactly as `mx.fast.cross_entropy` alone.
"""

import mlx.core as mx
import numpy as np

from akeratos.train.loop import token_ce

mx.set_default_device(mx.cpu)


def targets() -> mx.array:
    return mx.array(np.random.default_rng(0).integers(0, 11, (2, 5)), mx.int32)


def raw(x: mx.array) -> bytes:
    return np.array(x).tobytes()


def test_a_transposed_view_scores_as_its_contiguous_copy() -> None:
    zt = mx.random.normal((11, 5, 2), key=mx.random.key(3))
    z = zt.transpose(2, 1, 0)
    assert z.shape == (2, 5, 11)
    tg = targets()
    assert raw(token_ce(z, tg)) == raw(token_ce(mx.array(np.array(z)), tg))


def test_row_contiguous_logits_score_as_the_fused_kernel_alone() -> None:
    z = mx.random.normal((2, 5, 11), key=mx.random.key(4))
    tg = targets()
    assert raw(token_ce(z, tg)) == raw(mx.fast.cross_entropy(z, tg))
