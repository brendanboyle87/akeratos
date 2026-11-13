"""`attn.path` fused: the attention kernels' bounds, wiring and backward pass.

On the CPU `akeratos/flash.py` runs its FP32 reference with the kernels' backward formulas,
so these tests check the bounds, the wiring into the model and the hand-written backward. The
Metal kernels are checked against the same reference on a Mac.
"""

import mlx.core as mx
import numpy as np
import pytest

from akeratos.eval.heldout import scoring_config
from akeratos.flash import attend, bounds, keep_mask, kernels_apply, reference_forward
from akeratos.model import _full_mask, build_masks, forward, init_params
from tests.unit import model_ref as ref


@pytest.mark.parametrize(("window", "doc"), [(16, True), (None, True), (16, False), (8, True)])
def test_bounds_equal_the_model_mask_rule(window, doc) -> None:
    ids, _ = ref.batch(5)
    seg = ref.seg(mx.array(ids)) if doc else mx.zeros(ids.shape, mx.int32)
    b = bounds(seg, window)
    rule = _full_mask(seg, window, True)
    assert bool(mx.array_equal(keep_mask(b.lo), rule).item())
    p = mx.arange(ref.T)[None, :, None]
    last = mx.max(mx.where(rule[:, 0], p, -1), axis=1)  # the last query that sees each key
    assert bool(mx.array_equal(last, b.qend).item())


def test_the_fused_path_builds_bounds_and_equals_the_masked_path() -> None:
    ids, _ = ref.batch(2)
    seg = ref.seg(mx.array(ids))
    out = {}
    for path in ("fused", "masked"):
        cfg = ref.tiny(**{"table": False, "attn.path": path})
        out[path] = np.array(forward(init_params(cfg), cfg, mx.array(ids), seg, mx.float32))
    cfg = ref.tiny(**{"table": False, "attn.path": "fused"})
    keys = {"global.lo", "global.qend", "sliding.lo", "sliding.qend"}
    assert set(build_masks(seg, cfg)) == keys
    live = np.isfinite(out["masked"])
    assert np.abs(out["fused"][live] - out["masked"][live]).max() < 1e-5


@pytest.mark.parametrize(("heads", "window"), [((4, 2), 16), ((6, 2), None), ((2, 2), 8)])
def test_the_hand_backward_equals_autograd(heads, window) -> None:
    H, kv = heads
    B, T, D = 2, 64, 16
    ids, _ = ref.batch(7)
    b = bounds(ref.seg(mx.array(ids)), window)
    k1, k2, k3, k4 = mx.random.split(mx.random.key(3), 4)
    q = mx.random.normal((B, H, T, D), key=k1)
    k = mx.random.normal((B, kv, T, D), key=k2)
    v = mx.random.normal((B, kv, T, D), key=k3)
    do = mx.random.normal((B, T, H, D), key=k4)

    def auto(q_, k_, v_):
        return reference_forward(q_, k_, v_, b.lo, D**-0.5)[0]

    o_ref, g_ref = mx.vjp(auto, [q, k, v], [do])
    o, g = mx.vjp(lambda q_, k_, v_: attend(q_, k_, v_, b), [q, k, v], [do])
    assert np.array_equal(np.array(o[0]), np.array(o_ref[0]))
    for got, want in zip(g, g_ref, strict=True):
        assert np.abs(np.array(got) - np.array(want)).max() < 1e-4


def test_the_kernels_run_only_on_the_gpu_in_bfloat16() -> None:
    q = mx.zeros((1, 4, 64, 16), mx.bfloat16)
    k = mx.zeros((1, 2, 64, 16), mx.bfloat16)
    on_gpu = mx.default_device() == mx.gpu
    assert kernels_apply(q, k) is on_gpu
    assert not kernels_apply(q.astype(mx.float32), k.astype(mx.float32))
    assert not kernels_apply(q[:, :, :40], k[:, :, :40])


def test_readouts_score_a_fused_run_on_the_banded_path() -> None:
    cfg = ref.tiny(**{"attn.path": "fused"})
    assert scoring_config(cfg).attn_path == "banded"
    assert scoring_config(ref.tiny(**{"attn.path": "masked"})).attn_path == "masked"
