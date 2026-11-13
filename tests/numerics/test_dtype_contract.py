"""The model's dtype contract.

A probe wraps the model's matrix products, the attention kernel, the blocks and the head, and
records what they see in one BF16 step of the tiny model: every matmul operand and the head
output before the upcast are BF16, the residual stream leaves every block as BF16, the logits
into the cross-entropy and the loss are FP32, and the masters and the gradients at optimiser
entry are FP32.
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from akeratos import model
from akeratos.model import init_params
from akeratos.train import loop as loss_mod
from akeratos.train.loop import loss_sum, prepare_targets
from tests.unit import model_ref as ref


def recorder(seen: list[tuple[str, mx.Dtype]], name: str, fn: Callable[..., Any], inputs: int):
    """Wrap fn to record the dtypes of its first `inputs` arguments and of its output."""

    def wrapped(*args: Any, **kw: Any) -> Any:
        seen.extend((name, a.dtype) for a in args[:inputs])
        out = fn(*args, **kw)
        seen.append((name + "->", out.dtype))
        return out

    return wrapped


@pytest.mark.parametrize("table", [False, True])
def test_dtype_contract(monkeypatch: pytest.MonkeyPatch, table: bool) -> None:
    cfg = ref.tiny(table=table, **{"loss.z": 1e-4})
    params = init_params(cfg)
    ids, tg = ref.batch(14)
    x = mx.array(ids)
    t, w = prepare_targets(tg, None, cfg)
    tt = None
    if table:
        W = mx.array(
            np.random.default_rng(15).standard_normal((32, cfg.d_model)).astype(np.float32)
        )
        idx = mx.array(np.random.default_rng(16).integers(0, 32, size=(*ids.shape, 1)))
        tt = model.table_term(W, idx, cfg.d_model, mx.bfloat16)
    seen: list[tuple[str, mx.Dtype]] = []
    monkeypatch.setattr(model, "dense", recorder(seen, "dense", model.dense, 1))
    monkeypatch.setattr(
        mx.fast,
        "scaled_dot_product_attention",
        recorder(seen, "sdpa", mx.fast.scaled_dot_product_attention, 3),
    )
    monkeypatch.setattr(model, "block", recorder(seen, "block", model.block, 0))
    monkeypatch.setattr(model, "finish_logits", recorder(seen, "head", model.finish_logits, 1))
    monkeypatch.setattr(loss_mod, "token_ce", recorder(seen, "ce", loss_mod.token_ce, 1))

    def f(p):
        h = model.hidden(p, cfg, x, ref.seg(x), mx.bfloat16, tt)
        return loss_sum(p, cfg, h, mx.array(t), mx.array(w)) / float(w.sum())

    loss, grads = mx.value_and_grad(f)(params)
    by = {}
    for name, dt in seen:
        by.setdefault(name, set()).add(dt)
    assert by["dense->"] == {mx.bfloat16}
    assert by["dense"] == {mx.bfloat16}  # the activation; dense casts the FP32 master itself
    assert by["sdpa"] == {mx.bfloat16}
    assert by["block->"] == {mx.bfloat16}

    assert by["head"] == {mx.bfloat16}  # the product h·Eᵀ before the upcast
    assert by["head->"] == {mx.float32}
    assert by["ce"] == {mx.float32}
    assert loss.dtype == mx.float32
    assert {v.dtype for v in params.values()} == {mx.float32}
    assert {k for k, v in grads.items() if v.dtype != mx.float32} == set()
