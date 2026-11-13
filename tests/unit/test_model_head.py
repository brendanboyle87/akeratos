"""The model's embeddings, head, loss and multi-token prediction.

Ids 1-16 get zero probability and the logit cap matches float64; the chunked loss path, the
table term's entry and the same-head multi-token switch are checked too. The oracle is the
NumPy float64 model in `tests/unit/model_ref.py`.
"""

import mlx.core as mx
import numpy as np
import pytest

from akeratos.errors import DataMismatchError
from akeratos.model import (
    EMBED,
    forward,
    head_logits,
    hidden,
    init_params,
    kind_of,
    masked_id_mask,
    table_term,
)
from akeratos.objective import depth_targets, depth_weights, mtp_loss_sum
from akeratos.train.loop import _chunk_sum, loss_sum, prepare_targets, token_ce
from tests.unit import model_ref as ref


def to64(params: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float64) for k, v in params.items()}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-300))


def grads(cfg, params, ids, tg, lmask=None):
    targets, weights = prepare_targets(tg, lmask, cfg)
    t, w, x = mx.array(targets), mx.array(weights), mx.array(ids)
    n = mx.array(float(weights.sum()))

    def f(p):
        return loss_sum(p, cfg, hidden(p, cfg, x, ref.seg(x), mx.float32), t, w) / n

    loss, g = mx.value_and_grad(f)(params)
    return float(np.array(loss)), {k: np.array(v, dtype=np.float64) for k, v in g.items()}


# ---- masked ids --------------------------------------------------------------------------------


def test_masked_ids_get_zero_probability_and_no_gradient() -> None:
    """BF16 compute; the logits come out FP32 and ids 1-16 are -inf."""
    cfg = ref.tiny(table=False)
    ids, tg = ref.batch(2)
    z = forward(init_params(cfg), cfg, mx.array(ids), ref.seg(mx.array(ids)))
    assert z.dtype == mx.float32
    assert float(mx.softmax(z, axis=-1)[..., 1:17].max()) == 0.0
    g = mx.grad(lambda zz: token_ce(zz, mx.array(tg.astype(np.int32))).sum())(z)
    assert float(mx.abs(g[..., 1:17]).max()) == 0.0


def test_cap_matches_float64() -> None:
    """15·tanh(z/15) with m_logit 256/d (4 at d 64), FP32 against the NumPy float64 head."""
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    ids, _ = ref.batch(2)
    P = to64(params)
    raw = ref.hidden(P, cfg, ids) @ P[EMBED].T * (256 / cfg.d_model)
    capped = 15 * np.tanh(raw / 15)
    z = np.array(forward(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    assert rel(z[..., 17:], capped[..., 17:]) < 1e-5
    assert np.abs(capped).max() < 15
    off = ref.tiny(table=False, **{"loss.final_softcap": None})
    z_off = np.array(forward(params, off, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    assert rel(z_off[..., 17:], raw[..., 17:]) < 1e-5


def test_masked_ids_follow_the_key() -> None:
    """`head.masked_ids` sets which ids the head masks (post-training unmasks some)."""
    cfg = ref.tiny(**{"head.masked_ids": [3, 4]})
    m = np.array(masked_id_mask(cfg))
    assert np.flatnonzero(m).tolist() == [3, 4]


# ---- targets and the loss ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 512])
def test_out_of_range_targets_are_refused(bad: int) -> None:
    with pytest.raises(DataMismatchError, match="outside"):
        prepare_targets(np.array([[30, bad]]), None, ref.tiny())


def test_masked_targets_get_weight_zero() -> None:
    """A masked-id target and a masked position both get weight 0 and target 0."""
    targets, weights = prepare_targets(np.array([[30, 5, -1]]), np.array([[1, 1, 0]]), ref.tiny())
    assert targets.tolist() == [[30, 0, 0]]
    assert weights.tolist() == [[1.0, 0.0, 0.0]]


def test_a_reserved_target_gives_a_finite_loss() -> None:
    """The safety-data arm's tag as the next token."""
    cfg = ref.tiny(table=False)
    ids, tg = ref.batch(3)
    tg[0, 5] = 9
    loss, g = grads(cfg, init_params(cfg), ids, tg)
    assert np.isfinite(loss)
    assert all(np.isfinite(v).all() for v in g.values())


def test_the_loss_refuses_bf16_logits() -> None:
    with pytest.raises(TypeError, match="FP32"):
        token_ce(mx.zeros((2, 8), mx.bfloat16), mx.zeros((2,), mx.int32))


def test_loss_matches_float64_with_z_loss_and_a_mask() -> None:
    cfg = ref.tiny(table=False, **{"loss.z": 1e-4})
    params = init_params(cfg, seed=2)
    ids, tg = ref.batch(1)
    lmask = np.ones(ids.shape)
    lmask[0, :5] = 0
    loss, _ = grads(cfg, params, ids, tg, lmask)
    assert abs(loss - ref.loss(to64(params), cfg, ids, tg, lmask)) < 1e-5


def test_chunked_loss_equals_unchunked() -> None:
    """4 chunks of 32 under `mx.checkpoint`, z-loss on, 5 masked targets, FP32."""
    ids, tg = ref.batch(1)
    lmask = np.ones(ids.shape)
    lmask[0, :5] = 0
    out = {}
    for path in ("unchunked", "chunked"):
        cfg = ref.tiny(table=False, **{"loss.path": path, "loss.chunk_tokens": 32, "loss.z": 1e-4})
        out[path] = grads(cfg, init_params(cfg), ids, tg, lmask)
    (la, ga), (lb, gb) = out["unchunked"], out["chunked"]
    assert abs(la - lb) < 1e-5 * abs(la)
    assert max(rel(gb[k], ga[k]) for k in ga) < 1e-5


def test_late_bound_chunk_targets_give_a_wrong_gradient() -> None:
    """The control for the chunked loss's pitfall: a closure reads the last chunk's targets."""
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    ids, tg = ref.batch(1)
    h = hidden(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32)
    N, C = ids.size, 32
    hf, tf = mx.reshape(h, (N, -1)), mx.array(tg.reshape(N).astype(np.int32))
    w = mx.ones((N,))
    masked = masked_id_mask(cfg)

    def total(E, late):
        s = mx.array(0.0)
        cur = {}  # read when the checkpoint recomputes, as a late-binding closure reads
        for c0 in range(0, N, C):
            cur["tgt"], cur["msk"] = tf[c0 : c0 + C], w[c0 : c0 + C]

            def bound(hc, e, tgt=cur["tgt"], msk=cur["msk"]):
                return _chunk_sum(head_logits({EMBED: e}, cfg, hc, masked), tgt, msk, 0.0)

            def unbound(hc, e):
                z = head_logits({EMBED: e}, cfg, hc, masked)
                return _chunk_sum(z, cur["tgt"], cur["msk"], 0.0)

            s = s + mx.checkpoint(unbound if late else bound)(hf[c0 : c0 + C], E)
        return s

    good = np.array(mx.grad(lambda e: total(e, False))(params[EMBED]))
    bad = np.array(mx.grad(lambda e: total(e, True))(params[EMBED]))
    assert rel(bad, good) > 0.1


def test_gather_then_cast_keeps_a_frequent_tokens_gradient() -> None:
    """A row read 4,096 times at BF16 compute: its summed gradient against float64."""
    rng = np.random.default_rng(3)
    E = mx.array(rng.normal(0, 0.02, (8, 64)).astype(np.float32))
    ids = mx.array(np.full((4096,), 5, np.int32))
    c = mx.array(rng.normal(0, 1, (4096, 64)).astype(np.float32)).astype(mx.bfloat16)
    exact = np.array(c.astype(mx.float32), np.float64).sum(0)

    def gather_then_cast(e):
        return (mx.take(e, ids, axis=0).astype(mx.bfloat16) * c).astype(mx.float32).sum()

    def cast_then_gather(e):
        return (mx.take(e.astype(mx.bfloat16), ids, axis=0) * c).astype(mx.float32).sum()

    good = np.array(mx.grad(gather_then_cast)(E))[5]
    bad = np.array(mx.grad(cast_then_gather)(E))[5]
    assert rel(good, exact) < 1e-3
    assert rel(bad, exact) > rel(good, exact)


def test_tied_gradient_is_the_sum_of_both_paths() -> None:
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    ids, tg = ref.batch(4)
    _, g = grads(cfg, params, ids, tg)
    targets, weights = prepare_targets(tg, None, cfg)
    t, w, x = mx.array(targets), mx.array(weights), mx.array(ids)
    masked = masked_id_mask(cfg)

    def split(e_in, e_out):
        h = hidden({**params, EMBED: e_in}, cfg, x, ref.seg(x), mx.float32)
        z = head_logits({EMBED: e_out}, cfg, h, masked)
        return (token_ce(z, t) * w).sum() / float(weights.sum())

    E = params[EMBED]
    g_in, g_out = mx.grad(split, argnums=(0, 1))(E, E)
    assert rel(np.array(g_in) + np.array(g_out), g[EMBED]) < 1e-5


def test_value_embedding_gate_and_training() -> None:
    """The gate is 3·sigmoid(0) = 1.5 at initialisation; every table and gate gets a gradient."""
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    a_in = mx.random.normal((2, 3, cfg.d_model), key=mx.random.key(0))
    gate = 3.0 * mx.sigmoid(a_in[..., :12] @ params["layers.1.attn.ve_gate.weight"].T)
    assert np.allclose(np.array(gate), 1.5)
    ids, tg = ref.batch(5)
    _, g = grads(cfg, params, ids, tg)
    live = [
        k for k, v in g.items() if kind_of(k) in ("value_embed", "ve_gate") and np.abs(v).max() > 0
    ]
    assert len(live) == 2 * len(cfg.ve_layers())


# ---- the table term ----------------------------------------------------------------------------


@pytest.mark.parametrize("heads", [1, 2])
def test_table_term_matches_float64(heads: int) -> None:
    """Order-major rows, each order's heads side by side, the orders summed over √k."""
    rng = np.random.default_rng(6)
    W = rng.normal(0, 0.1, (40, 64 // heads)).astype(np.float32)
    idx = rng.integers(0, 40, (2, 8, 2 * heads)).astype(np.int32)
    got = np.array(table_term(mx.array(W), mx.array(idx), 64, mx.float32))
    assert rel(got, ref.table_term(W.astype(np.float64), idx, heads)) < 1e-6


def test_forward_with_the_table_term_matches_float64() -> None:
    cfg = ref.tiny(**{"table.sites": [0, 3]})
    params = init_params(cfg, seed=3)
    ids, _ = ref.batch(6)
    t = np.random.default_rng(7).normal(0, 0.5, (ref.B, ref.T, cfg.d_model)).astype(np.float32)
    z = np.array(
        forward(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32, mx.array(t))
    )
    z_ref = ref.logits(to64(params), cfg, ids, t.astype(np.float64))
    live = np.isfinite(z_ref)
    assert rel(z[live], z_ref[live]) < 1e-5
    assert params["tl"].shape == (2,)


# ---- multi-token prediction --------------------------------------------------------------------


def test_depth_weights_follow_the_stages() -> None:
    assert depth_weights(0.0) == (1.0, 0.5, 0.25)
    assert depth_weights(0.285 / 2) == pytest.approx((1.0, 0.5, 0.125))
    assert depth_weights(0.4455) == pytest.approx((1.0, 0.25, 0.0))
    assert depth_weights(0.606) is None
    assert depth_weights(0.9) is None


def test_mtp_loss_matches_float64() -> None:
    """Same-head depths 1-3 against the NumPy model, one offset at a time, with a padded row end."""
    cfg = ref.tiny(table=False)
    params = init_params(cfg, seed=1)
    ids, tg = ref.batch(4, mean_doc=9)
    lmask = np.ones(ids.shape, np.float32)
    lmask[1, -5:] = 0
    _, weights = prepare_targets(tg, lmask, cfg)
    tg3, w3 = depth_targets(tg, weights)
    dw = (1.0, 0.5, 0.2)
    h = hidden(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32)
    got = float(np.array(mtp_loss_sum(params, cfg, h, mx.array(tg3), mx.array(w3), mx.array(dw))))
    want = ref.mtp_loss_sum(ref.logits(to64(params), cfg, ids), tg, weights, dw)
    assert abs(got - want) < 1e-5 * abs(want)


def test_mtp_at_next_token_weights_equals_the_plain_loss() -> None:
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    ids, tg = ref.batch(4)
    targets, weights = prepare_targets(tg, None, cfg)
    tg3, w3 = depth_targets(tg, weights)
    h = hidden(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32)
    plain = float(np.array(loss_sum(params, cfg, h, mx.array(targets), mx.array(weights))))
    one = mx.array((1.0, 0.0, 0.0))
    mtp = float(np.array(mtp_loss_sum(params, cfg, h, mx.array(tg3), mx.array(w3), one)))
    assert abs(mtp - plain) < 1e-5 * plain


def test_no_depth_predicts_across_a_document() -> None:
    """Depth j at t is off when a separator lies among targets[t .. t + j - 1]."""
    tg = np.array([[20, 0, 21, 22, 23]])
    t3, w3 = depth_targets(tg, np.ones(tg.shape, np.float32))
    assert w3[0, :, 0].tolist() == [1, 1, 1, 1, 1]
    assert w3[0, :, 1].tolist() == [1, 0, 1, 1, 0]
    assert w3[0, :, 2].tolist() == [0, 0, 1, 0, 0]
    assert t3[0, 2, 2] == 23
    _, open_rows = depth_targets(tg, np.ones(tg.shape, np.float32), doc_mask=False)
    assert open_rows[0, :, 2].tolist() == [1, 1, 1, 0, 0]


def test_mtp_refuses_the_chunked_path() -> None:
    cfg = ref.tiny(table=False, **{"loss.path": "chunked"})
    z = mx.zeros((1, 4, cfg.d_model))
    with pytest.raises(ValueError, match="chunked"):
        mtp_loss_sum(
            init_params(cfg),
            cfg,
            z,
            mx.zeros((1, 4, 3), mx.int32),
            mx.ones((1, 4, 3)),
            mx.ones((3,)),
        )
