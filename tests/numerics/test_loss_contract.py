"""The model's loss: cross-entropy and z-loss in FP32 on BF16 head outputs.

The CE is FP32 on BF16 logits and matches float64; the z-loss shares the CE's log-partition;
the loss and its gradients do not depend on the micro-batch split or the CE's chunk size; a
masked target stays finite. The references are NumPy float64, and each bound is a named
constant below.
"""

import mlx.core as mx
import numpy as np
import pytest

from akeratos.configs import from_keys
from akeratos.model import finish_logits, hidden, init_params, masked_id_mask
from akeratos.train.loop import (
    loss_metrics,
    loss_sum,
    loss_terms,
    n_valid,
    prepare_targets,
    token_ce,
)
from tests.unit import model_ref as ref

mx.set_default_device(mx.cpu)

CE_F64_ABS = 1e-4  # per-token |CE - float64| on BF16 head outputs at V 49,152 (FP32 path)
LSE_ABS = 1e-5  # rebuilt log-partition against FP32 mx.logsumexp
LOSS_REL = 1e-6  # loss across chunk sizes and splits, FP32 compute
LSE_SQ_REL = 1e-5  # the summed lse² (about 40 per token) across chunk sizes
METRIC_CE_ABS = 1e-5  # train/ce_loss against float64
METRIC_ZU_ABS = 1e-4  # train/z_loss_unscaled (lse² about 40) against float64
GRAD_REL_FP32 = 1e-5  # gradients across groupings, FP32 compute
GRAD_REL_BF16 = 1e-2  # gradients across chunk sizes, BF16 compute (each chunk's matmul rounds)

V = 49_152


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(b), 1e-300))


def head_cfg(d_model: int):
    """A configuration for the head alone: V 49,152 and μP's multiplier 256/d."""
    return from_keys({"d_model": d_model, "vocab_size": V})


def bf16_head_output(n: int, seed: int) -> mx.array:
    """BF16 products h·Eᵀ [n, V] with the spread of a trained head (std 4)."""
    rng = np.random.default_rng(seed)
    return mx.array((rng.standard_normal((n, V)) * 4).astype(np.float32)).astype(mx.bfloat16)


def ref_ce(z_bf16: mx.array, targets: np.ndarray, d_model: int) -> np.ndarray:
    """Float64 per-token CE on the same BF16 values: scale, cap, mask, then log-softmax."""
    z = np.array(z_bf16.astype(mx.float32), np.float64) * (256 / d_model)
    z = 15 * np.tanh(z / 15)
    z[:, 1:17] = -np.inf
    m = z.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True)))[:, 0]
    return lse - z[np.arange(len(targets)), targets]


@pytest.mark.parametrize("d_model", [256, 1536])
def test_ce_fp32_on_bf16_logits(d_model: int) -> None:
    """The seam returns float32 and matches float64; at d 1,536 the multiplier is 1/6."""
    cfg = head_cfg(d_model)
    z = bf16_head_output(512, seed=d_model)
    tg = np.random.default_rng(1).integers(17, V, size=512)
    nll = token_ce(finish_logits(z, cfg, masked_id_mask(cfg)), mx.array(tg.astype(np.int32)))
    assert nll.dtype == mx.float32
    want = ref_ce(z, tg, d_model)
    assert np.abs(np.array(nll, np.float64) - want).max() < CE_F64_ABS
    # The control: the same steps in BF16, as without the upcast, miss the bound.
    zb = z * mx.array(256 / d_model, mx.bfloat16)
    zb = mx.where(masked_id_mask(cfg), -mx.inf, 15 * mx.tanh(zb / 15))
    bf = mx.fast.cross_entropy(zb, mx.array(tg.astype(np.int32)))
    assert np.abs(np.array(bf, np.float64) - want).max() > CE_F64_ABS


def test_zloss_shares_lse() -> None:
    """lse = nll + z[y] on capped, masked FP32 logits equals mx.logsumexp, with finite gradients."""
    cfg = head_cfg(256)
    masked = masked_id_mask(cfg)
    raw = bf16_head_output(256, seed=3)
    tg = mx.array(np.random.default_rng(4).integers(17, V, size=256).astype(np.int32))
    z = finish_logits(raw, cfg, masked)
    lse = token_ce(z, tg) + mx.take_along_axis(z, tg[:, None], axis=-1)[:, 0]
    assert float(mx.abs(lse - mx.logsumexp(z, axis=-1)).max()) < LSE_ABS

    def z_term(r: mx.array) -> mx.array:
        zz = finish_logits(r, cfg, masked)
        lz = token_ce(zz, tg) + mx.take_along_axis(zz, tg[:, None], axis=-1)[:, 0]
        return mx.square(lz).sum()

    g = mx.grad(z_term)(raw.astype(mx.float32))
    assert bool(mx.isfinite(g).all())
    assert float(mx.abs(g[:, 1:17]).max()) == 0.0


def split_grads(cfg, params, ids, tg, lmask, k, compiled):
    """Loss and gradients of one global batch summed over k row-wise micro-batches."""
    rows = np.array_split(np.arange(ids.shape[0]), k)
    parts = [prepare_targets(tg[r], lmask[r], cfg) for r in rows]
    _, n = n_valid(w for _, w in parts)
    total, acc = 0.0, None
    for r, (t, w) in zip(rows, parts, strict=True):
        x = mx.array(ids[r])
        loss, g = compiled(params, x, ref.seg(x), mx.array(t), mx.array(w), n)
        total += float(loss)
        g64 = {key: np.array(v, np.float64) for key, v in g.items()}
        acc = g64 if acc is None else {key: acc[key] + g64[key] for key in acc}
    return total, acc


def uneven_batch(seed: int):
    ids, tg = ref.batch(seed, b=8)
    rng = np.random.default_rng(seed)
    keep = rng.uniform(0.2, 1.0, size=(8, 1))  # each row keeps a different share of its targets
    return ids, tg, (rng.uniform(size=ids.shape) < keep).astype(np.float32)


def test_grad_invariant_to_microbatch_split() -> None:
    """k ∈ {1, 2, 4, 8} give one loss and gradient; the compiled step traces once per shape."""
    cfg = ref.tiny(table=False, n_layers=2, **{"loss.z": 1e-4})
    params = init_params(cfg, seed=5)
    traces = {"n": 0}

    def step(p, x, s, t, w, n):
        traces["n"] += 1
        return loss_sum(p, cfg, hidden(p, cfg, x, s, mx.float32), t, w) / n

    out = {}
    for k in (1, 2, 4, 8):
        compiled = mx.compile(mx.value_and_grad(step))
        traces["n"] = 0
        for seed in (6, 7):  # two steps with different masks, so different N_valid values
            ids, tg, lmask = uneven_batch(seed)
            out[k, seed] = split_grads(cfg, params, ids, tg, lmask, k, compiled)
        assert traces["n"] == 1, (
            f"k={k} traced {traces['n']} times: N_valid must be data, not a constant"
        )
    for seed in (6, 7):
        la, ga = out[1, seed]
        for k in (2, 4, 8):
            lb, gb = out[k, seed]
            assert abs(lb - la) <= LOSS_REL * abs(la)
            assert max(rel(gb[key], ga[key]) for key in ga) < GRAD_REL_FP32
    # The control: a mean of per-micro-batch means, as mlx-lm takes, misses the bound.
    ids, tg, lmask = uneven_batch(6)
    rows = np.array_split(np.arange(8), 4)
    means = 0.0
    for r in rows:
        t, w = prepare_targets(tg[r], lmask[r], cfg)
        x = mx.array(ids[r])
        h = hidden(params, cfg, x, ref.seg(x), mx.float32)
        means += float(loss_sum(params, cfg, h, mx.array(t), mx.array(w))) / float(w.sum()) / 4
    assert abs(means - out[1, 6][0]) > 100 * LOSS_REL * abs(out[1, 6][0])


def test_python_count_retraces() -> None:
    """The control for passing N_valid as a Python number: each new value is a new trace."""
    traces = {"n": 0}

    def f(x, n):
        traces["n"] += 1
        return x.sum() / n

    compiled = mx.compile(f)
    x = mx.ones((4,))
    for n in (3.0, 4.0, 5.0):
        compiled(x, n)
    assert traces["n"] == 3
    traces["n"] = 0
    for n in (3.0, 4.0, 5.0):
        compiled(x, mx.array(n, mx.float32))
    assert traces["n"] == 1


@pytest.mark.parametrize(
    ("dtype", "grad_bound"), [(mx.float32, GRAD_REL_FP32), (mx.bfloat16, GRAD_REL_BF16)]
)
def test_chunked_ce_equivalence(dtype: mx.Dtype, grad_bound: float) -> None:
    """Unchunked, 256-token and 64-token chunks: one loss, and gradients within the dtype's bound."""
    ids, tg = ref.batch(8, b=8)  # 512 tokens
    lmask = np.ones(ids.shape, np.float32)
    lmask[0, :7] = 0
    x = mx.array(ids)
    out = {}
    for path, chunk in (("unchunked", 256), ("chunked", 256), ("chunked", 64)):
        cfg = ref.tiny(
            table=False, **{"loss.path": path, "loss.chunk_tokens": chunk, "loss.z": 1e-4}
        )
        t, w = prepare_targets(tg, lmask, cfg)
        params = init_params(cfg, seed=9)

        def f(p, cfg=cfg, t=t, w=w):
            return loss_sum(p, cfg, hidden(p, cfg, x, ref.seg(x), dtype), mx.array(t), mx.array(w))

        loss, g = mx.value_and_grad(f)(params)
        terms = loss_terms(
            params, cfg, hidden(params, cfg, x, ref.seg(x), dtype), *map(mx.array, (t, w))
        )
        out[path, chunk] = (
            float(loss),
            float(terms.ce),
            float(terms.lse_sq),
            {key: np.array(v, np.float64) for key, v in g.items()},
        )
    base = out["unchunked", 256]
    for key in (("chunked", 256), ("chunked", 64)):
        got = out[key]
        for i, bound in enumerate((LOSS_REL, LOSS_REL, LSE_SQ_REL)):
            assert abs(got[i] - base[i]) <= bound * abs(base[i])
        assert max(rel(got[3][name], base[3][name]) for name in base[3]) < grad_bound


def test_masked_target_finite() -> None:
    """A masked-id target scores like a masked position: same loss, same gradients, all finite."""
    cfg = ref.tiny(table=False)
    params = init_params(cfg, seed=10)
    ids, tg = ref.batch(11)
    x = mx.array(ids)
    tagged = tg.copy()
    tagged[0, 5] = 9  # the safety arm's tag as the next token
    lmask = np.ones(ids.shape, np.float32)
    lmask[0, 5] = 0
    out = []
    for targets, m in ((tagged, None), (tg, lmask)):
        t, w = prepare_targets(targets, m, cfg)

        def f(p, t=t, w=w):
            return loss_sum(
                p, cfg, hidden(p, cfg, x, ref.seg(x), mx.float32), mx.array(t), mx.array(w)
            )

        loss, g = mx.value_and_grad(f)(params)
        out.append((float(loss), {key: np.array(v) for key, v in g.items()}))
    (la, ga), (lb, gb) = out
    assert np.isfinite(la)
    assert la == lb
    assert all(np.isfinite(ga[key]).all() and np.array_equal(ga[key], gb[key]) for key in ga)


def test_loss_metrics_match_float64() -> None:
    """`train/ce_loss` and `train/z_loss_unscaled` over two micro-batches against float64."""
    cfg = ref.tiny(table=False, n_layers=2, **{"loss.z": 1e-4})
    params = init_params(cfg, seed=12)
    ids, tg, lmask = uneven_batch(13)
    P = {key: np.array(v, np.float64) for key, v in params.items()}
    rows = np.array_split(np.arange(8), 2)
    parts = [prepare_targets(tg[r], lmask[r], cfg) for r in rows]
    count, n = n_valid(w for _, w in parts)
    terms, objective = [], 0.0
    for r, (t, w) in zip(rows, parts, strict=True):
        x = mx.array(ids[r])
        h = hidden(params, cfg, x, ref.seg(x), mx.float32)
        terms.append(loss_terms(params, cfg, h, mx.array(t), mx.array(w)))
        objective += float(loss_sum(params, cfg, h, mx.array(t), mx.array(w)) / n)
    got = loss_metrics(terms, count, cfg.z_loss)
    z = ref.logits(P, cfg, ids)
    t, w = prepare_targets(tg, lmask, cfg)
    lse = ref.lse(z)
    nll = lse - np.take_along_axis(z, t[..., None].astype(np.int64), -1)[..., 0]
    assert got["train/n_valid_tokens"] == int(w.sum())
    assert abs(got["train/ce_loss"] - (nll * w).sum() / w.sum()) < METRIC_CE_ABS
    assert abs(got["train/z_loss_unscaled"] - (lse**2 * w).sum() / w.sum()) < METRIC_ZU_ABS
    assert got["train/z_loss"] == pytest.approx(1e-4 * got["train/z_loss_unscaled"], rel=1e-6)
    assert objective == pytest.approx(got["train/ce_loss"] + got["train/z_loss"], rel=1e-6)
