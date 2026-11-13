"""The dense optimiser step: its groups, AdamW guards, skip flag, step order, μP and EMA.

The step runs a small functional stand-in with the model's group shapes: a tied
embedding, a value embedding, peri-norm feed-forward blocks with ReLU², gains and
residual scalars. Every check runs on MLX's CPU device, where float64 exists.
"""

from __future__ import annotations

import gc
import math

import mlx.core as mx
import numpy as np
import pytest

from akeratos.train.loop import Stepper
from akeratos.train.optim import (
    EMA,
    DenseOptim,
    ema_beta,
    ema_for,
    sq_norm,
    width_mult,
)
from akeratos.train.recipe import OptimConfig, RecipeConfig

mx.set_default_device(mx.cpu)

VOCAB, B, T = 256, 2, 32


def half_life_steps(tokens: float, tokens_per_update: int, half_life: float = 0.02) -> float:
    """The half-life in optimiser steps."""
    return half_life * tokens / tokens_per_update


def toy(d, layers=2, seed=0, ve=True, ffn=None):
    """Parameters and groups of the stand-in model.

    Embeddings start N(0, 0.02), hidden weights uniform with standard deviation 0.5/√fan-in.
    """
    rng = np.random.default_rng(seed)
    f = ffn or 4 * d

    def unif(out, inp):
        a = math.sqrt(3) * 0.5 / math.sqrt(inp)
        return rng.uniform(-a, a, (out, inp))

    p, g = {}, {}

    def add(name, value, group):
        p[name] = mx.array(np.asarray(value, np.float32))
        g[name] = group

    add("embed.weight", rng.normal(0, 0.02, (VOCAB, d)), "embed")
    if ve:
        add("ve.0.weight", rng.normal(0, 0.02, (VOCAB, d)), "value_embed")
        add("ve.0.gate", np.full((1,), 1.5), "scalar")
    for i in range(layers):
        add(f"layers.{i}.norm.weight", np.ones(d), "scalar")
        add(f"layers.{i}.ffn.up.weight", unif(f, d), "hidden")
        add(f"layers.{i}.ffn.down.weight", unif(d, f), "hidden")
        add(f"layers.{i}.xl", np.full((1,), 0.2), "scalar")
    add("norm_f.weight", np.ones(d), "scalar")
    return p, g


def rms(x, w):
    return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w


def forward(p, ids, m_logit=1.0):
    """Logits and every block's feed-forward pre-activation."""
    x = p["embed.weight"][ids]
    if "ve.0.weight" in p:
        x = x + p["ve.0.gate"] * p["ve.0.weight"][ids]
    pre = []
    i = 0
    while f"layers.{i}.xl" in p:
        u = rms(x, p[f"layers.{i}.norm.weight"]) @ p[f"layers.{i}.ffn.up.weight"].T
        pre.append(u)
        h = mx.maximum(u, 0) ** 2 @ p[f"layers.{i}.ffn.down.weight"].T
        x = x + p[f"layers.{i}.xl"] * h
        i += 1
    return (rms(x, p["norm_f.weight"]) @ p["embed.weight"].T) * m_logit, pre


def make_loss(m_logit=1.0):
    def loss_fn(p, ids, targets, poison):
        logits, _ = forward(p, ids, m_logit)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.take_along_axis(lp, targets[..., None], axis=-1).mean() * poison

    return loss_fn


def batch(seed, b=B, t=T):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, VOCAB, (b, t + 1))
    return mx.array(ids[:, :-1], mx.int32), mx.array(ids[:, 1:], mx.int32), mx.array(1.0)


def stepper(d=64, seed=1, o=None, r=None, compile=False, **kw):
    p, g = toy(d, seed=seed, **kw)
    o, r = o or OptimConfig(), r or RecipeConfig()
    opt = DenseOptim(p, g, d, o, r)
    return Stepper(make_loss(width_mult(d, r)), p, opt, o.clip, compile=compile), g


def np64(tree):
    return {k: np.array(v, np.float64) for k, v in tree.items()}


def opt_state(st):
    """Every optimiser array (step, rate and both moments of each weight), flattened."""
    out = {}
    for i, s in enumerate(st.opt.states()):
        for k, v in s.items():
            leaves = v.items() if isinstance(v, dict) else [("", v)]
            out |= {f"{i}.{k}.{kk}": np.array(vv) for kk, vv in leaves}
    return out


# ---- AdamW guards ------------------------------------------------------------------------------


def test_each_group_has_bias_correction_explicit_decay_and_the_beta2_rule():
    st, _ = stepper(d=64)
    o = OptimConfig()
    lam_ind = 1e-3 * 0.1  # η_ref·λ at the default 1B tokens and 8,192 per update
    assert set(st.opt.opts) == {"hidden", "embed", "value_embed", "scalar"}
    for g, opt in st.opt.opts.items():
        s = opt.settings
        assert s["bias_correction"] is True
        assert s["betas"] == (0.9, pytest.approx(0.95 ** (8192 / 524_288), abs=1e-15))
        assert s["eps"] == pytest.approx(1e-8)
        if g == "hidden":
            assert opt.learning_rate == pytest.approx(o.lr_peak * 256 / 64, rel=1e-6)
            assert s["weight_decay"] * st.opt.peak[g] == pytest.approx(lam_ind, rel=1e-12)
        else:
            assert s["weight_decay"] == 0.0


def test_gradients_and_every_moment_stay_fp32():
    st, _ = stepper()
    _, grads, *_ = st.grads(*batch(3))
    assert {v.dtype for v in grads.values()} == {mx.float32}
    st.step(*batch(4))
    for state in st.opt.states():
        for name, leaf in state.items():
            if isinstance(leaf, dict):
                assert {v.dtype for v in leaf.values()} == {mx.float32}, name


def test_a_bf16_weight_is_refused():
    p, g = toy(32)
    p["embed.weight"] = p["embed.weight"].astype(mx.bfloat16)
    with pytest.raises(TypeError, match="not FP32"):
        DenseOptim(p, g, 32, OptimConfig(), RecipeConfig())


def test_a_rate_change_to_one_group_leaves_the_others():
    st, _ = stepper()
    before = {g: o.learning_rate for g, o in st.opt.opts.items()}
    st.opt.opts["embed"].learning_rate = 0.123
    after = {g: o.learning_rate for g, o in st.opt.opts.items()}
    assert after.pop("embed") == pytest.approx(0.123)
    before.pop("embed")
    assert after == before
    st.opt.set_schedule({"hidden": 0.5, "embed": 1.0, "value_embed": 0.25, "scalar": 1.0})
    for g, s in {"hidden": 0.5, "value_embed": 0.25, "scalar": 1.0}.items():
        assert st.opt.opts[g].learning_rate == pytest.approx(st.opt.peak[g] * s, rel=1e-6)


def test_groups_are_built_only_with_members():
    p, g = toy(32, ve=False)
    opt = DenseOptim(p, g, 32, OptimConfig(), RecipeConfig())
    assert set(opt.opts) == {"hidden", "embed", "scalar"}
    assert opt.peak["hidden"] == pytest.approx(3e-3 * 256 / 32)


def test_every_parameter_needs_one_known_group():
    p, g = toy(32)
    del g["norm_f.weight"]
    with pytest.raises(ValueError, match="exactly one group"):
        DenseOptim(p, g, 32, OptimConfig(), RecipeConfig())
    g["norm_f.weight"] = "table"
    with pytest.raises(ValueError, match="unknown parameter groups"):
        DenseOptim(p, g, 32, OptimConfig(), RecipeConfig())


def test_a_stepper_without_the_table_refuses_table_micro():
    st, _ = stepper(32)
    before = {k: np.array(v) for k, v in st.params.items()}
    with pytest.raises(ValueError, match="table_micro needs a Stepper built with table=True"):
        st.step(table_micro=[lambda: batch(3)], on_table_grad=lambda i, g: None)
    assert st.step_no == 0
    assert all(np.array_equal(np.array(st.params[k]), v) for k, v in before.items())


def test_a_stepper_with_the_table_refuses_a_batch_and_micro_batches():
    p, g = toy(32)
    o, r = OptimConfig(), RecipeConfig()
    st = Stepper(make_loss(), p, DenseOptim(p, g, 32, o, r), o.clip, compile=False, table=True)
    with pytest.raises(ValueError, match="built with table=True takes table_micro alone"):
        st.step(*batch(3))
    with pytest.raises(ValueError, match="built with table=True takes table_micro alone"):
        st.step(micro=[batch(3), batch(4)])
    assert st.step_no == 0


@pytest.mark.parametrize(
    ("mode", "lr", "tpu"),
    [
        ("independent", 1e-3, 8192),
        ("independent", 1e-2, 8192),
        ("independent", 3e-3, 65_536),
        ("coupled", 3e-3, 8192),
        ("coupled", 3e-3, 65_536),
    ],
)
def test_decay_shrinks_only_hidden_weights_by_the_rule(mode, lr, tpu):
    d, tokens, s = 64, 2e8, 0.5
    lam = 0.1 * (1e9 / tokens) * (tpu / 8192)  # the decay rule written out
    want = {"independent": 1e-3 * lam * s, "coupled": lr * (256 / d) * s * lam}[mode]
    p, g = toy(d, seed=5)
    o = OptimConfig(lr_peak=lr, weight_decay_mode=mode, tokens=tokens, tokens_per_update=tpu)
    opt = DenseOptim(p, g, d, o, RecipeConfig())
    opt.set_schedule(s)
    zero = {n: mx.zeros_like(v) for n, v in p.items()}
    new = opt.apply(zero, p, mx.array(1.0))
    mx.eval(new)
    for n, v in p.items():
        a, b = np.array(v, np.float64), np.array(new[n], np.float64)
        if g[n] == "hidden":
            assert 1 - float((a * b).sum() / (a * a).sum()) == pytest.approx(want, rel=1e-3)
        else:
            assert np.array_equal(a, b), n


def test_independent_decay_does_not_follow_the_peak_rate():
    assert (
        len({OptimConfig(lr_peak=lr, tokens=2e8).hidden_decay() for lr in (1e-3, 3e-3, 1e-2)}) == 1
    )


# ---- the skip flag -----------------------------------------------------------------------------


@pytest.mark.parametrize("poison", [math.inf, math.nan])
def test_a_non_finite_step_updates_no_group_and_raises_the_flag(poison):
    st, _ = stepper()
    st.step(*batch(1))
    before = np64(st.params)
    states = opt_state(st)
    assert any(k.endswith(".v") for k in states)
    calls = []
    ids, tg, _ = batch(2)
    res = st.step(ids, tg, mx.array(poison), table_update=lambda: calls.append(1))
    assert res.skipped
    assert st.skipped == 1
    assert calls == []
    assert all(np.array_equal(before[k], np.array(v, np.float64)) for k, v in st.params.items())
    assert st.opt.steps() == dict.fromkeys(st.opt.opts, 1)
    after = opt_state(st)
    assert after.keys() == states.keys()
    assert all(np.array_equal(states[k], after[k]) for k in states)
    res = st.step(*batch(3), table_update=lambda: calls.append(1))
    assert not res.skipped
    assert calls == [1]
    assert st.opt.steps() == dict.fromkeys(st.opt.opts, 2)


# ---- the step order ----------------------------------------------------------------------------


def test_one_step_equals_a_float64_reference_step():
    """NumPy float64 from the step's own FP32 gradients: clip, then AdamW per group.

    MLX raises β to the step in FP32, so the reference takes 1 - fp32(β) for the
    bias correction.
    """
    o = OptimConfig(clip=0.05)  # small enough that the clip acts
    st, groups = stepper(d=64, o=o, compile=True)
    p0 = np64(st.params)
    g32 = np64(st.grads(*batch(11))[1])
    res = st.step(*batch(11))
    norm = math.sqrt(sum(float((v**2).sum()) for v in g32.values()))
    scale = min(1.0, o.clip / norm)
    assert scale < 1
    assert res.clip_scale == pytest.approx(scale, rel=1e-5)
    assert res.norms["dense"] == pytest.approx(norm, rel=1e-5)
    assert set(res.norms) == {*groups.values(), "dense"}
    b1, b2, eps = o.beta1, o.beta2(), o.eps
    c1, c2 = 1 - float(np.float32(b1)), 1 - float(np.float32(b2))
    worst_p, worst_ulp = 0.0, 0.0
    for k, p in p0.items():
        grp, gk = groups[k], g32[k] * scale
        lr = st.opt.peak[grp]
        mhat, vhat = (1 - b1) * gk / c1, (1 - b2) * gk * gk / c2
        want = p * (1 - st.opt.decay_ind[grp]) - lr * mhat / (np.sqrt(vhat) + eps)
        got = np.array(st.params[k], np.float64)
        worst_p = max(worst_p, float(np.linalg.norm(got - want) / np.linalg.norm(want)))
        ulp = np.spacing(np.maximum(np.abs(p), np.abs(want - p)).astype(np.float32)).astype(
            np.float64
        )
        worst_ulp = max(worst_ulp, float((np.abs(got - want) / ulp).max()))
    assert worst_p < 1e-6
    assert worst_ulp <= 8


def test_two_runs_from_one_seed_are_bitwise_equal():
    runs = []
    for _ in range(2):
        st, _ = stepper(seed=2)
        for i in range(3):
            st.step(*batch(20 + i))
        runs.append(np64(st.params))
    assert all(np.array_equal(runs[0][k], runs[1][k]) for k in runs[0])


def test_the_one_host_read_of_the_norms_equals_the_per_group_reads(monkeypatch):
    """`sq.tolist()` gives each group's norm as `float(sq[i]) ** 0.5`, value for value."""
    st, _ = stepper(seed=2)
    seen = []
    grads = st.grads

    def spy(*b):
        out = grads(*b)
        seen.append(out[2])
        return out

    monkeypatch.setattr(st, "grads", spy)
    result = st.step(*batch(40))
    assert not result.skipped
    (sq,) = seen
    want = {g: float(sq[i]) ** 0.5 for i, g in enumerate(st.opt.names)}
    got = {g: result.norms[g] for g in st.opt.names}
    assert [v.hex() for v in got.values()] == [v.hex() for v in want.values()]


def test_the_update_is_built_after_the_gradients_are_evaluated(monkeypatch):
    """By memory: after the gradient phase only the gradients are held; the update adds no activations.

    The gradient phase is the call to `grads`. The dense phase runs from `set_schedule`, which the
    step calls after the gradients and before the update, to the table update, which follows it.
    """
    st, _ = stepper(d=64, layers=2, ffn=256, compile=True)
    big = batch(13, b=16, t=512)
    st.step(*big)  # compile and allocate the optimiser state first
    rec = {}

    def begin(name):
        gc.collect()
        rec[name] = {"start": mx.get_active_memory()}
        mx.reset_peak_memory()

    def end(name):
        gc.collect()
        rec[name] |= {"peak": mx.get_peak_memory(), "end": mx.get_active_memory()}

    grads, set_schedule = st.grads, st.opt.set_schedule

    def gradient_phase(*b):
        begin("grads")
        out = grads(*b)
        end("grads")
        return out

    def dense_phase(s):
        begin("dense")
        set_schedule(s)

    def table_update():
        end("dense")
        begin("table")

    monkeypatch.setattr(st, "grads", gradient_phase)
    monkeypatch.setattr(st.opt, "set_schedule", dense_phase)
    st.step(*batch(14, b=16, t=512), table_update=table_update)
    grads_bytes = 4 * sum(v.size for v in st.params.values())
    g, d = rec["grads"], rec["dense"]
    held = g["end"] - g["start"]
    activations = g["peak"] - g["start"] - held
    assert held == pytest.approx(grads_bytes, rel=0.1)
    assert activations > 4 * grads_bytes
    assert d["peak"] - d["start"] < activations / 2
    assert list(rec) == ["grads", "dense", "table"]


def test_two_stage_norm_is_accurate_at_the_tied_matrix_shape():
    x = (np.random.default_rng(5).standard_normal((49_152, 256)) * 1e-2).astype(np.float32)
    ref = float((x.astype(np.float64) ** 2).sum())
    assert float(sq_norm(mx.array(x))) == pytest.approx(ref, rel=1e-5)


# ---- the μP coordinate check -------------------------------------------------------------------


def _probe(p, ids, m):
    logits, pre = forward(p, ids, m)
    out = {"logits": logits} | {f"u{i}": u for i, u in enumerate(pre)}
    mx.eval(out)
    return np64(out)


def _coord_run(d, kind, steps=3):
    r = RecipeConfig(param_kind=kind)
    o = OptimConfig(tokens_per_update=128)
    p, g = toy(d, seed=1, ve=False, ffn=d)
    m = width_mult(d, r)
    st = Stepper(make_loss(m), p, DenseOptim(p, g, d, o, r), o.clip, compile=False)
    ids = batch(5)[0]
    base = _probe(st.params, ids, m)
    out = []
    for t in range(steps):
        st.step(*batch(100 + t))
        now = _probe(st.params, ids, m)
        out.append({k: float(np.sqrt(np.mean((now[k] - base[k]) ** 2))) for k in base})
    return out


def _slopes(kind, widths=(256, 512, 768)):
    runs = {d: _coord_run(d, kind) for d in widths}
    x = np.log(widths)
    return [
        float(np.polyfit(x, np.log([runs[d][t][k] for d in widths]), 1)[0])
        for t in range(len(runs[widths[0]]))
        for k in runs[widths[0]][t]
    ]


def test_mup_coordinate_check_at_widths_256_512_768():
    """μP keeps every coordinate's change Θ(1) in width; sp grows."""
    assert max(abs(s) for s in _slopes("mup")) <= 0.3
    assert min(_slopes("sp")) >= 0.2


def test_mup_equals_sp_at_the_base_width():
    res = []
    for kind in ("mup", "sp"):
        st, _ = stepper(
            d=256, r=RecipeConfig(param_kind=kind), o=OptimConfig(tokens_per_update=128)
        )
        for i in range(2):
            st.step(*batch(40 + i))
        res.append(np64(st.params))
    assert all(np.array_equal(res[0][k], res[1][k]) for k in res[0])


# ---- EMA ---------------------------------------------------------------------------------------


def test_ema_matches_its_float64_closed_form():
    st, _ = stepper(seed=6)
    beta = 0.7
    ema = EMA(st.params, beta, st.params)
    hist = [np64(st.params)]
    for t in range(5):
        if not st.step(*batch(70 + t)).skipped:
            ema.update(st.params)
            hist.append(np64(st.params))
    k = len(hist) - 1
    for n in hist[0]:
        want = beta**k * hist[0][n] + sum(
            (1 - beta) * beta ** (k - i) * hist[i][n] for i in range(1, k + 1)
        )
        got = np.array(ema.avg[n], np.float64)
        assert np.linalg.norm(got - want) <= 1e-6 * max(np.linalg.norm(want), 1e-30)
    assert k == 5
    assert abs(half_life_steps(2.6e9, 8192) - 6347.66) < 0.01
    assert ema_beta(2.6e9, 8192) ** half_life_steps(2.6e9, 8192) == pytest.approx(0.5)


def test_ema_refuses_a_non_dense_name_and_is_off_when_disabled():
    st, groups = stepper()
    with pytest.raises(ValueError, match="dense weights only"):
        EMA(st.params | {"table.shard.0": mx.zeros((4, 4))}, 0.9, groups)
    assert ema_for(st.params, groups, OptimConfig(), RecipeConfig(ema=False)) is None
    ema = ema_for(st.params, groups, OptimConfig(), RecipeConfig())
    assert ema is not None
    assert ema.beta == ema_beta(1e9, 8192)


def test_training_with_and_without_the_ema_is_bitwise_equal():
    res = []
    for use in (False, True):
        st, groups = stepper(seed=7)
        ema = EMA(st.params, 0.9, groups) if use else None
        for t in range(3):
            st.step(*batch(90 + t))
            if ema is not None:
                ema.update(st.params)
        res.append(np64(st.params))
    assert all(np.array_equal(res[0][k], res[1][k]) for k in res[0])
