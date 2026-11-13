"""The model's trunk against float64 oracles.

Parameter counts equal the goldens; a block's forward pass and gradients match float64; banded
attention equals the full mask; an absent switch equals the switch off; the initial statistics
and values are right. The oracle is the NumPy float64 model in `tests/unit/model_ref.py`; the
MLX model runs in FP32 here (BF16 where a test says so), because float64 never runs in
`akeratos/`.
"""

import mlx.core as mx
import numpy as np
import pytest

from akeratos import model
from akeratos.configs import SWITCHES, ModelConfig, describe, from_keys
from akeratos.model import (
    RULES,
    apply_rope,
    build_masks,
    forward,
    hidden,
    init_params,
    kind_of,
    param_counts,
    param_shapes,
    rope_freqs,
)
from akeratos.train.loop import loss_sum, prepare_targets
from tests.unit import model_ref as ref

mx.set_default_device(mx.cpu)

# The seven presets rows and their goldens.
ROWS = {
    "core-10m": (
        (256, 8, 4, 64, 2, 1152),
        (12_582_912, 25_165_824, 1_572_864, 4_718_592, 9_472, 112, 44_049_776, 18_874_368),
    ),
    "core-30m": (
        (384, 16, 6, 64, 2, 1792),
        (18_874_368, 50_331_648, 6_291_456, 22_020_096, 27_008, 224, 97_544_800, 47_185_920),
    ),
    "core-150m": (
        (768, 18, 6, 128, 2, 3456),
        (37_748_736, 113_246_208, 28_311_552, 95_551_488, 60_672, 252, 274_918_908, 161_611_776),
    ),
    "core-340m": (
        (1024, 24, 8, 128, 2, 4608),
        (50_331_648, 150_994_944, 62_914_560, 226_492_416, 105_472, 336, 490_839_376, 339_738_624),
    ),
    "core-1b": (
        (1536, 32, 12, 128, 4, 6912),
        (
            75_497_472,
            402_653_184,
            201_326_592,
            679_477_248,
            206_336,
            832,
            1_359_161_664,
            956_301_312,
        ),
    ),
    "core-1b-wide": (
        (2048, 18, 16, 128, 4, 9216),
        (
            100_663_296,
            226_492_416,
            188_743_680,
            679_477_248,
            154_112,
            468,
            1_195_531_220,
            968_884_224,
        ),
    ),
    "core-1b-deep": (
        (1280, 46, 10, 128, 5, 5760),
        (
            62_914_560,
            723_517_440,
            226_099_200,
            678_297_600,
            248_576,
            1_472,
            1_691_078_848,
            967_311_360,
        ),
    ),
}
PARTS = ("embed", "value_embed", "attention", "ffn", "norms", "scalars", "dense", "active")
SHAPE = ("d_model", "n_layers", "n_heads", "head_dim", "n_kv_heads", "ffn")


def shape(name: str, **keys: object) -> ModelConfig:
    return from_keys({**dict(zip(SHAPE, ROWS[name][0], strict=True)), **keys})


def to64(params: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float64) for k, v in params.items()}


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-300))


def loss_and_grads(cfg, params, ids, tg, dtype=mx.float32):
    targets, weights = prepare_targets(tg, None, cfg)
    t, w, x = mx.array(targets), mx.array(weights), mx.array(ids)

    def f(p):
        return loss_sum(p, cfg, hidden(p, cfg, x, ref.seg(x), dtype), t, w) / float(weights.sum())

    loss, grads = mx.value_and_grad(f)(params)
    return float(np.array(loss)), {k: np.array(v, dtype=np.float64) for k, v in grads.items()}


# ---- parameter counts --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROWS)
def test_param_counts_equal_the_goldens(name: str) -> None:
    """Every part of each presets row, from the parameter shapes alone (no array is built)."""
    counts = param_counts(shape(name))
    assert tuple(counts[p] for p in PARTS) == ROWS[name][1]


def test_built_parameters_match_their_shapes() -> None:
    cfg = ref.tiny()
    params = init_params(cfg)
    assert {k: tuple(v.shape) for k, v in params.items()} == param_shapes(cfg)
    assert all(v.dtype == mx.float32 for v in params.values())


def test_glu_parity_width_gives_equal_counts() -> None:
    """GeGLU at 3·d against ReLU² at 4.5·d; the 30M row rounds its 1,728 to 1,792."""
    relu2, glu = (
        param_counts(shape("core-10m")),
        param_counts(shape("core-10m", **{"ffn.kind": "glu"})),
    )
    assert relu2["ffn"] == glu["ffn"]


def test_global_and_value_embedding_layers_follow_the_rule() -> None:
    cfg = shape("core-1b")
    assert [i for i in range(32) if cfg.is_global(i)] == [4, 9, 14, 19, 24, 29, 31]
    assert cfg.ve_layers() == list(range(1, 32, 2))


# ---- a block against float64 -------------------------------------------------------------------


@pytest.mark.parametrize("placement", ["peri", "pre", "post"])
def test_forward_matches_float64(placement: str) -> None:
    """FP32 logits against the NumPy float64 model, every layer type and value embeddings."""
    cfg = ref.tiny(**{"table": False, "norm.placement": placement})
    params = init_params(cfg)
    ids, _ = ref.batch(0)
    z_ref = ref.logits(to64(params), cfg, ids)
    z = np.array(forward(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    live = np.isfinite(z_ref)
    assert (~live).sum() == ids.size * 16
    assert rel(z[live], z_ref[live]) < 1e-5


@pytest.mark.parametrize("placement", ["peri", "pre", "post"])
def test_vjp_matches_float64_differences(placement: str) -> None:
    """FP32 gradients of every tensor kind against central differences of the float64 loss."""
    cfg = ref.tiny(**{"table": False, "norm.placement": placement})
    params = init_params(cfg, seed=1)
    ids, tg = ref.batch(1)
    loss, grads = loss_and_grads(cfg, params, ids, tg)
    P = to64(params)
    assert abs(loss - ref.loss(P, cfg, ids, tg)) < 1e-5
    rng = np.random.default_rng(1)
    worst = 0.0
    for name in sorted(P):
        g = grads[name]
        scale = np.abs(g).max()
        coords = np.flatnonzero(np.abs(g) > 1e-3 * scale)
        for j in rng.choice(coords, size=min(2, len(coords)), replace=False):
            h = 1e-5 * max(1.0, abs(P[name].flat[j]))
            hi, lo = dict(P), dict(P)
            hi[name], lo[name] = P[name].copy(), P[name].copy()
            hi[name].flat[j] += h
            lo[name].flat[j] -= h
            fd = (ref.loss(hi, cfg, ids, tg) - ref.loss(lo, cfg, ids, tg)) / (2 * h)
            worst = max(worst, abs(fd - g.flat[j]) / scale)
    assert worst < 1e-4


def test_rope_periods_match_numpy_and_base() -> None:
    cfg = ref.tiny()
    x = mx.random.normal((2, 3, 40, 16), key=mx.random.key(4))
    for is_global, theta in ((False, 1e4), (True, 1e6)):
        got = np.array(apply_rope(x, rope_freqs(cfg, is_global)))
        want = ref.rope(np.array(x, np.float64), theta)
        base = np.array(mx.fast.rope(x, 16, traditional=False, base=theta, scale=1.0, offset=0))
        assert np.abs(got - want).max() < 1e-5
        assert np.abs(got - base).max() < 1e-5


def test_partial_rotation_leaves_the_slow_pairs_unchanged() -> None:
    """`rope.global_fraction` 0.25 rotates the fastest 2 of 8 pairs on global layers only."""
    cfg = ref.tiny(**{"rope.global_fraction": 0.25})
    x = mx.random.normal((1, 2, 24, 16), key=mx.random.key(5))
    got = np.array(apply_rope(x, rope_freqs(cfg, True)))
    want = ref.rope(np.array(x, np.float64), 1e6, 0.25)
    assert np.abs(got - want).max() < 1e-5
    assert np.array_equal(got[..., 2:8], np.array(x)[..., 2:8])
    assert np.array_equal(got[..., 10:], np.array(x)[..., 10:])
    sliding = np.array(apply_rope(x, rope_freqs(cfg, False)))
    assert np.abs(sliding - ref.rope(np.array(x, np.float64), 1e4)).max() < 1e-5


# ---- banded attention equals the full mask -----------------------------------------------------


def test_banded_windows_equal_the_full_mask() -> None:
    """W 16 in 4 bands against the full [T, T] mask with the window, FP32."""
    ids, _ = ref.batch(2)
    out = {}
    for path, key in (("banded", "band"), ("masked", "sliding")):
        cfg = ref.tiny(**{"table": False, "attn.path": path})
        assert set(build_masks(ref.seg(mx.array(ids)), cfg)) == {"global", key}
        out[path] = np.array(
            forward(init_params(cfg), cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32)
        )
    live = np.isfinite(out["masked"])
    assert np.abs(out["banded"][live] - out["masked"][live]).max() < 1e-5


def test_banded_windows_refuse_a_ragged_length() -> None:
    cfg = ref.tiny(**{"table": False})
    with pytest.raises(ValueError, match="multiple of the window"):
        forward(
            init_params(cfg),
            cfg,
            mx.zeros((1, 40), mx.int32),
            ref.seg(mx.zeros((1, 40), mx.int32)),
            mx.float32,
        )


@pytest.mark.parametrize("path", ["banded", "masked", "fused"])
def test_no_attention_crosses_a_document(path: str) -> None:
    """Rewriting every earlier document of a row leaves the last documents' logits unchanged."""
    ids, _ = ref.batch(3)
    seps = np.flatnonzero(ids[0] == 0)
    s = int(seps[np.argmin(np.abs(seps - ref.T // 2))])
    ids2 = ids.copy()
    body = ids2[0, :s] != 0
    ids2[0, :s][body] = np.random.default_rng(9).integers(17, 512, size=int(body.sum()))
    for doc, moves in ((True, False), (False, True)):
        cfg = ref.tiny(
            **{"table": False, "attn.path": path, "mask.packing": "document" if doc else "causal"}
        )
        params = init_params(cfg)
        a = np.array(forward(params, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))[
            0, s:, 17:
        ]
        b = np.array(forward(params, cfg, mx.array(ids2), ref.seg(mx.array(ids2)), mx.float32))[
            0, s:, 17:
        ]
        diff = np.abs(a - b).max()
        assert (diff > 1e-3) if moves else (diff == 0.0)


# ---- initialisation ----------------------------------------------------------------------------


def test_init_statistics_within_two_percent() -> None:
    """The 10M core's random tensors, statistics in float64."""
    cfg = shape("core-10m")
    worst_std = worst_mean = 0.0
    for name, value in init_params(cfg).items():
        kind = kind_of(name)
        p = np.array(value, np.float64)
        if kind == "embed":
            target = 0.02
        elif kind == "value_embed":
            target = 0.5 / np.sqrt(cfg.d_model)
        elif kind in ("attn", "ffn"):
            target = 0.5 / np.sqrt(p.shape[1])
        else:
            continue
        worst_std = max(worst_std, abs(p.std() / target - 1))
        worst_mean = max(worst_mean, abs(p.mean()) / target)
    assert worst_std < 0.02
    assert worst_mean < 0.02


def test_fixed_initial_values() -> None:
    """xl 0.20 → 0.05 linear; tl 0.1; value-embedding gates 0 (gate 1.5); every gain 1."""
    P = to64(init_params(shape("core-10m")))
    assert np.allclose(P["xl"], np.linspace(0.2, 0.05, 8))
    assert np.all(P["tl"] == np.float32(0.1))
    assert all((v == 0).all() for k, v in P.items() if kind_of(k) == "ve_gate")
    assert all((v == 1).all() for k, v in P.items() if kind_of(k) == "gain")


def test_block_switches_build() -> None:
    """Pre-norm with zero output projections; output gains at 0.1."""
    pre = init_params(ref.tiny(**{"norm.placement": "pre", "init.out_proj": "zero"}))
    assert all(
        float(mx.abs(v).max()) == 0
        for k, v in pre.items()
        if k.split(".")[-2:-1] in (["o"], ["down"])
    )
    out = init_params(ref.tiny(**{"norm.out_gain_init": 0.1}))
    gains = {round(float(v[0]), 4) for k, v in out.items() if k.endswith("_out.weight")}
    assert gains == {0.1}


# ---- an absent switch equals the switch off ----------------------------------------------------


def test_absent_keys_equal_explicit_off() -> None:
    """Every switch at its off value, spelled out, gives bitwise the same parameters and step."""
    off = {
        "norm.placement": "peri",
        "norm.out_gain_init": 1.0,
        "ffn.kind": "relu2",
        "residual.x0_inject": True,
        "init.out_proj": "fan_in",
        "attn.window": 16,
        "attn.path": "banded",
        "rope.global_fraction": 1.0,
        "mask.packing": "document",
        "embed.input_scale": "norm",
        "value_embed.layers": "odd",
        "loss.final_softcap": 15.0,
        "head.masked_ids": list(range(1, 17)),
        "loss.z": 0.0,
        "loss.path": "unchunked",
        "loss.chunk_tokens": 256,
        "table": True,
        "table.sites": "all",
        "table.site_scale": True,
        "param.kind": "mup",
    }
    assert set(off) == set(SWITCHES)
    a, b = ref.tiny(), ref.tiny(**off)
    assert a == b
    ids, tg = ref.batch(4)
    pa, pb = init_params(a), init_params(b)
    assert all(np.array_equal(np.array(pa[k]), np.array(pb[k])) for k in pa)
    la, ga = loss_and_grads(a, pa, ids, tg)
    lb, gb = loss_and_grads(b, pb, ids, tg)
    assert la == lb
    assert all(np.array_equal(ga[k], gb[k]) for k in ga)


def test_x0_off_equals_zero_xl() -> None:
    ids, _ = ref.batch(4)
    off = ref.tiny(**{"residual.x0_inject": False})
    on = ref.tiny()
    p_on = init_params(on)
    p_on["xl"] = mx.zeros_like(p_on["xl"])
    a = np.array(forward(init_params(off), off, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    b = np.array(forward(p_on, on, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    assert np.array_equal(a, b)


def test_table_term_off_equals_zero_tl() -> None:
    """With tl = 0 the table term changes nothing."""
    ids, _ = ref.batch(4)
    cfg = ref.tiny()
    p = init_params(cfg)
    t = mx.random.normal((ref.B, ref.T, 64), key=mx.random.key(2))
    p0 = {**p, "tl": mx.zeros_like(p["tl"])}
    a = np.array(forward(p0, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32, t))
    b = np.array(forward(p, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))
    c = np.array(
        forward(
            init_params(ref.tiny(table=False)),
            ref.tiny(table=False),
            mx.array(ids),
            ref.seg(mx.array(ids)),
            mx.float32,
        )
    )
    assert np.array_equal(a, b)
    assert np.array_equal(b, c)


def test_switching_off_leaves_other_tensors_unchanged() -> None:
    """Initialisation is keyed by name, so removing value embeddings shifts nothing else."""
    on, off = (
        init_params(ref.tiny()),
        init_params(ref.tiny(**{"value_embed.layers": "off", "table": False})),
    )
    shared = [k for k in off if k in on]
    assert len(shared) == len(off)
    assert all(np.array_equal(np.array(on[k]), np.array(off[k])) for k in shared)
    assert not any(kind_of(k) in ("value_embed", "ve_gate") for k in off)


def test_a_zero_gate_is_not_off() -> None:
    ids, _ = ref.batch(4)
    cfg = ref.tiny()
    p = {
        k: (mx.zeros_like(v) if kind_of(k) == "ve_gate" else v) for k, v in init_params(cfg).items()
    }
    off = ref.tiny(**{"value_embed.layers": "off"})
    a = np.array(forward(p, cfg, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))[..., 17:]
    b = np.array(forward(init_params(off), off, mx.array(ids), ref.seg(mx.array(ids)), mx.float32))[
        ..., 17:
    ]
    assert np.abs(a - b).max() > 1e-3


# ---- configuration -----------------------------------------------------------------------------


def test_zero_output_projection_under_peri_norm_is_refused() -> None:
    with pytest.raises(ValueError, match=r"init\.out_proj zero"):
        ref.tiny(**{"init.out_proj": "zero"})


def test_unknown_key_is_refused() -> None:
    with pytest.raises(KeyError, match=r"no\.such\.switch"):
        ref.tiny(**{"no.such.switch": 1})


def test_one_log_line_per_switch() -> None:
    lines = describe(ref.tiny(**{"mask.packing": "causal"}))
    assert len(lines) == len(SWITCHES)
    assert "switch mask.packing = False" in lines


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"norm.placement": "pre", "ffn.kind": "glu"},
        {"norm.placement": "post", "table.sites": "all"},
        {"residual.x0_inject": False, "table": False},
    ],
)
def test_every_parameter_matches_exactly_one_rule(keys: dict[str, object]) -> None:
    for name in param_shapes(ref.tiny(**keys)):
        hits = [p for p, _ in RULES if model._matches(p, name)]
        assert len(hits) == 1, (name, hits)


def test_unmatched_and_doubly_matched_paths_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="matches 0 rules"):
        kind_of("layers.0.attn.sink")
    monkeypatch.setattr(model, "RULES", (*RULES, ("layers.*.attn.*.weight", "attn")))
    with pytest.raises(ValueError, match="matches 2 rules"):
        kind_of("layers.0.attn.q.weight")


def test_masks_follow_the_batch_segment_ids() -> None:
    """The masks come from `segment_ids`, never from the separators in ids."""
    ids, _ = ref.batch(5)
    cfg = ref.tiny(table=False)
    params = init_params(cfg)
    seg = np.cumsum(ids == 0, axis=1)

    def logits(segment_ids: np.ndarray) -> np.ndarray:
        z = forward(params, cfg, mx.array(ids), mx.array(segment_ids), mx.float32)
        return np.array(z)[..., 17:]

    base = logits(seg)
    assert np.array_equal(base, logits(seg + 7))  # only equality of ids matters
    assert np.abs(logits(np.zeros_like(seg)) - base).max() > 1e-3
    split = seg.copy()
    split[0, ref.T // 2 :] += 100  # a segment start with no separator in the ids
    assert np.abs(logits(split) - base).max() > 1e-3


@pytest.mark.parametrize(
    ("keys", "tl"),
    [
        ({"n_layers": 8}, 0.1),
        ({"n_layers": 16}, 0.05),
        ({"n_layers": 32}, 0.025),
        ({"n_layers": 16, "table.site_scale": False}, 0.1),
    ],
)
def test_entry_scalars_start_at_the_site_scaled_value(keys: dict[str, object], tl: float) -> None:
    """tl starts at 0.1 x 8/L under `table.site_scale`, 0.1 without it."""
    got = np.array(init_params(ref.tiny(**keys))["tl"])
    np.testing.assert_allclose(got, tl, rtol=1e-6)
