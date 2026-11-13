"""The model's schedules, β₂ rule, decay rule and recipe keys.

Each schedule equals its closed form, written out on its own in `closed_form`, and the rules
give their worked values.
"""

from __future__ import annotations

import math
from fractions import Fraction

import mlx.core as mx
import numpy as np
import pytest

from akeratos.train.optim import Schedule, schedule_for
from akeratos.train.recipe import (
    KEYS,
    OptimConfig,
    RecipeConfig,
    beta2_halflife,
    describe,
    from_keys,
    steps_for,
)

mx.set_default_device(mx.cpu)


def halflife_tokens(beta2: float, tokens_per_update: int) -> float:
    """Tokens over which the second moment's weight halves."""
    return math.log(0.5) / math.log(beta2) * tokens_per_update


def closed_form(kind, n, w, d, t, shape="1-sqrt", f=0.0):
    """The schedule written out on its own, in exact fractions where it can be."""
    if t < w:
        return float(Fraction(t + 1, w))
    if kind == "constant":
        return 1.0
    if kind == "wsd":
        if t < n - d:
            return 1.0
        u = min(Fraction(t - (n - d) + 1, d), Fraction(1))
    else:
        u = min(Fraction(t - w + 1, n - w), Fraction(1))
    if kind == "cosine":
        return f + (1 - f) * 0.5 * (1 + math.cos(math.pi * float(u)))
    k = math.sqrt(float(u)) if (kind == "wsd" and shape == "1-sqrt") else float(u)
    return f + (1 - f) * (1 - k)


CASES = [
    ("wsd", 1000, 50, 200, "1-sqrt", 0.0),
    ("wsd", 1000, 50, 200, "linear", 0.1),
    ("d2z", 1000, 50, 0, "1-sqrt", 0.0),
    ("cosine", 1000, 50, 0, "1-sqrt", 0.01),
    ("constant", 1000, 50, 0, "1-sqrt", 0.0),
    ("wsd", 1000, 0, 200, "1-sqrt", 0.0),
    ("wsd", 317_383, 3_174, 63_477, "1-sqrt", 0.0),
]


def boundaries(s):
    """The steps where the multiplier changes form."""
    b = {0, s.warmup, s.steps - 1, s.steps}
    if s.kind != "constant":
        b.add(s.span()[0])
    return sorted(x for x in b if x >= 0)


def probe_steps(s):
    if s.steps <= 2000:
        return list(range(s.steps + 2))
    return sorted({t for b in boundaries(s) for t in (b - 1, b, b + 1) if t >= 0})


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}-{c[1]}-{c[2]}-{c[4]}" for c in CASES])
def test_schedule_matches_the_float64_closed_form(case):
    kind, n, w, d, shape, f = case
    s = Schedule(n, w, d, kind, shape, f).validate()
    ts = probe_steps(s)
    ref = np.array([closed_form(kind, n, w, d, t, shape, f) for t in ts])
    assert np.abs(np.array([s(t) for t in ts]) - ref).max() <= 1e-15


@pytest.mark.parametrize(
    ("tokens", "tpu", "steps", "warmup", "decay", "beta2"),
    [
        (2.6e9, 8192, 317_383, 3_174, 63_477, 0.99920),
        (1e9, 8192, 122_071, 2_497, 24_415, 0.99920),
        (3e9, 65_536, 45_777, 1_000, 9_156, 0.99361),
        (8e9, 65_536, 122_071, 1_221, 24_415, 0.99361),
        (20e9, 65_536, 305_176, 3_052, 61_036, 0.99361),
    ],
)
def test_table_10_steps_warmup_decay_and_beta2(tokens, tpu, steps, warmup, decay, beta2):
    o = OptimConfig(tokens=tokens, tokens_per_update=tpu)
    s = schedule_for(RecipeConfig(), o)
    assert (s.steps, s.warmup, s.decay_steps, round(o.beta2(), 5)) == (steps, warmup, decay, beta2)


def test_warmup_steps_and_the_floor_switch():
    s = Schedule(1000, 50, 200)
    assert [s(t) for t in (48, 49)] == [0.98, 1.0]
    assert s(50) == 1.0
    assert s(5000) == 0.0
    with pytest.raises(ValueError, match="negative"):
        s(-1)


@pytest.mark.parametrize(("tpu", "want"), [(8192, 0.99920), (14_336, 0.99860), (65_536, 0.99361)])
def test_dense_beta2_holds_its_half_life_in_tokens(tpu, want):
    b = beta2_halflife(tpu)
    assert round(b, 5) == want
    assert halflife_tokens(b, tpu) == pytest.approx(
        math.log(0.5) / math.log(0.95) * 524_288, rel=1e-12
    )
    assert round(halflife_tokens(b, tpu) / 1e6, 2) == 7.08


def test_steps_round_up_and_beta2_fixed_rule():
    assert steps_for(2.6e9, 8192) == 317_383
    assert OptimConfig(beta2_rule="fixed", beta2_fixed=0.99).beta2() == 0.99


@pytest.mark.parametrize(
    ("kw", "want"),
    [
        ({"tokens": 2e8}, 5.0e-4),
        ({"tokens": 1e9}, 1.0e-4),
        ({"tokens": 2.6e9}, 3.85e-5),
        ({"tokens": 6e8, "decay_tokens": 7.2e8}, 1.39e-4),
        ({"tokens": 3e9, "tokens_per_update": 65_536}, 2.67e-4),
        ({"tokens": 2e10, "tokens_per_update": 65_536}, 4.0e-5),
        ({"tokens": 1e9, "tokens_per_update": 14_336}, 1.75e-4),
    ],
)
def test_decay_rule_worked_values_and_share(kw, want):
    """λ_ind = 1e-3·0.1·(1e9/D_tok)·(B/8,192); 1/λ_ind is 8.192% of D_tok/B steps."""
    o = OptimConfig(**kw).validate()
    d_tok = kw.get("decay_tokens", kw["tokens"])
    assert float(f"{o.hidden_decay():.3g}") == want
    assert 1 / (o.hidden_decay() * d_tok / o.tokens_per_update) == pytest.approx(0.08192, abs=1e-12)


@pytest.mark.parametrize(
    ("kw", "refused"),
    [
        ({"tokens": 1e5}, True),
        ({"tokens": 1.01e5}, False),
        ({"tokens": 8e5, "tokens_per_update": 65_536}, True),
        ({"tokens": 8.1e5, "tokens_per_update": 65_536}, False),
        ({"tokens": 1e5, "decay_ind": 3e-4}, False),
    ],
)
def test_the_validator_refuses_lambda_ind_at_or_above_one(kw, refused):
    if refused:
        with pytest.raises(ValueError, match="λ_ind"):
            OptimConfig(**kw).validate()
    else:
        OptimConfig(**kw).validate()


def test_a_branch_forks_only_from_the_stable_phase():
    wsd = schedule_for(RecipeConfig(), OptimConfig(tokens=7.2e8))
    assert wsd.branch(70_312).steps == 70_312 + 14_063
    with pytest.raises(ValueError, match="stable phase"):
        wsd.branch(70_313)
    with pytest.raises(ValueError, match="after warm-up"):
        wsd.branch(wsd.warmup - 1)
    trunk = schedule_for(RecipeConfig(schedule="constant"), OptimConfig(tokens=6e8))
    assert trunk.steps == 73_243
    assert trunk.branch(24_415).steps * 8192 / 1e9 == pytest.approx(0.240, abs=5e-4)
    assert trunk.branch(73_243).steps * 8192 / 1e9 == pytest.approx(0.720, abs=5e-4)


@pytest.mark.parametrize("kind", ["step", "wsd2"])
def test_unknown_schedules_and_misfits_are_refused(kind):
    with pytest.raises(ValueError, match="schedule"):
        Schedule(100, 5, 10, kind).validate()
    with pytest.raises(ValueError, match="do not fit"):
        Schedule(10, 8, 4).validate()


def test_recipe_keys_default_log_and_round_trip():
    o, r = from_keys({})
    assert (o, r) == (OptimConfig(), RecipeConfig())
    log = describe(o, r)
    assert len(log) == len(KEYS)
    assert all(any(line.startswith(f"switch {k} = ") for line in log) for k in KEYS)
    vals = {
        "lr.peak": 1e-2,
        "lr.schedule": "d2z",
        "lr.warmup": 0,
        "opt.weight_decay_mode": "coupled",
        "opt.clip_norm": 0.5,
        "param.kind": "sp",
        "param.mup_base_d": 384,
        "ema.enabled": False,
        "ema.half_life": 0.05,
        "opt.decay_tokens": 7.2e8,
    }
    o2, r2 = from_keys(vals)
    for k, v in vals.items():
        part, name = KEYS[k]
        assert getattr(o2 if part == "optim" else r2, name) == v


@pytest.mark.parametrize(
    ("keys", "exc"),
    [
        ({"lr.nope": 1}, KeyError),
        ({"param.depth_scaling": "completep"}, ValueError),
        ({"param.kind": "umup"}, ValueError),
        ({"opt.weight_decay_mode": "l2"}, ValueError),
        ({"opt.beta2_rule": "cooldown"}, ValueError),
        ({"ema.half_life": 1.5}, ValueError),
    ],
)
def test_recipe_keys_refuse_unknown_values(keys, exc):
    with pytest.raises(exc):
        from_keys(keys)
