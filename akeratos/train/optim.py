"""The optimiser: the learning-rate schedule, the parameter groups, AdamW per group and the EMA.

| Group | Members | Rate (s(t) multiplies it) | Decay |
|---|---|---|---|
| hidden | q, k, v, o; feed-forward up and down; experts | η_peak·(256/d) (μP) | independent: w·(1 - λ_ind·s(t)) |
| embed | the tied matrix E | η_emb (μP multiplier 1) | 0 |
| value_embed | the value-embedding tables | η_emb | 0 |
| scalar | norm gains, xl, tl, value-embedding gates, the attention output gate | η_peak | 0 |

A group's rate is its peak times s(t), and a group with no members is not built. The table's
rows are not a group: their gradient never enters an optimiser, the clip or a dense checkpoint;
the table has its own row-wise Adam. The clip is a global FP32 norm over the dense groups only.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from akeratos._compat.mlx_seam import AdamWGroup, evaluate
from akeratos.model import kind_of
from akeratos.numerics import assert_fp32
from akeratos.train.recipe import OptimConfig, RecipeConfig, steps_for

KINDS = ("wsd", "d2z", "cosine", "constant")
SHAPES = ("1-sqrt", "linear")
_MIN_WARMUP = 1000


def ceil_share(frac: float, n: int) -> int:
    """ceil(frac·n) without a spurious +1 from binary rounding."""
    return math.ceil(round(frac * n, 9))


def warmup_rule(steps: int, beta2: float) -> int:
    """W = max(ceil(1% of N), 1,000, ceil(2/(1 - β₂)))."""
    return max(ceil_share(0.01, steps), _MIN_WARMUP, math.ceil(round(2 / (1 - beta2), 9)))


@dataclass(frozen=True)
class Schedule:
    """One schedule s(t); `decay_steps` is WSD's cool-down D, unused by the other kinds.

    t counts optimiser steps from 0 and N is the run's steps. Warm-up, W steps: s = (t + 1)/W.
    wsd (the default): 1 until d0 = N - D, then f + (1 - f)·(1 - √u) with u = (t - d0 + 1)/D
    and f = `final_frac`; shape linear uses 1 - u. d2z: linear from the end of warm-up to f;
    cosine: f + (1 - f)·½(1 + cos πu); constant: 1. Steps past N hold at f.
    """

    steps: int
    warmup: int
    decay_steps: int
    kind: str = "wsd"
    shape: str = "1-sqrt"
    final_frac: float = 0.0

    def validate(self) -> Schedule:
        """Return self, or raise ValueError when the kind, shape or lengths do not fit."""
        if self.kind not in KINDS or self.shape not in SHAPES:
            raise ValueError(f"schedule {self.kind!r} / shape {self.shape!r}")
        tail = {"wsd": self.decay_steps, "constant": 0}.get(self.kind, 1)
        if self.warmup < 0 or self.warmup + tail > self.steps:
            msg = f"warm-up {self.warmup} and decay {tail} do not fit in {self.steps} steps"
            raise ValueError(msg)
        return self

    def span(self) -> tuple[int, int]:
        """(d0, length) of the decay: u = (t - d0 + 1)/length."""
        if self.kind == "wsd":
            return self.steps - self.decay_steps, self.decay_steps
        return self.warmup, self.steps - self.warmup

    def decay_start(self) -> int:
        """The first step whose multiplier may fall below 1 after warm-up."""
        return self.steps if self.kind == "constant" else self.span()[0]

    def __call__(self, t: int) -> float:
        """The host multiplier at step `t`, in float64."""
        if t < 0:
            raise ValueError("a negative step")
        if t < self.warmup:
            return (t + 1) / self.warmup
        if self.kind == "constant":
            return 1.0
        d0, n = self.span()
        u = min(1.0, max(0.0, (t - d0 + 1) / n))
        f = self.final_frac
        if self.kind == "cosine":
            return f + (1 - f) * 0.5 * (1 + math.cos(math.pi * u))
        return f + (1 - f) * (1 - self._shaped(u))

    def _shaped(self, u: float) -> float:
        return math.sqrt(u) if self.kind == "wsd" and self.shape == "1-sqrt" else u

    def branch(self, at: int, decay_steps: int | None = None) -> Schedule:
        """A WSD cool-down branch forked at step `at`: D steps from `at`, by default 20% of `at`.

        A branch forks only from the stable phase: after warm-up and before the run's own decay
        has taken a step.
        """
        if at < self.warmup:
            raise ValueError("a branch must fork after warm-up")
        if at > self.decay_start():
            msg = (
                f"a branch forks only from the stable phase: step {at} is past the "
                f"decay's start at {self.decay_start()}"
            )
            raise ValueError(msg)
        d = ceil_share(0.2, at) if decay_steps is None else decay_steps
        return replace(self, steps=at + d, decay_steps=d, kind="wsd").validate()


def schedule_for(r: RecipeConfig, o: OptimConfig) -> Schedule:
    """Steps from the token budget, W from the rule or `lr.warmup`, D from `lr.decay_frac`."""
    n = steps_for(o.tokens, o.tokens_per_update)
    w = warmup_rule(n, o.beta2()) if r.warmup is None else r.warmup
    d = ceil_share(r.decay_frac, n)
    return Schedule(n, w, d, r.schedule, r.shape, r.final_frac).validate()


# Width transfer: under `param.kind` mup the hidden matrices' rate multiplier is 256/d and every
# other group's is 1; under sp all are 1. The logit multiplier is also 256/d under μP, and the
# model reads it. At d 256 mup and sp are the same run.
SCALED_GROUPS = frozenset({"hidden"})


def width_mult(d: int, r: RecipeConfig) -> float:
    """m_N⁻¹: base width / d under μP; 1 under sp."""
    return r.mup_base_d / d if r.param_kind == "mup" else 1.0


def rate_mults(d: int, r: RecipeConfig, groups: tuple[str, ...]) -> dict[str, float]:
    m = width_mult(d, r)
    return {g: (m if g in SCALED_GROUPS else 1.0) for g in groups}


# Each dense parameter's group, by its kind (`kind_of`): attn and ffn are hidden, embed and
# value_embed their own groups, and gain, xl, tl and ve_gate scalar. A kind the table does not
# name is refused rather than falling into `scalar`, so a new kind from the model cannot pick up
# a rate by accident.
GROUP_OF_KIND = {
    "attn": "hidden",
    "ffn": "hidden",
    "embed": "embed",
    "value_embed": "value_embed",
    "gain": "scalar",
    "xl": "scalar",
    "tl": "scalar",
    "ve_gate": "scalar",
}


def groups_for(names: Iterable[str]) -> dict[str, str]:
    """Each parameter name's group, for `DenseOptim`."""
    out: dict[str, str] = {}
    for name in names:
        kind = kind_of(name)
        if kind not in GROUP_OF_KIND:
            raise ValueError(f"parameter {name} has kind {kind!r}, which no group takes")
        out[name] = GROUP_OF_KIND[kind]
    return out


GROUPS = ("hidden", "embed", "value_embed", "scalar")
Params = dict[str, mx.array]


def _members(params: Mapping[str, mx.array], groups: Mapping[str, str]) -> dict[str, list[str]]:
    if set(params) != set(groups):
        missing = sorted(set(params) ^ set(groups))[:8]
        raise ValueError(f"every dense parameter needs exactly one group: {missing}")
    unknown = sorted({g for g in groups.values() if g not in GROUPS})
    if unknown:
        raise ValueError(f"unknown parameter groups {unknown}; the groups are {GROUPS}")
    out = {g: sorted(n for n, k in groups.items() if k == g) for g in GROUPS}
    return {g: ns for g, ns in out.items() if ns}


class DenseOptim:
    """One AdamW per non-empty dense group, each with its own peak and schedule.

    Every weight in `params` is FP32 and has one group in `groups`, from `GROUPS`.
    """

    def __init__(
        self,
        params: Mapping[str, mx.array],
        groups: Mapping[str, str],
        d: int,
        o: OptimConfig,
        r: RecipeConfig,
    ) -> None:
        # Weights and gradients stay FP32, because MLX casts the rate to the gradient's dtype.
        assert_fp32(dict(params), "dense weights")
        self.names = _members(params, groups)
        mult = rate_mults(d, r, tuple(self.names))
        base = {"embed": o.emb_rate(), "value_embed": o.emb_rate()}
        self.peak = {g: base.get(g, o.lr_peak) * mult[g] for g in self.names}
        self.decay_ind = {g: (o.hidden_decay() if g == "hidden" else 0.0) for g in self.names}
        coupled = o.weight_decay_mode == "coupled"
        # MLX decays w·(1 - lr(t)·wd), so wd = λ_ind / η_group,peak gives the independent decay
        # λ_ind·s(t) whatever the group's rate; under `opt.weight_decay_mode` coupled, wd = λ.
        self.wd = {
            g: (o.decay_lambda() if coupled and g == "hidden" else self.decay_ind[g] / self.peak[g])
            for g in self.names
        }
        betas = (o.beta1, o.beta2())
        self.opts: dict[str, AdamWGroup] = {}
        # One `AdamWGroup` per group, never MLX's `MultiOptimizer`, whose rate setter overwrites
        # every group.
        for g, names in self.names.items():
            opt = AdamWGroup(self.peak[g], betas, o.eps, self.wd[g])
            opt.init({n: params[n] for n in names})
            self.opts[g] = opt

    def set_schedule(self, s: float | Mapping[str, float]) -> None:
        """Set each group's rate to its peak times s(t), one value or one per group."""
        for g, opt in self.opts.items():
            opt.learning_rate = self.peak[g] * (s if isinstance(s, float | int) else s[g])

    def apply(
        self, grads: Mapping[str, mx.array], params: Mapping[str, mx.array], scale: mx.array
    ) -> Params:
        """Each group's AdamW on its own members, gradients multiplied by the clip `scale`."""
        new: Params = {}
        for g, opt in self.opts.items():
            ns = self.names[g]
            new.update(opt.apply({n: grads[n] * scale for n in ns}, {n: params[n] for n in ns}))
        return new

    def states(self) -> list[dict[str, Any]]:
        """Every group's MLX state tree (for compile capture and checkpoints)."""
        return [opt.state for opt in self.opts.values()]

    def steps(self) -> dict[str, int]:
        """Updates each group has applied."""
        return {g: opt.steps() for g, opt in self.opts.items()}


def sq_norm(x: mx.array) -> mx.array:
    """FP32 squared norm summed in two stages: rows, then the row sums."""
    x = x.astype(mx.float32)
    x = mx.reshape(x, (x.shape[0], -1) if x.ndim > 1 else (1, -1))
    # A flat FP32 sum over the 10M tied matrix's squares is off by 2.2e-2, this one by 3.0e-6
    # (MLX 0.32.3, CPU).
    return (x * x).sum(-1).sum()


def group_sq_norms(
    grads: Mapping[str, mx.array], names: Mapping[str, list[str]]
) -> dict[str, mx.array]:
    """FP32 squared norms per group; each tensor is far below 2^31 elements.

    MLX's reductions over more than 2^31 elements go wrong (docs/mlx-notes.md).
    """
    out: dict[str, mx.array] = {}
    for g, ns in names.items():
        total = mx.array(0.0, mx.float32)
        for n in ns:
            total = total + sq_norm(grads[n])
        out[g] = total
    return out


def clip_scale(dense_sq: float, clip: float) -> tuple[float, float]:
    """(scale, norm): the factor that brings the global dense norm to at most `clip`."""
    norm = dense_sq**0.5
    return (min(1.0, clip / norm) if norm > 0 else 1.0), norm


def ema_beta(tokens: float, tokens_per_update: int, half_life: float = 0.02) -> float:
    """β for a half-life of `half_life` of the run's tokens.

    β = 0.5^(tokens_per_update / (half_life·tokens)).
    """
    return 0.5 ** (tokens_per_update / (half_life * tokens))


class EMA:
    """The FP32 average of the dense weights named in `names`.

    After every step that updated, avg ← avg + (1 - β)·(w - avg). It starts at the weights it is
    given. A run builds it over the optimiser's member names, tl and xl included, so a table
    shard cannot enter: a table average would cost 26-104 GB.
    """

    def __init__(self, params: Mapping[str, mx.array], beta: float, names: Iterable[str]) -> None:
        if not 0 < beta < 1:
            raise ValueError("β must be in (0, 1)")
        self.names = frozenset(names)
        self.beta = beta
        # (1 - β) is formed in float64 and applied in FP32: a β rounded to FP32 would move
        # (1 - β) by about 5e-4 of itself at β 0.99989.
        self.inc = mx.array(1.0 - beta, mx.float32)
        self.avg: dict[str, mx.array] = {}
        self.reset(params)

    def _check(self, params: Mapping[str, mx.array]) -> None:
        if set(params) != set(self.names):
            extra = sorted(set(params) ^ self.names)[:8]
            raise ValueError(f"the EMA averages the dense weights only: {extra}")

    def reset(self, params: Mapping[str, mx.array]) -> None:
        self._check(params)
        self.avg = {n: p.astype(mx.float32) for n, p in params.items()}
        evaluate(self.avg)

    def update(self, params: Mapping[str, mx.array]) -> None:
        """Move the average toward `params` after a step that updated."""
        self._check(params)
        for n, p in params.items():
            a = self.avg[n]
            self.avg[n] = a + (p.astype(mx.float32) - a) * self.inc
        evaluate(self.avg)


def ema_for(
    params: Mapping[str, mx.array], names: Iterable[str], o: OptimConfig, r: RecipeConfig
) -> EMA | None:
    """The run's EMA, or None with `ema.enabled` off."""
    if not r.ema:
        return None
    return EMA(params, ema_beta(o.tokens, o.tokens_per_update, r.ema_half_life), names)
