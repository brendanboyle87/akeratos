"""The optimiser's and the schedule's settings, named by their switch keys.

`OptimConfig` holds the rates, betas, clip, token counts and the dense decay rule;
`RecipeConfig` the schedule, the width transfer and the EMA. Every field that a run may change
has a switch key in `KEYS`, and `from_keys({})` gives the defaults ("absent equals off").

    steps = ceil(tokens / tokens per update)
    β₂ = 0.95^(B/524,288)                         # B tokens per update
    w <- w·(1 - λ_ind·s(t)),  λ_ind = η_ref·λ,  λ = 0.1·(1e9/D_tok)·(B/8,192)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

# Steps round up: 2.6B / 8,192 = 317,382.8 gives 317,383. Dense β₂ holds the second moment's
# half-life fixed in tokens: 0.99920 at 8,192 tokens per update, 0.99860 at 14,336 and 0.99361 at
# 65,536, and the half-life, ln 0.5 / ln β₂ · B, is 7.08M tokens at any B. β₂ is fixed for the
# run: MLX's AdamW holds its betas as Python floats, so a change would re-trace the compiled
# update.
BETA2_REF, BETA2_REF_TOKENS = 0.95, 524_288


def steps_for(tokens: float, tokens_per_update: int) -> int:
    """Optimiser steps for a token budget, rounded up."""
    return math.ceil(tokens / tokens_per_update)


def beta2_halflife(tokens_per_update: int) -> float:
    """β₂ = 0.95^(B/524,288)."""
    return BETA2_REF ** (tokens_per_update / BETA2_REF_TOKENS)


# The independent decay applies to the hidden matrices only, with η_ref 1e-3 for every run, so
# 1/λ_ind is 8.2% of D_tok/B steps. Under `opt.weight_decay_mode` coupled the decay is
# η_hidden(t)·λ.
MUP_BASE_D = 256
_LAMBDA_REF_TOKENS, _LAMBDA_REF_BATCH = 1e9, 8192


@dataclass(frozen=True)
class OptimConfig:
    """Dense AdamW settings shared by every group."""

    lr_peak: float = 3e-3
    eta_emb: float | None = None
    beta1: float = 0.9
    eps: float = 1e-8
    clip: float = 1.0
    tokens: float = 1e9
    tokens_per_update: int = 8192
    weight_decay_mode: str = "independent"
    eta_ref: float = 1e-3
    decay_tokens: float | None = None
    decay_ind: float | None = None
    beta2_rule: str = "halflife"
    beta2_fixed: float = 0.95

    def beta2(self) -> float:
        """Dense β₂: 0.95^(B/524,288) under the half-life rule."""
        if self.beta2_rule == "halflife":
            return beta2_halflife(self.tokens_per_update)
        return self.beta2_fixed

    def emb_rate(self) -> float:
        """η_emb: η_peak unless set apart (μP multiplier 1)."""
        return self.lr_peak if self.eta_emb is None else self.eta_emb

    def decay_lambda(self) -> float:
        """The coupled λ: 0.1 at 1B tokens and 8,192 tokens per update, ∝ B/D_tok."""
        d_tok = self.tokens if self.decay_tokens is None else self.decay_tokens
        return 0.1 * (_LAMBDA_REF_TOKENS / d_tok) * (self.tokens_per_update / _LAMBDA_REF_BATCH)

    def hidden_decay(self) -> float:
        """λ_ind, the independent decay per step at s(t) = 1."""
        return self.eta_ref * self.decay_lambda() if self.decay_ind is None else self.decay_ind

    def validate(self) -> OptimConfig:
        """Return self, or raise ValueError naming every bad field."""
        errs: list[str] = []
        if self.weight_decay_mode not in ("independent", "coupled"):
            errs.append(f"opt.weight_decay_mode {self.weight_decay_mode!r}")
        if self.beta2_rule not in ("halflife", "fixed"):
            errs.append(f"opt.beta2_rule {self.beta2_rule!r}")
        elif not (0 < self.beta2() < 1 and 0 <= self.beta1 < 1):
            errs.append("Adam betas outside [0, 1)")
        positive = (self.lr_peak, self.emb_rate(), self.eta_ref, self.eps, self.clip)
        if min(*positive, self.tokens, self.tokens_per_update) <= 0:
            errs.append("rates, ε, the clip and the token counts must be positive")
        elif self.weight_decay_mode == "independent" and not 0 <= self.hidden_decay() < 1:
            errs.append(
                f"independent decay λ_ind {self.hidden_decay():.4g} per step is outside [0, 1); "
                "set opt.decay_ind for a run this short"
            )
        if errs:
            raise ValueError("; ".join(errs))
        return self


@dataclass(frozen=True)
class RecipeConfig:
    """Schedule, width transfer and averaging."""

    schedule: str = "wsd"
    warmup: int | None = None
    decay_frac: float = 0.20
    shape: str = "1-sqrt"
    final_frac: float = 0.0
    param_kind: str = "mup"
    mup_base_d: int = MUP_BASE_D
    depth_scaling: str = "off"
    ema: bool = True
    ema_half_life: float = 0.02

    def validate(self) -> RecipeConfig:
        """Return self, or raise ValueError naming every bad field."""
        errs: list[str] = []
        if self.param_kind not in ("mup", "sp"):
            errs.append(f"param.kind {self.param_kind!r}")
        if self.depth_scaling != "off":
            errs.append(
                f"param.depth_scaling {self.depth_scaling!r}: only off is built "
                "(train/param.py says why)"
            )
        if self.mup_base_d <= 0 or not 0 < self.ema_half_life < 1:
            errs.append("param.mup_base_d must be positive and ema.half_life in (0, 1)")
        if errs:
            raise ValueError("; ".join(errs))
        return self


KEYS: dict[str, tuple[str, str]] = {
    "lr.peak": ("optim", "lr_peak"),
    "lr.emb": ("optim", "eta_emb"),
    "lr.schedule": ("recipe", "schedule"),
    "lr.warmup": ("recipe", "warmup"),
    "lr.decay_frac": ("recipe", "decay_frac"),
    "lr.shape": ("recipe", "shape"),
    "lr.final_frac": ("recipe", "final_frac"),
    "opt.clip_norm": ("optim", "clip"),
    "opt.weight_decay_mode": ("optim", "weight_decay_mode"),
    "opt.decay_tokens": ("optim", "decay_tokens"),
    "opt.decay_ind": ("optim", "decay_ind"),
    "opt.beta2_rule": ("optim", "beta2_rule"),
    "tokens.budget": ("optim", "tokens"),
    "batch.tokens_per_update": ("optim", "tokens_per_update"),
    "param.kind": ("recipe", "param_kind"),
    "param.mup_base_d": ("recipe", "mup_base_d"),
    "param.depth_scaling": ("recipe", "depth_scaling"),
    "ema.enabled": ("recipe", "ema"),
    "ema.half_life": ("recipe", "ema_half_life"),
}


def from_keys(values: Mapping[str, Any]) -> tuple[OptimConfig, RecipeConfig]:
    """Build both configs from switch keys; an absent key keeps its default."""
    unknown = sorted(set(values) - set(KEYS))
    if unknown:
        raise KeyError(f"unknown recipe keys: {unknown}")
    o = {f: values[k] for k, (part, f) in KEYS.items() if part == "optim" and k in values}
    r = {f: values[k] for k, (part, f) in KEYS.items() if part == "recipe" and k in values}
    return replace(OptimConfig(), **o).validate(), replace(RecipeConfig(), **r).validate()


def describe(o: OptimConfig, r: RecipeConfig) -> list[str]:
    """One log line per switch key, with its value in force."""
    parts: dict[str, OptimConfig | RecipeConfig] = {"optim": o, "recipe": r}
    return [f"switch {k} = {getattr(parts[part], f)!r}" for k, (part, f) in KEYS.items()]
