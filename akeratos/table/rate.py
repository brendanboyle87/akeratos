"""The table's learning rate, its decay and its share rho of the final residual.

η_table(t) = M·η_emb(t), with η_emb(t) the tied matrix's scheduled rate. In warm-up the table
follows the trunk's warm-up; after it a floor of floor·M·η_emb,peak applies. The decay follows
the table's own rate, decay(t) = d_peak·(η_table(t)/η_table,peak)², so at the floor of 0.3 it
is 0.09·d_peak; `table.decay` coupled is 0.025·η_table(t)², and zero is off.
`table.lr_schedule` gives the cool-down branches: `to0` drops the floor and `constant` holds
M·η_emb,peak after warm-up. `residual_share` is rho, the table term's size against the rest of
the final residual, logged every step as a health readout, not an abort.
"""

import mlx.core as mx

from akeratos.table.config import TableConfig

COUPLED = 0.025  # c in table.decay coupled


def table_rate(
    cfg: TableConfig, eta_emb_t: float, eta_emb_peak: float, after_warmup: bool
) -> float:
    """η_table at a step whose tied-matrix rate is `eta_emb_t`."""
    r = cfg.lr_mult * eta_emb_t
    if after_warmup and cfg.lr_schedule == "floor":
        r = max(r, cfg.floor * cfg.lr_mult * eta_emb_peak)
    elif after_warmup and cfg.lr_schedule == "constant":
        r = cfg.lr_mult * eta_emb_peak
    return r


def table_decay(cfg: TableConfig, rate: float, eta_emb_peak: float) -> float:
    """The gated decay at table rate `rate`."""
    if cfg.decay == "zero":
        return 0.0
    if cfg.decay == "coupled":
        return COUPLED * rate * rate
    return cfg.d_peak * (rate / (cfg.lr_mult * eta_emb_peak)) ** 2


def residual_share(tl: mx.array, t: mx.array, x_last: mx.array) -> mx.array:
    """The share rho in FP32 for entry scalars tl [S], table term t and residual x_L [B, T, d].

        rho = RMS(Σ tl_i·t) / RMS(x_L - Σ tl_i·t)

    with x_L the last layer's residual before the final norm. The term t is the same at every
    entry site, so Σ tl_i·t = (Σ tl_i)·t. On a toy model the per-site ratio read 0.03 while this
    cumulative share read 0.99, so rho is the one share metric. The result is lazy.
    """
    term = tl.astype(mx.float32).sum() * t.astype(mx.float32)
    rest = x_last.astype(mx.float32) - term
    return mx.sqrt(mx.mean(term * term)) / mx.sqrt(mx.mean(rest * rest))
