"""Same-head multi-token prediction, `objective.mtp.mode` same_head.

The tied head that predicts t+1 also scores t+2 and t+3 from the same final state, over the
capped, masked FP32 logits, with one `logsumexp` and one [B, T, k] gather:
    loss = Σ_j w_j · Σ_t m_j[t] · (logsumexp(z_t) - z_t[y_j[t]]),  y_j[t] = targets[t + j]
summed over depths unnormalised and divided by the step's next-token count like the plain loss.
- **Weights** (`depth_weights`): [1, .5, .25→0] to 28.5% of the run's steps, then [1, .5→0] to
  60.6%, then None: next-token alone, so the rest of the run, cool-down included, takes the
  plain path. They reach a compiled step as an `mx.array`, since a Python float re-traces it.
- **Targets** (`depth_targets`): depth j at t is masked past the row's end, where the
  next-token weight is 0 at t or at t + j, and, with `objective.mtp.doc_mask` (default on),
  when a separator lies strictly between t and t + j + 1, so no depth predicts across a
  document. The separator itself, which ends t's document, stays a target, as at depth 0.
The keys `objective.mtp.mode`, `.stages` and `.doc_mask` are read by the training step.

Shape legend: B batch, T positions, k depths, d width, V vocabulary.
"""

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from akeratos.configs import SEP_ID, ModelConfig
from akeratos.model import Params, head_logits, masked_id_mask

DEPTHS = 3
STAGES = (0.285, 0.606)  # objective.mtp.stages: the two stage ends, as shares of the steps


def depth_weights(frac: float, stages: tuple[float, float] = STAGES) -> tuple[float, ...] | None:
    """(w1, w2, w3) at a share `frac` of the run's steps, or None once next-token alone."""
    a, b = stages
    if frac < a:
        return (1.0, 0.5, 0.25 * (1.0 - frac / a))
    if frac < b:
        return (1.0, 0.5 * (1.0 - (frac - a) / (b - a)), 0.0)
    return None


def depth_targets(
    targets: npt.ArrayLike, weights: npt.ArrayLike, doc_mask: bool = True, k: int = DEPTHS
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Targets and weights [B, T, k] from next-token targets and weights [B, T].

    `targets` are the row's raw next-token ids and `weights` those of
    `akeratos.train.loop.prepare_targets`, so a masked id already has weight 0. A target
    whose depth weight is 0 is set to 0.
    """
    tg = np.asarray(targets, np.int64)
    w = np.asarray(weights, np.float32) > 0
    B, T = tg.shape
    sep = tg == SEP_ID
    out_t, out_w = np.zeros((B, T, k), np.int64), np.zeros((B, T, k), np.bool_)
    for j in range(min(k, T)):
        n = T - j
        ok = w[:, :n] & w[:, j:]
        if doc_mask:
            for s in range(
                j
            ):  # the tokens strictly between t and t + j + 1: targets[t .. t + j - 1]
                ok &= ~sep[:, s : s + n]
        out_t[:, :n, j], out_w[:, :n, j] = tg[:, j:], ok
    return np.where(out_w, out_t, 0).astype(np.int32), out_w.astype(np.float32)


def mtp_loss_sum(
    p: Params,
    cfg: ModelConfig,
    h: mx.array,
    targets: mx.array,
    weights: mx.array,
    depth_w: mx.array,
) -> mx.array:
    """Token-sum loss for the final-normed state h [B, T, d], targets and weights [B, T, k]."""
    if cfg.loss_path != "unchunked":
        raise ValueError("objective.mtp.mode same_head with loss.path chunked is not built")
    z = head_logits(p, cfg, h, masked_id_mask(cfg))
    lse = mx.logsumexp(z, axis=-1)
    picked = mx.take_along_axis(z, targets, axis=-1)
    w = weights.astype(mx.float32)
    per = ((lse[..., None] - picked) * w).sum(axis=(0, 1))  # [k]
    s = (per * depth_w.astype(mx.float32)).sum()
    if cfg.z_loss:
        s = s + mx.array(cfg.z_loss, mx.float32) * (mx.square(lse) * w[..., 0]).sum()
    return s
