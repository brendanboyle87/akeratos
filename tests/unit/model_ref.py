"""An independent NumPy float64 reference of the model, with test helpers.

It was written from the architecture's description (docs/architecture.md), not from the MLX
modules, and shares only the parameter names. Attention uses an explicit repeat-kv for GQA and
a full [T, T] mask for every layer type (the banded path's oracle). float64 lives only here and
in the tests, never in `akeratos/`.

The helpers are the tiny oracle configuration and synthetic batches: Zipf ids 17..V-1 with
geometric document lengths, id 0 opening every document.
"""

import mlx.core as mx
import numpy as np

from akeratos.configs import ModelConfig, from_keys

# Every layer type, value embeddings on a sliding and a global layer, GQA 4:2, banded windows
# (W 16, 4 bands at T 64), μP logit multiplier 4 (d 64).
TINY = {
    "d_model": 64,
    "n_layers": 6,
    "n_heads": 4,
    "head_dim": 16,
    "n_kv_heads": 2,
    "ffn": 288,
    "vocab_size": 512,
    "window": 16,
}
B, T = 2, 64


def tiny(**keys):
    return from_keys({**TINY, **keys})


def seg(ids):
    """The packer's segment ids [B, T] int32 for rows that open every document with id 0."""
    return mx.array(np.cumsum(np.array(ids) == 0, axis=1).astype(np.int32))


def batch(seed, b=B, t=T, vocab=512, mean_doc=20, alpha=1.1):
    """(ids, targets) int64 [b, t] from rows of t + 1 tokens."""
    rng = np.random.default_rng(seed)
    first = 17
    p = np.arange(1, vocab - first + 1, dtype=np.float64) ** -alpha
    toks = rng.choice(vocab - first, size=(b, t + 1), p=p / p.sum()) + first
    for r in range(b):
        pos = 0
        while pos < t + 1:
            toks[r, pos] = 0
            pos += 1 + int(rng.geometric(1.0 / mean_doc))
    return toks[:, :-1].astype(np.int64), toks[:, 1:].astype(np.int64)


def rms(x, g=None, eps=1e-5):
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if g is None else y * g


def rope(x, theta, p=1.0):
    """x [B, h, T, D], half-split layout, positions 0..T-1; pairs at or past p·D/2 stay unrotated."""
    D, n = x.shape[-1], x.shape[-2]
    per = theta ** (np.arange(0, D, 2) / D)
    ang = np.arange(n)[:, None] / per[None]
    ang[:, round(p * D / 2) :] = 0.0
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def mask(ids, window, doc):
    b, n = ids.shape
    p, q = np.arange(n)[:, None], np.arange(n)[None]
    m = q <= p
    if window is not None:
        m &= p - q < window
    m = np.broadcast_to(m, (b, n, n)).copy()
    if doc:
        seg = np.cumsum(ids == 0, axis=1)
        m &= seg[:, :, None] == seg[:, None, :]
    return m


def is_global(cfg: ModelConfig, i):
    return cfg.window is None or (i + 1) % cfg.global_every == 0 or i == cfg.n_layers - 1


def attention(P, pre, a, ids, cfg: ModelConfig, i, ve):
    b, n, _ = a.shape
    H, kv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q = rms((a @ P[pre + "q.weight"].T).reshape(b, n, H, hd), P[pre + "q_norm.weight"])
    k = rms((a @ P[pre + "k.weight"].T).reshape(b, n, kv, hd), P[pre + "k_norm.weight"])
    v = rms((a @ P[pre + "v.weight"].T).reshape(b, n, kv, hd))
    if ve is not None:
        g = 3.0 / (1.0 + np.exp(-(a[..., :12] @ P[pre + "ve_gate.weight"].T)))
        v = v + g[..., None] * ve.reshape(b, n, kv, hd)
    glob = is_global(cfg, i)
    theta = cfg.rope_theta_global if glob else cfg.rope_theta_local
    p = cfg.rope_global_fraction if glob else 1.0
    q = rope(q.transpose(0, 2, 1, 3), theta, p)
    k = np.repeat(rope(k.transpose(0, 2, 1, 3), theta, p), H // kv, axis=1)
    v = np.repeat(v.transpose(0, 2, 1, 3), H // kv, axis=1)
    s = q @ k.swapaxes(-1, -2) / np.sqrt(hd)
    s = np.where(mask(ids, None if glob else cfg.window, cfg.doc_mask)[:, None], s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True))
    o = ((s / s.sum(-1, keepdims=True)) @ v).transpose(0, 2, 1, 3).reshape(b, n, H * hd)
    return o @ P[pre + "o.weight"].T


def ffn(P, pre, x, kind):
    u = x @ P[pre + "up.weight"].T
    if kind == "glu":
        z = x @ P[pre + "gate.weight"].T
        h = 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3))) * u
    else:
        h = np.maximum(u, 0) ** 2
    return h @ P[pre + "down.weight"].T


def hidden(P, cfg: ModelConfig, ids, t=None):
    e = P["emb.embed.weight"][ids]
    scale = {"norm": None, "sqrt_d": np.sqrt(cfg.d_model), "none": 1.0}[cfg.input_scale]
    x0 = rms(e) if scale is None else e * scale
    x = x0
    sites = cfg.sites()
    for i in range(cfg.n_layers):
        if cfg.x0_inject:
            x = x + P["xl"][i] * x0
        if t is not None and i in sites:
            x = x + P["tl"][sites.index(i)] * t
        pre, fpre = f"layers.{i}.attn.", f"layers.{i}.ffn."
        name = f"emb.ve.L{i}.weight"
        ve = P[name][ids] if name in P else None

        def n(norm, y, i=i):
            return rms(y, P[f"layers.{i}.{norm}.weight"])

        if cfg.norm_placement == "peri":
            x = x + n("norm_attn_out", attention(P, pre, n("norm_attn_in", x), ids, cfg, i, ve))
            x = x + n("norm_ffn_out", ffn(P, fpre, n("norm_ffn_in", x), cfg.ffn_kind))
        elif cfg.norm_placement == "pre":
            x = x + attention(P, pre, n("norm_attn_in", x), ids, cfg, i, ve)
            x = x + ffn(P, fpre, n("norm_ffn_in", x), cfg.ffn_kind)
        else:
            x = n("norm_attn_out", x + attention(P, pre, x, ids, cfg, i, ve))
            x = n("norm_ffn_out", x + ffn(P, fpre, x, cfg.ffn_kind))
    return rms(x, P["final_norm.weight"])


def logits(P, cfg: ModelConfig, ids, t=None):
    z = hidden(P, cfg, ids, t) @ P["emb.embed.weight"].T * (256 / cfg.d_model)
    if cfg.logit_cap is not None:
        z = cfg.logit_cap * np.tanh(z / cfg.logit_cap)
    z[..., list(cfg.masked_ids)] = -np.inf
    return z


def table_term(W, idx, heads=1):
    """Each order's `heads` rows side by side (idx is order-major), then the orders summed over √k."""
    k = idx.shape[-1] // heads
    orders = [
        np.concatenate([W[idx[..., o * heads + j]] for j in range(heads)], axis=-1)
        for o in range(k)
    ]
    return sum(orders) / np.sqrt(k)


def lse(z):
    m = z.max(-1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))[..., 0]


def loss(P, cfg: ModelConfig, ids, targets, weights=None, t=None):
    """Mean loss over the weights; the targets must have weight 0 wherever they are masked ids."""
    z = logits(P, cfg, ids, t)
    w = np.ones(ids.shape) if weights is None else weights
    safe = np.where(w > 0, targets, 0)
    ls = lse(z)
    ce = ls - np.take_along_axis(z, safe[..., None], -1)[..., 0]
    s = (ce * w).sum()
    if cfg.z_loss:
        s = s + cfg.z_loss * ((ls**2) * w).sum()
    return s / w.sum()


def mtp_loss_sum(z, targets, weights, depth_w):
    """Σ_j w_j Σ_t (lse z_t - z_t[targets[t + j]]) over t where t + j is in the row, both
    next-token weights are on and no separator lies among targets[t .. t + j - 1]; one offset
    at a time."""
    ls = lse(z)
    b, n = targets.shape
    s = 0.0
    for j, wj in enumerate(depth_w):
        for r in range(b):
            for t in range(n - j):
                live = weights[r, t] and weights[r, t + j]
                if live and not (targets[r, t : t + j] == 0).any():
                    s += wj * (ls[r, t] - z[r, t, targets[r, t + j]])
    return s
