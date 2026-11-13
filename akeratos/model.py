"""The language model: the forward pass from token ids to next-token scores, its parameters and their starting values.

    x0 = rms_norm(E[id])                          # FP32 gather, then cast; no gain
    x = x0
    for i in 0..L-1:
        x = x + xl[i]·x0 + tl[i]·t                # residual scale fixed at 1
        x = Block_i(x)                            # peri-norm attention and ReLU² feed-forward
    z = (RMSNorm(x)·Eᵀ)·m_logit → 15·tanh(z/15) → ids 1-16 masked → cross-entropy in FP32

t = (r_bigram + r_trigram)/√2 is the table term, computed once per token from the working set
W, which is a function argument and never a parameter; the table itself is `akeratos.table`.
With `table` off there is no tl and no t; with `residual.x0_inject` off there is no xl and no
x0 term; `table.sites` keeps a tl, and the table term, only at the listed layers.

The compute dtype is BF16 by default and FP32 for the oracle checks; parameters are FP32 masters.

Shape legend: B batch, T positions, d width, V vocabulary, k·h table rows read per token,
U working-set rows.
"""

import hashlib
import math

import mlx.core as mx
import numpy as np

from akeratos.configs import ModelConfig
from akeratos.flash import Bounds, attend, bounds

# Every matrix product is `x @ W.T` with no bias: the model never uses MLX's `nn.Linear`, which
# has `bias=True`.
# Shape legend: B batch, T positions, D any last axis.
# A model's parameters, keyed by dotted name ("layers.0.attn.q.weight"); FP32 masters.
Params = dict[str, mx.array]


# - The tied matrix E [V, d], an FP32 master, N(0, 0.02²). The input gathers E[id] in FP32, then
#   casts to the compute dtype; x0 = rms_norm(E[id]) with no gain and ε 1e-5.
# - Value embeddings: one [V, kv·hd] table per value-embedding layer (the odd layers by default).
#   `value_embed.layers` off removes the tables and their gates, which a zero gate does not do:
#   3·sigmoid(0) = 1.5.
# Shape legend: B batch, T positions, d width, kv key-value heads, hd head size.
EMBED = "emb.embed.weight"  # the tied matrix's name; the head reads it too


def table_term(W: mx.array, idx: mx.array, d: int, dtype: mx.Dtype) -> mx.array:
    """The table term t [B, T, d] in `dtype` from the working set W [U, w] (FP32)."""
    B, T = idx.shape[:2]
    # idx [B, T, k·h] is order-major: reshaping to d joins each order's h heads of width w = d/h.
    r = mx.reshape(mx.take(W, mx.reshape(idx, (-1,)), axis=0), (B, T, -1, d))
    # The k orders are summed in FP32 over √k, which keeps the mean square, and then cast.
    return (r.sum(axis=2) * (r.shape[2] ** -0.5)).astype(dtype)


def hidden(
    p: Params,
    cfg: ModelConfig,
    ids: mx.array,
    segment_ids: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
    t: mx.array | None = None,
) -> mx.array:
    """The final-normed state [B, T, d] in `dtype` for ids and segment ids [B, T] and the table term t."""
    x = residual(p, cfg, ids, segment_ids, dtype, t)
    return rms_norm(x, p["final_norm.weight"], cfg.rms_norm_eps)


def residual(
    p: Params,
    cfg: ModelConfig,
    ids: mx.array,
    segment_ids: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
    t: mx.array | None = None,
) -> mx.array:
    """The last layer's residual x_L [B, T, d] before the final norm.

    It is the x_L of the table's share (`residual_share` in `akeratos/table/rate.py`).
    """
    masks = build_masks(segment_ids, cfg)
    e = x0(p, cfg, ids, dtype)
    x = e
    site = (
        {layer: j for j, layer in enumerate(cfg.sites())} if (cfg.table and t is not None) else {}
    )
    for i in range(cfg.n_layers):
        if cfg.x0_inject:
            x = x + p["xl"][i].astype(dtype) * e
        if i in site and t is not None:
            x = x + p["tl"][site[i]].astype(dtype) * t
        x = block(p, cfg, i, x, masks, ve_rows(p, i, ids, dtype))
    return x


def forward(
    p: Params,
    cfg: ModelConfig,
    ids: mx.array,
    segment_ids: mx.array,
    dtype: mx.Dtype = mx.bfloat16,
    t: mx.array | None = None,
) -> mx.array:
    """FP32 logits [B, T, V]: the capped, masked head over `hidden`."""
    return head_logits(p, cfg, hidden(p, cfg, ids, segment_ids, dtype, t), masked_id_mask(cfg))


# Default, peri-norm ("sandwich"): RMSNorm at the input and the output of each sublayer,
#     x = x + RMSNorm_ao(Attention(RMSNorm_ai(x)))
#     x = x + RMSNorm_fo(FFN(RMSNorm_fi(x)))
# with vector gains (initial 1, no decay) and ε 1e-5. The feed-forward is down(ReLU(up(x))²),
# no biases, width 4.5·d.
# Switches (each off by default; "absent equals off" is tested):
# - `norm.placement` pre: x = x + f(RMSNorm(x)); post: x = RMSNorm(x + f(x)).
# - `ffn.kind` glu: down(GELU-tanh(gate(x)) · up(x)), width 3·d.
# Shape legend: B batch, T positions, d width.
def _ffn(p: Params, pre: str, cfg: ModelConfig, x: mx.array) -> mx.array:
    """The feed-forward sublayer on its input x [B, T, d]; `pre` is "layers.i.ffn."."""
    u = dense(x, p[pre + "up.weight"])
    if cfg.ffn_kind == "glu":
        return dense(gelu_tanh(dense(x, p[pre + "gate.weight"])) * u, p[pre + "down.weight"])
    return dense(relu2(u), p[pre + "down.weight"])


def block(
    p: Params,
    cfg: ModelConfig,
    i: int,
    x: mx.array,
    masks: dict[str, mx.array],
    ve_rows: mx.array | None,
) -> mx.array:
    """Layer i on the residual stream x [B, T, d], after the x0 and table terms are added."""
    pre = f"layers.{i}."
    eps = cfg.rms_norm_eps

    def n(name: str, y: mx.array) -> mx.array:
        return rms_norm(y, p[f"{pre}{name}.weight"], eps)

    def attn(y: mx.array) -> mx.array:
        return attention(p, pre + "attn.", cfg, i, y, masks, ve_rows)

    def ffn(y: mx.array) -> mx.array:
        return _ffn(p, pre + "ffn.", cfg, y)

    if cfg.norm_placement == "peri":
        x = x + n("norm_attn_out", attn(n("norm_attn_in", x)))
        return x + n("norm_ffn_out", ffn(n("norm_ffn_in", x)))
    if cfg.norm_placement == "pre":
        x = x + attn(n("norm_attn_in", x))
        return x + ffn(n("norm_ffn_in", x))
    x = n("norm_attn_out", x + attn(x))
    return n("norm_ffn_out", x + ffn(x))


# q d → H·hd, k and v d → kv·hd, o H·hd → d, no biases; grouped-query attention, where query
# head h reads key-value head h // (H / kv). On value-embedding layers the value is
#     v = v_norm(v) + 3·sigmoid(g(a_in[..., :12]))·ve[id],
# with g = Linear(12, kv, bias=False), one gate per key-value head, reading the sublayer's
# normalised input a_in. Training runs `mx.fast.scaled_dot_product_attention` unfused on 0.32.3,
# which falls back while tracing gradients.
# Shape legend: B batch, T positions, H query heads, KV key-value heads, hd head size, d width.
def attention(
    p: Params,
    pre: str,
    cfg: ModelConfig,
    i: int,
    a_in: mx.array,
    masks: dict[str, mx.array],
    ve_rows: mx.array | None,
) -> mx.array:
    """Layer i's attention sublayer on its normalised input a_in [B, T, d].

    `pre` is the layer's prefix ("layers.i.attn."); `ve_rows` [B, T, kv·hd] are the layer's
    value-embedding rows, or None on a layer without them.
    """
    B, T, _ = a_in.shape
    H, kv, hd, eps = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rms_norm_eps
    q = rms_norm(
        mx.reshape(dense(a_in, p[pre + "q.weight"]), (B, T, H, hd)), p[pre + "q_norm.weight"], eps
    )
    k = rms_norm(
        mx.reshape(dense(a_in, p[pre + "k.weight"]), (B, T, kv, hd)), p[pre + "k_norm.weight"], eps
    )
    v = rms_norm(mx.reshape(dense(a_in, p[pre + "v.weight"]), (B, T, kv, hd)), None, eps)
    if ve_rows is not None:
        gate = 3.0 * mx.sigmoid(dense(a_in[..., : cfg.ve_gate_width], p[pre + "ve_gate.weight"]))
        v = v + gate[..., None] * mx.reshape(ve_rows, (B, T, kv, hd))  # gate [B, T, kv]
    is_global = cfg.is_global(i)
    freqs = rope_freqs(cfg, is_global)
    q = apply_rope(mx.transpose(q, (0, 2, 1, 3)), freqs)
    k = apply_rope(mx.transpose(k, (0, 2, 1, 3)), freqs)
    v = mx.transpose(v, (0, 2, 1, 3))
    if cfg.attn_path == "fused":
        kind = "global" if is_global else "sliding"
        o = attend(q, k, v, Bounds(masks[kind + ".lo"], masks[kind + ".qend"]))
        return dense(mx.reshape(o, (B, T, H * hd)), p[pre + "o.weight"])
    scale = hd**-0.5
    sdpa = mx.fast.scaled_dot_product_attention
    if is_global:
        o = sdpa(q, k, v, scale=scale, mask=masks["global"])
    elif "band" in masks:
        W = cfg.window or T
        bands = sdpa(
            query_bands(q, W), key_bands(k, W), key_bands(v, W), scale=scale, mask=masks["band"]
        )
        o = from_bands(bands, B)
    else:
        o = sdpa(q, k, v, scale=scale, mask=masks["sliding"])
    return dense(mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (B, T, H * hd)), p[pre + "o.weight"])


# - RoPE on all head dimensions, half-split layout: θ 10⁴ on sliding layers and 10⁶ on global
#   layers. `rope.global_fraction` p < 1 keeps the fastest p·hd/2 pairs' periods on global
#   layers and gives the rest an infinite one, so they never rotate (`rope(dims=…)` spreads the
#   frequencies differently).
# - Sliding layers use banded windows: each W-token query block attends to its own block and the
#   previous one, masked to exactly W keys, so the executed scores are T·2W per head instead of
#   T². `attn.path` masked keeps the full [B, 1, T, T] mask with the window as the oracle path.
# Shape legend: B batch, T positions, h heads, D head size, W window, nb = T / W bands.
def rope_freqs(cfg: ModelConfig, is_global: bool) -> mx.array:
    """The periods (wavelength / 2π) of each pair (i, i + hd/2) for one layer type, FP32."""
    D = cfg.head_dim
    theta = cfg.rope_theta_global if is_global else cfg.rope_theta_local
    periods = theta ** (np.arange(0, D, 2) / D)
    if is_global and cfg.rope_global_fraction < 1:
        kept = round(cfg.rope_global_fraction * D / 2)
        periods = np.where(np.arange(D // 2) < kept, periods, np.inf)
    return mx.array(periods.tolist(), dtype=mx.float32)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Rotate x [B, h, T, D] at positions 0..T-1."""
    return mx.fast.rope(
        x, x.shape[-1], traditional=False, base=None, scale=1.0, offset=0, freqs=freqs
    )


def _full_mask(seg: mx.array, window: int | None, doc: bool) -> mx.array:
    """[B or 1, 1, T, T]: key q ≤ query p, within `window` keys if given, same segment if `doc`."""
    T = seg.shape[1]
    p, q = mx.arange(T)[:, None], mx.arange(T)[None]
    m = q <= p
    if window is not None:
        m = m & (p - q < window)
    if not doc:
        return m[None, None]
    return (m[None] & (seg[:, :, None] == seg[:, None, :]))[:, None]


def _band_mask(seg: mx.array, W: int, doc: bool) -> mx.array:
    """[B·nb, 1, W, 2W] for query blocks of W against (previous block, own block)."""
    B, T = seg.shape
    nb = T // W
    a, b = mx.arange(W)[:, None], mx.arange(2 * W)[None]
    rel = (b > a) & (b <= a + W)  # key (j-1)·W + b is within W of query j·W + a, not after it
    has_prev = mx.arange(nb)[:, None, None] > 0  # block 0's first W keys are padding
    static = rel[None] & (has_prev | (b >= W)[None])  # [nb, W, 2W]
    if not doc:
        return mx.reshape(mx.broadcast_to(static[None], (B, nb, W, 2 * W)), (B * nb, 1, W, 2 * W))
    segq = mx.reshape(seg, (B, nb, W))
    pad = mx.full((B, W), -1, seg.dtype)
    segprev = mx.reshape(mx.concatenate([pad, seg[:, : T - W]], axis=1), (B, nb, W))
    segk = mx.concatenate([segprev, segq], axis=2)
    m = static[None] & (segq[..., :, None] == segk[..., None, :])
    return mx.reshape(m, (B * nb, 1, W, 2 * W))


def query_bands(x: mx.array, W: int) -> mx.array:
    """[B, h, T, D] -> [B·nb, h, W, D]."""
    B, h, T, D = x.shape
    nb = T // W
    y = mx.transpose(mx.reshape(x, (B, h, nb, W, D)), (0, 2, 1, 3, 4))
    return mx.reshape(y, (B * nb, h, W, D))


def key_bands(x: mx.array, W: int) -> mx.array:
    """[B, h, T, D] -> [B·nb, h, 2W, D]: block j-1 (zeros for j = 0), then block j."""
    B, h, T, D = x.shape
    nb = T // W
    prev = mx.concatenate([mx.zeros((B, h, W, D), x.dtype), x[:, :, : T - W]], axis=2)
    kb = mx.concatenate(
        [mx.reshape(prev, (B, h, nb, W, D)), mx.reshape(x, (B, h, nb, W, D))], axis=3
    )
    return mx.reshape(mx.transpose(kb, (0, 2, 1, 3, 4)), (B * nb, h, 2 * W, D))


def from_bands(o: mx.array, B: int) -> mx.array:
    """[B·nb, h, W, D] -> [B, h, T, D]."""
    n, h, W, D = o.shape
    nb = n // B
    y = mx.transpose(mx.reshape(o, (B, nb, h, W, D)), (0, 2, 1, 3, 4))
    return mx.reshape(y, (B, h, nb * W, D))


def build_masks(segment_ids: mx.array, cfg: ModelConfig) -> dict[str, mx.array]:
    """The masks of one forward pass, built once and shared by the layers of each type.

    `segment_ids` [B, T] int32 is the packer's `Batch.segment_ids`; the model never recounts
    separators in the ids. The masks are "global" [B or 1, 1, T, T] and, on models with sliding
    layers, "band" [B·nb, 1, W, 2W] (`attn.path` banded) or "sliding" [B or 1, 1, T, T]
    (masked). Under `attn.path` fused they are int32 bounds [B, T] instead: "global.lo" and
    "global.qend", and on models with sliding layers "sliding.lo" and "sliding.qend".
    """
    T = segment_ids.shape[1]
    seg = segment_ids.astype(mx.int32) if cfg.doc_mask else mx.zeros(segment_ids.shape, mx.int32)
    W = cfg.window
    sliding = W is not None and not all(cfg.is_global(i) for i in range(cfg.n_layers))
    if cfg.attn_path == "fused":
        kinds = {"global": None, "sliding": W} if sliding else {"global": None}
        return {
            f"{kind}.{name}": a
            for kind, w in kinds.items()
            for name, a in bounds(seg, w)._asdict().items()
        }
    out = {"global": _full_mask(seg, None, cfg.doc_mask)}
    if W is None or not sliding:
        return out
    if cfg.attn_path == "masked":
        out["sliding"] = _full_mask(seg, W, cfg.doc_mask)
    elif T % W:
        raise ValueError(f"banded windows need T ({T}) to be a multiple of the window ({W})")
    else:
        out["band"] = _band_mask(seg, W, cfg.doc_mask)
    return out


def ve_name(i: int) -> str:
    """The name of layer i's value-embedding table."""
    return f"emb.ve.L{i}.weight"


def x0(p: Params, cfg: ModelConfig, ids: mx.array, dtype: mx.Dtype) -> mx.array:
    """The scaled input embedding [B, T, d] in `dtype` for ids [B, T]."""
    e = gather(p[EMBED], ids, dtype)
    if cfg.input_scale == "norm":
        return rms_norm(e, None, cfg.rms_norm_eps)
    if cfg.input_scale == "sqrt_d":
        return e * (cfg.d_model**0.5)
    return e


def ve_rows(p: Params, i: int, ids: mx.array, dtype: mx.Dtype) -> mx.array | None:
    """Layer i's value-embedding rows [B, T, kv·hd] in `dtype`, or None without a table."""
    name = ve_name(i)
    return gather(p[name], ids, dtype) if name in p else None


#     z = h @ E.astype(compute dtype).T      # h = RMSNorm(x), the final-normed state
#     z = z.astype(FP32) · m_logit           # m_logit = 256/d under μP (1/6 at d = 1536)
#     z = 15·tanh(z/15)                      # loss.final_softcap; None turns the cap off
#     z[head.masked_ids] = -inf              # ids 1-16 by default: zero probability
# The upcast comes before the multiplier, because 1/6 is not a BF16 number.
# Shape legend: d width, V vocabulary.
def masked_id_mask(cfg: ModelConfig) -> mx.array:
    """[V] bool: True at the ids the head masks to -inf."""
    m = np.zeros((cfg.vocab_size,), np.bool_)
    m[list(cfg.masked_ids)] = True
    return mx.array(m.tolist(), mx.bool_)


def finish_logits(z: mx.array, cfg: ModelConfig, masked: mx.array) -> mx.array:
    """The product h·Eᵀ [..., V] to FP32 logits: the upcast, m_logit, the cap and the mask."""
    z = z.astype(mx.float32) * mx.array(cfg.m_logit(), mx.float32)
    if cfg.logit_cap is not None:
        c = mx.array(cfg.logit_cap, mx.float32)
        z = c * mx.tanh(z / c)
    return mx.where(masked, -mx.inf, z)


def head_logits(p: Params, cfg: ModelConfig, h: mx.array, masked: mx.array) -> mx.array:
    """FP32 logits [..., V] from the final-normed state h [..., d] in the compute dtype."""
    return finish_logits(h @ p[EMBED].astype(h.dtype).T, cfg, masked)


def rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """RMSNorm over the last axis, with an optional gain cast to x's dtype."""
    return mx.fast.rms_norm(x, None if weight is None else weight.astype(x.dtype), eps)


def dense(x: mx.array, weight: mx.array) -> mx.array:
    """Return x Wᵀ for W [out, in], the FP32 master cast to x's dtype (the BF16 copy)."""
    return x @ weight.astype(x.dtype).T


def gather(table: mx.array, ids: mx.array, dtype: mx.Dtype) -> mx.array:
    """Rows of an FP32 table by id: gather in FP32, then cast."""
    # Casting before the gather would sum its gradient in BF16: 11% off on a frequent token (CPU).
    return mx.take(table, ids, axis=0).astype(dtype)


def relu2(x: mx.array) -> mx.array:
    return mx.square(mx.maximum(x, 0))


def gelu_tanh(x: mx.array) -> mx.array:
    """GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))."""
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


# | Tensor | Initialisation |
# |---|---|
# | Hidden matrices (q, k, v, o; up, down, gate), output projections included | U(±√3·0.5/√d_in), std 0.5/√d_in |
# | Output projections under `init.out_proj` zero (pre-norm only) | 0 |
# | Tied embedding | N(0, 0.02²) |
# | Value-embedding rows; gates | like the value projection, U(±√3·0.5/√d); 0 (gate 1.5) |
# | Norm gains, QK-norm gains | 1 (output norms: `norm.out_gain_init`) |
# | xl; tl | 0.20 → 0.05 by layer; 0.1 |
# Each tensor draws from its own key, so a switch that adds or removes a tensor never shifts
# another tensor's values, and a run re-initialises identically in any process.
OUT_PROJ = {"o", "down"}
OUT_NORMS = {"norm_attn_out", "norm_ffn_out"}
_COUNT_ROW = {
    "attn": "attention",
    "gain": "norms",
    "xl": "scalars",
    "tl": "scalars",
    "ve_gate": "scalars",
}


def _layer_shapes(cfg: ModelConfig, i: int, has_ve: bool) -> dict[str, tuple[int, ...]]:
    d, hd, f = cfg.d_model, cfg.head_dim, cfg.ffn_width()
    H, kv = cfg.n_heads * hd, cfg.n_kv_heads * hd
    a, m = f"layers.{i}.attn.", f"layers.{i}.ffn."
    out = {
        a + "q.weight": (H, d),
        a + "k.weight": (kv, d),
        a + "v.weight": (kv, d),
        a + "o.weight": (d, H),
        a + "q_norm.weight": (hd,),
        a + "k_norm.weight": (hd,),
        m + "up.weight": (f, d),
        m + "down.weight": (d, f),
    }
    if has_ve:
        out[a + "ve_gate.weight"] = (cfg.n_kv_heads, cfg.ve_gate_width)
    if cfg.ffn_kind == "glu":
        out[m + "gate.weight"] = (f, d)
    norms = {"peri": ("in", "out"), "pre": ("in",), "post": ("out",)}[cfg.norm_placement]
    for side in norms:
        out[f"layers.{i}.norm_attn_{side}.weight"] = (d,)
        out[f"layers.{i}.norm_ffn_{side}.weight"] = (d,)
    return out


def param_shapes(cfg: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every parameter's name and shape, for this configuration's switches."""
    ve = cfg.ve_layers()
    out: dict[str, tuple[int, ...]] = {EMBED: (cfg.vocab_size, cfg.d_model)}
    for i in ve:
        out[ve_name(i)] = (cfg.vocab_size, cfg.n_kv_heads * cfg.head_dim)
    for i in range(cfg.n_layers):
        out |= _layer_shapes(cfg, i, i in ve)
    out["final_norm.weight"] = (cfg.d_model,)
    if cfg.x0_inject:
        out["xl"] = (cfg.n_layers,)
    if cfg.table:
        out["tl"] = (len(cfg.sites()),)
    return out


# (pattern, kind): `*` matches exactly one dotted segment, and a pattern matches the whole path.
RULES: tuple[tuple[str, str], ...] = (
    (EMBED, "embed"),
    ("emb.ve.*.weight", "value_embed"),
    ("layers.*.attn.ve_gate.weight", "ve_gate"),
    *((f"layers.*.attn.{m}.weight", "attn") for m in ("q", "k", "v", "o")),
    *((f"layers.*.ffn.{m}.weight", "ffn") for m in ("up", "down", "gate")),
    *((f"layers.*.attn.{m}.weight", "gain") for m in ("q_norm", "k_norm")),
    *((f"layers.*.{n}.weight", "gain") for n in (*OUT_NORMS, "norm_attn_in", "norm_ffn_in")),
    ("final_norm.weight", "gain"),
    ("xl", "xl"),
    ("tl", "tl"),
)


def _matches(pattern: str, name: str) -> bool:
    want, got = pattern.split("."), name.split(".")
    return len(want) == len(got) and all(w in ("*", g) for w, g in zip(want, got, strict=True))


def kind_of(name: str) -> str:
    """The parameter's kind by `RULES`, used by initialisation, the optimiser groups and `part_of`."""
    kinds = [kind for pattern, kind in RULES if _matches(pattern, name)]
    if len(kinds) != 1:
        raise ValueError(f"parameter {name} matches {len(kinds)} rules, not exactly one")
    return kinds[0]


def _name_key(seed: int, name: str) -> mx.array:
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return mx.random.key(int.from_bytes(h, "little"))


def _init_one(cfg: ModelConfig, name: str, shape: tuple[int, ...], seed: int) -> mx.array:
    kind, key = kind_of(name), _name_key(seed, name)
    leaf = name.split(".")[-2] if "." in name else name
    if kind == "embed":
        return mx.random.normal(shape, mx.float32, scale=cfg.embed_std, key=key, stream=mx.cpu)
    if kind in ("attn", "ffn", "value_embed"):
        if cfg.init_out_proj == "zero" and leaf in OUT_PROJ:
            return mx.zeros(shape)
        d_in = cfg.d_model if kind == "value_embed" else shape[1]
        a = (3**0.5) * 0.5 / d_in**0.5
        return mx.random.uniform(-a, a, shape, mx.float32, key=key, stream=mx.cpu)
    if kind == "ve_gate":
        return mx.zeros(shape)
    if kind == "xl":
        return mx.array(cfg.xl_values())
    if kind == "tl":
        return mx.full(shape, cfg.tl_value())
    return mx.full(shape, cfg.norm_out_gain_init if leaf in OUT_NORMS else 1.0)


def init_params(cfg: ModelConfig, seed: int = 0) -> Params:
    """Initialise every parameter by name as the table above says, as FP32 arrays."""
    return {
        name: _init_one(cfg, name, shape, seed).astype(mx.float32)
        for name, shape in param_shapes(cfg).items()
    }


def part_of(name: str) -> str:
    """The part of the parameter count a parameter belongs to."""
    kind = kind_of(name)
    return _COUNT_ROW.get(kind, kind)


def param_counts(cfg: ModelConfig) -> dict[str, int]:
    """Parameter counts for a configuration, one per part (`part_of`), with `dense` and `active`.

    `active` counts the parameters every token uses: attention, feed-forward and the tied
    embedding. It counts from `param_shapes`, so no parameter is built.
    """
    rows: dict[str, int] = dict.fromkeys(
        ("embed", "value_embed", "attention", "ffn", "norms", "scalars"), 0
    )
    for name, shape in param_shapes(cfg).items():
        rows[part_of(name)] += math.prod(shape)
    rows["dense"] = sum(rows.values())
    rows["active"] = rows["attention"] + rows["ffn"] + rows["embed"]
    return rows
