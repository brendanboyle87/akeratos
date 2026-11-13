"""The 5-minute speedrun's training file: the one file a contributor or their agent edits.

`bake speedrun` runs this file once and calls `build(seed)`. It then trains the object `build`
returns with `step`, one batch of `ROWS` rows of `SEQ_LEN` tokens at a time, until 300 seconds
of step time have passed, and scores the held-out slice with `logits`. The rules this file
keeps are in docs/speedrun.md: it reads no clock, file, environment variable or network,
directly or through another process; it replaces no function or attribute outside itself; its
data comes only from `step`'s rows, and `build` does not train; a warm-up step does no more
training than a later step; `logits` changes nothing; every random draw derives from `build`'s
seed; and it imports only what `uv sync --frozen` installs.

This reference is Akeratos's 10M model without the table, whole: its weights, forward pass,
loss, optimiser and schedule. It trains in BF16, on the fused attention kernels where they run
and on the banded path elsewhere, and it scores in FP32 on the banded path.
"""

from __future__ import annotations

import hashlib
import math

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from numpy.typing import NDArray

from akeratos.flash import Bounds, attend, bounds, kernels_fit
from akeratos.train.source import StepRows

ROWS = 4  # rows per step
SEQ_LEN = 2048  # tokens per row
# The weights' shape: width, layers, query heads, head size, key-value heads, feed-forward width.
MODEL = {"d_model": 256, "n_layers": 8, "n_heads": 4, "head_dim": 64, "n_kv_heads": 2, "ffn": 1152}
WINDOW = 512  # the keys a sliding layer sees; it changes no weight's shape
WARMUP = 100  # steps of rate warm-up
DECAY = 0.2  # the share of the budget, at its end, over which the rate decays to 0
TOKENS = 14e6  # about what an M4 Max trains on in 300 s; the weight decay's rate follows from it
LR = 3e-3  # the peak learning rate

VOCAB = 49_152  # SmolLM2's vocabulary
SEP = 0  # <|endoftext|>: it opens each document
MASKED_IDS = tuple(range(1, 17))  # reserved ids, never emitted: the head masks them out
GLOBAL_EVERY = 5  # every 5th layer and the last attend to the whole document
THETA_LOCAL, THETA_GLOBAL = 1e4, 1e6  # RoPE bases on sliding and global layers
EPS = 1e-5  # every RMSNorm's epsilon
EMBED_STD = 0.02  # the tied embedding's starting standard deviation
XL_INIT = (0.20, 0.05)  # the x0 term's starting scale at the first layer and at the last
GATE_WIDTH = 12  # the value-embedding gate reads this many channels of its input
LOGIT_CAP = 15.0  # logits are capped as 15·tanh(z/15)
BASE_D = 256  # the width the rates were tuned at; the hidden rate and the logits scale by 256/d
BETA1, ADAM_EPS, CLIP = 0.9, 1e-8, 1.0  # AdamW's beta1 and epsilon; the clip on the gradient norm

EMBED = "emb.embed.weight"  # the tied matrix: the input embedding and the output head
# Each parameter kind's optimiser group: only the hidden matrices take 256/d and the decay.
GROUP_OF_KIND = {
    "attn": "hidden",
    "ffn": "hidden",
    "embed": "embed",
    "value_embed": "value_embed",
    "gain": "scalar",
    "xl": "scalar",
    "ve_gate": "scalar",
}
GROUPS = ("hidden", "embed", "value_embed", "scalar")  # the order the step sums their norms in

Params = dict[str, mx.array]  # FP32 masters by flat dotted name, cast to the compute dtype at use


# ---------------------------------------------------------------- the parameters


def is_global(i: int) -> bool:
    return (i + 1) % GLOBAL_EVERY == 0 or i == MODEL["n_layers"] - 1


def ve_layers() -> list[int]:
    """The layers with a value embedding: the odd ones."""
    return list(range(1, MODEL["n_layers"], 2))


def ve_name(i: int) -> str:
    return f"emb.ve.L{i}.weight"


def param_shapes() -> dict[str, tuple[int, ...]]:
    d, hd, f, kv_heads = MODEL["d_model"], MODEL["head_dim"], MODEL["ffn"], MODEL["n_kv_heads"]
    q, kv = MODEL["n_heads"] * hd, kv_heads * hd
    out: dict[str, tuple[int, ...]] = {EMBED: (VOCAB, d)}
    out |= {ve_name(i): (VOCAB, kv) for i in ve_layers()}
    for i in range(MODEL["n_layers"]):
        a, m = f"layers.{i}.attn.", f"layers.{i}.ffn."
        out |= {a + "q.weight": (q, d), a + "k.weight": (kv, d), a + "v.weight": (kv, d)}
        out |= {a + "o.weight": (d, q), a + "q_norm.weight": (hd,), a + "k_norm.weight": (hd,)}
        out |= {m + "up.weight": (f, d), m + "down.weight": (d, f)}
        if i in ve_layers():
            out[a + "ve_gate.weight"] = (kv_heads, GATE_WIDTH)
        for side in ("in", "out"):
            out[f"layers.{i}.norm_attn_{side}.weight"] = (d,)
            out[f"layers.{i}.norm_ffn_{side}.weight"] = (d,)
    out["final_norm.weight"] = (d,)
    out["xl"] = (MODEL["n_layers"],)
    return out


def kind_of(name: str) -> str:
    leaf = name.split(".")[-2] if "." in name else name
    if name == EMBED:
        return "embed"
    if name.startswith("emb.ve."):
        return "value_embed"
    if leaf == "ve_gate":
        return "ve_gate"
    if leaf in ("q", "k", "v", "o"):
        return "attn"
    if leaf in ("up", "down"):
        return "ffn"
    return "xl" if name == "xl" else "gain"


def xl_values() -> list[float]:
    """The x0 term's starting scales, linear from `XL_INIT[0]` at layer 0 to `XL_INIT[1]`."""
    a, b = XL_INIT
    last = max(MODEL["n_layers"] - 1, 1)
    return [a + (b - a) * i / last for i in range(MODEL["n_layers"])]


def init_seed(seed: int) -> int:
    """The init stream's seed: SHA-256 over a domain, the run's seed and the stream's name."""
    message = b"metalforge.rng/v1\0" + seed.to_bytes(8, "little") + b"init\0"
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


def name_key(seed: int, name: str) -> mx.array:
    """One key per seed and name, so adding or removing a tensor never moves another's values."""
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return mx.random.key(int.from_bytes(h, "little"))


def init_one(name: str, shape: tuple[int, ...], seed: int) -> mx.array:
    """One tensor's starting value; every draw is FP32 on the CPU stream, whatever the device."""
    kind, key = kind_of(name), name_key(seed, name)
    if kind == "embed":
        return mx.random.normal(shape, mx.float32, scale=EMBED_STD, key=key, stream=mx.cpu)
    if kind in ("attn", "ffn", "value_embed"):
        d_in = MODEL["d_model"] if kind == "value_embed" else shape[1]
        a = (3**0.5) * 0.5 / d_in**0.5  # uniform with standard deviation 0.5/√d_in
        return mx.random.uniform(-a, a, shape, mx.float32, key=key, stream=mx.cpu)
    if kind == "ve_gate":
        return mx.zeros(shape)  # the gate starts at 3·sigmoid(0) = 1.5
    if kind == "xl":
        return mx.array(xl_values())
    return mx.full(shape, 1.0)


def init_params(seed: int) -> Params:
    return {
        name: init_one(name, shape, seed).astype(mx.float32)
        for name, shape in param_shapes().items()
    }


# ---------------------------------------------------------------- the forward pass


def rms_norm(x: mx.array, weight: mx.array | None) -> mx.array:
    return mx.fast.rms_norm(x, None if weight is None else weight.astype(x.dtype), EPS)


def dense(x: mx.array, weight: mx.array) -> mx.array:
    """The product x Wᵀ for W [out, in], the FP32 master cast to x's dtype; no bias anywhere."""
    return x @ weight.astype(x.dtype).T


def gather(table: mx.array, ids: mx.array, dtype: mx.Dtype) -> mx.array:
    rows = mx.take(table, ids, axis=0)  # in FP32, so the gradient sums in FP32, not BF16
    return rows.astype(dtype)


def ve_rows(p: Params, i: int, ids: mx.array, dtype: mx.Dtype) -> mx.array | None:
    """Layer i's value-embedding rows [B, T, kv·hd], or None on an even layer."""
    name = ve_name(i)
    return gather(p[name], ids, dtype) if name in p else None


def x0(p: Params, ids: mx.array, dtype: mx.Dtype) -> mx.array:
    """The input embedding [B, T, d]: the tied matrix's rows, normed with no gain."""
    return rms_norm(gather(p[EMBED], ids, dtype), None)


def rope_freqs(glob: bool) -> mx.array:
    """The period of each pair (i, i + hd/2) of the half-split layout, FP32."""
    D = MODEL["head_dim"]
    theta = THETA_GLOBAL if glob else THETA_LOCAL
    periods = theta ** (np.arange(0, D, 2) / D)
    return mx.array(periods.tolist(), dtype=mx.float32)


def apply_rope(x: mx.array, freqs: mx.array) -> mx.array:
    """Rotate x [B, h, T, D] at positions 0..T-1, which run on across packed documents."""
    return mx.fast.rope(
        x, x.shape[-1], traditional=False, base=None, scale=1.0, offset=0, freqs=freqs
    )


def global_mask(seg: mx.array) -> mx.array:
    """[B, 1, T, T], True keeps: key ≤ query, in the same document."""
    T = seg.shape[1]
    p, q = mx.arange(T)[:, None], mx.arange(T)[None]
    return ((q <= p)[None] & (seg[:, :, None] == seg[:, None, :]))[:, None]


def band_mask(seg: mx.array) -> mx.array:
    """[B·nb, 1, W, 2W]: each block of W queries against the previous block and its own.

    It keeps exactly the W keys up to each query, in the query's document, so a sliding layer
    computes T·2W scores per head instead of T².
    """
    B, T = seg.shape
    W = WINDOW
    nb = T // W
    a, b = mx.arange(W)[:, None], mx.arange(2 * W)[None]
    rel = (b > a) & (b <= a + W)  # key (j-1)·W + b is within W of query j·W + a, not after it
    has_prev = mx.arange(nb)[:, None, None] > 0  # block 0's first W keys are padding
    static = rel[None] & (has_prev | (b >= W)[None])  # [nb, W, 2W]
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


def build_masks(segment_ids: mx.array, fused: bool) -> dict[str, mx.array]:
    """The masks of one forward pass, shared by the layers of each type.

    The fused kernels read int32 bounds [B, T] per layer type and build no mask array.

    Raises:
        ValueError: the banded path with T not a multiple of `WINDOW`.
    """
    seg = segment_ids.astype(mx.int32)
    if fused:
        kinds = {"global": None, "sliding": WINDOW}
        return {
            f"{kind}.{name}": a
            for kind, w in kinds.items()
            for name, a in bounds(seg, w)._asdict().items()
        }
    if seg.shape[1] % WINDOW:
        raise ValueError(f"banded windows need T ({seg.shape[1]}) to be a multiple of {WINDOW}")
    return {"global": global_mask(seg), "band": band_mask(seg)}


def attention(
    p: Params, i: int, a_in: mx.array, masks: dict[str, mx.array], ve: mx.array | None, fused: bool
) -> mx.array:
    """Layer i's attention on its normed input a_in [B, T, d], with grouped keys and values.

    Query head h reads key-value head h // (H / kv). On an odd layer the value adds the
    value-embedding rows through a gate of 3·sigmoid, one per key-value head.
    """
    pre = f"layers.{i}.attn."
    B, T, _ = a_in.shape
    H, kv, hd = MODEL["n_heads"], MODEL["n_kv_heads"], MODEL["head_dim"]
    q = mx.reshape(dense(a_in, p[pre + "q.weight"]), (B, T, H, hd))
    k = mx.reshape(dense(a_in, p[pre + "k.weight"]), (B, T, kv, hd))
    v = mx.reshape(dense(a_in, p[pre + "v.weight"]), (B, T, kv, hd))
    q, k = rms_norm(q, p[pre + "q_norm.weight"]), rms_norm(k, p[pre + "k_norm.weight"])
    v = rms_norm(v, None)
    if ve is not None:
        gate = 3.0 * mx.sigmoid(dense(a_in[..., :GATE_WIDTH], p[pre + "ve_gate.weight"]))
        v = v + gate[..., None] * mx.reshape(ve, (B, T, kv, hd))  # one gate per kv head
    glob = is_global(i)
    freqs = rope_freqs(glob)
    q = apply_rope(mx.transpose(q, (0, 2, 1, 3)), freqs)
    k = apply_rope(mx.transpose(k, (0, 2, 1, 3)), freqs)
    v = mx.transpose(v, (0, 2, 1, 3))
    if fused:
        kind = "global" if glob else "sliding"
        o = attend(q, k, v, Bounds(masks[kind + ".lo"], masks[kind + ".qend"]))  # [B, T, H, hd]
        return dense(mx.reshape(o, (B, T, H * hd)), p[pre + "o.weight"])
    scale = hd**-0.5
    sdpa = mx.fast.scaled_dot_product_attention
    if glob:
        o = sdpa(q, k, v, scale=scale, mask=masks["global"])
    else:
        qb, kb, vb = query_bands(q, WINDOW), key_bands(k, WINDOW), key_bands(v, WINDOW)
        o = from_bands(sdpa(qb, kb, vb, scale=scale, mask=masks["band"]), B)
    return dense(mx.reshape(mx.transpose(o, (0, 2, 1, 3)), (B, T, H * hd)), p[pre + "o.weight"])


def ffn(p: Params, i: int, x: mx.array) -> mx.array:
    """down(ReLU(up(x))²)."""
    u = dense(x, p[f"layers.{i}.ffn.up.weight"])
    return dense(mx.square(mx.maximum(u, 0)), p[f"layers.{i}.ffn.down.weight"])


def block(
    p: Params, i: int, x: mx.array, masks: dict[str, mx.array], ve: mx.array | None, fused: bool
) -> mx.array:
    """Layer i, with an RMSNorm before and after each sublayer; the residual scale is 1."""
    pre = f"layers.{i}."
    a = attention(p, i, rms_norm(x, p[pre + "norm_attn_in.weight"]), masks, ve, fused)
    x = x + rms_norm(a, p[pre + "norm_attn_out.weight"])
    f = ffn(p, i, rms_norm(x, p[pre + "norm_ffn_in.weight"]))
    return x + rms_norm(f, p[pre + "norm_ffn_out.weight"])


def hidden(
    p: Params, ids: mx.array, segment_ids: mx.array, dtype: mx.Dtype, fused: bool
) -> mx.array:
    """The final-normed state [B, T, d] in `dtype` for ids and segment ids [B, T]."""
    masks = build_masks(segment_ids, fused)
    e = x0(p, ids, dtype)
    x = e
    for i in range(MODEL["n_layers"]):
        x = x + p["xl"][i].astype(dtype) * e  # the x0 term: each layer adds xl[i]·x0
        x = block(p, i, x, masks, ve_rows(p, i, ids, dtype), fused)
    return rms_norm(x, p["final_norm.weight"])


def masked_id_mask() -> mx.array:
    m = np.zeros((VOCAB,), np.bool_)
    m[list(MASKED_IDS)] = True
    return mx.array(m.tolist(), mx.bool_)


def head_logits(p: Params, h: mx.array, masked: mx.array) -> mx.array:
    """FP32 logits [..., V] from the final-normed state h [..., d] in the compute dtype."""
    z = h @ p[EMBED].astype(h.dtype).T
    z = z.astype(mx.float32)  # upcast before the multiplier: 256/d need not be a BF16 number
    z = z * mx.array(BASE_D / MODEL["d_model"], mx.float32)
    c = mx.array(LOGIT_CAP, mx.float32)
    return mx.where(masked, -mx.inf, c * mx.tanh(z / c))


def segment_ids(ids: NDArray[np.int32]) -> NDArray[np.int32]:
    """Per row: 0 for the first document, then one more at each separator after position 0."""
    opens = (ids == SEP).astype(np.int32)
    opens[:, 0] = 0
    return np.cumsum(opens, axis=1, dtype=np.int32)


# ---------------------------------------------------------------- the loss


def prepare(rows: StepRows) -> tuple[mx.array, ...]:
    """The step's inputs after the weights: ids, segment ids, targets, weights, live targets.

    A target leaves the loss only by its weight: a masked id gets weight 0 and is set to 0,
    because its logit is -inf and inf·0 is NaN.
    """
    t = np.asarray(rows.targets, np.int64)
    live = np.ones(t.shape, np.bool_) if rows.lmask is None else np.asarray(rows.lmask) > 0
    keep = live & ~np.isin(t, MASKED_IDS)
    targets, weights = np.where(keep, t, 0).astype(np.int32), keep.astype(np.float32)
    n_valid = mx.array(float(weights.sum()), mx.float32)
    return (
        mx.array(rows.ids),
        mx.array(rows.segment_ids),
        mx.array(targets),
        mx.array(weights),
        n_valid,
    )


def loss_fn(
    p: Params,
    ids: mx.array,
    segment_ids: mx.array,
    targets: mx.array,
    weights: mx.array,
    n_valid: mx.array,
    fused: bool,
) -> mx.array:
    """The mean token cross-entropy over the live targets, from BF16 activations."""
    z = head_logits(p, hidden(p, ids, segment_ids, mx.bfloat16, fused), masked_id_mask())
    nll = mx.fast.cross_entropy(mx.contiguous(z), targets)
    return (nll * weights.astype(mx.float32)).sum() / n_valid


# ---------------------------------------------------------------- the optimiser


def group_of(names: list[str]) -> dict[str, list[str]]:
    """Each non-empty group's members, sorted, in the order of `GROUPS`."""
    out = {g: sorted(n for n in names if GROUP_OF_KIND[kind_of(n)] == g) for g in GROUPS}
    return {g: ns for g, ns in out.items() if ns}


def width_mult() -> float:
    return BASE_D / MODEL["d_model"]


def peak_rates(groups: list[str]) -> dict[str, float]:
    """Each group's peak rate: `LR`, times 256/d for the hidden matrices."""
    return {g: LR * (width_mult() if g == "hidden" else 1.0) for g in groups}


def beta2() -> float:
    """Adam's beta2, 0.95^(B / 524,288) at B tokens per update: the same half-life in tokens."""
    return 0.95 ** (ROWS * SEQ_LEN / 524_288)


def hidden_decay() -> float:
    """The hidden matrices' decay per step at full rate: 1e-3 x 0.1 x (1e9 / D) x (B / 8,192).

    D is `TOKENS`, the run's tokens, and B the tokens per update.
    """
    return 1e-3 * (0.1 * (1e9 / TOKENS) * (ROWS * SEQ_LEN / 8192))


def sq_norm(x: mx.array) -> mx.array:
    """FP32 squared norm in two stages, rows and then their sums: a flat sum loses digits."""
    x = x.astype(mx.float32)
    x = mx.reshape(x, (x.shape[0], -1) if x.ndim > 1 else (1, -1))
    return (x * x).sum(-1).sum()


def group_sq_norms(grads: Params, names: dict[str, list[str]]) -> mx.array:
    """[groups] FP32: each group's squared gradient norm, in the order of `names`."""
    out = []
    for ns in names.values():
        total = mx.array(0.0, mx.float32)
        for n in ns:
            total = total + sq_norm(grads[n])
        out.append(total)
    return mx.stack(out)


def rate(step: int, progress: float) -> float:
    """The rate's multiplier at `step`, and at `progress`, the share of the budget spent so far.

    It warms up over `WARMUP` steps, holds at 1, then falls as 1 - √u to 0 over the last `DECAY`
    of the budget, as u runs from 0 to 1.
    """
    warm = min(1.0, (step + 1) / WARMUP)
    cool = 1 - math.sqrt(max(0.0, progress - (1 - DECAY)) / DECAY)
    return warm * cool


# ---------------------------------------------------------------- the reference


class Reference:
    """Akeratos's model and optimiser, trained on the batches `bake speedrun` gives it."""

    def __init__(self, seed: int, fused: bool) -> None:
        self.fused, self.steps = fused, 0
        self.params = init_params(init_seed(seed))
        self.names = group_of(list(self.params))
        self.peak = peak_rates(list(self.names))
        self.decay = {g: hidden_decay() if g == "hidden" else 0.0 for g in self.names}
        self.opts: dict[str, optim.AdamW] = {}
        for g, ns in self.names.items():
            # MLX decays w by lr·wd, so wd = decay / peak keeps the decay at decay·s(t).
            opt = optim.AdamW(
                learning_rate=self.peak[g],
                betas=[BETA1, beta2()],
                eps=ADAM_EPS,
                weight_decay=self.decay[g] / self.peak[g],
                bias_correction=True,  # MLX's AdamW leaves it off by default
            )
            opt.init({n: self.params[n] for n in ns})
            self.opts[g] = opt
        self.states = [opt.state for opt in self.opts.values()]
        self.value_and_grad = mx.value_and_grad(loss_fn)
        self._grad_step = mx.compile(self.grad_step)
        self._apply = mx.compile(self.apply, inputs=self.states, outputs=self.states)

    def grad_step(self, p: Params, *batch: mx.array) -> tuple[mx.array, Params, mx.array, mx.array]:
        """The loss, the FP32 gradients, each group's squared norm, and whether all are finite."""
        loss, grads = self.value_and_grad(p, *batch, self.fused)
        sq = group_sq_norms(grads, self.names)
        return loss, grads, sq, mx.isfinite(loss) & mx.all(mx.isfinite(sq))

    def apply(self, grads: Params, p: Params, scale: mx.array) -> Params:
        """Each group's AdamW on its own members, the gradients first scaled by the clip."""
        new: Params = {}
        for g, opt in self.opts.items():
            ns = self.names[g]
            new.update(
                opt.apply_gradients({n: grads[n] * scale for n in ns}, {n: p[n] for n in ns})
            )
        return new

    def step(self, rows: StepRows, progress: float) -> float:
        """One optimiser step on `rows`; returns the loss. A non-finite step updates no weight."""
        s = rate(self.steps, progress)
        self.steps += 1
        loss, grads, sq, finite = self._grad_step(self.params, *prepare(rows))
        mx.eval(loss, grads, sq, finite)
        if not bool(finite):
            return float(loss)
        norm = sum(float(x) for x in sq.tolist()) ** 0.5
        scale = min(1.0, CLIP / norm) if norm > 0 else 1.0
        for g, opt in self.opts.items():
            opt.learning_rate = self.peak[g] * s
        self.params = self._apply(grads, self.params, mx.array(scale, mx.float32))
        mx.eval(self.params, self.states)
        return float(loss)

    def arrays(self) -> object:
        """The weights and the optimiser's state, which each step must have computed."""
        return self.params, self.states

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        """FP32 logits [B, T, 49,152] of ids [B, T], on the banded path."""
        x = ids.astype(np.int32)
        h = hidden(self.params, mx.array(x), mx.array(segment_ids(x)), mx.float32, fused=False)
        return head_logits(self.params, h, masked_id_mask())


def build(seed: int) -> Reference:
    """The reference model and optimiser for `seed`, untrained."""
    heads = (MODEL["n_heads"], MODEL["n_kv_heads"], MODEL["head_dim"])
    return Reference(seed, kernels_fit(*heads, SEQ_LEN, mx.bfloat16))
