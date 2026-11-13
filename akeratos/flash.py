"""Fused attention for training on the Mac's GPU: three Metal kernels, no score matrix in memory.

`attn.path` fused runs every layer's attention core through three Metal kernels
(`mx.fast.metal_kernel`), wrapped in one `mx.custom_function`:

  forward   O = softmax(scale·Q Kᵀ + mask) V and the row log-sum-exp, flash style: no score
            matrix and no mask array in memory;
  dq        dQ = scale·dS K, with S and dP recomputed per key tile;
  dkdv      dK = scale·dSᵀ Q and dV = Pᵀ dO, summed over the query heads of each key-value head,
            with S and dP recomputed per query tile;
  where P = exp(S - lse), dP = dO Vᵀ and dS = P·(dP - rowsum(dO·O)). Every sum runs in a fixed
  order, so the result is deterministic (no atomics).

MLX 0.32.3 runs training attention unfused: `ScaledDotProductAttention::use_fallback` is true
while tracing gradients and `ScaledDotProductAttentionVJP::eval_gpu` is "NYI" (`mlx/fast.cpp`,
`mlx/backend/metal/scaled_dot_product_attention.cpp` at v0.32.3). These kernels replace it.

Layouts: q [B, H, T, D], k and v [B, KV, T, D] (after RoPE); o [B, T, H, D]; lse [B, H, T] FP32.
"""

from collections.abc import Callable
from typing import NamedTuple, Protocol, cast

import mlx.core as mx

from akeratos.flash_metal import DKDV_SRC, DQ_SRC, FWD_SRC, HEADER


class Bounds(NamedTuple):
    """One layer type's mask as bounds, int32 [B, T]: keys lo[p]..p; queries c..qend[c]."""

    lo: mx.array
    qend: mx.array


class Tiles(NamedTuple):
    """Tile sizes: forward and dQ use (bq, bk); dK and dV use (bq2 query rows, bk2 keys)."""

    bq: int = 32
    bk: int = 32
    bq2: int = 16
    bk2: int = 32


TILES = Tiles()  # fixed in code: a different tile sums in a different order (other bits)
MAX_HEAD_DIM = 128  # dK/dV needs 192·D bytes of threadgroup memory; 32 KiB is the limit


def bounds(segment_ids: mx.array, window: int | None) -> Bounds:
    """The bounds of one layer type: `window` None for global layers.

    The kernels compute the mask from them: query p sees key c if and only if lo[p] ≤ c ≤ p, and
    key c is seen by queries c..qend[c]. That is the rule of `build_masks` in
    `akeratos/model.py`: key ≤ query, same segment, within W keys on sliding layers. Key tiles
    no query of a tile can see are never loaded, so sliding layers execute about W + 32 keys per
    query (the banded path executes 2W) and cross-document tiles are skipped.

    `segment_ids` [B, T] are nondecreasing per row; only equality matters, so any array whose
    equal values form one run per row works (all zeros: a plain causal mask).
    """
    seg = segment_ids.astype(mx.int32)
    B, T = seg.shape
    t = mx.broadcast_to(mx.arange(T, dtype=mx.int32)[None], (B, T))
    first = mx.concatenate([mx.ones((B, 1), mx.bool_), seg[:, 1:] != seg[:, :-1]], axis=1)
    last = mx.concatenate([seg[:, 1:] != seg[:, :-1], mx.ones((B, 1), mx.bool_)], axis=1)
    sstart = mx.cummax(mx.where(first, t, 0), axis=1)
    send = mx.cummin(mx.where(last, t, T - 1), axis=1, reverse=True)
    if window is None:
        return Bounds(sstart.astype(mx.int32), send.astype(mx.int32))
    lo = mx.maximum(sstart, t - window + 1)
    qend = mx.minimum(send, t + window - 1)
    return Bounds(lo.astype(mx.int32), qend.astype(mx.int32))


def keep_mask(lo: mx.array) -> mx.array:
    """Boolean [B, 1, T, T] (True keeps) equivalent to `lo`: lo[p] ≤ c ≤ p."""
    T = lo.shape[1]
    c = mx.arange(T, dtype=mx.int32)
    return ((c[None, None, :] >= lo[:, :, None]) & (c[None, None, :] <= c[None, :, None]))[:, None]


# ---------------------------------------------------------------- reference (pure MLX)


def reference_forward(
    q: mx.array, k: mx.array, v: mx.array, lo: mx.array, scale: float
) -> tuple[mx.array, mx.array]:
    """FP32 masked attention: o [B, T, H, D] in q's dtype and lse [B, H, T] FP32."""
    _, H, _, _ = q.shape
    R = H // k.shape[1]
    qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
    kf, vf = mx.repeat(kf, R, axis=1), mx.repeat(vf, R, axis=1)
    s = (qf @ mx.swapaxes(kf, -1, -2)) * scale
    s = mx.where(keep_mask(lo), s, -mx.inf)
    lse = mx.logsumexp(s, axis=-1)
    o = mx.exp(s - lse[..., None]) @ vf
    return mx.transpose(o, (0, 2, 1, 3)).astype(q.dtype), lse


def reference_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    lo: mx.array,
    scale: float,
    lse: mx.array,
    o: mx.array,
    do: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """The kernels' backward formulas in FP32 MLX ops: (dq, dk, dv) in the inputs' dtypes."""
    B, H, T, D = q.shape
    kv = k.shape[1]
    R = H // kv
    qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
    kr, vr = mx.repeat(kf, R, axis=1), mx.repeat(vf, R, axis=1)
    dof = mx.transpose(do.astype(mx.float32), (0, 2, 1, 3))  # [B, H, T, D]
    of = mx.transpose(o.astype(mx.float32), (0, 2, 1, 3))
    s = (qf @ mx.swapaxes(kr, -1, -2)) * scale
    p = mx.where(keep_mask(lo), mx.exp(s - lse[..., None]), 0.0)
    dvec = mx.sum(dof * of, axis=-1, keepdims=True)
    dp = dof @ mx.swapaxes(vr, -1, -2)
    ds = p * (dp - dvec)
    dq = (ds @ kr) * scale
    dk = mx.sum(mx.reshape(mx.swapaxes(ds, -1, -2) @ qf, (B, kv, R, T, D)), axis=2) * scale
    dv = mx.sum(mx.reshape(mx.swapaxes(p, -1, -2) @ dof, (B, kv, R, T, D)), axis=2)
    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)


# ---------------------------------------------------------------- kernels


_Kernel = Callable[..., list[mx.array]]


_SPECS = {
    "fwd": (FWD_SRC, ["q", "k", "v", "lo", "params", "fparams"], ["o", "lse"]),
    "dq": (DQ_SRC, ["q", "k", "v", "dout", "lse", "dvec", "lo", "params", "fparams"], ["dq"]),
    "dkdv": (
        DKDV_SRC,
        ["q", "k", "v", "dout", "lse", "dvec", "lo", "qend", "params", "fparams"],
        ["dk", "dv"],
    ),
}
_KERNELS: dict[str, _Kernel] = {}


def _kernel(name: str) -> _Kernel:
    if name not in _KERNELS:
        src, ins, outs = _SPECS[name]
        _KERNELS[name] = cast(
            _Kernel,
            mx.fast.metal_kernel(
                name=f"mf_flash_{name}",
                input_names=ins,
                output_names=outs,
                source=src,
                header=HEADER,
            ),
        )
    return _KERNELS[name]


def _fwd(q: mx.array, k: mx.array, v: mx.array, b: Bounds, scale: float) -> list[mx.array]:
    B, H, T, D = q.shape
    params = mx.array([B, H, k.shape[1], T], dtype=mx.int32)
    fparams = mx.array([scale], dtype=mx.float32)
    nt = (TILES.bq // 8) * 32
    return _kernel("fwd")(
        inputs=[q, k, v, b.lo, params, fparams],
        template=[("BQ", TILES.bq), ("BK", TILES.bk), ("HD", D)],
        grid=((T // TILES.bq) * nt, H, B),
        threadgroup=(nt, 1, 1),
        output_shapes=[(B, T, H, D), (B, H, T)],
        output_dtypes=[q.dtype, mx.float32],
    )


def _bwd(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    b: Bounds,
    scale: float,
    lse: mx.array,
    o: mx.array,
    do: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    B, H, T, D = q.shape
    kv = k.shape[1]
    do = do.astype(q.dtype)
    dvec = mx.transpose(mx.sum(do.astype(mx.float32) * o.astype(mx.float32), axis=-1), (0, 2, 1))
    params = mx.array([B, H, kv, T], dtype=mx.int32)
    fparams = mx.array([scale], dtype=mx.float32)
    nt = (TILES.bq // 8) * 32
    (dq,) = _kernel("dq")(
        inputs=[q, k, v, do, lse, dvec, b.lo, params, fparams],
        template=[("BQ", TILES.bq), ("BK", TILES.bk), ("HD", D)],
        grid=((T // TILES.bq) * nt, H, B),
        threadgroup=(nt, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    nt2 = (TILES.bk2 // 8) * 32
    dk, dv = _kernel("dkdv")(
        inputs=[q, k, v, do, lse, dvec, b.lo, b.qend, params, fparams],
        template=[("BQ", TILES.bq2), ("BK", TILES.bk2), ("HD", D)],
        grid=((T // TILES.bk2) * nt2, kv, B),
        threadgroup=(nt2, 1, 1),
        output_shapes=[k.shape, v.shape],
        output_dtypes=[k.dtype, v.dtype],
    )
    return dq, dk, dv


def kernels_apply(q: mx.array, k: mx.array) -> bool:
    """Whether `attend` runs the Metal kernels rather than the FP32 reference.

    It does on the GPU, for bfloat16 q, k and v, T a multiple of every tile and a head size
    that is a multiple of 8, up to 128.
    """
    _, H, T, D = q.shape
    return (
        mx.default_device() == mx.gpu
        and q.dtype == mx.bfloat16
        and k.dtype == mx.bfloat16
        and H % k.shape[1] == 0
        and D % 8 == 0
        and D <= MAX_HEAD_DIM
        and all(T % t == 0 for t in TILES)
    )


def kernels_fit(heads: int, kv_heads: int, head_dim: int, seq_len: int, dtype: mx.Dtype) -> bool:
    """Whether a run of this shape and compute dtype would run the kernels (`kernels_apply`).

    `train/entry.py` reads it once, at a fresh run's start, to pick the default `attn.path`.
    """
    return (
        mx.default_device() == mx.gpu
        and dtype == mx.bfloat16
        and heads % kv_heads == 0
        and head_dim % 8 == 0
        and head_dim <= MAX_HEAD_DIM
        and all(seq_len % t == 0 for t in TILES)
    )


# ---------------------------------------------------------------- the custom function


_Fwd = Callable[..., tuple[mx.array, mx.array]]
_Vjp = Callable[
    [tuple[mx.array, ...], tuple[mx.array, ...], tuple[mx.array, ...]], tuple[mx.array, ...]
]


class _Custom(Protocol):
    def __call__(self, *args: mx.array) -> tuple[mx.array, mx.array]: ...

    def vjp(self, f: _Vjp) -> _Vjp: ...


def _core_fwd(
    q: mx.array, k: mx.array, v: mx.array, lo: mx.array, qend: mx.array
) -> tuple[mx.array, mx.array]:
    scale = q.shape[-1] ** -0.5
    if kernels_apply(q, k):
        o, lse = _fwd(q, k, v, Bounds(lo, qend), scale)
        return o, lse
    return reference_forward(q, k, v, lo, scale)


def _core_vjp(
    primals: tuple[mx.array, ...], cotangents: tuple[mx.array, ...], outputs: tuple[mx.array, ...]
) -> tuple[mx.array, ...]:
    q, k, v, lo, qend = primals
    o, lse = outputs
    scale = q.shape[-1] ** -0.5
    if kernels_apply(q, k):
        dq, dk, dv = _bwd(q, k, v, Bounds(lo, qend), scale, lse, o, cotangents[0])
    else:
        dq, dk, dv = reference_backward(q, k, v, lo, scale, lse, o, cotangents[0])
    return dq, dk, dv, mx.zeros_like(lo), mx.zeros_like(qend)


_core = cast(_Custom, mx.custom_function(cast(_Fwd, _core_fwd)))
_core.vjp(_core_vjp)


def attend(q: mx.array, k: mx.array, v: mx.array, b: Bounds) -> mx.array:
    """Masked attention o [B, T, H, D] under `b`, differentiable in q, k and v.

    q [B, H, T, D] and k, v [B, KV, T, D] after RoPE; scale 1/√D. Where the kernels do not
    apply (`kernels_apply`), it runs `reference_forward` and `reference_backward`, the same
    formulas in FP32 MLX ops, so the CPU tests check the wiring and the gradients.
    """
    return _core(q, k, v, b.lo, b.qend)[0]
