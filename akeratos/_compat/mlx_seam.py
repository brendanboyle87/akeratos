"""The calls into MLX that go through a typed seam.

MLX 0.32.3's stubs leave several calls partly unknown under strict pyright: `mx.eval` has no
parameter type, `array.at[...].add` takes a partly unknown NumPy type, `mx.save_safetensors`
returns an unannotated value and `mx.load` returns a union of three shapes. This file sits in the
one root where pyright relaxes those reports, and wraps each call in an annotated function:
`evaluate`, `scatter_add`, `save_safetensors` and `load_safetensors`. It also holds `AdamWGroup`,
the seam over `mlx.optimizers.AdamW`, and the only reads of MLX's allocator counters: `memory`,
`reset_peak` and `active_bytes`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim


def evaluate(*trees: object) -> None:
    """`mx.eval` over arrays and nested lists, tuples or dicts of arrays."""
    mx.eval(*trees)


def scatter_add(a: mx.array, rows: mx.array, values: mx.array) -> mx.array:
    """`a` with `values` added at `rows` along axis 0 (lazy)."""
    return a.at[rows].add(values)


def save_safetensors(
    path: Path, arrays: dict[str, mx.array], metadata: dict[str, str] | None = None
) -> None:
    """Write `arrays` to one safetensors file at `path`, with string `__metadata__`."""
    mx.save_safetensors(str(path), arrays, metadata=metadata)


def load_safetensors(path: Path) -> tuple[dict[str, mx.array], dict[str, str]]:
    """The named arrays of the safetensors file at `path` and its `__metadata__`."""
    loaded = cast(Any, mx.load(str(path), return_metadata=True))
    arrays, metadata = loaded
    if not isinstance(arrays, dict):
        raise ValueError(f"{path} holds no named arrays")
    return cast(dict[str, mx.array], arrays), {str(k): str(v) for k, v in dict(metadata).items()}


# MLX 0.32.3's `AdamW` defaults to `bias_correction=False` and `weight_decay=0.01` and casts
# the rate to the gradient's dtype. `AdamWGroup` takes no defaults: the rate, both betas, ε and
# the decay are required, and bias correction is always on. `MultiOptimizer` is never used,
# because its rate setter overwrites every group; each group owns one `AdamWGroup`.
Params = dict[str, mx.array]


class AdamWGroup:
    """One parameter group's AdamW, with bias correction on and decay explicit.

    MLX applies decoupled decay as w·(1 - lr(t)·weight_decay), so the caller
    passes the decay already divided by the group's peak rate when it wants
    the decay independent of the rate.
    """

    def __init__(
        self, lr: float, betas: tuple[float, float], eps: float, weight_decay: float
    ) -> None:
        self._opt = optim.AdamW(
            learning_rate=lr,
            betas=[betas[0], betas[1]],
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=True,
        )

    def init(self, params: Params) -> None:
        """Create the zero moments for `params` (a flat name-to-array dict)."""
        self._opt.init(params)

    def apply(self, grads: Params, params: Params) -> Params:
        """Return the updated parameters; the moments and the step count move in place."""
        out: Params = self._opt.apply_gradients(grads, params)
        return out

    @property
    def learning_rate(self) -> float:
        """The rate the next update uses."""
        return float(self._opt.learning_rate)

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self._opt.learning_rate = value

    @property
    def settings(self) -> dict[str, Any]:
        """The constructor's guarded arguments, read back from MLX."""
        o = self._opt
        return {
            "betas": tuple(float(b) for b in o.betas),
            "eps": float(o.eps),
            "weight_decay": float(o.weight_decay),
            "bias_correction": bool(o.bias_correction),
        }

    @property
    def state(self) -> dict[str, Any]:
        """MLX's state tree: the step, the rate and each parameter's moments."""
        out: dict[str, Any] = self._opt.state
        return out

    def steps(self) -> int:
        """Updates applied so far."""
        return int(self._opt.step)


# The step metrics carry the allocator counters as `active_bytes`, `peak_bytes` and
# `cache_bytes`, the buffer cache MLX holds for reuse. `active_bytes` alone is the counter
# injected into the table's `ShardStore`.
def memory() -> dict[str, int]:
    """MLX's allocator counters in bytes: in use, the peak since start, and the buffer cache."""
    return {
        "active_bytes": mx.get_active_memory(),
        "peak_bytes": mx.get_peak_memory(),
        "cache_bytes": mx.get_cache_memory(),
    }


def reset_peak() -> None:
    """Restart the peak counter, so `memory()["peak_bytes"]` covers only what follows."""
    mx.reset_peak_memory()


def active_bytes() -> int:
    """MLX's bytes in use now: the counter the table's write-back check reads."""
    return mx.get_active_memory()
