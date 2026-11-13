"""Numerics guards: an FP32 check over a tree of arrays, and the manifest's numerics lines.

The optimiser keeps everything it looks at in FP32: master weights, gradients and both moments.
In BF16, Adam's 0.9992 rounds to 1.0, and MLX 0.32.3's AdamW casts the rate to the gradient's
dtype. `assert_fp32` checks a tree; `record` gives the effective `MLX_ENABLE_TF32` and the
versions it was read under.

This file imports MLX: import it only after `env.tf32_off()` has run.
"""

from __future__ import annotations

import importlib.metadata
import platform
from collections.abc import Iterable, Mapping
from typing import cast

import mlx.core as mx
import numpy as np

from akeratos import env
from akeratos.env import TF32_VAR

_SHOWN = 8


def _leaves(tree: object, prefix: str = "") -> Iterable[tuple[str, mx.array]]:
    if isinstance(tree, mx.array):
        yield prefix, tree
    elif isinstance(tree, Mapping):
        for k, v in cast("Mapping[object, object]", tree).items():
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))


def assert_fp32(
    tree: object, what: str = "tree", skip: tuple[str, ...] = ("step", "learning_rate")
) -> None:
    """Raise TypeError naming every array in `tree` that is not FP32.

    `tree` is an array or nested mappings of arrays (an optimiser's state);
    leaves whose last name is in `skip` are not checked.
    """
    bad = [
        (n, str(v.dtype))
        for n, v in _leaves(tree)
        if v.dtype != mx.float32 and n.rsplit(".", 1)[-1] not in skip
    ]
    if bad:
        more = " ..." if len(bad) > _SHOWN else ""
        msg = f"{what}: not FP32: {bad[:_SHOWN]}{more}"
        raise TypeError(msg)


def record() -> dict[str, str | None]:
    """The effective `MLX_ENABLE_TF32` and the versions it was read under."""
    return {
        TF32_VAR: env.get(TF32_VAR),
        "mlx": importlib.metadata.version("mlx"),
        "device": str(mx.default_device()),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
