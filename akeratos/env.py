"""The package's only access to the environment, and TF32 off before MLX is imported.

It reads and writes one variable, `MLX_ENABLE_TF32`. `get` reads it and raises on any other
name, so a typo cannot read silently. `set_tf32` writes it, and refuses once `mlx.core` is
imported, because MLX reads the variable once per process and a later change would have no
effect. `tf32_off` sets it to "0" and runs first in any entry point, before `import mlx`.
This file imports nothing from MLX.
"""

from __future__ import annotations

import os
import sys

from akeratos.errors import ConfigError

REGISTERED = frozenset({"MLX_ENABLE_TF32"})


def get(name: str) -> str | None:
    """The value of a registered environment variable, or None when it is unset."""
    if name not in REGISTERED:
        raise KeyError(f"{name} is not a registered environment variable")
    return os.environ.get(name)


def set_tf32(enabled: bool) -> None:
    """Set `MLX_ENABLE_TF32` before MLX is imported."""
    if "mlx.core" in sys.modules:
        msg = "MLX_ENABLE_TF32 is read once per process; set it before the first `import mlx`"
        raise ConfigError(msg)
    os.environ["MLX_ENABLE_TF32"] = "1" if enabled else "0"


# MLX 0.32.3's `enable_tf32()` (`mlx/include/mlx/utils.h`) reads `MLX_ENABLE_TF32` into a
# function-local static with default 1. `numerics.record()` writes the value to the run manifest.
TF32_VAR = "MLX_ENABLE_TF32"


def tf32_off() -> None:
    """Set `MLX_ENABLE_TF32=0` before MLX is imported, or raise if it is too late to."""
    if get(TF32_VAR) == "0":
        return
    set_tf32(False)
