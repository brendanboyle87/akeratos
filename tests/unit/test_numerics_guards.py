"""The numerics guards: the FP32 check and TF32.

`assert_fp32` names every array that is not FP32. TF32 is set off before MLX is imported and
recorded, a late write is refused, and `akeratos/env.py` reads only registered names.
"""

from __future__ import annotations

import os
import subprocess
import sys

import mlx.core as mx
import pytest

from akeratos import env
from akeratos import numerics as dtypes
from akeratos.errors import ConfigError

mx.set_default_device(mx.cpu)


def test_assert_fp32_names_the_offenders():
    dtypes.assert_fp32({"a": mx.zeros((2,)), "s": {"step": mx.array(1, mx.uint64)}})
    with pytest.raises(TypeError, match=r"g\.v"):
        dtypes.assert_fp32({"g": {"v": mx.zeros((2,), mx.bfloat16)}}, "state")


def _run(code, env):
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, check=False
    )
    return out.stdout.strip()


def test_tf32_is_set_off_before_mlx_and_recorded():
    """MLX reads MLX_ENABLE_TF32 once per process (utils.h enable_tf32, default 1)."""
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    pre = "from akeratos.errors import ConfigError\nfrom akeratos.env import tf32_off\n"
    on = _run(
        pre + "tf32_off()\nfrom akeratos.numerics import record\n"
        "print(record()['MLX_ENABLE_TF32'])",
        env,
    )
    late = _run(
        pre + "import mlx.core\ntry:\n    tf32_off(); print('accepted')\n"
        "except ConfigError:\n    print('refused')",
        env,
    )
    already = _run(
        pre + "import mlx.core\ntf32_off(); print('accepted')", env | {"MLX_ENABLE_TF32": "0"}
    )
    assert [on, late, already] == ["0", "refused", "accepted"]


def test_env_reads_registered_names_only_and_refuses_a_late_tf32_write():
    with pytest.raises(KeyError, match="registered"):
        env.get("MLX_ENABLE_TF33")
    assert env.get("MLX_ENABLE_TF32") in (None, "0", "1")
    with pytest.raises(ConfigError, match="once per process"):
        env.set_tf32(False)  # mlx.core is imported in this process
