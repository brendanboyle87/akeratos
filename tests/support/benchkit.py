"""A tiny `bake bench` for the CPU tests: test_bench.py and test_bench_rows.py.

`TINY` is `test_cli.py`'s TINY model sizes with a table of 4,096 rows of width 64, run for 2
warm-up and 3 timed steps with the table off and then on. `M4_MAX` is an M4 Max MacBook Pro's
hardware class, as `test_hardware_class.py` reads it.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable, Mapping
from typing import Any

from akeratos import bench, macops
from akeratos.macops import Hardware
from tests.fakes.clock import TickingClock

D = 64
TINY_KEYS = {
    "d_model": D,
    "n_layers": 2,
    "n_heads": 2,
    "head_dim": 32,
    "n_kv_heads": 1,
    "ffn": 128,
    "vocab_size": 512,
    "window": 16,
}
M4_MAX = Hardware("Apple M4 Max", 40, "applegpu_g16s", 48, "26.6.2", "laptop", "Mac16,5")
ON_AC = {"power": macops.AC, "thermal_level": 0}


class Tiny(bench.Shape):
    """A shape whose model is the tiny one, not a preset's."""

    def model_keys(self) -> dict[str, Any]:
        return dict(TINY_KEYS)


TINY = Tiny(
    version=0,
    preset="tiny",
    table_params=4096 * D,
    rows=2,
    seq_len=32,
    warmup_steps=2,
    steps=3,
    seed=0,
    cache_limit_gib=2,
)


def tiny_row(
    conditions: Callable[[], Mapping[str, object]] = lambda: ON_AC, person: str | None = "BhodiSea"
) -> tuple[dict[str, Any], str]:
    """A tiny bench's row on `TickingClock`, and what it wrote on stderr."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        row = bench.run(M4_MAX, conditions, TickingClock(), person, TINY)
    return row, err.getvalue()
