"""`bake bench`: what the n-gram table costs per training step on this Mac, as one row.

`run` trains a model on synthetic token ids twice in one process, with the table off and then
on, at one fixed shape (`SHAPE`), and returns a `metalforge.bench-row/v1` row. Each run is a
fresh `train.loop.Run` on `SynthSource` in a temporary directory, built as `train/entry.py`
builds a fresh run: the 10M model (`configs.MODEL_10M`) with its table term off or on,
`attn.path` fused where the kernels fit (`flash.kernels_fit`) and banded elsewhere, BF16
compute and the compiled step. The table holds `table_params` numbers at the model's width.

Each run's `figures` cover its timed steps, after `warmup_steps` steps that compile and are not
counted, with MLX's peak counter restarted before the run. `on_off_ratio` is the on run's step
time over the off run's. The CLI passes the hardware class and `conditions`, macops' power
source and thermal level, which `run` reads at its start and its end.
"""

from __future__ import annotations

import platform
import re
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from statistics import median
from typing import Any

import akeratos
from akeratos._compat import mlx_seam as memory
from akeratos.clock import Clock
from akeratos.configs import MODEL_10M, ModelConfig, from_keys
from akeratos.flash import kernels_fit
from akeratos.macops import AC, Hardware, run_command
from akeratos.table.config import TableConfig
from akeratos.table.table import Table
from akeratos.train.loop import DTYPES, Run
from akeratos.train.recipe import from_keys as recipe_keys
from akeratos.train.source import SynthSource

SCHEMA = "metalforge.bench-row/v1"
DTYPE = "bfloat16"  # a training run's compute dtype
NUMBER_BYTES = {"F2": 2, "F1": 1, "F32": 4}  # bytes per stored number, by table format
GB = 1e9
HANDLE = re.compile(r"(?=.{1,39}\Z)[A-Za-z0-9](?:-?[A-Za-z0-9])*")  # a GitHub user name
_HEX40 = re.compile(r"[0-9a-f]{40}")

Conditions = Callable[[], Mapping[str, object]]


@dataclass(frozen=True)
class Shape:
    """What the bench runs; a change to any field is a new `version`, so rows stay comparable."""

    version: int
    preset: str  # the model's name in the row: "core-10m" is `configs.MODEL_10M`
    table_params: float  # the numbers the table stores
    rows: int  # sequences per step
    seq_len: int  # tokens per sequence
    warmup_steps: int  # untimed: they compile
    steps: int  # timed
    seed: int
    cache_limit_gib: int  # the cap on MLX's buffer cache, which the CLI installs

    def model_keys(self) -> dict[str, Any]:
        """The model's `ModelConfig` fields."""
        return dict(MODEL_10M)

    def describe(self) -> dict[str, object]:
        """The shape as JSON-ready data, for the row."""
        return {
            "version": self.version,
            "preset": self.preset,
            "table_params": self.table_params,
            "rows": self.rows,
            "seq_len": self.seq_len,
            "warmup_steps": self.warmup_steps,
            "steps": self.steps,
            "seed": self.seed,
            "cache_limit_gib": self.cache_limit_gib,
        }


SHAPE = Shape(
    version=1,
    preset="core-10m",
    table_params=1.0e9,
    rows=2,
    seq_len=2048,
    warmup_steps=10,
    steps=300,
    seed=0,
    cache_limit_gib=2,
)


def run(
    hardware: Hardware, conditions: Conditions, clock: Clock, person: str | None, shape: Shape
) -> dict[str, Any]:
    """Run the bench, table off and then on, and return its row; tests pass a tiny `shape`."""
    utc, start = clock.utc_now(), clock.monotonic_ns()
    first = _read(conditions, "start")
    off = from_keys(shape.model_keys() | {"table": False})
    fits = kernels_fit(off.n_heads, off.n_kv_heads, off.head_dim, shape.seq_len, DTYPES[DTYPE])
    off = replace(off, attn_path="fused" if fits else "banded")
    on = replace(off, table=True)
    table = TableConfig(params=shape.table_params, width=on.d_model, seed=shape.seed).validate()
    row_bytes = table.row_width() * NUMBER_BYTES[table.fmt]
    arms = {
        "off": figures(_arm(shape, off, None, clock), shape.warmup_steps),
        "on": figures(_arm(shape, on, table, clock), shape.warmup_steps, row_bytes),
    }
    last = _read(conditions, "end")
    seconds = (clock.monotonic_ns() - start) / 1e9
    return {
        "schema": SCHEMA,
        "person": person,
        "hardware": hardware.describe(),
        "versions": _versions(),
        "conditions": {key: [first[key], last[key]] for key in ("power", "thermal_level")},
        "shape": shape.describe() | {"attn_path": off.attn_path},
        "arms": arms,
        "on_off_ratio": arms["on"]["step_s"] / arms["off"]["step_s"],
        "seconds": seconds,
        "utc": utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _arm(
    shape: Shape, cfg: ModelConfig, table: TableConfig | None, clock: Clock
) -> list[dict[str, Any]]:
    """The log lines of one fresh run of the shape's steps, with `table` or with none."""
    n, tokens = shape.warmup_steps + shape.steps, shape.rows * shape.seq_len
    o, r = recipe_keys(
        {
            "tokens.budget": n * tokens,
            "batch.tokens_per_update": tokens,
            "lr.warmup": shape.warmup_steps,
            "opt.decay_ind": 0.0,  # costs what any decay does; the rule's would shrink a short run
        }
    )
    sys.stderr.write(f"bench: table {'off' if table is None else 'on'}, {n} steps\n")
    memory.reset_peak()  # the run's peak covers its own table and steps only
    store = None if table is None else Table(table, memory.active_bytes)
    source = SynthSource(shape.rows, shape.seq_len, cfg.vocab_size, seed=shape.seed)
    lines: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="bake-bench-") as out:
        loop = Run(
            cfg, o, r, source, out, seed=shape.seed, compute_dtype=DTYPE, table=store, clock=clock
        )
        loop.train(n, lambda _, line: lines.append(line))
    return lines


def figures(
    lines: Sequence[Mapping[str, Any]], warmup: int, row_bytes: int | None = None
) -> dict[str, float]:
    """One run's figures over its timed steps: the log lines after the first `warmup`.

    The medians of `step_s` and `tokens_per_s` and the largest `peak_bytes`. With `row_bytes`,
    the bytes of one stored table row (the table's width times the stored format's bytes per
    number), also the median of the table rows updated per step and `table_gbps_estimate`: those
    rows times `row_bytes` over the median step time, in GB of 10^9 bytes per second. It is an
    estimate, because the log counts the rows updated and not the rows read.
    """
    timed = lines[warmup:]
    step_s = median(x["step_s"] for x in timed)
    arm: dict[str, float] = {
        "step_s": step_s,
        "tokens_per_s": median(x["tokens_per_s"] for x in timed),
        "peak_bytes": max(x["peak_bytes"] for x in timed),
    }
    if row_bytes is not None:
        rows = median(x["table"]["rows"] for x in timed)
        arm |= {"rows_updated": rows, "table_gbps_estimate": rows * row_bytes / step_s / GB}
    return arm


def file_name(hardware_class: str, mlx: str) -> str:
    """A row's file in `results/bench/`, so a hardware class holds one row per MLX version."""
    return f"{hardware_class}-mlx{mlx}.json"


def summary(row: Mapping[str, Any]) -> str:
    """The row as a short table: each run's step time, tokens per second and peak memory."""
    lines = [
        f"{row['hardware']['class']}, MLX {row['versions']['mlx']}",
        f"{'':10}{'ms/step':>9}{'tokens/s':>11}{'peak GiB':>10}",
    ]
    for name, arm in row["arms"].items():
        rate, peak = arm["tokens_per_s"], arm["peak_bytes"] / 2**30
        lines.append(f"table {name:<4}{arm['step_s'] * 1e3:9.1f}{rate:11,.0f}{peak:10.2f}")
    gbps = row["arms"]["on"]["table_gbps_estimate"]
    lines.append(
        f"on/off {row['on_off_ratio']:.3f}; table bandwidth {gbps:.3g} GB/s "
        "(an estimate: rows updated x row bytes / step time)"
    )
    return "\n".join(lines)


def _read(conditions: Conditions, when: str) -> Mapping[str, object]:
    """The conditions now; a warning on stderr when the power is not AC."""
    now = conditions()
    if now["power"] != AC:
        sys.stderr.write(
            f"bench: the power at the {when} is {now['power'] or 'unknown'}, not AC; "
            "results/bench/ takes only rows run on AC power\n"
        )
    return now


def _versions() -> dict[str, str | None]:
    """Akeratos's, the commit's, MLX's and Python's; the commit is None outside a clone."""
    root = Path(akeratos.__file__).resolve().parents[1]
    git = ("git", "-C", str(root), "rev-parse", "HEAD")
    head = (run_command(git) if (root / ".git").exists() else None) or ""
    return {
        "metalforge": akeratos.__version__,
        "commit": head.strip() if _HEX40.fullmatch(head.strip()) else None,
        "mlx": metadata.version("mlx"),
        "python": platform.python_version(),
    }
