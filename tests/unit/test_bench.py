"""`akeratos/bench.py` and `bake bench`, on MLX's CPU device.

`bench.run` runs `tests/support/benchkit.py`'s tiny shape on `TickingClock`, which moves one
second on every monotonic read, so every step takes exactly 1 s. `figures` gets made-up log
lines in which the warm-up steps differ from the timed ones, and a spy on it shows that `run`
passes each run's log with the warm-up count and divides the on run's step time by the off
run's. With `kernels_fit` patched to say yes, as on the GPU, the row records the fused path; on
the CPU that path runs its FP32 reference. The CLI tests replace the hardware reads, the cache
cap and `bench.run`, so nothing here reads this machine's hardware or touches Metal.
"""

from __future__ import annotations

import errno
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

import akeratos
from akeratos import bench, cli, digest, macops
from akeratos.clock import SystemClock
from akeratos.table.config import TableConfig
from tests.support.benchkit import M4_MAX, ON_AC, TINY, D, tiny_row

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["schema", "person", "hardware", "versions", "conditions", "shape", "arms"]
FIELDS += ["on_off_ratio", "seconds", "utc"]


def head() -> str | None:
    """This checkout's HEAD, or None outside a clone, as the row records it."""
    if not (ROOT / ".git").exists():
        return None
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    return subprocess.run(git, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(scope="module")
def tiny() -> tuple[dict[str, Any], str]:
    """One tiny bench, on battery at its start and on AC at its end, and its stderr."""
    readings = iter([{"power": macops.BATTERY, "thermal_level": 1}, {**ON_AC, "thermal_level": 2}])
    return tiny_row(lambda: next(readings))


def test_a_tiny_bench_fills_every_field_of_the_row(tiny) -> None:
    row, _ = tiny
    assert list(row) == FIELDS
    assert row["schema"] == "metalforge.bench-row/v1"
    assert row["person"] == "BhodiSea"
    assert row["hardware"] == {**M4_MAX._asdict(), "class": "m4-max-40c-48gb-macos26-laptop"}
    versions = row["versions"]
    assert list(versions) == ["metalforge", "commit", "mlx", "python"]
    assert versions["metalforge"] == akeratos.__version__
    assert versions["mlx"] == metadata.version("mlx")
    assert versions["python"] == platform.python_version()
    assert versions["commit"] == head()
    assert row["shape"] == {**TINY.describe(), "attn_path": "banded"}  # no kernels on the CPU
    assert row["utc"] == "2026-01-01T00:00:00Z"  # the fake clock's start
    # Two reads a step after the start read, then the closing read.
    assert row["seconds"] == 2 * 2 * (TINY.warmup_steps + TINY.steps) + 1


def test_each_run_is_timed_by_its_steps_and_the_ratio_divides_them(tiny) -> None:
    row, _ = tiny
    tokens = TINY.rows * TINY.seq_len
    off, on = row["arms"]["off"], row["arms"]["on"]
    assert list(off) == ["step_s", "tokens_per_s", "peak_bytes"]
    assert list(on) == [*off, "rows_updated", "table_gbps_estimate"]
    for arm in (off, on):
        assert arm["step_s"] == 1.0  # one tick between a step's two reads
        assert arm["tokens_per_s"] == tokens
        assert isinstance(arm["peak_bytes"], int)
        assert arm["peak_bytes"] > 0
    table = TableConfig(params=TINY.table_params, width=D)
    assert on["peak_bytes"] >= table.bytes()  # the table is built after the counter restarts
    assert 0 < on["rows_updated"] <= 2 * tokens  # two orders: at most two rows a token
    assert on["table_gbps_estimate"] == on["rows_updated"] * D * 2 / 1e9  # BF16 rows of width d
    assert row["on_off_ratio"] == on["step_s"] / off["step_s"]


def test_run_passes_each_runs_log_with_its_warm_up_and_divides_on_by_off(monkeypatch) -> None:
    calls: list[tuple[int, int, int | None]] = []
    figures = bench.figures

    def spy(lines, warmup, row_bytes=None):
        calls.append((len(lines), warmup, row_bytes))
        arm = figures(lines, warmup, row_bytes)
        if row_bytes is not None:
            arm["step_s"] *= 2  # the table on: its steps take twice as long
        return arm

    monkeypatch.setattr(bench, "figures", spy)
    row, _ = tiny_row()
    n, row_bytes = TINY.warmup_steps + TINY.steps, D * 2  # a table row: D numbers of 2 bytes (F2)
    assert calls == [(n, TINY.warmup_steps, None), (n, TINY.warmup_steps, row_bytes)]
    assert row["on_off_ratio"] == 2.0


def test_the_attention_path_is_fused_where_the_kernels_fit(monkeypatch) -> None:
    asked: list[tuple[object, ...]] = []

    def fit(*shape: object) -> bool:
        asked.append(shape)
        return True

    monkeypatch.setattr(bench, "kernels_fit", fit)
    row, _ = tiny_row()
    keys = TINY.model_keys()
    want = (keys["n_heads"], keys["n_kv_heads"], keys["head_dim"], TINY.seq_len, mx.bfloat16)
    assert asked == [want]
    assert row["shape"]["attn_path"] == "fused"


def test_power_and_heat_are_read_at_the_start_and_end_and_battery_warns(tiny) -> None:
    row, err = tiny
    assert row["conditions"] == {"power": ["battery", "ac"], "thermal_level": [1, 2]}
    assert [line for line in err.splitlines() if "not AC" in line] == [
        "bench: the power at the start is battery, not AC; "
        "results/bench/ takes only rows run on AC power"
    ]
    assert "bench: table off, 5 steps\n" in err
    assert "bench: table on, 5 steps\n" in err


def line(step_s: float, peak: int, rows: int) -> dict[str, Any]:
    """A log line with only the fields `figures` reads."""
    return {
        "step_s": step_s,
        "tokens_per_s": 64 / step_s,
        "peak_bytes": peak,
        "table": {"rows": rows},
    }


def test_warm_up_steps_are_left_out_of_every_figure() -> None:
    warm = [line(50.0, 999, 9), line(40.0, 998, 8)]
    timed = [line(2.0, 30, 300), line(1.0, 40, 100), line(4.0, 35, 200)]
    want = {"step_s": 2.0, "tokens_per_s": 32.0, "peak_bytes": 40}
    assert bench.figures([*warm, *timed], 2) == want
    assert bench.figures([*warm, *timed], 2, 512) == {
        **want,
        "rows_updated": 200,
        "table_gbps_estimate": 200 * 512 / 2.0 / 1e9,
    }


def test_the_peak_counter_restarts_before_each_run(monkeypatch) -> None:
    events: list[str] = []

    class Spy(bench.Run):
        def train(self, until=None, on_step=None):
            events.append("train with the table" if self.table else "train")
            return super().train(until, on_step)

    monkeypatch.setattr(bench, "Run", Spy)
    monkeypatch.setattr(bench.memory, "reset_peak", lambda: events.append("reset"))
    tiny_row(person=None)
    assert events == ["reset", "train", "reset", "train with the table"]


def test_a_rows_file_is_named_by_its_class_and_mlx_version() -> None:
    name = bench.file_name("m4-max-40c-48gb-macos26-laptop", "0.32.3")
    assert name == "m4-max-40c-48gb-macos26-laptop-mlx0.32.3.json"


# ---- bake bench --------------------------------------------------------------------------------

ARM = {"step_s": 0.0812, "tokens_per_s": 50443.3, "peak_bytes": 6 * 2**30}
ROW: dict[str, Any] = {
    "hardware": {"class": "m4-max-40c-48gb-macos26-laptop"},
    "versions": {"mlx": "0.32.3"},
    "arms": {
        "off": ARM,
        "on": {**ARM, "step_s": 0.1218, "rows_updated": 7987.5, "table_gbps_estimate": 0.0336},
    },
    "on_off_ratio": 1.5,
}
SUMMARY = (
    "m4-max-40c-48gb-macos26-laptop, MLX 0.32.3\n"
    "            ms/step   tokens/s  peak GiB\n"
    "table off      81.2     50,443      6.00\n"
    "table on      121.8     50,443      6.00\n"
    "on/off 1.500; table bandwidth 0.0336 GB/s (an estimate: rows updated x row bytes / step time)"
)


@pytest.fixture
def calls(monkeypatch) -> list[tuple[Any, ...]]:
    """What `bake bench` asks of macops and of the bench, which give M4_MAX and ROW."""
    seen: list[tuple[Any, ...]] = []
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")  # MLX is imported already, with TF32 off

    def fake_run(*args: Any) -> dict[str, Any]:
        seen.append(("run", *args))
        return ROW

    monkeypatch.setattr(macops, "read_hardware", lambda: M4_MAX)
    monkeypatch.setattr(macops, "apply_cache_limit", lambda n: seen.append(("cache", n)))
    monkeypatch.setattr(bench, "run", fake_run)
    return seen


def test_bake_bench_caps_the_cache_then_runs_the_bench_and_prints_a_table(calls, capsys) -> None:
    assert cli.main(["bench"]) == 0
    assert capsys.readouterr().out == SUMMARY + "\n"
    assert [c[0] for c in calls] == ["cache", "run"]
    assert calls[0] == ("cache", 2 * 2**30)
    hardware, conditions, clock, person, shape = calls[1][1:]
    assert (hardware, conditions, person, shape) == (
        M4_MAX,
        macops.read_conditions,
        None,
        bench.SHAPE,
    )
    assert isinstance(clock, SystemClock)


def test_json_prints_the_row_alone_once(calls, capsys) -> None:
    assert cli.main(["bench", "--json", "--person", "BhodiSea"]) == 0
    assert capsys.readouterr().out == digest.pretty(ROW)  # one final newline, pretty's own
    assert calls[1][4] == "BhodiSea"


def test_out_writes_the_row_and_refuses_a_file_that_exists(calls, capsys, tmp_path) -> None:
    argv = ["bench", "--out", str(tmp_path / "bench"), "--person", "BhodiSea"]
    assert cli.main(argv) == 0
    path = tmp_path / "bench" / bench.file_name(M4_MAX.key(), metadata.version("mlx"))
    assert path.read_text(encoding="utf-8") == digest.pretty(ROW)
    assert f"bench: wrote {path}\n" in capsys.readouterr().err
    runs = len(calls)
    assert cli.main(argv) == 78
    assert capsys.readouterr().err == (
        f"configuration error: {path} exists: a class holds one row per MLX version\n"
    )
    assert len(calls) == runs  # refused before the bench ran
    assert path.read_text(encoding="utf-8") == digest.pretty(ROW)


def test_out_needs_a_person(calls, capsys, tmp_path) -> None:
    assert cli.main(["bench", "--out", str(tmp_path)]) == 78
    assert capsys.readouterr().err == (
        "configuration error: --out needs --person HANDLE, the GitHub handle the row credits\n"
    )
    assert calls == []


@pytest.mark.parametrize("out", [False, True], ids=["printed", "written"])
def test_a_person_that_is_not_a_github_handle_is_refused_before_the_run(
    calls, capsys, tmp_path, out
) -> None:
    argv = ["bench", "--person", "@BhodiSea", *(["--out", str(tmp_path)] if out else [])]
    assert cli.main(argv) == 78
    assert capsys.readouterr().err == (
        "configuration error: --person @BhodiSea is not a GitHub handle\n"
    )
    assert calls == []


@pytest.mark.parametrize(
    ("where", "code"),
    [("file", errno.EEXIST), ("file/bench", errno.ENOTDIR)],
    ids=["a file", "in a file"],
)
def test_an_out_that_cannot_be_a_folder_is_refused_before_the_run(
    calls, capsys, tmp_path, where, code
) -> None:
    (tmp_path / "file").write_text("x", encoding="utf-8")
    out = tmp_path / where
    assert cli.main(["bench", "--out", str(out), "--person", "BhodiSea"]) == 78
    assert capsys.readouterr().err == (
        f"configuration error: --out: [Errno {code}] {os.strerror(code)}: '{out}'\n"
    )
    assert calls == []


def test_the_summary_shows_each_run_then_the_ratio_and_the_estimate() -> None:
    assert bench.summary(ROW) == SUMMARY


@pytest.mark.parametrize("argv", [["--help"], ["bench", "--help"]], ids=["bake", "bench"])
def test_help_lists_bench_and_loads_no_mlx(argv, children) -> None:
    command = [sys.executable, "-X", "importtime", "-m", "akeratos", *argv]
    proc = children.run(command, cwd=ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "bench" in proc.stdout
    assert " mlx" not in proc.stderr
