"""The bench rows in `results/bench/` and their page.

`speedrun/records.py`, loaded by path, checks every row and renders `results/bench/README.md`
from them; the real folder must check out, with a page that may lag by rows it does not list
yet. A page that lags passes only under `may_lag`, and one that is not the render of the rows it
lists fails under both. A good row, built here from `bench.SHAPE` with
made-up measurements and a made-up time, passes, and each bad variant of it is refused for its
one fault. Its command line runs on a temporary folder. A row from a tiny bench on MLX's CPU
device passes every check but the shape, which is the test's own, so the bench writes what the
check reads.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from akeratos import bench, digest
from tests.support.benchkit import M4_MAX, tiny_row
from tests.support.byfile import ROOT, load

mx.set_default_device(mx.cpu)

BENCH = ROOT / "results" / "bench"
NAME = "m4-max-40c-48gb-macos26-laptop-mlx0.32.3.json"
records = load("speedrun/records.py", "speedrun_records")


def good_row() -> dict[str, Any]:
    """A row as `bake bench` writes one, with made-up measurements."""
    off = {"step_s": 0.0812, "tokens_per_s": 50443.3, "peak_bytes": 6 * 2**30}
    on = {"step_s": 0.1218, "tokens_per_s": 33627.5, "peak_bytes": 8 * 2**30}
    on |= {"rows_updated": 7987.5, "table_gbps_estimate": 7987.5 * 512 / 0.1218 / 1e9}
    return {
        "schema": bench.SCHEMA,
        "person": "BhodiSea",
        "hardware": M4_MAX.describe(),
        "versions": {
            "metalforge": "0.0.1a1",
            "commit": "d86f0deaf33e8bea343b4d05856c596ae06301d7",
            "mlx": "0.32.3",
            "python": "3.12.16",
        },
        "conditions": {"power": ["ac", "ac"], "thermal_level": [0, 1]},
        "shape": {**bench.SHAPE.describe(), "attn_path": "fused"},
        "arms": {"off": off, "on": on},
        "on_off_ratio": 0.1218 / 0.0812,
        "seconds": 95.4,
        "utc": "2026-01-02T03:04:05Z",
    }


def write(folder: Path, row: dict[str, Any], text: str | None = None, name: str = "") -> Path:
    """Write `row` (or `text`) under its own file name (or `name`)."""
    folder.mkdir(exist_ok=True)
    path = folder / (name or bench.file_name(row["hardware"]["class"], row["versions"]["mlx"]))
    path.write_text(digest.pretty(row) if text is None else text, encoding="utf-8")
    return path


def test_the_rows_in_results_bench_and_their_page_check_out() -> None:
    # The page may lag, so that a pull request may add one row file and nothing else.
    assert records.check_bench(BENCH, may_lag=True) == []


def test_a_good_row_and_its_page_check_out(tmp_path) -> None:
    assert write(tmp_path, good_row()).name == NAME
    (tmp_path / "README.md").write_text(records.render_bench(tmp_path), encoding="utf-8")
    assert records.check_bench(tmp_path) == []


def edit(where: str, value: object) -> Callable[[dict[str, Any]], None]:
    """A change of the field at the dotted path `where`."""
    *parents, leaf = where.split(".")

    def apply(row: dict[str, Any]) -> None:
        for key in parents:
            row = row[key]
        row[leaf] = value

    return apply


def drop(where: str) -> Callable[[dict[str, Any]], None]:
    """The removal of the field at the dotted path `where`."""
    *parents, leaf = where.split(".")

    def apply(row: dict[str, Any]) -> None:
        for key in parents:
            row = row[key]
        del row[leaf]

    return apply


NOT_AC = "the power was not AC at the start and the end"
UNKNOWN_SHAPE = "shape is not a known version of bench.SHAPE"
NOT_UTC = "utc is not an ISO 8601 time in UTC"
STALE = (
    "README.md: not the page the rows give; write it with uv run python speedrun/records.py --write"
)
FAULTS = {
    "battery at the end": (edit("conditions.power", ["ac", "battery"]), NOT_AC),
    "battery at the start": (edit("conditions.power", ["battery", "ac"]), NOT_AC),
    "power unknown": (edit("conditions.power", ["ac", None]), NOT_AC),
    "other steps": (edit("shape.steps", bench.SHAPE.steps + 1), UNKNOWN_SHAPE),
    "another shape version": (edit("shape.version", bench.SHAPE.version + 1), UNKNOWN_SHAPE),
    "no attention path": (drop("shape.attn_path"), "shape.attn_path is not fused or banded"),
    "short commit": (edit("versions.commit", "d86f0dea"), "commit is not 40 hex or null"),
    "a name, not a handle": (edit("person", "Thomas N"), "person is not a GitHub handle"),
    "a handle of 40": (edit("person", "a" * 40), "person is not a GitHub handle"),
    "another schema": (
        edit("schema", "metalforge.bench-row/v2"),
        "schema is not metalforge.bench-row/v1",
    ),
    "local time": (edit("utc", "2026-01-02T05:04:05+02:00"), NOT_UTC),
    "no offset": (edit("utc", "2026-01-02T03:04:05"), NOT_UTC),
    "class not its key": (
        edit("hardware.class", "m4-max-40c-48gb-macos26-desktop"),
        "hardware.class is not m4-max-40c-48gb-macos26-laptop",
    ),
    "unknown form": (edit("hardware.form", "tablet"), "hardware.form is not laptop or desktop"),
    "thermal level 5": (
        edit("conditions.thermal_level", [0, 5]),
        "conditions.thermal_level is not [start, end], each 0-4 or null",
    ),
    "ratio off": (
        edit("on_off_ratio", 1.0),
        "on_off_ratio is not arms.on.step_s / arms.off.step_s",
    ),
    "zero step time": (edit("arms.off.step_s", 0.0), "a measurement is not positive"),
    "cores as text": (edit("hardware.gpu_cores", "40"), "hardware.gpu_cores is not int"),
    "cores as a bool": (edit("hardware.gpu_cores", True), "hardware.gpu_cores is not int"),
    "no rows updated": (drop("arms.on.rows_updated"), "arms.on has no rows_updated"),
    "an extra field": (edit("notes", "x"), "the row has an unknown field notes"),
}


@pytest.mark.parametrize("fault", list(FAULTS))
def test_a_row_with_one_fault_is_refused_for_it(tmp_path, fault) -> None:
    change, problem = FAULTS[fault]
    row = good_row()
    change(row)
    path = write(tmp_path, row)
    assert records.check_bench(tmp_path, page=False) == [f"{path.name}: {problem}"]


@pytest.mark.parametrize(
    ("text", "name", "problem"),
    [
        (None, "m4-max-40c-48gb-macos26-laptop-mlx0.32.4.json", f"the file name is not {NAME}"),
        (json.dumps(good_row()), "", "not in the pretty profile that bake bench --out writes"),
        ('{"schema": 1, "schema": 2}', NAME, "not strict JSON: duplicate key 'schema'"),
    ],
    ids=["wrong name", "not pretty", "duplicate key"],
)
def test_a_row_file_with_a_fault_of_its_own_is_refused(tmp_path, text, name, problem) -> None:
    path = write(tmp_path, good_row(), text, name)
    assert records.check_bench(tmp_path, page=False) == [f"{path.name}: {problem}"]


def test_a_stale_page_and_a_stray_file_are_refused(tmp_path) -> None:
    write(tmp_path, good_row())
    (tmp_path / "README.md").write_text("# Bench rows\n", encoding="utf-8")
    assert records.check_bench(tmp_path) == [STALE]
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert records.check_bench(tmp_path) == [
        "notes.txt: only rows (.json) and README.md belong here"
    ]


def three_rows() -> list[dict[str, Any]]:
    """Rows of classes made up from M4_MAX, for the order alone.

    By file name, 128gb would come before 48gb; by memory first, the M2 Ultra's 192 GB would
    come last.
    """
    rows = [good_row(), good_row(), good_row()]
    rows[1]["hardware"] = M4_MAX._replace(memory_gb=128).describe()
    rows[2]["hardware"] = M4_MAX._replace(
        chip="Apple M2 Ultra", gpu_cores=76, memory_gb=192, form="desktop"
    ).describe()
    return rows


def test_the_page_lists_the_rows_by_chip_then_memory(tmp_path) -> None:
    rows = three_rows()
    names = [write(tmp_path, row).name for row in rows]
    assert sorted(names) == [names[2], names[1], names[0]]
    page = records.render_bench(tmp_path)
    table = [line for line in page.splitlines() if line.startswith("| Apple")]
    assert [line.split(" | ")[:3] for line in table] == [
        ["| Apple M2 Ultra", "76", "192 GB"],
        ["| Apple M4 Max", "40", "48 GB"],
        ["| Apple M4 Max", "40", "128 GB"],
    ]
    assert table[1] == (
        "| Apple M4 Max | 40 | 48 GB | 26.6.2 | laptop | 0.32.3 | 81.2 | 121.8 | 1.500 | 0.0336 "
        "| 8.00 | [BhodiSea](https://github.com/BhodiSea) |"
    )
    assert "No rows yet." not in page


def lagging_page(folder: Path) -> str:
    """The page of the three rows' first two, with the third written after it."""
    first, second, third = three_rows()
    write(folder, first)
    write(folder, second)
    page = records.render_bench(folder)
    write(folder, third)
    return page


def test_a_page_that_lags_by_a_row_passes_only_when_it_may_lag(tmp_path) -> None:
    page = lagging_page(tmp_path)
    (tmp_path / "README.md").write_text(page, encoding="utf-8")
    assert records.check_bench(tmp_path) == [STALE]
    assert records.check_bench(tmp_path, may_lag=True) == []


def test_a_page_with_no_rows_lags_by_one(tmp_path) -> None:
    page = records.render_bench(tmp_path)
    write(tmp_path, good_row())
    (tmp_path / "README.md").write_text(page, encoding="utf-8")
    assert records.check_bench(tmp_path, may_lag=True) == []


def no_file(lines: list[str], folder: Path) -> list[str]:
    """A line for a row that has no file: the 48 GB row's, after its file is gone."""
    (folder / NAME).unlink()
    return lines


EDITS: dict[str, Callable[[list[str], Path], list[str]]] = {
    "an edited cell": lambda lines, _: [line.replace("| 81.2 |", "| 71.2 |") for line in lines],
    "a line for a row with no file": no_file,
    "rows out of order": lambda lines, _: [*lines[:-2], lines[-1], lines[-2]],
}


@pytest.mark.parametrize("change", list(EDITS))
def test_a_page_that_is_not_a_render_of_the_rows_it_lists_fails_even_when_it_may_lag(
    tmp_path, change
) -> None:
    lines = lagging_page(tmp_path).splitlines(keepends=True)
    (tmp_path / "README.md").write_text("".join(EDITS[change](lines, tmp_path)), encoding="utf-8")
    assert records.check_bench(tmp_path, may_lag=True) == [STALE]


def test_check_and_write_run_on_the_rows_folder(tmp_path, monkeypatch, capsys) -> None:
    rows, speedrun = tmp_path / "bench", tmp_path / "speedrun"
    speedrun.mkdir()
    (speedrun / "README.md").write_text(records.render(speedrun), encoding="utf-8")  # no claims
    monkeypatch.setattr(records, "SPEEDRUN", speedrun)
    monkeypatch.setattr(records, "BENCH", rows)
    write(rows, good_row())
    page = rows / "README.md"
    assert records.main(["--check"]) == 1  # no page yet
    assert capsys.readouterr().err == f"results/bench/{STALE}\n"
    assert records.main(["--write"]) == 0
    assert page.read_text(encoding="utf-8") == records.render_bench(rows)
    assert records.main(["--check"]) == 0
    written = page.read_text(encoding="utf-8")
    later = good_row()
    later["hardware"] = M4_MAX._replace(memory_gb=128).describe()
    path = write(rows, later)
    assert records.main(["--check"]) == 1  # the page lags by a row
    assert capsys.readouterr().err == f"results/bench/{STALE}\n"
    assert records.main(["--check", "--page-may-lag"]) == 0
    path.unlink()
    bad = good_row()
    bad["hardware"] = M4_MAX._replace(memory_gb=128).describe()
    edit("conditions.power", ["ac", "battery"])(bad)
    path = write(rows, bad)
    assert records.main(["--write"]) == 1  # refused, and the page is left as it was
    assert capsys.readouterr().err == f"results/bench/{path.name}: {NOT_AC}\n"
    assert page.read_text(encoding="utf-8") == written


def test_with_no_rows_the_page_says_so(tmp_path) -> None:
    assert records.render_bench(tmp_path).endswith("\nNo rows yet.\n")


def test_a_row_from_the_bench_passes_every_check_but_its_test_shape(tmp_path) -> None:
    row, _ = tiny_row()
    path = write(tmp_path, row)
    assert records.check_bench(tmp_path, page=False) == [f"{path.name}: {UNKNOWN_SHAPE}"]


def test_page_may_lag_goes_only_with_check(capsys) -> None:
    with pytest.raises(SystemExit) as stop:
        records.main(["--write", "--page-may-lag"])
    assert stop.value.code == 2
    assert "--page-may-lag goes with --check" in capsys.readouterr().err
