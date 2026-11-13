"""Each scorer makes the folders of its output files before any work.

A path that is a folder, or that lies under a file, is refused before the scorer opens anything,
so a bad path costs no scoring. Each scorer's entry-point test writes into folders that do not
exist yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from akeratos import export
from akeratos.eval import heldout, mc
from akeratos.eval.heldout import EXIT_CONFIG
from akeratos.post import sample

RUN = ["--ckpt", "run"]
DATA = [*RUN, "--data", "data"]
MC = [*RUN, "--set", "set", "--store", "store"]


@pytest.mark.parametrize(
    ("main", "argv", "flag"),
    [
        pytest.param(heldout.main, DATA, "--out", id="heldout"),
        pytest.param(mc.main, MC, "--out", id="mc"),
        pytest.param(mc.main, MC, "--samples", id="mc-samples"),
        pytest.param(sample.main, DATA, "--out", id="post-sample"),
    ],
)
def test_a_bad_output_path_exits_78_before_any_work(
    main: Callable[[Sequence[str]], int],
    argv: list[str],
    flag: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*_: object) -> None:
        raise AssertionError("work started before the output path was refused")

    monkeypatch.setattr(export, "checkpoint", never)  # every checkpoint scorer's first step
    a_file = tmp_path / "a-file"
    a_file.write_text("")
    for bad in (a_file / "out.json", tmp_path):
        assert main([*argv, flag, str(bad)]) == EXIT_CONFIG
        assert str(bad) in capsys.readouterr().err
