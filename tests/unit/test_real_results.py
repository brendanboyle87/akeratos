"""This repository's own results/speedrun/ and results/bench/ check out, pages included.

`speedrun/records.py --check --page-may-lag`, loaded by path, runs on the real folders: every
claim and bench row keeps its rules, and each page equals a fresh render, except that the bench
page may leave out rows it does not list yet. So a pull request may add one bench row file
alone, and `--write` brings the page up to date later. The check asks git whether a receipt's
commit is on `origin/main`. Only a clone with its whole history can answer that, so in a shallow
clone or a downloaded archive this test takes each commit as on main and checks the rest. CI
clones the whole history and checks the commits too.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from akeratos.macops import run_command
from tests.support.byfile import load

records = load("speedrun/records.py", "speedrun_records_real")


def _knows_main(repo: Path) -> bool:
    """Whether git can say what is on `origin/main`: `repo` is a clone with its whole history."""

    def git(*args: str) -> str | None:
        out = run_command(("git", "-C", str(repo), *args))
        return None if out is None else out.strip()

    top = git("rev-parse", "--show-toplevel")
    return (
        top is not None
        and Path(top).resolve() == repo.resolve()
        and git("rev-parse", "--is-shallow-repository") == "false"
        and git("rev-parse", "--verify", "--quiet", "origin/main^{commit}") is not None
    )


def test_the_real_results_and_both_pages_check_out(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _knows_main(records.REPO) and "CI" not in os.environ:
        monkeypatch.setattr(records, "_on_main", lambda repo, main, commit: True)
    assert records.main(["--check", "--page-may-lag"]) == 0
    assert capsys.readouterr().err == ""
