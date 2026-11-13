"""`scripts/data/prepare_store.py` run from its own folder, and what `layout.py` gives it.

A probe loads a committed copy of `scripts/data` in a fresh `python -I -B` process, as the
script is started on the Mac: under `-I` the folder is on `sys.path` only because the script puts
it there. The copy has a folder beside it, `scripts/other`, because the script draws its line at
its own folder and not at the one above. No test here needs the dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from akeratos.data import packer
from tests.support.byfile import ROOT, load

DATA = ROOT / "scripts" / "data"
# Loads the script as the module `prepare_store`, imports what its prepare mode imports and one
# module from the folder beside the script's, and prints what the script then finds.
PROBE = """\
import importlib.util, json, sys

spec = importlib.util.spec_from_file_location("prepare_store", sys.argv[1])
script = importlib.util.module_from_spec(spec)
sys.modules["prepare_store"] = script
spec.loader.exec_module(script)
script.import_libs("prepare")
sys.path.insert(0, sys.argv[2])
import neighbour

try:
    script.check_imports()
    refused = None
except script.Stop as stop:
    refused = stop.detail
found = {
    "layout": script.store_fmt.__file__,
    "bpe": script.bpe.__file__,
    "refused": refused,
    "root": script.repo_root(),
    **script.script_commit(),
}
print(json.dumps(found))
"""


def git(repo: Path, *args: str) -> str:
    env = os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    who = ("-c", "user.name=test", "-c", "user.email=test@example.invalid")
    done = subprocess.run(
        ["git", "-C", str(repo), *who, *args], capture_output=True, text=True, check=True, env=env
    )
    return done.stdout.strip()


def python(*args: str | Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run `python -I -B` with `args`, started in `cwd`."""
    cmd = [sys.executable, "-I", "-B", *map(str, args)]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=False)


def probe(folder: str, clone: Path) -> dict[str, Any]:
    """What the clone's script finds when the probe in `scripts/<folder>` loads it, in a process
    started outside the clone."""
    scripts = clone / "scripts"
    script = scripts / "data" / "prepare_store.py"
    done = python(scripts / folder / "probe.py", script, scripts / "other", cwd=clone.parent)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def clone(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A git repository holding a copy of `scripts/data` and, beside it, `scripts/other`. Each
    folder holds the probe. `other` also holds the module the probe imports, which changes after
    the commit: the clone's one uncommitted change is outside the script's folder."""
    root = tmp_path_factory.mktemp("clone")
    data, other = root / "scripts" / "data", root / "scripts" / "other"
    shutil.copytree(DATA, data, ignore=shutil.ignore_patterns("__pycache__"))
    other.mkdir()
    for folder in (data, other):
        (folder / "probe.py").write_text(PROBE)
    (other / "neighbour.py").write_text("NAME = 'as committed'\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "The data scripts")
    (other / "neighbour.py").write_text("NAME = 'changed after the commit'\n")
    return root


@pytest.fixture(scope="module")
def found(clone: Path) -> dict[str, Any]:
    """What the script finds when the probe in the folder beside its own loads it."""
    return probe("other", clone)


def test_the_script_imports_its_siblings_from_its_own_folder(
    clone: Path, found: dict[str, Any]
) -> None:
    assert Path(found["layout"]).samefile(clone / "scripts" / "data" / "layout.py")
    assert Path(found["bpe"]).samefile(clone / "scripts" / "data" / "bpe.py")


def test_the_import_check_refuses_only_what_is_outside_the_folder(
    clone: Path, found: dict[str, Any]
) -> None:
    refused = found["refused"]
    assert refused is not None
    for outsider in ("'__main__'", "'neighbour'"):  # the probe in `scripts/other` and its module
        assert outsider in refused
    for own in ("'prepare_store'", "'layout'", "'bpe'"):
        assert own not in refused
    beside = probe("data", clone)["refused"]
    assert "'neighbour'" in beside
    assert "'__main__'" not in beside  # started from the folder, as the script is


def test_the_script_reports_its_clone_its_commit_and_a_change_in_its_folder(
    clone: Path, found: dict[str, Any]
) -> None:
    assert Path(found["root"]).samefile(clone)
    # Not dirty: the clone's one change so far is in `scripts/other`.
    assert (found["commit"], found["dirty"]) == (git(clone, "rev-parse", "HEAD"), False)
    tracked, new = (clone / "scripts" / "data" / name for name in ("requirements.txt", "new.py"))
    before = tracked.read_text()
    tracked.write_text(before + "# a change inside the folder\n")
    edited = probe("other", clone)["dirty"]
    tracked.write_text(before)
    new.write_text("")
    added = probe("other", clone)["dirty"]
    new.unlink()
    assert (edited, added) == (True, True)


@pytest.fixture(scope="module")
def download(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same copy as `clone`, with a pyproject.toml above `scripts` and no .git anywhere
    above it: the repository as an archive download holds it."""
    root = tmp_path_factory.mktemp("download")
    assert not any((d / ".git").exists() for d in (root, *root.parents))
    data, other = root / "scripts" / "data", root / "scripts" / "other"
    shutil.copytree(DATA, data, ignore=shutil.ignore_patterns("__pycache__"))
    other.mkdir()
    for folder in (data, other):
        (folder / "probe.py").write_text(PROBE)
    (other / "neighbour.py").write_text("NAME = 'as downloaded'\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'download'\n")
    return root


def prepare(root: Path, out: Path) -> subprocess.CompletedProcess[str]:
    """The copy's script in prepare mode, which reaches the location check without the Hub."""
    script = root / "scripts" / "data" / "prepare_store.py"
    args = ("prepare", "--sample", "sample/10BT", "--cap-tokens", "1000", "--allow-env-drift")
    return python(script, *args, "--out", out, cwd=root.parent)


def test_without_git_the_root_is_the_folder_holding_pyproject_toml(download: Path) -> None:
    assert Path(probe("other", download)["root"]).samefile(download)


def test_without_git_an_output_root_inside_the_copy_is_refused(
    download: Path, tmp_path: Path
) -> None:
    inside = prepare(download, download / "mf-data")
    assert inside.returncode == 73, inside.stdout + inside.stderr
    assert "stopped: inside-clone (exit 73)" in inside.stdout + inside.stderr
    outside = prepare(download, tmp_path / "mf-data")
    assert "inside-clone" not in outside.stdout + outside.stderr
    assert outside.returncode != 0  # stopped later: nothing was fetched


def test_help_names_the_script(tmp_path: Path) -> None:
    done = python(DATA / "prepare_store.py", "--help", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("usage: prepare_store.py ")


@pytest.fixture(scope="module")
def layout() -> ModuleType:
    return load("scripts/data/layout.py", "scripts_data_layout")


def test_layout_gives_the_ids_the_package_packs_with(layout: ModuleType) -> None:
    assert (layout.SEP_ID, layout.RESERVED) == (packer.SEP_ID, packer.RESERVED)
    assert layout.FIRST_ID == layout.RESERVED[1] + 1 == 17
    assert layout.FORMAT == "metalforge-a21-standin.split/v0"
    assert layout.content_ids([17, 99], 100).tolist() == [17, 99]
    for ids in ([16], [100]):
        with pytest.raises(ValueError, match="outside"):
            layout.content_ids(ids, 100)
    with pytest.raises(TypeError, match="integers only"):
        layout.content_ids([17.0], 100)


def test_layout_checks_a_shard_s_offsets_and_cuts_the_shards(layout: ModuleType) -> None:
    layout.check_idx(np.array([0, 3, 5], dtype=layout.U64), 5, "a shard")
    for idx in ([0, 3, 3, 5], [1, 3, 5], [0, 3, 4]):  # repeated; not from 0; not up to the tokens
        with pytest.raises(ValueError, match="a shard: offsets"):
            layout.check_idx(np.array(idx, dtype=layout.U64), 5, "a shard")
    assert layout.close_points([3, 3, 3], 5) == [0, 1, 2, 3]
    assert layout.close_points([2, 2, 9, 1], 5) == [0, 2, 3, 4]  # 9 is longer than a shard
