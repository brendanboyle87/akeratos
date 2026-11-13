"""Test setup: an offline, single-threaded environment and the fixtures the suites share."""

import os
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.proc import HF_CACHE_VARS, OFFLINE_ENV, THREAD_ENV, Children

ROOT = Path(__file__).resolve().parent.parent

# Set here, before any test module is imported: huggingface_hub reads its variables at
# import. The temporary home keeps the tests away from your own Hugging Face cache.
_HF_HOME = tempfile.TemporaryDirectory(prefix="akeratos-tests-")
for _name in HF_CACHE_VARS:
    os.environ.pop(_name, None)
os.environ.update(OFFLINE_ENV | THREAD_ENV | {"HF_HOME": _HF_HOME.name})


def pytest_unconfigure() -> None:
    _HF_HOME.cleanup()


@pytest.fixture
def children(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Children]:
    """Start child processes; their process groups are killed when the test ends."""
    kids = Children(tmp_path_factory.getbasetemp())
    yield kids
    kids.close()


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="session")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the wheel once, keyed by distribution name."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    out = tmp_path_factory.mktemp("dist")
    # Without the outer run's virtual environment or uv's frozen and offline switches.
    dropped = {"VIRTUAL_ENV", "UV_FROZEN", "UV_OFFLINE", "UV_NATIVE_TLS"}
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    proc = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(out)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    built = {whl.name.split("-")[0].replace("_", "-"): whl for whl in out.glob("*.whl")}
    assert set(built) == {"akeratos"}, built
    return built
