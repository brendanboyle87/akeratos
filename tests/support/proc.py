"""Child processes for tests: an offline environment, and no process left behind."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The Hugging Face libraries read these once, at import. No test downloads anything.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_MODE": "disabled",
}
# One thread per library and a fixed hash seed, so a child's run repeats.
THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "0",
}
# These follow HF_HOME once they are unset.
HF_CACHE_VARS = ("HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE")


def child_env(base: Mapping[str, str], basetemp: Path) -> dict[str, str]:
    """`base` made offline and single-threaded, with a Hugging Face home under `basetemp`."""
    env = dict(base)
    for name in HF_CACHE_VARS:
        env.pop(name, None)
    env.update(OFFLINE_ENV)
    env.update(THREAD_ENV)
    env["HF_HOME"] = str(Path(basetemp) / "hf-home")
    return env


def kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


class Children:
    """Starts children for one test and kills their process groups when it ends."""

    def __init__(self, basetemp: Path) -> None:
        self.basetemp = Path(basetemp)
        self._pids: list[int] = []

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        env = child_env(os.environ, self.basetemp)
        env.update(extra or {})
        return env

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        **popen: Any,
    ) -> subprocess.Popen[Any]:
        # A new session, so a test stopped by its timeout still takes its children with it.
        proc = subprocess.Popen(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            env=self.env() if env is None else dict(env),
            start_new_session=True,
            **popen,
        )
        self._pids.append(proc.pid)
        return proc

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        proc = self.spawn(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = proc.communicate(input=input, timeout=timeout)
        finally:
            kill_group(proc.pid)
            code = proc.wait()
        return subprocess.CompletedProcess(list(argv), code, out, err)

    def close(self) -> None:
        for pid in self._pids:
            kill_group(pid)
        self._pids.clear()
