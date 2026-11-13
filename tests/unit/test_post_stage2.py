"""Stage 2 of post-training, the safety fork (`post.safety`).

Stage 2 runs from a stage-1 checkpoint only, each stage from the one before, and keeps the
table frozen as stage 1 does. The fixtures, the tiny table-on base and the SFT store format are
`test_post_sft.py`'s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from akeratos.model import EMBED
from akeratos.post import sft
from akeratos.post.run import SFTRun, marker_rows
from tests.unit.test_post_sft import (
    STEP_TOKENS,
    base_run,
    digests,
    newest,
    run_sft,
    sft_args,
    store,
)

mx.set_default_device(mx.cpu)


@pytest.fixture(autouse=True)
def tf32_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    with pytest.MonkeyPatch.context() as m:
        m.setenv("MLX_ENABLE_TF32", "0")
        return base_run(tmp_path_factory.mktemp("base") / "run")


@pytest.fixture(scope="module")
def stage1(base: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("stage1")
    data = store(tmp / "train")
    with pytest.MonkeyPatch.context() as m:  # module scope runs before the autouse fixture
        m.setenv("MLX_ENABLE_TF32", "0")
        assert run_sft(sft_args(base, data, tmp / "sft", 4 * STEP_TOKENS, "--rate", "1")) == 0
    return tmp / "sft"


# ---- stage 2 ----------------------------------------------------------------------------------


def test_stage_two_runs_from_stage_one_and_keeps_the_table(stage1: Path, tmp_path: Path) -> None:
    c = newest(stage1)
    before = digests(c.path / "table")
    data = store(tmp_path / "safety", seed=3)
    argv = sft_args(stage1, data, tmp_path / "s2", 4 * STEP_TOKENS, "--rate", "1", "--stage", "2")
    assert run_sft(argv) == 0
    done = newest(tmp_path / "s2")
    assert digests(done.path / "table") == before
    assert np.array_equal(
        np.load(done.path / "table" / "reads.npy"), np.load(c.path / "table" / "reads.npy")
    )
    post = done.state["config"]["post"]
    assert (post["stage"], post["base_step"], post["unmasked"]) == (2, c.step, [1, 2])
    assert done.state["config"]["model"]["masked_ids"] == list(range(3, 17))
    a, b = c.arrays("dense"), done.arrays("dense")
    assert any(not np.array_equal(np.array(a[n]), np.array(b[n])) for n in a)


def test_each_stage_starts_from_the_one_before(
    base: Path,
    stage1: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = store(tmp_path / "train")
    two_on_base = sft_args(
        base, data, tmp_path / "a", 2 * STEP_TOKENS, "--rate", "1", "--stage", "2"
    )
    assert run_sft(two_on_base) == sft.EXIT_CONFIG
    assert (
        "stage 2 starts from a stage-1 checkpoint; this is a pretraining" in capsys.readouterr().err
    )
    one_on_one = sft_args(stage1, data, tmp_path / "b", 2 * STEP_TOKENS, "--rate", "1")
    assert run_sft(one_on_one) == sft.EXIT_CONFIG
    assert "stage 1 starts from a pretraining checkpoint" in capsys.readouterr().err
    two = sft_args(stage1, data, tmp_path / "c", 2 * STEP_TOKENS, "--rate", "1", "--stage", "2")
    assert run_sft(two) == 0
    again = sft_args(
        tmp_path / "c", data, tmp_path / "d", 2 * STEP_TOKENS, "--rate", "1", "--stage", "2"
    )
    assert run_sft(again) == sft.EXIT_CONFIG
    assert "this is a stage-2 checkpoint" in capsys.readouterr().err


def test_only_stage_one_resets_the_marker_rows(
    stage1: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage 2 starts from the marker rows stage 1 trained, not from the table mean."""
    c = newest(stage1)
    trained = {n: np.array(w) for n, w in c.arrays("dense").items()}
    started: list[dict[str, np.ndarray]] = []
    init = SFTRun.__init__

    def spy(self: SFTRun, *args: Any, **kw: Any) -> None:
        init(self, *args, **kw)
        started.append({n: np.array(w) for n, w in self.stepper.params.items()})

    monkeypatch.setattr(SFTRun, "__init__", spy)
    data = store(tmp_path / "safety", seed=4)
    argv = sft_args(stage1, data, tmp_path / "s2", 2 * STEP_TOKENS, "--rate", "1", "--stage", "2")
    assert run_sft([*argv, "--steps", "1"]) == sft.EXIT_STOPPED
    for n, w in trained.items():
        assert np.array_equal(started[0][n], w), n
    assert np.array_equal(started[0][EMBED][1:3], trained[EMBED][1:3])
    assert not np.array_equal(marker_rows(c.arrays("dense"))[EMBED], trained[EMBED])
