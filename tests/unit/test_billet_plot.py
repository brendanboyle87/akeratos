"""The 100-step means that `results/billet/plot.py` draws its figure from.

A finite log gives one mean per window, non-finite losses are left out of their window, and a
window with no finite loss has no point. The script is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "results" / "billet" / "plot.py"
NAN, INF = float("nan"), float("inf")


def _stand_in(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    vars(module).update(attrs)
    return module


@pytest.fixture(scope="module")
def plot() -> ModuleType:
    """The script, loaded with stand-ins for matplotlib, which the unit suite does not install."""
    spec = importlib.util.spec_from_file_location("billet_plot", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "matplotlib", _stand_in("matplotlib", use=lambda backend: None))
        patch.setitem(sys.modules, "matplotlib.pyplot", _stand_in("matplotlib.pyplot"))
        spec.loader.exec_module(module)
    return module


def _log(path: Path, losses: list[float]) -> Path:
    lines = (json.dumps({"step": step, "loss": loss}) for step, loss in enumerate(losses))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_finite_log_gives_one_mean_per_window(plot: ModuleType, tmp_path: Path) -> None:
    losses = [5.0 - step / 1000 for step in range(1000)]
    steps, means = plot.windowed_loss(_log(tmp_path / "log.jsonl", losses))
    np.testing.assert_allclose(steps, [49.5 + 100 * window for window in range(10)])
    np.testing.assert_allclose(means, [np.mean(losses[w * 100 : (w + 1) * 100]) for w in range(10)])


def test_non_finite_losses_are_left_out_of_their_window(plot: ModuleType, tmp_path: Path) -> None:
    losses = [float(step % 7) for step in range(300)]
    for step, bad in ((3, NAN), (150, INF), (199, -INF)):
        losses[step] = bad
    steps, means = plot.windowed_loss(_log(tmp_path / "log.jsonl", losses))
    kept = [
        [(s, v) for s, v in enumerate(losses) if s // 100 == w and np.isfinite(v)] for w in range(3)
    ]
    assert [len(window) for window in kept] == [99, 98, 100]
    np.testing.assert_allclose(steps, [np.mean([s for s, _ in window]) for window in kept])
    np.testing.assert_allclose(means, [np.mean([v for _, v in window]) for window in kept])


def test_a_window_with_no_finite_loss_has_no_point(plot: ModuleType, tmp_path: Path) -> None:
    losses = [1.0] * 100 + [NAN] * 100 + [3.0] * 100
    steps, means = plot.windowed_loss(_log(tmp_path / "log.jsonl", losses))
    np.testing.assert_allclose(steps, [49.5, 249.5])
    np.testing.assert_allclose(means, [1.0, 3.0])
