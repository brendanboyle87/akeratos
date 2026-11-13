"""speedrun/train.py on CPU. A tiny copy of the file trains end to end through `bake speedrun`'s
driver, and `speedrun/records.py` refuses its receipt as a record only for the test's own values.
The real file builds, and its learning rate follows its warm-up, hold and decay. The rest check
that it is Akeratos's model: its starting weights, its logits and three of its steps equal the
package's bit for bit.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from akeratos import digest, speedrun
from akeratos.configs import ModelConfig, from_keys
from akeratos.data.packer import segment_ids
from akeratos.model import forward, init_params
from akeratos.rng import run_seeds
from akeratos.train.loop import Stepper, make_loss_fn, prepare
from akeratos.train.optim import DenseOptim, groups_for
from akeratos.train.recipe import from_keys as recipe_from_keys
from akeratos.train.source import StepRows
from tests.fakes.clock import TickingClock
from tests.support.byfile import load
from tests.support.speedrun_cut import cut, fixture_commits_on_main, options

mx.set_default_device(mx.cpu)

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "speedrun" / "train.py"
train = load("speedrun/train.py", "speedrun_train_file")  # the real file, loaded by path
records = load("speedrun/records.py", "speedrun_records_smoke")
# The smoke run's copy: one row of 64 tokens, and a model small enough for any CPU in seconds.
TINY_MODEL = (
    '{"d_model": 64, "n_layers": 2, "n_heads": 2, "head_dim": 32, "n_kv_heads": 1, "ffn": 128}'
)
COPY = {"ROWS": "1", "SEQ_LEN": "64", "MODEL": TINY_MODEL, "WINDOW": "16"}


@pytest.fixture(autouse=True)
def _commits_on_main(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_commits_on_main(records, monkeypatch)


def test_a_tiny_copy_of_the_real_file_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    text = TRAIN.read_text(encoding="utf-8")
    for name, value in COPY.items():
        text, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", text, flags=re.MULTILINE)
        assert n == 1, name
    (tmp_path / "train.py").write_text(text, encoding="utf-8")
    c = cut(tmp_path / "cut", 64, 200)
    o = options(c, TickingClock(), tmp_path, 64, 200, budget_s=3.0)
    line = speedrun.run(o)
    folder = tmp_path / "out" / "20260101T000000Z-seed-0"
    r = json.loads((folder / "receipt.json").read_text())
    log = [json.loads(x) for x in (folder / "metrics.jsonl").read_text().splitlines()]
    # Each step takes 1 s on the ticking clock, so the first timed step brings the timed total
    # to 1 s plus the 10 warm-up steps at the median, 1 s each.
    assert (r["steps"], r["timed_s"], len(log)) == (11, 11.0, 11)
    assert (r["batch"], r["tokens"]) == ({"rows": 1, "seq_len": 64}, 11 * 64)
    assert r["train_py"] == {"sha256": digest.digest_bytes(text.encode()), "source": text}
    assert r["practice"] is True
    assert r["heldout"] == c.heldout | {"seq_len": 64, "doc_mask": False}
    assert all(math.isfinite(x["loss"]) for x in log)
    assert math.isfinite(r["bpb"])
    assert line == f"11 steps, 11.000 s timed, {r['bpb']:.4f} bits per byte: {folder}/receipt.json"
    # The checker accepts the receipt's format and arithmetic, and refuses it as a record for
    # the test's slices, evaluation values and budget, which make it a practice run.
    run = f"{r['hardware']['class']}/{r['claim']}/seed-0"
    shutil.copytree(folder, tmp_path / "results" / run)
    assert records.check(tmp_path / "results", page=False) == [
        f"{run}/receipt.json: {problem}"
        for problem in (
            "data is not the track's slices",
            "heldout is not the track's scored set",
            "rules are not the track's",
            "practice is not false: a practice run is never a record",
        )
    ]


@pytest.fixture(scope="module")
def reference() -> Any:
    """The real file's trainer for seed 0, built on CPU and not trained."""
    return train.build(0)


def test_the_real_file_builds_on_cpu(reference: Any) -> None:
    assert reference.fused is False  # the fused kernels run only on the GPU: the banded path
    # The recipe's rule: λ_ind = 1e-3 x 0.1 x (1e9 / tokens) x (tokens per update / 8,192).
    want = 1e-4 * (1e9 / train.TOKENS) * (train.ROWS * train.SEQ_LEN / 8192)
    assert reference.decay["hidden"] == pytest.approx(want)


@pytest.mark.parametrize(
    ("step", "progress", "want"),
    [
        (0, 0.0, 0.01),
        (49, 0.5, 0.5),
        (99, 0.8, 1.0),
        (100, 0.9, 1 - math.sqrt(0.5)),
        (101, 0.99, 1 - math.sqrt(0.95)),
    ],
)
def test_the_reference_schedule(step: int, progress: float, want: float) -> None:
    """A warm-up over 100 steps, a hold, then 1 - √u to 0 over the last 20% of the budget."""
    assert train.rate(step, progress) == pytest.approx(want, abs=1e-12)


def same_bits(a: mx.array, b: mx.array) -> bool:
    """Whether two arrays have one dtype and shape and the same bytes."""
    if (a.dtype, tuple(a.shape)) != (b.dtype, tuple(b.shape)):
        return False
    return bool(np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)))


def package_config(mod: Any) -> ModelConfig:
    """The package's configuration of a train file's model, on the banded path."""
    return from_keys(mod.MODEL | {"window": mod.WINDOW, "table": False})


@pytest.fixture(scope="module", params=[0, 7], ids=["seed-0", "seed-7"])
def seeded(request: pytest.FixtureRequest) -> tuple[int, Any]:
    """A seed and the real file's trainer for it, built on CPU and not trained."""
    return request.param, train.build(request.param)


def test_the_file_starts_from_the_packages_weights(seeded: tuple[int, Any]) -> None:
    seed, ref = seeded
    want = init_params(package_config(train), run_seeds(seed)["init_seed"])
    assert [(n, a.shape) for n, a in ref.params.items()] == [(n, a.shape) for n, a in want.items()]
    assert [n for n, a in want.items() if not same_bits(ref.params[n], a)] == []


# Each case runs the real file's model twice on a row of 1,024 tokens. A CI runner runs two xdist
# workers on two vCPUs: there a case took 32-56 s on a fast runner and 86-112 s on a slow one, so
# it went over the global 60 s. Its own ceiling covers the slow runner; a hang still fails.
REAL_MODEL_TIMEOUT_S = 300


@pytest.mark.timeout(REAL_MODEL_TIMEOUT_S)
def test_the_file_scores_as_the_package_does(seeded: tuple[int, Any]) -> None:
    seed, ref = seeded
    ids = np.random.default_rng(seed).integers(17, train.VOCAB, size=(1, 1024))
    ids[0, [5, 300, 301, 777]] = 0  # separators: four documents start after position 0
    x = ids.astype(np.int32)
    cfg = package_config(train)
    want = forward(ref.params, cfg, mx.array(x), mx.array(segment_ids(x)), mx.float32)
    assert cfg.attn_path == "banded"
    assert same_bits(ref.logits(ids), want)


def small_copy(folder: Path) -> Any:
    """The smoke run's copy of the real file, written to `folder` and loaded from there."""
    text = TRAIN.read_text(encoding="utf-8")
    for name, value in COPY.items():
        text, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", text, flags=re.MULTILINE)
        assert n == 1, name
    path = folder / "train.py"
    path.write_text(text, encoding="utf-8")
    return load(str(path), "speedrun_train_small_copy")


def batches(n: int, rows: int, seq_len: int) -> list[StepRows]:
    """`n` batches of packed rows from a seeded generator, with separators and a masked target."""
    rng = np.random.default_rng(3)
    out: list[StepRows] = []
    for _ in range(n):
        toks = rng.integers(17, train.VOCAB, size=(rows, seq_len + 1)).astype(np.int32)
        toks[:, [9, 40]] = 0  # two separators
        toks[:, 20] = 5  # a reserved id: as a target it has weight 0
        ids = np.ascontiguousarray(toks[:, :-1])
        out.append(StepRows(ids, np.ascontiguousarray(toks[:, 1:]), segment_ids(ids)))
    return out


def test_three_steps_match_the_package(tmp_path: Path) -> None:
    small = small_copy(tmp_path)
    mine = small.build(0)
    cfg = package_config(small)
    recipe = {"tokens.budget": small.TOKENS, "batch.tokens_per_update": small.ROWS * small.SEQ_LEN}
    o, r = recipe_from_keys(recipe | {"param.kind": cfg.param_kind})
    params = init_params(cfg, run_seeds(0)["init_seed"])
    opt = DenseOptim(params, groups_for(params), cfg.d_model, o, r)
    stepper = Stepper(make_loss_fn(cfg, mx.bfloat16), params, opt, o.clip)
    seen, want = [], []
    for k, (rows, progress) in enumerate(
        zip(batches(3, small.ROWS, small.SEQ_LEN), [0.0, 0.5, 0.9], strict=True)
    ):
        seen.append(mine.step(rows, progress))
        want.append(stepper.step(*prepare(rows, cfg), s=small.rate(k, progress)).loss)
    assert seen == want
    assert list(mine.params) == list(stepper.params)
    assert [n for n, a in stepper.params.items() if not same_bits(mine.params[n], a)] == []
