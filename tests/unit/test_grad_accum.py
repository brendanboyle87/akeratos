"""Dense gradient accumulation in the training loop and the stepper.

K micro-batches of B rows summed into one optimiser step must equal one step on the K·B rows
stacked, because every micro-batch's loss is divided by the whole step's N_valid. One
`Run.step` stays one `Stepper.step`, so `step_no` counts optimiser updates, and `on_step` runs
only after the update, where the checkpoint saves hook in.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from akeratos.configs import ModelConfig
from akeratos.errors import ConfigError
from akeratos.train import entry
from akeratos.train.loop import N_VALID, Run, prepare
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.saver import Checkpointer
from akeratos.train.source import StepRows, SynthSource
from tests.fakes.clock import FakeClock

mx.set_default_device(mx.cpu)

V, B, T, K = 512, 2, 32, 3
TINY = ModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=2,
    head_dim=32,
    n_kv_heads=1,
    ffn=128,
    vocab_size=V,
    window=16,
    table=False,
    param_kind="sp",
)


class Listed:
    """Serves the given batches in order."""

    def __init__(self, batches):
        self.batches, self.i = list(batches), 0

    def next(self):
        self.i += 1
        return self.batches[self.i - 1]

    def state(self):
        return {"kind": "listed", "i": self.i}


def stack(batches):
    def cat(name):
        return np.concatenate([getattr(b, name) for b in batches])

    lm = None
    if any(b.lmask is not None for b in batches):
        lm = np.concatenate(
            [np.ones(b.ids.shape, np.float32) if b.lmask is None else b.lmask for b in batches]
        )
    return StepRows(cat("ids"), cat("targets"), cat("segment_ids"), lm)


def make(tmp_path, name, source, accum, compile, steps=4):
    tpu = K * B * T
    o = OptimConfig(lr_peak=3e-3, tokens=steps * tpu, tokens_per_update=tpu, decay_ind=0.0)
    r = RecipeConfig(warmup=1, param_kind="sp")
    kw = {"compute_dtype": "float32", "compile": compile, "accum": accum, "clock": FakeClock()}
    return Run(TINY, o, r, source, tmp_path / name, **kw)


def micro_batches(n, seed=1, mask_from=None):
    src = SynthSource(B, T, V, seed=seed)
    out = []
    for i in range(n):
        b = src.next()
        if mask_from is not None and i % K == 0:
            # Uneven N_valid across micro-batches: mask out the second half of each row.
            lm = np.ones(b.ids.shape, np.float32)
            lm[:, mask_from:] = 0.0
            b = StepRows(b.ids, b.targets, b.segment_ids, lm)
        out.append(b)
    return out


@pytest.mark.parametrize("compile", [False, True])
@pytest.mark.parametrize("mask_from", [None, T // 2])
def test_accumulated_steps_equal_one_step_on_the_stacked_rows(tmp_path, compile, mask_from):
    micro = micro_batches(2 * K, mask_from=mask_from)
    big = [stack(micro[i : i + K]) for i in range(0, len(micro), K)]
    a = make(tmp_path, "a", Listed(micro), K, compile)
    b = make(tmp_path, "b", Listed(big), 1, compile)
    for _ in range(2):
        la, lb = a.step(), b.step()
        assert la["loss"] == pytest.approx(lb["loss"], rel=1e-5)
        for g, v in la["grad_norm"].items():
            assert v == pytest.approx(lb["grad_norm"][g], rel=1e-4)
        assert la["batch"] == lb["batch"]
    assert a.step_no == b.step_no == 2
    # Adam divides by sqrt(v), so float32 summation-order noise in near-zero gradients reaches
    # the weights at a few per cent of one update (lr 3e-3); the gradients are checked tighter.
    for k, v in a.stepper.params.items():
        np.testing.assert_allclose(np.array(v), np.array(b.stepper.params[k]), atol=1e-4)


@pytest.mark.parametrize("compile", [False, True])
def test_summed_gradients_equal_the_stacked_batch_s(tmp_path, compile):
    micro = micro_batches(K, mask_from=T // 2)
    a = make(tmp_path, "a", Listed([]), K, compile)
    args = [prepare(m, TINY) for m in micro]
    n_valid = sum((x[N_VALID] for x in args), mx.array(0.0, mx.float32))
    args = [(*x[:N_VALID], n_valid, *x[N_VALID + 1 :]) for x in args]
    loss, grads, sq, finite = a.stepper.accumulated_grads(args)
    loss1, grads1, sq1, _ = a.stepper.grads(*prepare(stack(micro), TINY))
    assert bool(finite)
    assert float(loss) == pytest.approx(float(loss1), rel=1e-6)
    np.testing.assert_allclose(np.array(sq), np.array(sq1), rtol=1e-5)
    for k, g in grads.items():
        np.testing.assert_allclose(np.array(g), np.array(grads1[k]), rtol=1e-4, atol=1e-8)


def test_on_step_runs_once_per_update(tmp_path):
    a = make(tmp_path, "a", Listed(micro_batches(4 * K)), K, False)
    seen = []
    a.train(
        on_step=lambda run, rec: seen.append((run.step_no, rec["step"], run.source.state()["i"]))
    )
    assert seen == [(i + 1, i, (i + 1) * K) for i in range(4)]
    lines = (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(x)["tokens"] for x in lines] == [(i + 1) * K * B * T for i in range(4)]
    assert json.loads((tmp_path / "a" / "run.json").read_text())["config"]["accum"] == K


def test_a_non_finite_micro_batch_skips_the_whole_step(tmp_path):
    poisoned = SynthSource(B, T, V, seed=1, poison={1})
    a = make(tmp_path, "a", poisoned, K, False)
    rec = a.step()
    assert rec["skipped"] is False  # one all-zero mask among K: N_valid is still the step's
    allzero = SynthSource(B, T, V, seed=1, poison=set(range(K)))
    b = make(tmp_path, "b", allzero, K, False)
    assert b.step()["skipped"] is True
    assert b.step_no == 1


def test_accumulation_below_one_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="at least 1"):
        make(tmp_path, "a", Listed([]), 0, False)


def test_a_step_short_of_its_tokens_is_refused(tmp_path):
    a = make(tmp_path, "a", Listed(micro_batches(K)), K - 1, False)
    with pytest.raises(ConfigError, match="tokens_per_update"):
        a.step()


# ---- the entry point -----------------------------------------------------------------------

TINY_KEYS = [
    "table=false",
    "d_model=64",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    f"vocab_size={V}",
    "window=16",
    "param.kind=sp",
]


@pytest.fixture
def tf32_set(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


@pytest.mark.usefixtures("tf32_set")
def test_tokens_and_steps_count_optimiser_steps_under_accum(tmp_path, capsys):
    # The sweep's form: --tokens sets the schedule's horizon, --steps stops it early.
    args = ["--data", "synthetic", "--rows", str(B), "--seq-len", str(T), "--accum", str(K)]
    args += ["--tokens", str(10 * K * B * T), "--steps", "3", "--out", str(tmp_path)]
    args += ["--compute-dtype", "float32", *[f"--model={k}" for k in TINY_KEYS]]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    assert entry.main(args, Run, Checkpointer) == entry.EXIT_STOPPED
    man = json.loads((tmp_path / "run.json").read_text())
    assert man["schedule"]["steps"] == 10
    assert man["config"]["optim"]["tokens_per_update"] == K * B * T
    assert man["source"]["i"] == 0  # the manifest is written before the first step
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [0, 1, 2]
    assert lines[-1]["tokens"] == 3 * K * B * T
    assert "stopped at step 3 of 10" in capsys.readouterr().err
