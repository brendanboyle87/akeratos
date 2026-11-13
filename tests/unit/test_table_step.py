"""The table in the training step: the training loop's wiring of the n-gram table.

The step hashes and gathers each micro-batch's working set, returns g_w beside the dense
gradients, sums it over the micro-batches and runs one replay event after the dense update. A
non-finite g_w skips the step as a non-finite loss does, and a skipped step moves no row, no
moment and no stamp. The checkpoint carries the table under
`table/`, and a killed table-on run resumes bitwise. Everything runs at tiny sizes on MLX's CPU
device: 4,096 rows of width 64 in 4 shards.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from akeratos.configs import ModelConfig
from akeratos.errors import ConfigError
from akeratos.table.config import TableConfig
from akeratos.table.table import Table
from akeratos.train.loop import N_VALID, Run, Stepper, prepare
from akeratos.train.optim import DenseOptim
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.source import StepRows, SynthSource
from tests.fakes.clock import FakeClock

mx.set_default_device(mx.cpu)

V, D, B, T, K = 512, 64, 2, 32, 3
PARAMS, SHARD = 4096 * D, 1024 * D  # 4,096 rows of width d in 4 shards
TINY = ModelConfig(
    d_model=D,
    n_layers=2,
    n_heads=2,
    head_dim=32,
    n_kv_heads=1,
    ffn=128,
    vocab_size=V,
    window=16,
    table=True,
    param_kind="sp",
)


def table(fmt="F2", seed=0):
    cfg = TableConfig(params=PARAMS, width=D, fmt=fmt, seed=seed, max_shard_elements=SHARD)
    return Table(cfg, mx.get_active_memory)


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

    return StepRows(cat("ids"), cat("targets"), cat("segment_ids"), None)


def make(tmp_path, name, source, accum, *, tab=None, steps=4, lr=3e-3, compile=False):
    tpu = accum * B * T
    o = OptimConfig(lr_peak=lr, tokens=steps * tpu, tokens_per_update=tpu, decay_ind=0.0)
    r = RecipeConfig(warmup=1, param_kind="sp")
    kw = {"compute_dtype": "float32", "compile": compile, "accum": accum, "clock": FakeClock()}
    return Run(TINY, o, r, source, tmp_path / name, table=tab or table(), **kw)


def bits(t: Table) -> list[dict[str, bytes]]:
    out = []
    for s in range(len(t.store)):
        names = t.store.fields(s)
        got = t.store.take(s, names, mx.arange(t.shard_rows(s)))
        out.append({k: np.array(v.view(mx.uint8)).tobytes() for k, v in got.items()})
    return out


def weights(t: Table) -> np.ndarray:
    parts = []
    for s in range(len(t.store)):
        got = t.store.take(s, ("rows",), mx.arange(t.shard_rows(s)))
        parts.append(np.array(got["rows"].astype(mx.float32)))
    return np.concatenate(parts)


# ---- the step ------------------------------------------------------------------------------


@pytest.mark.parametrize("compile", [False, True])
def test_a_table_on_run_learns_and_logs_the_table(tmp_path, compile):
    run = make(tmp_path, "a", SynthSource(B, T, V, seed=1), 1, steps=40, lr=1e-2, compile=compile)
    lines = [run.step() for _ in range(40)]
    losses = [x["loss"] for x in lines]
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.3, losses
    assert run.table is not None
    assert run.table.event == 40
    assert np.abs(weights(run.table)).max() > 0
    tl = lines[-1]["table"]
    assert set(tl) == {"rate", "decay", "event", "rows", "rho"}
    assert tl["event"] == 40
    assert 0 < tl["rows"] <= B * T * 2  # k = 2 rows per token
    assert tl["rho"] > 0
    assert lines[-1]["grad_norm"]["table"] > 0  # the summed g_w, outside the clip
    cfg, peak = run.table.cfg, run.stepper.opt.peak["embed"]
    assert tl["rate"] == pytest.approx(cfg.floor * cfg.lr_mult * peak)  # the cool-down, floored
    logged = json.loads((tmp_path / "a" / "metrics.jsonl").read_text().splitlines()[-1])
    assert logged["table"]["event"] == 40


def test_the_working_set_is_an_argument_not_a_parameter(tmp_path):
    run = make(tmp_path, "a", SynthSource(B, T, V, seed=1), 1)
    names = set(run.stepper.params)
    run.step()
    assert set(run.stepper.params) == names
    assert not any("table." in n and n.endswith(".w") for n in names)
    assert run.config()["table"]["params"] == PARAMS


@pytest.mark.parametrize("compile", [False, True])
def test_k_accumulated_micro_batches_equal_one_k_times_batch(tmp_path, compile):
    src = SynthSource(B, T, V, seed=3)
    micro = [src.next() for _ in range(2 * K)]
    big = [stack(micro[i : i + K]) for i in range(0, len(micro), K)]
    tpu = K * B * T
    o = OptimConfig(lr_peak=3e-3, tokens=2 * tpu, tokens_per_update=tpu, decay_ind=0.0)
    r = RecipeConfig(warmup=1, param_kind="sp")
    kw = {"compute_dtype": "float32", "compile": compile, "clock": FakeClock()}
    a = Run(TINY, o, r, Listed(micro), tmp_path / "a", accum=K, table=table("F32"), **kw)
    b = Run(TINY, o, r, Listed(big), tmp_path / "b", accum=1, table=table("F32"), **kw)
    for _ in range(2):
        la, lb = a.step(), b.step()
        assert la["loss"] == pytest.approx(lb["loss"], rel=1e-5)
        for g, v in la["grad_norm"].items():
            assert v == pytest.approx(lb["grad_norm"][g], rel=1e-4)
        assert la["table"]["rows"] == lb["table"]["rows"]
    assert a.table is not None
    assert b.table is not None
    assert a.table.event == b.table.event == 2
    wa, wb = weights(a.table), weights(b.table)
    assert np.abs(wa).max() > 0
    np.testing.assert_allclose(wa, wb, atol=1e-4)


def test_summed_working_set_gradients_equal_the_stacked_batch_s(tmp_path):
    src = SynthSource(B, T, V, seed=4)
    micro = [src.next() for _ in range(K)]
    t = table("F32")
    run = make(tmp_path, "a", Listed([]), K, tab=t)
    acc = run.acc
    assert acc is not None
    args = [prepare(m, TINY) for m in micro]
    n_valid = sum((x[N_VALID] for x in args), mx.array(0.0, mx.float32))
    args = [(*x[:N_VALID], n_valid, *x[N_VALID + 1 :]) for x in args]
    live = {}

    def build(i):
        def batch():
            live[i] = ws = t.gather(t.keys(micro[i].ids))
            return (ws.W, ws.idx, *args[i])

        return batch

    run.stepper.table_grads([build(i) for i in range(K)], lambda i, g: acc.add(g, live.pop(i)))
    g_sum, plan = acc.take()
    whole = stack(micro)
    one = t.gather(t.keys(whole.ids))
    got = []
    run.stepper.table_grads(
        [lambda: (one.W, one.idx, *prepare(whole, TINY))], lambda i, g: got.append(g)
    )
    assert [(p.shard, p.local.tolist()) for p in plan] == [
        (p.shard, p.local.tolist()) for p in one.plan
    ]
    n = g_sum.shape[0]  # the accumulator holds the union alone; the working set adds the pad
    np.testing.assert_allclose(np.array(g_sum), np.array(got[0][1 : 1 + n]), rtol=1e-4, atol=1e-8)
    assert not np.array(got[0][1 + n :]).any()


# ---- the skipped step ------------------------------------------------------------------------


def test_a_skipped_step_moves_no_row_moment_or_stamp(tmp_path):
    poisoned = SynthSource(B, T, V, seed=1, poison=set(range(K, 2 * K)))
    run = make(tmp_path, "a", poisoned, K)
    assert run.table is not None
    assert run.acc is not None
    assert run.step()["skipped"] is False  # step 0 trains, so rows, moments and stamps move
    before, event = bits(run.table), run.table.event
    rec = run.step()  # every micro-batch of step 1 has N_valid 0: a non-finite loss
    assert rec["skipped"] is True
    assert run.table.event == event == 1
    assert bits(run.table) == before
    assert rec["table"]["rows"] == 0
    assert run.acc.rows() == 0  # the sum is dropped, so the next step starts clean


def test_a_non_finite_working_set_gradient_alone_skips_the_step():
    # The loss is finite and the dense gradients are finite; d/dW sqrt(max(W, 0)) at W = 0 is
    # inf, times 0 is nan, so only g_w is non-finite.
    def loss_fn(p, W, idx, x):
        del idx
        dense = mx.sum(p["a"] * x)
        return dense + 0.0 * mx.sum(mx.sqrt(mx.maximum(W, 0.0))), dense

    p = {"a": mx.ones((4,), mx.float32)}
    o, r = OptimConfig(), RecipeConfig()
    opt = DenseOptim(p, {"a": "hidden"}, 4, o, r)
    st = Stepper(loss_fn, p, opt, o.clip, compile=False, table=True)
    W, idx, x = mx.zeros((3, 4), mx.float32), mx.zeros((1,), mx.int32), mx.ones((4,), mx.float32)
    seen, updates = [], []
    res = st.step(
        table_micro=[lambda: (W, idx, x)],
        on_table_grad=lambda i, g: seen.append(bool(mx.all(mx.isfinite(g)))),
        table_update=lambda: updates.append(1),
    )
    assert np.isfinite(res.loss)
    assert seen == [False]
    assert res.skipped is True
    assert updates == []
    assert np.array_equal(np.array(st.params["a"]), np.ones(4, np.float32))


def test_the_run_and_the_model_must_agree_on_the_table(tmp_path):
    o = OptimConfig(tokens=B * T, tokens_per_update=B * T)
    r = RecipeConfig(warmup=1, param_kind="sp")
    with pytest.raises(ConfigError, match="on together"):
        Run(TINY, o, r, Listed([]), tmp_path / "a", compute_dtype="float32", clock=FakeClock())
    wide = Table(
        TableConfig(params=PARAMS * 2, width=2 * D, max_shard_elements=SHARD * 2),
        mx.get_active_memory,
    )
    kw = {"compute_dtype": "float32", "table": wide, "clock": FakeClock()}
    with pytest.raises(ConfigError, match="not the model's d"):
        Run(TINY, o, r, Listed([]), tmp_path / "b", **kw)
