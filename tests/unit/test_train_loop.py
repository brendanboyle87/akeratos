"""The training loop: parameter groups, sources, the loop and the entry point.

Also the stop seam and the data pre-flight. Everything runs on MLX's CPU device on a tiny
model.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

from akeratos.configs import ModelConfig
from akeratos.data.packer import ExhaustedError, Packer
from akeratos.data.store import Split
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.model import init_params, param_shapes
from akeratos.rng import derive, run_seeds, seed_hex
from akeratos.train import entry
from akeratos.train.loop import Run
from akeratos.train.optim import GROUP_OF_KIND, groups_for
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.saver import Checkpointer, batch_hash
from akeratos.train.source import PackedSource, StepRows, SynthSource, preflight
from tests.fakes.clock import FakeClock, TickingClock

mx.set_default_device(mx.cpu)

V, B, T = 512, 2, 32
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


def configs(steps=20, lr=3e-3, **recipe):
    o = OptimConfig(lr_peak=lr, tokens=steps * B * T, tokens_per_update=B * T, decay_ind=0.0)
    kw = {"warmup": 2, "param_kind": "sp"} | recipe
    return o, RecipeConfig(**kw)


def run(tmp_path, steps=20, source=None, cfg=TINY, **kw):
    o, r = configs(steps)
    src = source or SynthSource(B, T, V, seed=1)
    opts = {"compute_dtype": "float32", "compile": False, "clock": FakeClock()} | kw
    return Run(cfg, o, r, src, tmp_path / "run", **opts)


class Repeat:
    """One batch, served every step."""

    def __init__(self, batch):
        self.batch = batch

    def next(self):
        return self.batch

    def state(self):
        return {"kind": "repeat"}


# ---- groups --------------------------------------------------------------------------------


@pytest.mark.parametrize("table", [False, True])
def test_every_parameter_has_one_group(table):
    cfg = replace(ModelConfig(), table=table)
    groups = groups_for(param_shapes(cfg))
    assert set(groups) == set(param_shapes(cfg))
    assert groups["emb.embed.weight"] == "embed"
    assert groups["layers.0.attn.q.weight"] == "hidden"
    assert groups["layers.0.ffn.down.weight"] == "hidden"
    assert groups["layers.1.attn.ve_gate.weight"] == "scalar"
    assert groups["xl"] == "scalar"
    assert ("tl" in groups) == table
    assert set(GROUP_OF_KIND.values()) <= {"hidden", "embed", "value_embed", "scalar"}


def test_unknown_kind_has_no_group(monkeypatch):
    monkeypatch.delitem(GROUP_OF_KIND, "xl")
    with pytest.raises(ValueError, match="no group takes"):
        groups_for(["xl"])


# ---- sources -------------------------------------------------------------------------------


def test_synthetic_rows_open_documents_and_repeat_by_index():
    a, b = SynthSource(B, T, V, seed=3), SynthSource(B, T, V, seed=3)
    x = a.next()
    assert x.ids.shape == x.targets.shape == x.segment_ids.shape == (B, T)
    assert (x.ids[:, 0] == 0).all()
    assert ((x.ids == 0) | (x.ids >= 17)).all()
    assert x.ids.max() < V
    np.testing.assert_array_equal(x.ids[:, 1:], x.targets[:, :-1])
    assert batch_hash(x.ids) == batch_hash(b.next().ids)
    assert a.state() == {"kind": "synthetic", "i": 1, "seed": seed_hex(3)}
    assert batch_hash(a.next().ids) != batch_hash(x.ids)


def write_split(path, docs, vocab=V):
    """A one-shard store split from id lists without their separator."""
    path.mkdir(parents=True)
    body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
    idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
    body.tofile(path / "000000.bin")
    idx.tofile(path / "000000.idx")
    manifest = {
        "dtype": "uint16",
        "eos": 0,
        "vocab": vocab,
        "shards": ["000000.bin"],
        "tokens": [len(body)],
        "documents": [len(docs)],
        "sep_position": "start",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return body, idx


def docs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(17, V, size=int(rng.integers(5, 50))) for _ in range(n)]


def test_split_reads_documents_and_feeds_the_packer(tmp_path):
    d = docs()
    write_split(tmp_path / "train", d)
    split = Split(tmp_path / "train", vocab=V)
    assert len(split.lengths) == len(d)
    assert split.tokens == sum(len(x) + 1 for x in d)
    np.testing.assert_array_equal(split.read(3, 1000), np.concatenate([[0], d[3]]))
    np.testing.assert_array_equal(split.read(3, 2), [0, d[3][0]])
    src = PackedSource(Packer(split, B, T, vocab=V, buffer=8))
    x = src.next()
    assert x.ids.shape == (B, T)
    assert (x.ids[:, 0] == 0).all()
    assert src.state()["kind"] == "packed"


@pytest.mark.parametrize("damage", ["size", "offsets", "vocab", "name"])
def test_split_refuses_a_broken_store(tmp_path, damage):
    path = tmp_path / "train"
    body, idx = write_split(path, docs(5))
    if damage == "size":
        body[:-1].tofile(path / "000000.bin")
    elif damage == "offsets":
        idx[2] = idx[1]
        idx.tofile(path / "000000.idx")
    elif damage == "name":
        m = json.loads((path / "manifest.json").read_text())
        (path / "000000.bin").rename(path / "0.bin")
        m["shards"] = ["0.bin"]
        (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(DataMismatchError):
        Split(path, vocab=V + 1 if damage == "vocab" else V)


# ---- the loop ------------------------------------------------------------------------------


def test_steps_log_and_write_the_manifest(tmp_path):
    r = run(tmp_path, steps=6)
    r.train(until=3)
    lines = [json.loads(x) for x in (r.out / "metrics.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [0, 1, 2]
    assert lines[-1]["tokens"] == 3 * B * T
    assert all(np.isfinite(x["loss"]) and not x["skipped"] for x in lines)
    assert set(lines[0]["grad_norm"]) == {"hidden", "embed", "value_embed", "scalar"}
    assert lines[0]["mfu"] is None
    assert lines[-1]["peak_bytes"] >= lines[-1]["active_bytes"] >= 0
    assert lines[0]["lr"]["hidden"] == pytest.approx(3e-3 / 2)
    man = json.loads((r.out / "run.json").read_text())
    assert man["schedule"]["steps"] == 6
    assert man["config"]["model"]["table"] is False
    assert any(s.startswith("switch opt.clip_norm") for s in man["switches"])
    assert man["numerics"]["mlx"]


def test_step_time_comes_from_the_clock(tmp_path):
    rec = run(tmp_path, clock=TickingClock()).step()
    assert rec["tokens_per_s"] == pytest.approx(B * T)


def test_a_system_sleep_on_the_run_clock_is_a_stall(tmp_path):
    clock = FakeClock()
    loop = run(tmp_path, clock=clock)
    assert loop.step()["stall"] is None
    clock.sleep_system(61_000_000_000)
    assert loop.step()["stall"] == 61.0


def test_a_run_without_a_clock_is_refused(tmp_path):
    with pytest.raises(TypeError, match="needs a clock"):
        run(tmp_path, clock=None)


def test_loss_falls_on_a_repeated_batch(tmp_path):
    batch = SynthSource(B, T, V, seed=5).next()
    o, r = configs(steps=40, lr=1e-2, schedule="constant", warmup=0)
    loop = Run(
        TINY,
        o,
        r,
        Repeat(batch),
        tmp_path,
        compute_dtype="float32",
        compile=False,
        clock=FakeClock(),
    )
    losses = [loop.step()["loss"] for _ in range(40)]
    assert losses[-1] < 0.5 * losses[0]


# The first compiled BF16 step builds 61 CPU kernels with g++, one at a time, into $TMPDIR/mlx.
# With that cache cold the test took 45-47 s on a 4-vCPU cloud VM (0.24 s warm), and 102 s
# pinned to one vCPU beside a busy process. A CI runner starts cold and runs two xdist workers
# on two vCPUs, so the test went over the global 60 s there. Its own ceiling covers the cold
# build; a hang still fails.
COLD_JIT_TIMEOUT_S = 180


@pytest.mark.timeout(COLD_JIT_TIMEOUT_S)
def test_compiled_bf16_step_matches_the_eager_one(tmp_path):
    batch = SynthSource(B, T, V, seed=6).next()
    a = run(tmp_path / "a", source=Repeat(batch), compute_dtype="bfloat16", compile=True)
    b = run(tmp_path / "b", source=Repeat(batch), compute_dtype="bfloat16", compile=False)
    la, lb = [x.step()["loss"] for x in (a, a)], [x.step()["loss"] for x in (b, b)]
    np.testing.assert_allclose(la, lb, rtol=1e-3)


def test_a_poisoned_step_is_skipped_and_still_advances(tmp_path):
    r = run(tmp_path, source=SynthSource(B, T, V, poison=[1]))
    assert r.ema is not None
    r.step()
    before = {k: np.array(v) for k, v in r.ema.avg.items()}
    params = {k: np.array(v) for k, v in r.stepper.params.items()}
    rec = r.step()
    assert rec["skipped"]
    assert rec["skipped_total"] == 1
    assert r.step_no == 2
    for k, v in r.ema.avg.items():
        np.testing.assert_array_equal(np.array(v), before[k])
    for k, v in r.stepper.params.items():
        np.testing.assert_array_equal(np.array(v), params[k])
    assert not r.step()["skipped"]


def test_multi_token_weights_stop_at_the_cool_down(tmp_path):
    r = run(tmp_path, steps=20, mtp=True)
    assert r.depth_weights(0) == pytest.approx((1.0, 0.5, 0.25))
    assert r.depth_weights(r.sched.decay_start()) is None
    assert run(tmp_path / "off", steps=20).depth_weights(0) is None
    rec = r.step()
    assert np.isfinite(rec["loss"])
    assert not rec["skipped"]


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (replace(TINY, table=True), "on together"),  # the model's table term without a table
        (replace(TINY, param_kind="mup"), "param.kind"),
    ],
)
def test_configurations_the_slice_refuses(tmp_path, cfg, match):
    with pytest.raises(ConfigError, match=match):
        run(tmp_path, cfg=cfg)


def test_a_batch_of_the_wrong_size_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="tokens_per_update"):
        run(tmp_path, source=SynthSource(B + 1, T, V)).step()
    with pytest.raises(ConfigError, match="compute dtype"):
        run(tmp_path, compute_dtype="float16")


# ---- the stop seam -------------------------------------------------------------------------


def test_on_step_sees_each_line_and_stop_ends_the_run_between_steps(tmp_path):
    r = run(tmp_path, steps=10)
    seen = []

    def hook(run_, rec):
        seen.append((run_.step_no, rec["step"]))
        if rec["step"] == 2:
            run_.stop()

    assert r.train(on_step=hook) == 3
    assert seen == [(1, 0), (2, 1), (3, 2)]
    assert len((r.out / "metrics.jsonl").read_text().splitlines()) == 3


def test_train_stops_at_the_budget(tmp_path):
    r = run(tmp_path, steps=4)
    assert r.train(until=99) == 4


def test_exhausted_data_raises_at_a_step_boundary(tmp_path):
    write_split(tmp_path / "train", docs(12))
    split = Split(tmp_path / "train", vocab=V)
    r = run(tmp_path, steps=50, source=PackedSource(Packer(split, B, T, vocab=V, buffer=4)))
    with pytest.raises(ExhaustedError):
        r.train()
    assert 0 < r.step_no < 50


def test_initial_weights_follow_the_derived_init_seed(tmp_path):
    a, b = run(tmp_path / "a", seed=4), run(tmp_path / "b", seed=4)
    ref = init_params(TINY, derive(4, "init"))
    for k, v in a.stepper.params.items():
        np.testing.assert_array_equal(np.array(v), np.array(b.stepper.params[k]))
        np.testing.assert_array_equal(np.array(v), np.array(ref[k]))
    alone = init_params(TINY, 9)
    c = run(tmp_path / "c", seed=4, init_seed=9)
    for k, v in c.stepper.params.items():
        np.testing.assert_array_equal(np.array(v), np.array(alone[k]))


def test_the_run_records_its_three_seeds_in_hex(tmp_path):
    man = run(tmp_path, seed=1, data_seed=7).manifest()["config"]
    assert man["seed"] == "0x0000000000000001"
    assert man["init_seed"] == f"0x{derive(1, 'init'):016x}"
    assert man["data_seed"] == "0x0000000000000007"
    assert all(re.fullmatch(r"0x[0-9a-f]{16}", man[k]) for k in ("seed", "init_seed", "data_seed"))


def test_run_seeds_derive_each_sub_seed_unless_given():
    s = run_seeds(5)
    assert s == {"seed": 5, "init_seed": derive(5, "init"), "data_seed": derive(5, "data")}
    assert run_seeds(5, data_seed=2)["data_seed"] == 2
    assert run_seeds(6)["init_seed"] != s["data_seed"]
    with pytest.raises(OverflowError, match="init_seed"):
        run_seeds(0, init_seed=-1)
    assert seed_hex(2**64 - 1) == "0xffffffffffffffff"


# ---- the entry point -----------------------------------------------------------------------


def cli_args(tmp_path, data):
    args = ["--data", str(data), "--tokens", str(4 * B * T), "--rows", str(B), "--seq-len", str(T)]
    args += ["--out", str(tmp_path / "out"), "--compute-dtype", "float32", "--print-every", "2"]
    args += [f"--model={k}" for k in TINY_KEYS]
    return [*args, "--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]


def cli(tmp_path, data, *extra, run_cls=Run):
    return entry.main([*cli_args(tmp_path, data), *extra], run_cls, Checkpointer)


@pytest.fixture
def tf32_set(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


@pytest.mark.usefixtures("tf32_set")
def test_entry_point_runs_synthetic_and_store_data(tmp_path, capsys):
    assert cli(tmp_path / "s", "synthetic", "--mtp") == entry.EXIT_OK
    assert len((tmp_path / "s" / "out" / "metrics.jsonl").read_text().splitlines()) == 4
    write_split(tmp_path / "train", docs(200))
    args = ("--steps", "2", "--buffer", "8", "--seed", "3")
    assert cli(tmp_path / "p", tmp_path / "train", *args) == entry.EXIT_STOPPED
    lines = (tmp_path / "p" / "out" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert "step 2 loss" in capsys.readouterr().err
    man = json.loads((tmp_path / "p" / "out" / "run.json").read_text())
    assert man["config"]["data_seed"] == f"0x{derive(3, 'data'):016x}"
    split = Split(tmp_path / "train", vocab=V)
    same = Packer(split, B, T, vocab=V, data_seed=derive(3, "data"), buffer=8)
    assert man["source"]["fingerprint"] == same.fingerprint


class SignalledRun(Run):
    """A run that receives SIGTERM during its first step."""

    def step(self):
        rec = super().step()
        if rec["step"] == 0:
            signal.raise_signal(signal.SIGTERM)
        return rec


@pytest.mark.usefixtures("tf32_set")
def test_a_signal_stops_the_run_after_the_step_with_exit_3(tmp_path, capsys):
    # With checkpoints off a signal stop is STOPPED; with one saved it is `EXIT_RESUMABLE`
    # (tests/unit/test_ckpt.py).
    args = ("--ckpt-every", "0")
    assert cli(tmp_path, "synthetic", *args, run_cls=SignalledRun) == entry.EXIT_STOPPED
    assert len((tmp_path / "out" / "metrics.jsonl").read_text().splitlines()) == 1
    assert "stopping after this step" in capsys.readouterr().err
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


@pytest.mark.usefixtures("tf32_set")
def test_setup_runs_once_per_launch_before_the_source_and_model(tmp_path, capsys):
    # The CLI's hook for macops' cache cap: once per launch, resume included, before
    # the source, the table (built between them) and the model.
    order = []

    class Saver(Checkpointer):
        def source_state(self):
            order.append("source")
            return super().source_state()

    class Model(Run):
        def __init__(self, *a, **kw):
            order.append("model")
            super().__init__(*a, **kw)

    args = [*cli_args(tmp_path, "synthetic"), "--ckpt-every", "1"]
    for steps in ("1", "2"):
        code = entry.main(
            [*args, "--steps", steps], Model, Saver, setup=lambda: order.append("setup")
        )
        assert code == entry.EXIT_STOPPED
    assert order == ["setup", "source", "model"] * 2
    assert "resumed from" in capsys.readouterr().err
    assert cli(tmp_path / "plain", "synthetic", "--steps", "1") == entry.EXIT_STOPPED  # optional


@pytest.mark.usefixtures("tf32_set")
def test_entry_point_exit_codes(tmp_path):
    assert cli(tmp_path / "t", "synthetic", "--model=table=true") == entry.EXIT_CONFIG
    assert cli(tmp_path / "k", "synthetic", "--model=nonsense") == entry.EXIT_CONFIG
    assert cli(tmp_path / "u", "synthetic", "--model=no.such.key=1") == entry.EXIT_CONFIG
    write_split(tmp_path / "train", docs(5), vocab=V + 1)
    assert cli(tmp_path / "d", tmp_path / "train") == entry.EXIT_DATA


def test_batch_holds_numpy_rows():
    x = SynthSource(B, T, V).next()
    assert isinstance(x, StepRows)
    assert x.ids.dtype == np.int32
    assert x.lmask is None


def test_the_entry_point_module_loads_no_mlx():

    code = "import sys, akeratos.train.entry; print('mlx.core' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_module_entry_turns_tf32_off_and_runs_the_loop(tmp_path):
    cmd = [sys.executable, "-m", "akeratos.train.entry", *cli_args(tmp_path, "synthetic")]
    env = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"}
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    assert out.returncode == entry.EXIT_OK, out.stderr
    man = json.loads((tmp_path / "out" / "run.json").read_text())
    assert man["numerics"]["MLX_ENABLE_TF32"] == "0"


@pytest.mark.usefixtures("tf32_set")
def test_an_invalid_recipe_exits_78(tmp_path, capsys):
    assert cli(tmp_path, "synthetic", "--recipe=opt.decay_ind=2.0") == entry.EXIT_CONFIG
    assert "decay" in capsys.readouterr().err


@pytest.mark.usefixtures("tf32_set")
def test_entry_point_refuses_a_split_too_small_for_the_budget(tmp_path, capsys):
    write_split(tmp_path / "train", docs(12))
    args = ("--buffer", "4", "--tokens", str(50 * B * T))
    assert cli(tmp_path / "x", tmp_path / "train", *args) == entry.EXIT_CONFIG
    assert "short by" in capsys.readouterr().err
    assert not (tmp_path / "x" / "out").exists()


def steps_fed(split, rows, **kw):
    """The whole steps of `rows` x T that one epoch of `split` feeds: counted by reading it."""
    packer, batches = Packer(split, 1, T, vocab=V, **kw), 0
    try:
        while True:
            next(packer)
            batches += 1
    except ExhaustedError:
        return batches // rows


@pytest.mark.usefixtures("tf32_set")
def test_entry_point_counts_a_step_as_every_row_of_its_micro_batches(tmp_path, capsys):
    write_split(tmp_path / "train", docs(400))
    seed = run_seeds(0, None, None)["data_seed"]
    split = Split(tmp_path / "train", vocab=V)
    fed = steps_fed(split, 3 * B, data_seed=seed, buffer=8)
    args = ("--buffer", "8", "--accum", "3", "--tokens", str((fed + 1) * 3 * B * T))
    assert cli(tmp_path / "x", tmp_path / "train", *args) == entry.EXIT_CONFIG
    assert f"one epoch of it feeds {fed}; short by 1\n" in capsys.readouterr().err


PASSED = {"sampled_rows": 0, "steps": 0, "steps_fed": 0}


@pytest.mark.usefixtures("tf32_set")
def test_entry_point_reports_exhausted_data(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(entry, "preflight", lambda *_, **__: PASSED)
    write_split(tmp_path / "train", docs(12))
    args = ("--buffer", "4", "--tokens", str(50 * B * T))
    assert cli(tmp_path / "x", tmp_path / "train", *args) == entry.EXIT_CONFIG
    assert "the data ran out" in capsys.readouterr().err


# ---- the data pre-flight -------------------------------------------------------------------


def test_preflight_passes_the_steps_one_epoch_feeds_and_refuses_one_more(tmp_path):
    write_split(tmp_path / "train", docs(400))
    split = Split(tmp_path / "train", vocab=V)
    fed = steps_fed(split, B, buffer=8)
    assert fed > 10
    fit = preflight(split, B, T, fed * B * T, vocab=V, buffer=8, sample_rows=50)
    assert fit == {"sampled_rows": 50, "steps": fed, "steps_fed": fed}
    with pytest.raises(ConfigError, match=rf"one epoch of it feeds {fed}; short by 1$"):
        preflight(split, B, T, (fed + 1) * B * T, vocab=V, buffer=8, sample_rows=50)


def test_preflight_counts_a_step_as_every_row_of_its_micro_batches(tmp_path):
    write_split(tmp_path / "train", docs(400))
    split = Split(tmp_path / "train", vocab=V)
    fed = steps_fed(split, 3 * B, data_seed=5, buffer=8)
    fit = preflight(split, 3 * B, T, 2 * 3 * B * T, vocab=V, data_seed=5, buffer=8)
    assert (fit["steps"], fit["steps_fed"]) == (2, fed)


def test_preflight_refuses_with_the_shortfall(tmp_path):
    write_split(tmp_path / "train", docs(40))
    split = Split(tmp_path / "train", vocab=V)
    with pytest.raises(ConfigError, match=r"short by [\d,]+$") as e:
        preflight(split, B, T, 1000 * B * T, vocab=V, buffer=8)
    assert f"cannot feed 1,000 steps of {B} x {T}" in str(e.value)
