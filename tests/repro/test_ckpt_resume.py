"""Bitwise resume, through the checkpointed parts and through the training entry point.

A resume through the checkpointed parts equals the run it continues; the entry point's crash,
stop and signal resumes, its resume refusals, its cadence and the pre-decay pin are checked
through `akeratos/train/entry.py`. The checkpoint format's own checks are in
`tests/unit/test_ckpt.py`, and the child-process SIGKILL cases in `test_resume.py`. Everything
runs at tiny sizes on MLX's CPU device.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from akeratos import ckpt
from akeratos.configs import ModelConfig
from akeratos.data.packer import Packer
from akeratos.train import entry, saver
from akeratos.train.loop import Run, Stepper
from akeratos.train.optim import EMA, DenseOptim, Schedule
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.saver import Checkpointer
from akeratos.train.source import SynthSource
from tests.fakes.clock import FakeClock
from tests.support.packdocs import TokenDocs
from tests.support.splits import write_split

mx.set_default_device(mx.cpu)
V, D, B, T = 128, 32, 2, 16


def newest(out: Path) -> Path:
    path = ckpt.latest(out)
    assert path is not None
    return path


# ---- bitwise resume ------------------------------------------------------------------------------


def docs() -> TokenDocs:
    rng = np.random.default_rng(0)
    body = [
        np.concatenate([[0], 17 + rng.integers(0, V - 17, rng.integers(3, 40))]) for _ in range(400)
    ]
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in body])])
    return TokenDocs(np.concatenate(body).astype(np.uint16), offsets)


def model() -> tuple[dict[str, mx.array], dict[str, str]]:
    rng = np.random.default_rng(1)
    p = {
        "embed.weight": rng.normal(0, 0.02, (V, D)),
        "up.weight": rng.uniform(-0.1, 0.1, (4 * D, D)),
        "down.weight": rng.uniform(-0.1, 0.1, (D, 4 * D)),
        "norm.weight": np.ones(D),
    }
    g = {
        "embed.weight": "embed",
        "up.weight": "hidden",
        "down.weight": "hidden",
        "norm.weight": "scalar",
    }
    return {k: mx.array(v.astype(np.float32)) for k, v in p.items()}, g


def loss_fn(p, ids, targets):
    x = p["embed.weight"][ids]
    x = x + mx.maximum(x @ p["up.weight"].T, 0) ** 2 @ p["down.weight"].T
    x = x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6) * p["norm.weight"]
    logits = x @ p["embed.weight"].T
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.take_along_axis(lp, targets[..., None], axis=-1).mean()


class Toy:
    """The parts of a run a checkpoint holds: weights, optimiser, EMA and the data cursor."""

    def __init__(self, packer_state=None):
        p, g = model()
        o, r = OptimConfig(), RecipeConfig()
        self.stepper = Stepper(loss_fn, p, DenseOptim(p, g, D, o, r), o.clip, compile=True)
        self.ema = EMA(p, 0.9, g)
        self.packer = Packer(docs(), B, T, vocab=V, buffer=16, state=packer_state)

    def step(self, t):
        rows = next(self.packer)
        s = 0.5 + 0.5 * math.cos(t)  # a moving schedule, so the rate is state too
        res = self.stepper.step(mx.array(rows.inputs), mx.array(rows.targets), s=s)
        if not res.skipped:
            self.ema.update(self.stepper.params)

    def save(self, out):
        st = self.stepper
        arrays = {"dense": st.params, "ema": self.ema.avg}
        arrays |= {f"optim-{g}": ckpt.flatten(o.state) for g, o in st.opt.opts.items()}
        state = {"step": st.step_no, "skipped": st.skipped, "source": self.packer.state()}
        return ckpt.save(out, st.step_no, arrays=arrays, state=state)

    @classmethod
    def load(cls, path):
        c = ckpt.load(path)
        run = cls(packer_state=c.state["source"])
        st = run.stepper
        st.params = c.arrays("dense")
        for g, o in st.opt.opts.items():
            ckpt.restore(o.state, c.arrays(f"optim-{g}"))
        run.ema.avg = {n: c.arrays("ema")[n] for n in run.ema.avg}
        st.step_no, st.skipped = c.state["step"], c.state["skipped"]
        return run

    def snapshot(self):
        st = self.stepper
        out = {f"p.{k}": v for k, v in st.params.items()} | {
            f"e.{k}": v for k, v in self.ema.avg.items()
        }
        for g, o in st.opt.opts.items():
            out |= {f"o.{g}.{k}": v for k, v in ckpt.flatten(o.state).items()}
        return {k: np.array(v) for k, v in out.items()}


def test_kill_and_resume_is_bitwise_on_cpu(tmp_path):
    whole = Toy()
    for t in range(6):
        whole.step(t)
    first = Toy()
    for t in range(3):
        first.step(t)
    first.save(tmp_path)
    del first
    resumed = Toy.load(newest(tmp_path))
    assert resumed.stepper.step_no == 3
    for t in range(3, 6):
        resumed.step(t)
    assert whole.packer.state() == resumed.packer.state()
    a, b = whole.snapshot(), resumed.snapshot()
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k]) for k in a), [
        k for k in a if not np.array_equal(a[k], b[k])
    ]


def test_the_packer_state_round_trips_through_state_json(tmp_path):
    run = Toy()
    for t in range(4):
        run.step(t)
    path = run.save(tmp_path)
    again = Packer(docs(), B, T, vocab=V, buffer=16, state=ckpt.load(path).state["source"])
    for _ in range(3):
        x, y = next(run.packer), next(again)
        assert np.array_equal(x.inputs, y.inputs)
        assert np.array_equal(x.segment_ids, y.segment_ids)


# ---- the entry point: resume, fallback and the cadence ----------------------------------------

TINY_KEYS = [
    "table=false",
    "d_model=64",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    "vocab_size=512",
    "window=16",
    "param.kind=sp",
]
STEPS = 6
# The clock and the allocator vary between runs.
VARY = ("step_s", "tokens_per_s", "stall", "active_bytes", "peak_bytes", "cache_bytes")


def cli(out, data, *extra, run_cls=Run):
    args = ["--data", str(data), "--tokens", str(STEPS * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--out", str(out), "--compute-dtype", "float32"]
    args += [f"--model={k}" for k in TINY_KEYS]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    return entry.main([*args, "--print-every", "0", *extra], run_cls, Checkpointer)


def final_state(out):
    """Every array of the newest checkpoint, its JSON state, and the log without wall-clock fields."""
    c = ckpt.load(newest(out))
    arrays = {
        f"{f.stem}/{k}": np.array(v)
        for f in sorted(c.path.glob("*.safetensors"))
        for k, v in c.arrays(f.stem).items()
    }
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    return c.step, arrays, c.state, [{k: v for k, v in x.items() if k not in VARY} for x in lines]


def assert_same(a, b):
    assert a[0] == b[0]
    assert a[1].keys() == b[1].keys()
    assert [k for k in a[1] if not np.array_equal(a[1][k], b[1][k])] == []
    assert a[2] == b[2]
    assert a[3] == b[3]


@pytest.fixture
def tf32_set(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


@pytest.mark.usefixtures("tf32_set")
@pytest.mark.parametrize(("data", "accum"), [("synthetic", 1), ("store", 1), ("synthetic", 2)])
def test_a_crash_and_rerun_matches_the_uninterrupted_run_bitwise(tmp_path, data, accum):
    src = "synthetic" if data == "synthetic" else write_split(tmp_path / "train")
    extra = ("--buffer", "8", "--ckpt-every", "2", "--accum", str(accum))
    extra += ("--tokens", str(STEPS * B * T * accum))  # STEPS optimiser steps of K micro-batches
    assert cli(tmp_path / "whole", src, *extra) == 0
    assert [p.name for p in ckpt.checkpoints(tmp_path / "whole")] == [
        "ckpt-00000004",
        "ckpt-00000006",
    ]
    # A crash after step 5: its end-of-run save is torn, so the rerun falls back to step 4,
    # drops the log lines of step 4 onward and reads the same rows again.
    assert cli(tmp_path / "cut", src, *extra, "--steps", "5") == entry.EXIT_STOPPED
    (tmp_path / "cut" / "ckpt-00000005" / "ema.safetensors").unlink(missing_ok=True)
    assert newest(tmp_path / "cut").name == "ckpt-00000004"
    assert cli(tmp_path / "cut", src, *extra) == 0
    assert_same(final_state(tmp_path / "whole"), final_state(tmp_path / "cut"))


@pytest.mark.usefixtures("tf32_set")
def test_a_stopped_run_saves_where_it_stopped_and_resumes(tmp_path, capsys):
    assert cli(tmp_path / "whole", "synthetic", "--ckpt-every", "100") == 0
    assert (
        cli(tmp_path / "cut", "synthetic", "--ckpt-every", "100", "--steps", "3")
        == entry.EXIT_STOPPED
    )
    assert newest(tmp_path / "cut").name == "ckpt-00000003"
    assert cli(tmp_path / "cut", "synthetic", "--ckpt-every", "100") == 0
    assert "resumed from" in capsys.readouterr().err
    assert_same(final_state(tmp_path / "whole"), final_state(tmp_path / "cut"))


class SignalAt3(Run):
    """A run that receives SIGTERM during step 3, as `kill` would send it."""

    def step(self):
        rec = super().step()
        if self.step_no == 3:
            os.kill(os.getpid(), signal.SIGTERM)
        return rec


@pytest.mark.usefixtures("tf32_set")
def test_a_signal_saves_the_step_in_flight_and_exits_resumable(tmp_path):
    assert cli(tmp_path / "whole", "synthetic") == entry.EXIT_OK
    cut = tmp_path / "cut"
    assert cli(cut, "synthetic", run_cls=SignalAt3) == entry.EXIT_RESUMABLE
    assert newest(cut).name == "ckpt-00000003"
    assert cli(cut, "synthetic") == entry.EXIT_OK
    assert_same(final_state(tmp_path / "whole"), final_state(cut))
    off = tmp_path / "off"
    assert cli(off, "synthetic", "--ckpt-every", "0", run_cls=SignalAt3) == entry.EXIT_STOPPED


@pytest.mark.usefixtures("tf32_set")
def test_a_resume_refuses_other_configs_and_sources(tmp_path):
    assert cli(tmp_path / "a", "synthetic", "--steps", "2") == entry.EXIT_STOPPED
    assert cli(tmp_path / "a", "synthetic", "--recipe=lr.peak=0.01") == entry.EXIT_CONFIG
    assert cli(tmp_path / "a", "synthetic", "--data-seed", "5") == entry.EXIT_CONFIG
    assert cli(tmp_path / "a", write_split(tmp_path / "train")) == entry.EXIT_CONFIG
    assert cli(tmp_path / "a", "synthetic", "--model=attn.path=fused") == entry.EXIT_CONFIG


@pytest.mark.usefixtures("tf32_set")
def test_a_fused_run_resumes_fused_and_refuses_the_eager_path(tmp_path):
    """`attn.path` is in the saved configs, so a resume cannot flip it."""
    fused = "--model=attn.path=fused"
    assert cli(tmp_path / "f", "synthetic", fused, "--steps", "2") == entry.EXIT_STOPPED
    assert cli(tmp_path / "f", "synthetic", "--model=attn.path=banded") == entry.EXIT_CONFIG
    assert cli(tmp_path / "f", "synthetic", "--steps", "3") == entry.EXIT_STOPPED  # keeps fused
    assert cli(tmp_path / "f", "synthetic", fused) == entry.EXIT_OK
    assert attn_path(tmp_path / "f") == "fused"


def attn_path(out):
    return json.loads((out / "run.json").read_text())["config"]["model"]["attn_path"]


@pytest.mark.usefixtures("tf32_set")
def test_a_fresh_run_defaults_to_fused_only_where_the_kernels_run(tmp_path, monkeypatch, capsys):
    """With no `attn.path` switch, a fresh run is fused where `kernels_fit` holds."""
    assert cli(tmp_path / "cpu", "synthetic", "--steps", "1") == entry.EXIT_STOPPED
    assert attn_path(tmp_path / "cpu") == "banded"  # the CPU: no kernels, so no FP32 fallback
    assert "attn.path banded (default)" in capsys.readouterr().err
    monkeypatch.setattr(saver, "kernels_fit", lambda *_: True)
    assert cli(tmp_path / "gpu", "synthetic", "--steps", "1") == entry.EXIT_STOPPED
    assert attn_path(tmp_path / "gpu") == "fused"
    assert cli(tmp_path / "set", "synthetic", "--steps", "1", "--model=attn.path=banded") == 3
    assert attn_path(tmp_path / "set") == "banded"  # an explicit switch wins


@pytest.mark.usefixtures("tf32_set")
def test_a_banded_run_resumes_banded_under_the_fused_default(tmp_path, monkeypatch, capsys):
    """A run started on the banded path resumes on it with the same flags."""
    assert cli(tmp_path / "arm", "synthetic", "--steps", "2") == entry.EXIT_STOPPED
    assert attn_path(tmp_path / "arm") == "banded"
    monkeypatch.setattr(saver, "kernels_fit", lambda *_: True)
    capsys.readouterr()
    assert cli(tmp_path / "arm", "synthetic") == entry.EXIT_OK
    assert "attn.path banded (the checkpoint)" in capsys.readouterr().err
    assert attn_path(tmp_path / "arm") == "banded"


@pytest.mark.usefixtures("tf32_set")
def test_ckpt_every_zero_saves_nothing(tmp_path):
    assert cli(tmp_path / "a", "synthetic", "--ckpt-every", "0") == 0
    assert ckpt.latest(tmp_path / "a") is None


def test_a_registered_writer_is_saved_checked_and_found_on_resume(tmp_path):
    """The save hook the table uses: `writers[name]` runs at every save, inside the checkpoint."""
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        n_kv_heads=1,
        ffn=128,
        vocab_size=512,
        window=16,
        table=False,
        param_kind="sp",
    )
    o = OptimConfig(lr_peak=3e-3, tokens=4 * B * T, tokens_per_update=B * T, decay_ind=0.0)
    r = RecipeConfig(warmup=1, param_kind="sp")
    kw = {"compute_dtype": "float32", "compile": False, "clock": FakeClock()}
    run = Run(cfg, o, r, SynthSource(B, T, 512), tmp_path, **kw)
    saver = Checkpointer(tmp_path, every=2)
    saver.writers["extra"] = lambda d: (d / "x.bin").write_bytes(bytes(run.step_no))
    saver.attach(run)
    run.train(4, lambda run_, _: saver.after_step(run_))
    assert [p.name for p in ckpt.checkpoints(tmp_path)] == ["ckpt-00000002", "ckpt-00000004"]
    top = tmp_path / "ckpt-00000004"
    assert (top / "extra" / "x.bin").read_bytes() == bytes(4)
    resumed = Checkpointer(tmp_path).resumed
    assert resumed is not None
    assert (resumed.path / "extra" / "x.bin").stat().st_size == 4
    (top / "extra" / "x.bin").write_bytes(b"")  # a torn writer file fails the size check
    assert newest(tmp_path).name == "ckpt-00000002"


# ---- the pre-decay pin through the entry point ---------------------------------------------------

DECAY_AT = 4  # STEPS 6 at the default 20% decay share: D = 2, so the decay starts at step 4


def pins(out):
    return [p.name for p in ckpt.checkpoints(out) if ckpt.pinned(p)]


@pytest.mark.usefixtures("tf32_set")
def test_pin_decay_keeps_the_last_stable_checkpoint_with_keep_1(tmp_path, capsys):
    out = tmp_path / "pinned"
    assert cli(out, "synthetic", "--ckpt-every", "1", "--keep", "1", "--pin-decay") == 0
    assert [p.name for p in ckpt.checkpoints(out)] == ["ckpt-00000004", "ckpt-00000006"]
    assert pins(out) == ["ckpt-00000004"]
    assert "will be pinned" in capsys.readouterr().err
    # It is the state the run held at the decay's start, and a stable-phase branch may fork there.
    assert cli(tmp_path / "cut", "synthetic", "--steps", str(DECAY_AT)) == entry.EXIT_STOPPED
    c, ref = ckpt.load(out / "ckpt-00000004"), ckpt.load(tmp_path / "cut" / "ckpt-00000004")
    assert c.state == ref.state
    for stem in sorted(f.stem for f in ref.path.glob("*.safetensors")):
        a, b = c.arrays(stem), ref.arrays(stem)
        assert [k for k in a if not np.array_equal(np.array(a[k]), np.array(b[k]))] == []
    sched = Schedule(**json.loads((out / "run.json").read_text())["schedule"])
    assert sched.decay_start() == DECAY_AT
    assert sched.branch(c.step).steps > c.step
    # The pin resumes like any checkpoint, and the run it resumes ends where the whole one did.
    branch = tmp_path / "from-pin"
    branch.mkdir()
    shutil.copytree(c.path, branch / c.path.name)
    assert cli(tmp_path / "whole", "synthetic") == 0
    assert cli(branch, "synthetic") == 0
    whole, again = final_state(tmp_path / "whole"), final_state(branch)
    assert_same((*whole[:3], []), (*again[:3], []))  # only the checkpoint was copied, not the log


@pytest.mark.usefixtures("tf32_set")
def test_a_run_started_without_the_flag_can_add_it_on_resume(tmp_path, capsys):
    early, at, late = tmp_path / "early", tmp_path / "at", tmp_path / "late"
    for out, steps in ((early, 2), (at, DECAY_AT), (late, 5)):
        assert cli(out, "synthetic", "--steps", str(steps)) == entry.EXIT_STOPPED
        assert cli(out, "synthetic", "--pin-decay") == entry.EXIT_OK
    assert pins(early) == ["ckpt-00000004"]
    assert pins(at) == ["ckpt-00000004"]  # the resumed checkpoint, pinned in place
    assert pins(late) == []
    err = capsys.readouterr().err
    assert "pinned " + str(at / "ckpt-00000004") in err
    assert "past the decay's start at 4; nothing is pinned" in err
    # A later resume, with or without the flag, keeps the pin.
    assert cli(early, "synthetic", "--pin-decay") == entry.EXIT_OK
    assert pins(early) == ["ckpt-00000004"]


@pytest.mark.usefixtures("tf32_set")
def test_pin_decay_needs_checkpoints_on(tmp_path, capsys):
    assert cli(tmp_path / "a", "synthetic", "--ckpt-every", "0", "--pin-decay") == (
        entry.EXIT_CONFIG
    )
    assert "--pin-decay needs checkpoints on" in capsys.readouterr().err
