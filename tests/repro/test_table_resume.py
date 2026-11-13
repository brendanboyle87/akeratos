"""A killed table-on run resumes bitwise through the entry point.

The table here is 4,096 rows of width 64 in 4 shards, on MLX's CPU device, and the entry
point's table switches are checked beside it.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from akeratos import ckpt
from akeratos.train import entry
from akeratos.train.loop import Run
from akeratos.train.saver import Checkpointer

mx.set_default_device(mx.cpu)

V, D, B, T, K = 512, 64, 2, 32, 3
PARAMS, SHARD = 4096 * D, 1024 * D  # 4,096 rows of width d in 4 shards


# ---- the entry point and table-on resume ------------------------------------------------------

TINY_KEYS = [
    "table=true",
    f"d_model={D}",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    f"vocab_size={V}",
    "window=16",
    "param.kind=sp",
]
TABLE_KEYS = [f"params={PARAMS}", f"max_shard_elements={SHARD}"]
STEPS = 6
VARY = (
    "step_s",
    "tokens_per_s",
    "stall",
    "active_bytes",
    "peak_bytes",
    "cache_bytes",
)  # clock and allocator


def cli(out, *extra, table_keys=TABLE_KEYS):
    args = ["--data", "synthetic", "--tokens", str(STEPS * K * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--accum", str(K), "--out", str(out)]
    args += ["--compute-dtype", "float32", *[f"--model={k}" for k in TINY_KEYS]]
    args += [f"--table={k}" for k in table_keys]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    return entry.main([*args, "--print-every", "0", *extra], Run, Checkpointer)


def final_state(out):
    """The newest checkpoint's files byte for byte, its state, and the log without clock fields."""
    path = ckpt.latest(out)
    assert path is not None
    files = {
        str(f.relative_to(path)): f.read_bytes()
        for f in sorted(path.rglob("*"))
        if f.is_file() and f.name != ckpt.STATE
    }
    c = ckpt.load(path)
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    return c.step, files, c.state, [{k: v for k, v in x.items() if k not in VARY} for x in lines]


@pytest.fixture
def tf32_set(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


@pytest.mark.usefixtures("tf32_set")
def test_a_killed_table_on_run_resumes_bitwise(tmp_path, capsys):
    assert cli(tmp_path / "whole", "--ckpt-every", "2") == 0
    # A crash after step 5: its end-of-run save is torn, so the rerun falls back to step 4,
    # loads the table saved there and reads the same rows again.
    assert cli(tmp_path / "cut", "--ckpt-every", "2", "--steps", "5") == entry.EXIT_STOPPED
    (tmp_path / "cut" / "ckpt-00000005" / "table" / "table.json").unlink()
    assert cli(tmp_path / "cut", "--ckpt-every", "2") == 0
    assert "resumed from" in capsys.readouterr().err
    a, b = final_state(tmp_path / "whole"), final_state(tmp_path / "cut")
    assert a[0] == b[0] == STEPS
    assert "table/table.json" in a[1]
    assert sum(k.startswith("table/shard_") for k in a[1]) == 4
    assert a[1].keys() == b[1].keys()
    assert [k for k in a[1] if a[1][k] != b[1][k]] == []
    assert a[2] == b[2]
    assert a[2]["table_event"] == STEPS
    assert a[3] == b[3]


@pytest.mark.usefixtures("tf32_set")
def test_the_entry_point_refuses_bad_table_switches(tmp_path):
    assert cli(tmp_path / "a", table_keys=[]) == entry.EXIT_CONFIG  # no size
    assert cli(tmp_path / "b", table_keys=[*TABLE_KEYS, "rows=3"]) == entry.EXIT_CONFIG
    off = ["--model=table=false", f"--table=params={PARAMS}"]
    assert cli(tmp_path / "c", *off) == entry.EXIT_CONFIG
    assert (
        cli(tmp_path / "d", "--steps", "1", table_keys=[*TABLE_KEYS, "format=F32"])
        == entry.EXIT_STOPPED
    )
    assert json.loads((tmp_path / "d" / "run.json").read_text())["config"]["table"]["fmt"] == "F32"


@pytest.mark.usefixtures("tf32_set")
def test_a_resume_refuses_another_table(tmp_path):
    assert cli(tmp_path / "a", "--steps", "2") == entry.EXIT_STOPPED
    other = [*TABLE_KEYS, "lr_mult=10"]
    assert cli(tmp_path / "a", table_keys=other) == entry.EXIT_CONFIG
