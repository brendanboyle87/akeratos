"""The table's per-row read counts and their replay.

A fresh table-on run counts every lookup per row and saves the counts as `table/reads.npy`;
`python -m akeratos.train.reads` rebuilds them from the data alone and must equal them. A run
resumed from a checkpoint saved without counts (before the counter existed) still resumes and
keeps none. Everything runs at tiny sizes on MLX's CPU backend.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from akeratos import ckpt
from akeratos.errors import ConfigError
from akeratos.table.config import TableConfig
from akeratos.table.hasher import keys, multipliers
from akeratos.table.table import READS, Table
from akeratos.train import entry, reads
from akeratos.train.loop import Run
from akeratos.train.saver import Checkpointer
from akeratos.train.source import SynthSource
from tests.support.splits import write_split

mx.set_default_device(mx.cpu)

V, D, B, T, K = 512, 64, 2, 32, 3
PARAMS, SHARD = 4096 * D, 1024 * D  # 4,096 rows of width d in 4 shards
STEPS = 6
LOOKUPS = 2  # orders (2, 3), one head
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


def cfg() -> TableConfig:
    return TableConfig(params=PARAMS, width=D, max_shard_elements=SHARD, seed=3)


def cli(out, *extra, data="synthetic"):
    args = ["--data", str(data), "--tokens", str(STEPS * K * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--accum", str(K), "--out", str(out), "--buffer", "8"]
    args += ["--compute-dtype", "float32", *[f"--model={k}" for k in TINY_KEYS]]
    args += [f"--table={k}" for k in TABLE_KEYS]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    return entry.main([*args, "--print-every", "0", *extra], Run, Checkpointer)


def replay(out, run, *extra, data="synthetic"):
    args = ["--ckpt", str(run), "--data", str(data), "--rows", str(B), "--seq-len", str(T)]
    return reads.main([*args, "--buffer", "8", "--out", str(out), *extra])


def inrun(run):
    path = ckpt.latest(run)
    assert path is not None
    return np.load(path / "table" / READS)


@pytest.fixture
def tf32_set(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


# ---- the counter on the table ---------------------------------------------------------------


def test_gather_counts_each_lookup_per_row_only_when_tracking() -> None:
    t = Table(cfg(), mx.get_active_memory)
    ids = SynthSource(B, T, V, seed=1).next().ids
    rows = t.keys(ids)
    t.gather(rows)
    assert t.reads is None  # evaluation's tables count nothing
    t.track_reads()
    t.gather(rows)
    t.gather(rows)
    assert t.reads is not None
    want = 2 * np.bincount(rows.reshape(-1), minlength=t.rows)
    assert np.array_equal(t.reads, want)
    assert int(t.reads.sum()) == 2 * B * T * LOOKUPS


def test_counts_are_saved_beside_the_shards_and_restored_on_request(tmp_path) -> None:
    t = Table(cfg(), mx.get_active_memory)
    t.track_reads()
    t.gather(t.keys(SynthSource(B, T, V, seed=1).next().ids))
    assert t.write(tmp_path) == [READS, "table.json"]  # event 0: no shard yet
    assert t.reads is not None
    back = Table.load(tmp_path, cfg(), mx.get_active_memory, reads=True)
    assert back.reads is not None
    assert np.array_equal(back.reads, t.reads)
    assert Table.load(tmp_path, cfg(), mx.get_active_memory).reads is None
    (tmp_path / READS).unlink()
    assert Table.load(tmp_path, cfg(), mx.get_active_memory, reads=True).reads is None


@pytest.mark.parametrize(
    "bad", [np.zeros(4096, np.int32), np.zeros(4095, np.int64)], ids=["dtype", "rows"]
)
def test_load_refuses_counts_that_are_not_one_int64_per_row(tmp_path, bad) -> None:
    Table(cfg(), mx.get_active_memory).write(tmp_path)
    np.save(tmp_path / READS, bad)
    with pytest.raises(ConfigError, match="expected int64"):
        Table.load(tmp_path, cfg(), mx.get_active_memory, reads=True)


def test_the_replay_counts_equal_the_hashed_ids() -> None:
    c = cfg()
    src = SynthSource(B, T, V, seed=9)
    got = reads.replay(SynthSource(B, T, V, seed=9), c, 4)
    mults = multipliers(c.seed, c.orders, c.heads)
    want = sum(
        np.bincount(
            keys(src.next().ids, c.orders, mults, c.rows_per_slot()).ravel(), minlength=c.rows()
        )
        for _ in range(4)
    )
    assert np.array_equal(got, want)


# ---- in the run, and the replay tool --------------------------------------------------------


@pytest.mark.usefixtures("tf32_set")
@pytest.mark.parametrize("data", ["synthetic", "store"])
def test_the_replay_equals_the_in_run_counts(tmp_path, data, capsys):
    src = "synthetic" if data == "synthetic" else write_split(tmp_path / "train")
    assert cli(tmp_path / "run", "--ckpt-every", "2", data=src) == 0
    counts = inrun(tmp_path / "run")
    assert counts.dtype == np.int64
    assert int(counts.sum()) == STEPS * K * B * T * LOOKUPS
    assert replay(tmp_path / "r.npy", tmp_path / "run", data=src) == reads.EXIT_OK
    assert np.array_equal(np.load(tmp_path / "r.npy"), counts)
    assert "equal the in-run counts" in capsys.readouterr().err
    # An older checkpoint replays to its own step.
    old = ckpt.checkpoints(tmp_path / "run")[0]
    assert replay(tmp_path / "old.npy", old, data=src) == reads.EXIT_OK
    assert int(np.load(tmp_path / "old.npy").sum()) == 4 * K * B * T * LOOKUPS


@pytest.mark.usefixtures("tf32_set")
def test_a_run_saved_without_counts_resumes_and_keeps_none(tmp_path, capsys):
    """A run started before the counter: its checkpoints lack `reads.npy`, and resume still works."""
    assert cli(tmp_path / "run", "--ckpt-every", "2", "--steps", "2") == entry.EXIT_STOPPED
    path = ckpt.latest(tmp_path / "run")
    assert path is not None
    (path / "table" / READS).unlink()
    state = json.loads((path / ckpt.STATE).read_text())
    del state["files"][f"table/{READS}"]
    (path / ckpt.STATE).write_text(json.dumps(state))
    assert ckpt.complete(path)
    assert cli(tmp_path / "run", "--ckpt-every", "2") == 0
    assert "keeps none" in capsys.readouterr().err
    last = ckpt.latest(tmp_path / "run")
    assert last is not None
    assert not (last / "table" / READS).exists()
    # The replay is what gives such a run its counts.
    assert replay(tmp_path / "r.npy", tmp_path / "run") == reads.EXIT_OK
    assert int(np.load(tmp_path / "r.npy").sum()) == STEPS * K * B * T * LOOKUPS


@pytest.mark.usefixtures("tf32_set")
def test_the_replay_refuses_flags_that_are_not_the_run_s(tmp_path):
    split = write_split(tmp_path / "train")
    assert cli(tmp_path / "run", data=split) == 0
    assert replay(tmp_path / "r.npy", tmp_path / "run", "--rows", "1", data=split) == 78
    assert replay(tmp_path / "r.npy", tmp_path / "run", "--buffer", "9", data=split) == 78
    assert replay(tmp_path / "r.npy", tmp_path / "run") == 78  # synthetic, for a store run
    assert not (tmp_path / "r.npy").exists()


@pytest.mark.usefixtures("tf32_set")
def test_the_replay_reports_counts_that_differ(tmp_path):
    assert cli(tmp_path / "run") == 0
    path = ckpt.latest(tmp_path / "run")
    assert path is not None
    counts = np.load(path / "table" / READS)
    counts[0] += 1
    np.save(path / "table" / READS, counts)  # same size, so the checkpoint stays complete
    assert replay(tmp_path / "r.npy", tmp_path / "run") == reads.EXIT_DATA
