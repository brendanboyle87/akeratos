"""Atomic checkpoints, the completeness check, fallback and bitwise resume.

Also the torn and short-file fallbacks, pruning, the stop request, the stall clock and the
pre-decay pin. The table-on resume is in `tests/unit/test_table_step.py`. Everything runs at
tiny sizes on MLX's CPU device.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from akeratos import ckpt
from akeratos.train.saver import Log, StallClock
from tests.fakes.clock import FakeClock

mx.set_default_device(mx.cpu)
V, D, B, T = 128, 32, 2, 16


def arrays(seed: int) -> dict[str, mx.array]:
    rng = np.random.default_rng(seed)
    return {"a": mx.array(rng.normal(size=(4, 3)).astype(np.float32)), "b": mx.arange(5)}


def write(out: Path, step: int, keep: int = 2) -> Path:
    def shards(d: Path) -> None:
        for s in range(2):
            mx.save_safetensors(str(d / f"shard-{s:05d}.safetensors"), arrays(step + s))
        (d / "table.json").write_text(json.dumps({"event": step}))

    return ckpt.save(
        out,
        step,
        arrays={"dense": arrays(step)},
        state={"step": step},
        dirs={"table": shards},
        keep=keep,
    )


def newest(out: Path) -> Path:
    path = ckpt.latest(out)
    assert path is not None
    return path


# ---- the format ----------------------------------------------------------------------------------


def test_a_checkpoint_round_trips_arrays_state_and_directories(tmp_path):
    path = write(tmp_path, 7)
    assert path.name == "ckpt-00000007"
    assert ckpt.complete(path)
    c = ckpt.load(path)
    assert c.step == 7
    assert c.state == {"step": 7}
    got, want = c.arrays("dense"), arrays(7)
    assert all(np.array_equal(np.array(got[k]), np.array(want[k])) for k in want)
    assert json.loads((c.path / "table" / "table.json").read_text()) == {"event": 7}
    files = json.loads((path / ckpt.STATE).read_text())["files"]
    assert set(files) == {
        "dense.safetensors",
        "table/shard-00000.safetensors",
        "table/shard-00001.safetensors",
        "table/table.json",
    }
    assert not list(tmp_path.glob(".tmp-*"))


def test_a_torn_tmp_directory_is_ignored_and_removed_by_the_next_save(tmp_path):
    write(tmp_path, 10)
    torn = tmp_path / ".tmp-ckpt-00000020"
    torn.mkdir()
    (torn / "dense.safetensors").write_bytes(b"\0" * 10)
    assert newest(tmp_path).name == "ckpt-00000010"
    write(tmp_path, 30)
    assert not torn.exists()
    assert [p.name for p in ckpt.checkpoints(tmp_path)] == ["ckpt-00000010", "ckpt-00000030"]


@pytest.mark.parametrize("damage", ["missing", "short", "no-state", "bad-state", "wrong-step"])
def test_an_incomplete_checkpoint_falls_back_to_the_previous_one(tmp_path, damage):
    write(tmp_path, 10)
    path = write(tmp_path, 20)
    shard = path / "table" / "shard-00001.safetensors"
    if damage == "missing":
        shard.unlink()
    elif damage == "short":
        shard.write_bytes(shard.read_bytes()[:-1])
    elif damage == "no-state":
        (path / ckpt.STATE).unlink()
    elif damage == "bad-state":
        (path / ckpt.STATE).write_text('{"format": 1, "st')
    else:
        doc = json.loads((path / ckpt.STATE).read_text())
        (path / ckpt.STATE).write_text(json.dumps(doc | {"step": 21}))
    assert not ckpt.complete(path)
    assert newest(tmp_path).name == "ckpt-00000010"
    with pytest.raises(ValueError, match="incomplete"):
        ckpt.load(path)


def test_no_checkpoint_gives_none(tmp_path):
    assert ckpt.latest(tmp_path) is None
    assert ckpt.latest(tmp_path / "absent") is None


def test_pruning_keeps_the_newest_two_complete_and_drops_torn_ones(tmp_path):
    for step in (10, 20):
        write(tmp_path, step)
    (tmp_path / "ckpt-00000020" / ckpt.STATE).unlink()
    write(tmp_path, 30)
    assert [p.name for p in ckpt.checkpoints(tmp_path)] == ["ckpt-00000010", "ckpt-00000030"]
    assert not (tmp_path / "ckpt-00000020").exists()
    write(tmp_path, 40)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt-00000030", "ckpt-00000040"]


def test_a_re_save_refuses_a_complete_checkpoint_and_replaces_a_torn_one(tmp_path):
    first = write(tmp_path, 10)
    with pytest.raises(FileExistsError):
        write(tmp_path, 10)
    assert ckpt.complete(first)
    (first / "dense.safetensors").unlink()
    write(tmp_path, 10)
    assert [p.name for p in ckpt.checkpoints(tmp_path)] == ["ckpt-00000010"]


def test_restore_writes_in_place_and_refuses_a_different_tree():
    inner = {"m": mx.zeros(2), "v": mx.zeros(2)}
    live = {"step": mx.array(0), "w": inner}
    flat = ckpt.flatten({"step": mx.array(3), "w": {"m": mx.ones(2), "v": mx.ones(2)}})
    assert set(flat) == {"step", "w::m", "w::v"}
    ckpt.restore(live, flat)
    assert live["w"] is inner
    assert float(inner["m"].sum()) == 2
    assert int(live["step"]) == 3
    with pytest.raises(KeyError, match="differ"):
        ckpt.restore(live, {"step": mx.array(1)})


def test_the_log_drops_lines_from_the_resumed_step_and_a_torn_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps({"step": i}) + "\n" for i in range(5)) + '{"step": 5, "lo')

    log = Log(path)
    log.truncate(3)
    assert [json.loads(x)["step"] for x in path.read_text().splitlines()] == [0, 1, 2]


def test_a_full_disk_is_warned_about(tmp_path, monkeypatch):
    path = write(tmp_path, 1)
    full = shutil.disk_usage(tmp_path)._replace(free=0)
    assert ckpt.disk_note(path) is None
    monkeypatch.setattr(ckpt.shutil, "disk_usage", lambda _: full)
    note = ckpt.disk_note(path)
    assert note is not None
    assert "next save may fail" in note


# ---- the stall field -----------------------------------------------------------------------------


def test_a_gap_above_sixty_seconds_is_a_stall():
    clock = FakeClock()
    stall = StallClock(clock.boot_ns)
    clock.advance(500_000_000)
    assert stall.stall() is None
    clock.sleep_system(60_500_000_000)  # a system sleep: only the boot clock moves
    assert stall.stall() == 60.5
    clock.advance(1_000_000_000)
    stall.mark()  # a save, not a stall
    clock.advance(138_000_000_000)
    assert stall.stall() == 138.0
    clock.advance(500_000_000)
    assert stall.stall() is None


# ---- the pre-decay pin ---------------------------------------------------------------------------


def test_a_pinned_checkpoint_outlives_the_rotation_and_does_not_count(tmp_path):
    ckpt.save(tmp_path, 10, arrays={"dense": arrays(10)}, state={}, keep=1, pin=True)
    for step in (20, 30, 40):
        write(tmp_path, step, keep=1)
    assert [p.name for p in ckpt.checkpoints(tmp_path)] == ["ckpt-00000010", "ckpt-00000040"]
    assert [ckpt.pinned(p) for p in ckpt.checkpoints(tmp_path)] == [True, False]
    assert ckpt.latest(tmp_path) == tmp_path / "ckpt-00000040"


def test_pin_marks_a_complete_checkpoint_in_place_and_refuses_a_torn_one(tmp_path):
    path = write(tmp_path, 10)
    assert not ckpt.pinned(path)  # a checkpoint without the key, as every older one is
    ckpt.pin(path)
    assert ckpt.pinned(path)
    assert ckpt.complete(path)
    assert ckpt.load(path).state == {"step": 10}
    assert not list(path.glob(".tmp-*"))
    torn = write(tmp_path, 20)
    (torn / "dense.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        ckpt.pin(torn)
    assert not ckpt.pinned(torn)
