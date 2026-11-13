"""Kill -9 and resume equals the uninterrupted run, bitwise.

A kill between steps, mid-step, mid-checkpoint and mid-accumulation each resumes to the
uninterrupted run; every kill test prints its plan and the command that reproduces it; and two
runs of one seed are bitwise equal.

Each run is a child process. The baseline and the resumed run are the trainer's entry, run on
MLX's CPU device through `tests.support.cpurun`, and the killed run is the same flags under
`tests.support.killrun`, which trains on the CPU too and SIGKILLs itself at the named point. The
resume is the Mac's: the same command again with the same `--out`. The comparison takes every
array of the newest checkpoint bit for bit, its JSON state (step, skip count, source cursor,
configs, loss window) and the log without its wall-clock fields; and the steps the resume ran
again must reproduce the killed run's own lines for them. Every comparison is bitwise, on MLX's
CPU device.

The kill step is derived from the seed and the test id, and is constrained to land after the
first checkpoint and inside the compared steps; `REPRO_KILL_STEP` overrides it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from akeratos import ckpt
from tests._env import DETERMINISM_ENV
from tests.support.proc import Children
from tests.support.splits import write_split

SEED = 0
STEPS, EVERY, B, T, D = 8, 2, 2, 16, 64
TINY_KEYS = (
    f"d_model={D}",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    "vocab_size=512",
    "window=16",
    "param.kind=sp",
)
# The clock and the allocator vary between runs.
VARY = ("step_s", "tokens_per_s", "stall", "active_bytes", "peak_bytes", "cache_bytes")
FIRST_RUN = (sys.executable, "-m", "tests.support.cpurun")
KILLRUN = (sys.executable, "-m", "tests.support.killrun")
ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_S = 240
# The table on, as on the Mac: 4,096 rows of width 64 in 4 shards (`test_table_resume.py`'s).
TABLE_KEYS = (f"params={4096 * D}", f"max_shard_elements={1024 * D}")

# The newest checkpoint's step, its files byte for byte, its JSON state, and the log.
State = tuple[int, dict[str, bytes], dict[str, Any], list[dict[str, Any]]]


@dataclass(frozen=True)
class Case:
    """One run's shape: the micro-batches per step, the table switch, and the data."""

    accum: int = 1
    table: bool = False
    data: str = "synthetic"  # or a split directory


def flags(out: Path, case: Case) -> list[str]:
    a = ["--data", case.data, "--tokens", str(STEPS * B * T * case.accum), "--rows", str(B)]
    a += ["--seq-len", str(T), "--accum", str(case.accum), "--seed", str(SEED)]
    a += ["--out", str(out), "--compute-dtype", "float32", "--buffer", "8"]
    a += [f"--model={k}" for k in (f"table={str(case.table).lower()}", *TINY_KEYS)]
    a += [f"--table={k}" for k in TABLE_KEYS if case.table]
    a += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    return [*a, "--ckpt-every", str(EVERY), "--print-every", "0"]


def kill_step(nodeid: str) -> int:
    """`1 + sha256(f"{seed}:{nodeid}") % (total_steps - 2)`, folded onto the valid steps.

    A valid step is a checkpoint step after the first checkpoint and before the last step. The
    resume then loads the checkpoint before it and runs again at least one step the killed run
    logged, so the re-run comparison is never empty, and a kill inside a save has a save to land
    in.
    """
    override = os.environ.get("REPRO_KILL_STEP")
    if override:
        return int(override)
    valid = [k for k in range(EVERY + 1, STEPS) if k % EVERY == 0]
    h = 1 + int(hashlib.sha256(f"{SEED}:{nodeid}".encode()).hexdigest(), 16) % (STEPS - 2)
    return valid[h % len(valid)]


def plan(phase: str, step: int, argv: list[str]) -> str:
    """The printed fault plan and the command that reproduces the kill."""
    cmd = shlex.join([*KILLRUN[1:], phase, str(step), "--", *argv])
    return f"KILL PLAN phase={phase} step={step}; reproduce: REPRO_KILL_STEP={step} python {cmd}"


def run(children: Children, env: dict[str, str], argv: list[str], what: str) -> int:
    proc = children.run(list(argv), cwd=ROOT, env=env, timeout=TIMEOUT_S)
    if proc.returncode not in (0, -9):
        raise AssertionError(f"{what} exited {proc.returncode}:\n{proc.stderr[-3000:]}")
    return proc.returncode


def log_lines(out: Path) -> list[dict[str, Any]]:
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines() if x]
    return [{k: v for k, v in x.items() if k not in VARY} for x in lines]


def final_state(out: Path) -> State:
    """The newest checkpoint: every file but `state.json` byte for byte (weights, each group's
    step, rate and moments, the EMA, the table's shards), its parsed state (step, skip count,
    configs, source cursor, loss window, table event), and the log without clock fields."""
    path = ckpt.latest(out)
    assert path is not None, f"no complete checkpoint under {out}"
    files = {
        f.relative_to(path).as_posix(): f.read_bytes()
        for f in sorted(path.rglob("*"))
        if f.is_file() and f.name != ckpt.STATE
    }
    return ckpt.load(path).step, files, ckpt.load(path).state, log_lines(out)


def assert_same(want: State, got: State, note: str) -> None:
    assert want[0] == got[0] == STEPS, note
    assert want[1].keys() == got[1].keys(), note
    assert [k for k in want[1] if want[1][k] != got[1][k]] == [], note
    assert want[2] == got[2], note
    assert [x["step"] for x in got[3]] == list(range(STEPS)), note
    assert want[3] == got[3], note


@pytest.fixture(scope="module")
def repro_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The shared helper's child environment plus `DETERMINISM_ENV`."""
    return Children(tmp_path_factory.getbasetemp()).env(DETERMINISM_ENV)


@pytest.fixture(scope="module")
def baselines(tmp_path_factory: pytest.TempPathFactory, repro_env: dict[str, str]):
    """The uninterrupted run per case, computed fresh once per module."""
    kids = Children(tmp_path_factory.getbasetemp())
    done: dict[Case, State] = {}

    def get(case: Case) -> State:
        if case not in done:
            out = tmp_path_factory.mktemp("whole")
            run(kids, repro_env, [*FIRST_RUN, *flags(out, case)], "the uninterrupted run")
            done[case] = final_state(out)
        return done[case]

    yield get
    kids.close()


def kill_and_resume(
    children: Children, env: dict[str, str], tmp_path: Path, phase: str, step: int, case: Case
) -> tuple[State, str]:
    out = tmp_path / "cut"
    argv = flags(out, case)
    note = plan(phase, step, argv)  # in every assertion message below
    code = run(children, env, [*KILLRUN, phase, str(step), "--", *argv], "the killed run")
    assert code == -9, f"the run was not killed: {note}"
    before = {x["step"]: x for x in log_lines(out)}
    resume_from = ckpt.latest(out)
    assert resume_from is not None, f"no checkpoint to resume from: {note}"
    c = ckpt.load(resume_from).step
    assert c == (step - 1) // EVERY * EVERY, note  # the newest save before the kill
    if phase == "mid-checkpoint":
        assert (out / f"{ckpt.TMP}{ckpt.PREFIX}{step:08d}").is_dir(), note
    run(children, env, [*FIRST_RUN, *argv], "the resumed run")
    got = final_state(out)
    # The steps the resume ran again reproduce the killed run's lines for them, bit for bit.
    again = [x for x in got[3] if x["step"] in before and x["step"] >= c]
    assert again, f"the resume ran no logged step again: {note}"
    assert [before[x["step"]] for x in again] == again, note
    assert not list(out.glob(f"{ckpt.TMP}*")), note
    return got, note


@pytest.fixture(scope="module")
def split(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(write_split(tmp_path_factory.mktemp("data") / "train"))


@pytest.mark.parametrize("phase", ["between-steps", "mid-step", "mid-checkpoint"])
def test_kill9_bitwise(children, repro_env, baselines, tmp_path, request, phase):
    step = kill_step(request.node.nodeid)
    got, note = kill_and_resume(children, repro_env, tmp_path, phase, step, Case())
    assert_same(baselines(Case()), got, note)


def test_kill9_mid_accumulation(children, repro_env, baselines, tmp_path, request):
    case, step = Case(accum=2), kill_step(request.node.nodeid)
    got, note = kill_and_resume(children, repro_env, tmp_path, "mid-accumulation", step, case)
    assert_same(baselines(case), got, note)


@pytest.mark.parametrize("phase", ["between-steps", "mid-checkpoint"])
def test_kill9_table_on_a_split(children, repro_env, baselines, split, tmp_path, request, phase):
    """The Mac's shape: the table on (its shards saved under `table/`) and the packer on a split."""
    case, step = Case(table=True, data=split), kill_step(request.node.nodeid)
    got, note = kill_and_resume(children, repro_env, tmp_path, phase, step, case)
    assert sum(k.startswith("table/shard_") for k in got[1]) == 4, note
    assert got[2]["table_event"] == STEPS, note
    assert_same(baselines(case), got, note)


def test_same_seed_bitwise(children, repro_env, baselines, tmp_path):
    out = tmp_path / "again"
    run(children, repro_env, [*FIRST_RUN, *flags(out, Case())], "the second run")
    assert_same(baselines(Case()), final_state(out), "two runs with the same seed")
