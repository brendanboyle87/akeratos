"""akeratos/speedrun.py, the 5-minute track's driver behind `bake speedrun`.

The slices are the real cutter's output over a fixture store, with the fixture's own pins
(tests/support/speedrun_cut.py). A fake trainer stands in for the train file and moves a
`FakeClock` inside its steps, so the timing rule is checked in exact nanoseconds; the train
file's own loading is checked on small files. The fixed-code check is a stub, except where its
own tests run git on a temporary repository. `bake speedrun`'s tests replace the driver or the
hardware read, so nothing here reads this machine's hardware. Everything runs on MLX's CPU
device.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from numpy.typing import NDArray

from akeratos import bench, cli, digest, macops, speedrun
from akeratos.clock import SystemClock
from akeratos.configs import ModelConfig
from akeratos.data.packer import segment_ids
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout import LN2, HeldOut, score, scoring_config
from akeratos.model import forward, init_params
from akeratos.rng import run_seeds
from akeratos.train.source import PackedSource, StepRows
from tests.fakes.clock import FakeClock
from tests.support.byfile import ROOT
from tests.support.proc import Children
from tests.support.slice_store import VOCAB, write_store
from tests.support.speedrun_cut import COMMIT, HARDWARE, Cut, cut, options

mx.set_default_device(mx.cpu)

T, TARGETS = 32, 200  # the tests' evaluation values: rows of 33 tokens, 200 targets
W, S = speedrun.WARMUP_STEPS, 10**9  # warm-up steps; ns per second
CLAIM = "2026-01-01-BhodiSea"  # the UTC day of a new `FakeClock`, as `options` claims
FOLDER = Path(HARDWARE.key()) / CLAIM / "seed-0"
FIELDS = {
    *("schema", "category", "claim", "kind", "rerun_of", "person", "ai_system", "seed"),
    *("practice", "commit", "metalforge", "mlx", "python", "train_py", "hardware"),
    *("conditions", "peak_bytes", "data", "heldout", "rules", "batch", "steps", "tokens"),
    *("timed_s", "warmup_s", "build_s", "eval_s", "bpb", "utc_start", "utc_end"),
}


@pytest.fixture(autouse=True)
def _tf32_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")  # MLX is loaded here, so `run` only checks it


@pytest.fixture
def slices(tmp_path: Path) -> Cut:
    return cut(tmp_path / "cut", T, TARGETS)


class Clock(FakeClock):
    """A fake clock that notes each monotonic read in `events`."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def monotonic_ns(self) -> int:
        self.events.append("clock")
        return super().monotonic_ns()


class Fake:
    """A trainer whose step k takes `ns(k)` on its clock and returns `loss(k)`; a unigram's logits."""

    def __init__(
        self, ns: Callable[[int], int], loss: Callable[[int], float] = lambda _: 2.0
    ) -> None:
        self.events: list[str] = []
        self.clock = Clock(self.events)
        self.ns, self.loss = ns, loss
        self.progress: list[float] = []
        self.seen: list[StepRows] = []
        self.w = mx.zeros(3)
        self.bias = mx.array(np.linspace(-2.0, 2.0, VOCAB, dtype=np.float32))

    def step(self, rows: StepRows, progress: float) -> float:
        k = len(self.progress)
        self.events.append("step")
        self.progress.append(progress)
        self.seen.append(rows)
        self.clock.advance(self.ns(k))
        self.w = self.w + 1.0
        return self.loss(k)

    def arrays(self) -> object:
        self.events.append("arrays")
        return [self.w]

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        return mx.broadcast_to(self.bias, (*ids.shape, VOCAB))


def warm_then(timed_s: int, warm_s: int = 5) -> Callable[[int], int]:
    """Step times: `warm_s` for the warm-up steps, as compilation would take, then `timed_s`."""
    return lambda k: (warm_s if k < W else timed_s) * S


def use(monkeypatch: pytest.MonkeyPatch, fake: Fake, source: str = "ROWS = 2\n") -> None:
    """Make `fake` the train file's trainer, with batches of 2 rows of `T` tokens."""
    tf = speedrun._TrainFile(source, digest.digest_bytes(source.encode()), 2, T, lambda _: fake)
    monkeypatch.setattr(speedrun, "_load_train", lambda _: tf)


def run(c: Cut, tmp_path: Path, fake: Fake, **changes: object) -> str:
    return speedrun.run(options(c, fake.clock, tmp_path, T, TARGETS, **changes))


def receipt(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "receipt.json").read_text())


# ---- timing ----------------------------------------------------------------------------------


def test_the_warm_up_steps_count_at_the_median_of_the_steps_after_them() -> None:
    times = [int(x) for x in np.random.default_rng(0).integers(1, 10 * S, size=41)]
    timing = speedrun.Timing()
    for k, ns in enumerate(times):
        timing.add(ns)
        timed = times[W : k + 1]
        want = sum(timed) + W * statistics.median(timed) if timed else 0
        assert timing.charged_ns() == want
    assert timing.steps == times


def test_the_run_stops_after_the_step_that_reaches_the_budget(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))  # timed total after timed step j: 7j + 10 x 7
    use(monkeypatch, fake)
    line = run(slices, tmp_path, fake)
    r = receipt(tmp_path / "out" / FOLDER)
    assert (r["steps"], r["timed_s"], r["warmup_s"]) == (W + 33, 301.0, 50.0)  # 7 x 32 + 70 < 300
    assert fake.progress == [0.0] * (W + 1) + [(7 * j + 70) / 300 for j in range(1, 33)]
    path = tmp_path / "out" / FOLDER / "receipt.json"
    assert line == f"43 steps, 301.000 s timed, {r['bpb']:.4f} bits per byte: {path}"


def test_each_step_evaluates_its_arrays_before_the_clock_stops(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    real = speedrun.evaluate
    monkeypatch.setattr(speedrun, "evaluate", lambda *t: (fake.events.append("eval"), real(*t)))
    run(slices, tmp_path, fake)
    steps = [i for i, e in enumerate(fake.events) if e == "step"]
    assert len(steps) == W + 33
    for i in steps:
        assert fake.events[i - 1 : i + 4] == ["clock", "step", "arrays", "eval", "clock"]


def test_build_batch_reads_and_scoring_are_not_timed(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    tf = speedrun._TrainFile("", digest.digest_bytes(b""), 2, T, lambda _: build(fake))
    monkeypatch.setattr(speedrun, "_load_train", lambda _: tf)
    read = PackedSource.next
    monkeypatch.setattr(PackedSource, "next", lambda s: (fake.clock.advance(3 * S), read(s))[1])
    scored = fake.logits
    calls: list[int] = []
    monkeypatch.setattr(fake, "logits", lambda ids: (calls.append(1), slow(fake), scored(ids))[2])
    run(slices, tmp_path, fake)
    r = receipt(tmp_path / "out" / FOLDER)
    assert (r["timed_s"], r["build_s"], r["eval_s"]) == (301.0, 100.0, 50.0 * len(calls))


def build(fake: Fake) -> Fake:
    fake.clock.advance(100 * S)
    return fake


def slow(fake: Fake) -> None:
    fake.clock.advance(50 * S)


def test_a_practice_run_says_so_and_its_progress_is_over_its_own_budget(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(1))  # timed total after timed step j: j + 10 x 1
    use(monkeypatch, fake)
    run(slices, tmp_path, fake, budget_s=30.0)
    r = receipt(tmp_path / "out" / "20260101T000000Z-seed-0")
    assert r["practice"] is True
    assert r["rules"] == {"budget_s": 30.0, "warmup_steps": W, "warmup_rule": "median"}
    assert r["timed_s"] == 30.0
    assert fake.progress == [0.0] * (W + 1) + [(j + W) / 30 for j in range(1, 20)]


@pytest.mark.parametrize(
    ("budget", "folder"),
    [
        (300.0, Path("results/speedrun") / FOLDER),
        (20.0, Path("runs/local/speedrun/20260101T000000Z-seed-0")),
        (400.0, Path("runs/local/speedrun/20260101T000000Z-seed-0")),
    ],
)
def test_the_default_folders(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float, folder: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    fake = Fake(warm_then(10))
    use(monkeypatch, fake)
    speedrun.run(options(slices, fake.clock, tmp_path, T, TARGETS, out=None, budget_s=budget))
    assert (tmp_path / folder / "receipt.json").is_file()


# ---- the receipt and the log ------------------------------------------------------------------


def test_the_receipt_has_every_field(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake, source="ROWS = 2  # the file's text\n")
    reads = iter([{"power": "ac", "thermal_level": 0}, {"power": "battery", "thermal_level": 2}])
    run(slices, tmp_path, fake, conditions=lambda: next(reads))
    r = receipt(tmp_path / "out" / FOLDER)
    assert set(r) == FIELDS
    assert r["schema"] == "metalforge.speedrun-receipt/v1"
    assert (r["category"], r["claim"], r["kind"], r["rerun_of"]) == (
        "no-table",
        CLAIM,
        "record",
        None,
    )
    assert (r["person"], r["ai_system"], r["seed"], r["practice"]) == (
        "BhodiSea",
        "Claude Code",
        0,
        False,
    )
    assert r["commit"] == COMMIT
    assert all(isinstance(r[k], str) and r[k] for k in ("metalforge", "mlx", "python"))
    source = "ROWS = 2  # the file's text\n"
    assert r["train_py"] == {"sha256": digest.digest_bytes(source.encode()), "source": source}
    assert r["hardware"] == HARDWARE.describe()
    assert r["conditions"] == {"power": ["ac", "battery"], "thermal_level": [0, 2]}
    assert isinstance(r["peak_bytes"], int)
    assert r["data"] == slices.pins
    assert r["heldout"] == slices.heldout | {"seq_len": T, "doc_mask": False}
    assert r["rules"] == {"budget_s": 300.0, "warmup_steps": W, "warmup_rule": "median"}
    assert r["batch"] == {"rows": 2, "seq_len": T}
    assert (r["steps"], r["tokens"]) == (43, 43 * 2 * T)
    assert (r["utc_start"], r["utc_end"]) == ("2026-01-01T00:00:00Z", "2026-01-01T00:04:41Z")
    held = HeldOut(slices.path / "heldout", T, TARGETS, doc_mask=False, vocab=VOCAB)
    bias = np.linspace(-2.0, 2.0, VOCAB, dtype=np.float32).astype(np.float64)
    lse = math.log(np.exp(bias).sum())
    nats = sum(float((lse - bias[tg][m > 0]).sum()) for _, tg, m in held.batches(2))
    assert math.isclose(r["bpb"], nats / (LN2 * held.bytes), rel_tol=1e-5)


def test_each_step_writes_one_log_line(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7), loss=lambda k: 3.0 - k / 100)
    use(monkeypatch, fake)
    run(slices, tmp_path, fake)
    lines = [
        json.loads(x)
        for x in (tmp_path / "out" / FOLDER / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(lines) == receipt(tmp_path / "out" / FOLDER)["steps"]
    for k, line in enumerate(lines):
        assert line == {
            "step": k,
            "step_s": 5.0 if k < W else 7.0,
            "warmup": k < W,
            "loss": 3.0 - k / 100,
            "progress": fake.progress[k],
        }


def test_the_run_says_when_it_trains_and_when_it_scores(
    slices: Cut,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    capsys.readouterr()
    run(slices, tmp_path, fake)
    folder = tmp_path / "out" / FOLDER
    steps = receipt(folder)["steps"]
    assert capsys.readouterr().err == (
        "speedrun: training for 300 s of step time, with no output until it ends; "
        f"each step is logged in {folder / 'metrics.jsonl'}\n"
        f"speedrun: {steps} steps trained; scoring the held-out set, which is not timed\n"
    )


def test_a_rerun_names_the_claim_it_reruns(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    other = f"{HARDWARE.key()}/2026-11-08-someone-else"
    run(slices, tmp_path, fake, rerun_of=other)
    r = receipt(tmp_path / "out" / FOLDER)
    assert (r["kind"], r["rerun_of"]) == ("rerun", other)


def test_the_seed_fixes_the_data_order(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    for seed, out in ((0, "a"), (0, "b"), (1, "c")):
        fake = Fake(warm_then(100))
        use(monkeypatch, fake)
        run(slices, tmp_path, fake, seed=seed, out=tmp_path / out)
        seen.append(np.concatenate([rows.ids for rows in fake.seen]))
    assert np.array_equal(seen[0], seen[1])
    assert not np.array_equal(seen[0], seen[2])


# ---- refusals ---------------------------------------------------------------------------------


def _flip(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda c: {"slices": c.pins | {"sha256sums": "sha256:" + "0" * 64}}, "the track pins"),
        (lambda c: {"heldout": c.heldout | {"targets": c.heldout["targets"] + 1}}, "scored set"),
        (lambda c: {"data": c.path.parent / "nowhere"}, "the slices' folder"),
        (lambda c: _flip(c.path / "train" / "000000.bin") or {}, "differs from"),
        (lambda c: _flip(c.path / "heldout" / "000000.bytes") or {}, "differs from"),
    ],
    ids=["sha256sums", "counts", "no-folder", "train-bin", "heldout-bytes"],
)
def test_slices_that_differ_from_their_pins_are_refused(
    slices: Cut,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[Cut], dict[str, object]],
    match: str,
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    with pytest.raises(DataMismatchError, match=match):
        run(slices, tmp_path, fake, **change(slices))
    assert not (tmp_path / "out").exists()
    assert fake.progress == []


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"category": "closed"}, "--category"),
        ({"claim": "BhodiSea"}, "--claim"),
        ({"claim": "2026-11-08-someone-else"}, "--claim"),
        ({"claim": "2026-02-30-BhodiSea"}, "--claim"),
        ({"claim": "2026-11-08-Bhodi--Sea", "person": "Bhodi--Sea"}, "--person"),
        ({"rerun_of": "m3-pro-18c-36gb-macos26-laptop/2026-11-08-someone-else"}, "--rerun-of"),
        ({"rerun_of": f"{HARDWARE.key()}/{CLAIM}"}, "--rerun-of"),
        ({"rerun_of": HARDWARE.key()}, "--rerun-of"),
        ({"rerun_of": f"{HARDWARE.key()}/2026-02-30-someone-else"}, "--rerun-of"),
        ({"rerun_of": f"{HARDWARE.key()}/2026-11-08-some--one"}, "--rerun-of"),
        ({"ai_system": " "}, "--ai-system"),
        ({"seed": -1}, "--seed"),
        ({"budget_s": 0.0}, "--seconds"),
        ({"budget_s": math.inf}, "--seconds"),
        ({"budget_s": 1e300}, "--seconds"),  # finite, but not in nanoseconds
    ],
)
def test_options_a_receipt_cannot_hold_are_refused(
    slices: Cut,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    match: str,
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    with pytest.raises(ConfigError, match=match):
        run(slices, tmp_path, fake, **changes)
    assert not (tmp_path / "out").exists()


def test_a_claim_dated_after_the_runs_utc_day_is_refused_before_it_trains(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    fake.clock.set_utc(datetime(2026, 1, 1, 23, 30, tzinfo=UTC))  # 07:30 on 2026-01-02 in Perth
    use(monkeypatch, fake)
    with pytest.raises(ConfigError, match=r"--claim '2026-01-02-BhodiSea': .*after 2026-01-01"):
        run(slices, tmp_path, fake, claim="2026-01-02-BhodiSea")
    assert fake.progress == []
    assert not (tmp_path / "out").exists()


def test_a_later_seed_may_run_after_its_claims_date(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    fake.clock.set_utc(datetime(2026, 1, 3, tzinfo=UTC))  # the checker judges the claim's date
    use(monkeypatch, fake)
    run(slices, tmp_path, fake, seed=1)
    assert receipt(tmp_path / "out" / HARDWARE.key() / CLAIM / "seed-1")["claim"] == CLAIM


def test_the_handle_pattern_copies_bench_handle() -> None:
    assert speedrun.HANDLE.pattern == bench.HANDLE.pattern  # a copy: speedrun cannot import bench


@pytest.mark.parametrize(
    ("claim", "handle"),
    [
        ("2026-11-08-a", "a"),
        ("2028-02-29-Bhodi-Sea", "Bhodi-Sea"),  # a leap day
        ("2026-11-08-" + "a-" * 19 + "b", "a-" * 19 + "b"),  # 39 characters
        ("2026-11-08-Bhodi--Sea", None),  # two hyphens in a row
        ("2026-11-08--BhodiSea", None),
        ("2026-11-08-BhodiSea-", None),
        ("2026-11-08-" + "a" * 40, None),
        ("2026-11-08-Bhodi_Sea", None),
        ("2026-02-29-BhodiSea", None),  # not a leap year
        ("2026-13-01-BhodiSea", None),
    ],
)
def test_a_claim_is_a_real_date_then_a_github_handle(claim: str, handle: str | None) -> None:
    assert speedrun._handle(claim) == handle


def test_a_folder_that_exists_is_refused(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "out" / FOLDER).mkdir(parents=True)
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    with pytest.raises(ConfigError, match="exists") as refused:
        run(slices, tmp_path, fake)
    assert refused.match("another --seed")
    assert fake.progress == []
    assert not any((tmp_path / "out" / FOLDER).iterdir())


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("SEQ_LEN = 32\ndef build(seed): return seed\n", "must define"),
        ("ROWS = 2\ndef build(seed): return seed\n", "must define"),
        ("ROWS = 2\nSEQ_LEN = 32\n", "must define"),
        ("ROWS = 0\nSEQ_LEN = 32\ndef build(seed): return seed\n", "must define"),
        ("ROWS = True\nSEQ_LEN = 32\ndef build(seed): return seed\n", "must define"),
        ("ROWS = '2'\nSEQ_LEN = 32\ndef build(seed): return seed\n", "must define"),
        ("ROWS = 2\nSEQ_LEN = 32\nbuild = 3\n", "must define"),
        ("ROWS = 2\nSEQ_LEN = = 32\n", "invalid syntax"),
        (None, "No such file"),
    ],
)
def test_a_malformed_train_file_is_refused(
    slices: Cut, tmp_path: Path, text: str | None, match: str
) -> None:
    if text is not None:
        (tmp_path / "train.py").write_text(text)
    with pytest.raises(ConfigError, match=match):
        speedrun.run(options(slices, FakeClock(), tmp_path, T, TARGETS))
    assert not (tmp_path / "out").exists()


def test_a_train_file_that_is_not_python_is_refused(slices: Cut, tmp_path: Path) -> None:
    (tmp_path / "train.txt").write_text("ROWS = 2\n")
    o = options(slices, FakeClock(), tmp_path, T, TARGETS, train=tmp_path / "train.txt")
    with pytest.raises(ConfigError, match="not a Python file"):
        speedrun.run(o)


def test_a_train_file_may_define_a_dataclass(tmp_path: Path) -> None:
    text = "from __future__ import annotations\nfrom dataclasses import dataclass\n\n\n"
    text += "@dataclass\nclass Batch:\n    rows: int = 2\n\n\n"
    text += "ROWS, SEQ_LEN = Batch().rows, 32\n\n\ndef build(seed):\n    return seed\n"
    (tmp_path / "train.py").write_text(text)
    tf = speedrun._load_train(tmp_path / "train.py")
    assert (tf.source, tf.rows, tf.seq_len) == (text, 2, 32)


@pytest.mark.parametrize(
    ("built", "match"),
    [
        (object(), "object lacks step, arrays, logits"),
        (SimpleNamespace(step=len, arrays=list), "SimpleNamespace lacks logits"),
    ],
    ids=["object", "no-logits"],
)
def test_a_build_that_returns_no_trainer_is_refused(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, built: Any, match: str
) -> None:
    tf = speedrun._TrainFile("", digest.digest_bytes(b""), 2, T, lambda _: built)
    monkeypatch.setattr(speedrun, "_load_train", lambda _: tf)
    with pytest.raises(ConfigError, match=f"build\\(seed\\) must return .*; {match}$"):
        speedrun.run(options(slices, FakeClock(), tmp_path, T, TARGETS))
    assert not (tmp_path / "out").exists()


class Peeking(Fake):
    """Logits that read the next id."""

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        nxt = mx.array(np.roll(ids, -1, axis=1).astype(np.float32))
        return super().logits(ids) + nxt[..., None]


class Narrow(Fake):
    """Logits over 512 ids, not the track's 49,152."""

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        return mx.zeros((*ids.shape, 512))


class Diverged(Fake):
    """Logits that are not numbers."""

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        return mx.full((*ids.shape, VOCAB), float("nan"))


@pytest.mark.parametrize(
    ("fake", "match"),
    [
        (Peeking(warm_then(7)), "read later tokens"),
        (Narrow(warm_then(7)), r"logits are \(2, 32, 512\)"),
        (Diverged(warm_then(7)), "the score is not finite"),
        (Fake(warm_then(7), loss=lambda k: math.nan if k == 12 else 2.0), "not finite at step 12"),
        (Fake(lambda _: 0), "ran out at step"),  # no step time: the slice runs out first
    ],
    ids=["peeking", "narrow", "diverged", "loss", "exhausted"],
)
def test_runs_that_cannot_give_a_record_write_no_receipt(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake: Fake, match: str
) -> None:
    use(monkeypatch, fake)
    with pytest.raises(ConfigError, match=match):
        run(slices, tmp_path, fake)
    assert (tmp_path / "out" / FOLDER / "metrics.jsonl").is_file()
    assert not (tmp_path / "out" / FOLDER / "receipt.json").exists()


# ---- the fixed-code check ---------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    env = os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    who = ("-c", "user.name=test", "-c", "user.email=test@example.invalid")
    proc = subprocess.run(
        ["git", "-C", str(repo), *who, *args], capture_output=True, text=True, check=True, env=env
    )
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    root = tmp_path / "repo"
    for rel in ("akeratos/__init__.py", "pyproject.toml", "uv.lock", "speedrun/train.py"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")
    return root


def test_the_fixed_code_check_records_head_and_allows_a_changed_train_file(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD").strip()
    assert speedrun.fixed_commit(repo / "akeratos") == head
    (repo / "speedrun" / "train.py").write_text("ROWS = 1\n")
    assert speedrun.fixed_commit(repo / "akeratos") == head


@pytest.mark.parametrize(
    "rel", ["akeratos/__init__.py", "akeratos/new.py", "pyproject.toml", "uv.lock"]
)
def test_the_fixed_code_check_refuses_uncommitted_fixed_code(repo: Path, rel: str) -> None:
    (repo / rel).write_text("changed\n")
    with pytest.raises(ConfigError, match=rel):
        speedrun.fixed_commit(repo / "akeratos")


def test_the_fixed_code_check_refuses_code_outside_a_checkout(repo: Path, tmp_path: Path) -> None:
    loose = tmp_path / "loose" / "akeratos"
    loose.mkdir(parents=True)
    with pytest.raises(ConfigError, match="not the akeratos/ folder of a git checkout") as got:
        speedrun.fixed_commit(loose)
    assert got.match("git clone")
    with pytest.raises(ConfigError, match="not the akeratos/ folder") as inner:
        speedrun.fixed_commit(repo / "speedrun")  # in a checkout, but not its akeratos/
    assert inner.match("git clone")


# ---- scoring ----------------------------------------------------------------------------------


class PackageModel(Fake):
    """Akeratos's model from the package, untrained, with the FP32 logits of its scoring path."""

    def __init__(self, cfg: ModelConfig, seed: int) -> None:
        super().__init__(warm_then(1))
        self.cfg, self.params = scoring_config(cfg), init_params(cfg, run_seeds(seed)["init_seed"])

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        x = ids.astype(np.int32)
        return forward(self.params, self.cfg, mx.array(x), mx.array(segment_ids(x)), mx.float32)


def test_scoring_equals_the_held_out_scorer_on_an_akeratos_model(tmp_path: Path) -> None:
    """An Akeratos model's logits, scored by the driver, give `heldout.score`'s bits per byte."""
    store = write_store(tmp_path / "store", vocab=512)
    held = HeldOut(store.path / "heldout", 32, None, doc_mask=False, vocab=512)
    assert held.docs == len(store.heldout)  # the whole split, two long documents in windows
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
    trainer = PackageModel(cfg, 0)
    want = held.bpb(score(trainer.params, scoring_config(cfg), held, 2))
    assert speedrun.bpb_of(trainer, held, 2) == want


class Half(Fake):
    """A unigram's logits in BF16."""

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        return super().logits(ids).astype(mx.bfloat16)


def test_bf16_logits_are_scored_in_float32(slices: Cut) -> None:
    held = HeldOut(slices.path / "heldout", T, TARGETS, doc_mask=False, vocab=VOCAB)
    half, widened = Half(warm_then(1)), Fake(warm_then(1))
    widened.bias = half.bias.astype(mx.bfloat16).astype(mx.float32)  # Half's values, in float32
    assert speedrun.bpb_of(half, held, 2) == speedrun.bpb_of(widened, held, 2)


def test_the_run_scores_one_document_per_row(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = Fake(warm_then(7))
    use(monkeypatch, fake)
    seen: list[NDArray[np.int64]] = []
    scored = fake.logits
    monkeypatch.setattr(fake, "logits", lambda ids: (seen.append(ids.copy()), scored(ids))[1])
    run(slices, tmp_path, fake, eval_seq_len=64)
    split = slices.path / "heldout"
    held = HeldOut(split, 64, TARGETS, doc_mask=False, vocab=VOCAB)
    # In rows of 65 tokens, packing would put documents together.
    assert len(HeldOut(split, 64, TARGETS, doc_mask=True, vocab=VOCAB).ids) < len(held.ids)
    want = [ids for ids, _, _ in held.batches(2)]
    assert len(seen) == 2 + len(want)  # the causality check's two calls, then the score's
    assert all(np.array_equal(a, b) for a, b in zip(seen[2:], want, strict=True))


# ---- bake speedrun ----------------------------------------------------------------------------

RERUN = f"{HARDWARE.key()}/2026-11-01-someone-else"
EVERY_FLAG = ["--train", "t.py", "--seed", "3", "--out", "o", "--category", "no-table"]
EVERY_FLAG += ["--rerun-of", RERUN, "--seconds", "60"]


@pytest.mark.parametrize(
    ("flags", "changes"),
    [
        ([], {}),
        (
            EVERY_FLAG,
            {
                "train": Path("t.py"),
                "seed": 3,
                "out": Path("o"),
                "category": "no-table",
                "rerun_of": RERUN,
                "budget_s": 60.0,
            },
        ),
    ],
    ids=["defaults", "every-flag"],
)
def test_bake_speedrun_passes_its_flags_to_the_driver(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    changes: dict[str, object],
) -> None:
    seen: list[speedrun.Options] = []
    monkeypatch.setattr(macops, "read_hardware", lambda: HARDWARE)
    monkeypatch.setattr(speedrun, "run", lambda o: (seen.append(o), "the summary line")[1])
    argv = ["speedrun", "--data", "d", "--claim", CLAIM, "--person", "BhodiSea"]
    assert cli.main([*argv, "--ai-system", "Claude Code", *flags]) == 0
    assert capsys.readouterr().out == "the summary line\n"
    (o,) = seen
    assert isinstance(o.clock, SystemClock)
    defaults = speedrun.Options(
        data=Path("d"),
        train=Path("speedrun/train.py"),
        seed=0,
        out=None,
        category="open",
        claim=CLAIM,
        person="BhodiSea",
        ai_system="Claude Code",
        rerun_of=None,
        hardware=HARDWARE,
        conditions=macops.read_conditions,
        clock=o.clock,
        budget_s=speedrun.BUDGET_S,
        eval_seq_len=speedrun.EVAL_SEQ_LEN,
        eval_rows=speedrun.EVAL_ROWS,
        eval_targets=speedrun.EVAL_TARGETS,
        slices=speedrun.SLICES,
        heldout=speedrun.HELDOUT,
        fixed_code=speedrun.fixed_commit,
    )
    assert o == replace(defaults, **changes)  # every field, the track's constants included


def test_bake_speedrun_exits_22_on_other_slices_and_78_on_a_malformed_claim(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(macops, "read_hardware", lambda: HARDWARE)
    argv = ["speedrun", "--data", str(slices.path), "--person", "BhodiSea", "--ai-system", "none"]
    argv += ["--out", str(tmp_path / "out")]
    assert cli.main([*argv, "--claim", CLAIM]) == 22  # a fixture's slices: not the track's pins
    assert capsys.readouterr().err.startswith("data error: ")
    assert cli.main([*argv, "--claim", "2026-11-08"]) == 78
    assert capsys.readouterr().err.startswith("configuration error: --claim")
    assert not (tmp_path / "out").exists()


def test_bake_speedrun_exits_78_on_a_claim_dated_after_today_before_reading_the_slices(
    slices: Cut, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(macops, "read_hardware", lambda: HARDWARE)
    claim = f"{datetime.now(UTC) + timedelta(days=2):%Y-%m-%d}-BhodiSea"
    argv = ["speedrun", "--data", str(slices.path), "--person", "BhodiSea", "--ai-system", "none"]
    argv += ["--out", str(tmp_path / "out"), "--claim", claim]
    assert cli.main(argv) == 78  # before the fixture's slices are read
    assert capsys.readouterr().err.startswith(f"configuration error: --claim '{claim}'")
    assert not (tmp_path / "out").exists()


def test_bake_speedrun_help_loads_no_mlx(children: Children) -> None:
    command = [sys.executable, "-X", "importtime", "-m", "akeratos", "speedrun", "--help"]
    proc = children.run(command, cwd=ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("usage: bake speedrun")
    assert " mlx" not in proc.stderr
