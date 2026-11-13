"""The laptop run guard in `akeratos/macops.py`.

The rails run on `Guard` alone, with times in nanoseconds and no clock at all; the runner runs
on `tests/fakes/clock.py`'s `FakeClock` with fake probes and fake children, so nothing sleeps
and no trainer starts. The probe parsers read output captured on an M4 Max MacBook Pro, and
the checkpoints the disk rails size are written by `akeratos.ckpt` itself, on MLX's CPU
device. One test launches a real child, a stub that saves on SIGTERM, through the shared
`children` helper, and one reads this machine's probes when it is a Mac.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlx.core as mx
import pytest

from akeratos import ckpt, macops
from akeratos.clock import SystemClock
from akeratos.macops import Act, Guard, Readings, Runner
from tests.fakes.clock import FakeClock

mx.set_default_device(mx.cpu)
ROOT = Path(__file__).resolve().parents[2]
S = macops.S
GB = 10**9
CKPT = 17 * GB
SAFE = Readings(
    power="ac",
    charge=90,
    thermal=0,
    thermal_state=0,
    pressure=1,
    swap_used=0,
    free_disk=1000 * GB,
    display="asleep",
)
BATT = SAFE._replace(power="battery")

# Captured on an M4 Max MacBook Pro on 2026-10-21.
PMSET_AC = (
    "Now drawing from 'AC Power'\n"
    " -InternalBattery-0 (id=22544483)\t100%; finishing charge; 0:00 remaining present: true\n"
)
PMSET_BATTERY = (
    "Now drawing from 'Battery Power'\n"
    " -InternalBattery-0 (id=22544483)\t85%; discharging; 7:08 remaining present: true\n"
)
SWAPUSAGE = "total = 2048.00M  used = 729.81M  free = 1318.19M  (encrypted)\n"


def polls(guard: Guard, r: Readings, start_s: int, end_s: int, every_s: int = 10) -> list[Act]:
    """Every action from polling `guard` with `r` each `every_s` s from `start_s` to `end_s`."""
    acts: list[Act] = []
    for t in range(start_s, end_s + 1, every_s):
        acts += guard.poll(r, t * S)
    return acts


def kinds(acts: list[Act]) -> list[str]:
    return [a.kind for a in acts]


def stopped(why: str) -> list[Act]:
    """A permanent stop's actions: the one SIGTERM, then STOP."""
    return [Act("term", why), Act("stop", why)]


class FakeProbes:
    """Readings from `script`, a function of the fake clock's seconds."""

    def __init__(self, script, clock: FakeClock | None = None) -> None:
        self.script, self.clock = script, clock

    def read(self, out: Path) -> Readings:
        del out
        return self.script(0 if self.clock is None else self.clock.boot_ns() // S)


class FakeChild:
    """A child that exits `code` on its `exit_after`-th wait, or `on_term` after a SIGTERM."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        on_term: int | None = 75,
        exit_after: int | None = None,
        code: int = 0,
        during_wait=None,
    ) -> None:
        self.clock, self.on_term, self.exit_after, self.code = clock, on_term, exit_after, code
        self.last_output_ns = clock.boot_ns()
        self.terms = self.waits = 0
        self.exit: int | None = None
        self.during_wait = during_wait

    def wait(self, timeout_s: float) -> int | None:
        self.waits += 1
        self.clock.advance(int(timeout_s * S))
        if self.during_wait is not None:
            self.during_wait(self)
        if self.exit_after is not None and self.waits >= self.exit_after:
            return self.code
        return self.exit

    def terminate(self) -> None:
        self.terms += 1
        if self.on_term is not None:
            self.exit = self.on_term


class Launcher:
    """Hands out the given children in turn and records each command."""

    def __init__(self, *children: FakeChild) -> None:
        self.children, self.argvs = list(children), []

    def __call__(self, argv):
        self.argvs.append(list(argv))
        return self.children.pop(0)


class FakeAwake:
    """Stands in for caffeinate's launch: records each command, or fails like a missing binary."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail, self.argvs, self.started = fail, [], []

    def __call__(self, argv):
        if self.fail:
            raise FileNotFoundError(2, "No such file or directory", "caffeinate")
        self.argvs.append(list(argv))
        self.started.append(FakeChild(FakeClock(), on_term=None))
        return self.started[-1]


def runner(tmp_path: Path, clock: FakeClock, script, *children: FakeChild, **kw) -> Runner:
    launch = kw.pop("launch", Launcher(*children))
    guard = kw.pop("guard", Guard(CKPT))
    kw.setdefault("awake", FakeAwake())  # no real caffeinate from a unit test
    return Runner(
        ["trainer", "--out", str(tmp_path)],
        tmp_path,
        guard,
        FakeProbes(script, clock),
        clock,
        launch=launch,
        pause=lambda s: clock.advance(int(s * S)),
        **kw,
    )


# ------------------------------------------------------------------ parsers and probes


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (PMSET_AC, ("ac", 100)),
        (PMSET_BATTERY, ("battery", 85)),
        ("Now drawing from 'AC Power'\n", ("ac", None)),  # a Mac with no battery
        ("Now drawing from 'UPS Power'\n", ("battery", None)),
        ("pmset: garbled\n", (None, None)),
        (None, (None, None)),
    ],
    ids=["ac", "battery", "desktop", "ups", "garbled", "failed"],
)
def test_parse_batt(text, want) -> None:
    assert macops.parse_batt(text) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [("1\n", 1), ("2\n", 2), ("4\n", 4), ("3\n", None), ("", None), ("warn", None), (None, None)],
)
def test_parse_pressure(text, want) -> None:
    assert macops.parse_pressure(text) == want


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (SWAPUSAGE, int(729.81 * 2**20)),
        ("total = 4096.00M  used = 1.50G  free = 2.50G  (encrypted)\n", int(1.5 * 2**30)),
        ("total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)\n", 0),
        ("vm.swapusage: no such key\n", None),
        (None, None),
    ],
    ids=["megabytes", "gigabytes", "none-used", "garbled", "failed"],
)
def test_parse_swap(text, want) -> None:
    assert macops.parse_swap(text) == want


def test_system_probes_parse_the_mac_commands(tmp_path: Path) -> None:
    out = {macops.PMSET_BATT: PMSET_BATTERY, macops.PRESSURE: "2\n", macops.SWAPUSAGE: SWAPUSAGE}
    r = macops.SystemProbes(lambda argv: out[tuple(argv)], mac=True).read(tmp_path / "run")
    assert (r.power, r.charge, r.pressure, r.swap_used) == ("battery", 85, 2, int(729.81 * 2**20))
    assert r.free_disk is not None
    assert r.free_disk > 0


def test_failed_reads_are_unknown_never_raised(tmp_path: Path) -> None:
    r = macops.SystemProbes(lambda argv: None, mac=True).read(tmp_path)
    assert (r.power, r.charge, r.pressure, r.swap_used) == (None, None, None, None)
    line = macops.status(r)
    assert "power=unknown charge=unknown" in line
    assert "None" not in line
    assert macops.run_command(["/nonexistent/akeratos-probe"]) is None
    assert macops.free_disk(Path("/nonexistent/a/b")) is not None  # "/" still has a filesystem


def test_off_a_mac_only_the_disk_reads(tmp_path: Path) -> None:
    asked: list[object] = []
    r = macops.SystemProbes(asked.append, mac=False).read(tmp_path / "not" / "yet")
    assert asked == []
    assert r.free_disk is not None
    assert r == Readings(free_disk=r.free_disk)


@pytest.mark.skipif(sys.platform != "darwin", reason="reads this Mac's own probes")
def test_darwin_probes_read_this_mac(tmp_path: Path) -> None:
    r = macops.SystemProbes().read(tmp_path)
    assert r.power in ("ac", "battery")
    assert r.charge is None or 0 <= r.charge <= 100
    assert r.thermal in (0, 1, 2, 3, 4)
    assert r.thermal_state in (None, 0, 1, 2, 3)
    assert r.pressure in (1, 2, 4)
    assert r.swap_used is not None
    assert r.swap_used >= 0
    assert r.display in (None, "awake", "asleep")


def test_newest_complete_checkpoint_sizes_the_disk_rails(tmp_path: Path) -> None:
    assert macops.newest_ckpt_bytes(tmp_path / "missing") is None
    good = ckpt.save(tmp_path, 10, arrays={"dense": {"w": mx.zeros((64,))}}, state={})
    torn = ckpt.save(tmp_path, 20, arrays={"dense": {"w": mx.zeros((64,))}}, state={})
    (torn / "dense.safetensors").write_bytes(b"short")
    (tmp_path / "ckpt-00000030").mkdir()  # no state.json
    (tmp_path / ".tmp-ckpt-00000040").mkdir()
    other = tmp_path / "ckpt-00000050"
    other.mkdir()
    (other / "state.json").write_text(json.dumps({"format": 2, "step": 50, "files": {}}))
    want = sum(p.stat().st_size for p in good.iterdir())
    assert macops.newest_ckpt_bytes(tmp_path) == want


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"power": "battery"}, "on battery power"),
        ({"power": None}, "power source unknown"),
        ({"thermal": 3}, "thermal pressure level 3"),
        ({"thermal": 4}, "thermal pressure level 4"),
        ({"thermal": None}, "thermal level unknown"),
        ({"pressure": 4}, "memory pressure critical"),
        ({"pressure": None}, "memory pressure unknown"),
        ({"free_disk": 3 * CKPT - 1}, "below 3 checkpoints of 17.0 GB"),
        ({"free_disk": None}, "free disk unknown"),
        ({"charge": 20}, "charge 20%, 20% or below"),
    ],
    ids=str,
)
def test_refusals_on_readings(change, why) -> None:
    assert why in (Guard(CKPT).refusal(SAFE._replace(**change)) or "")


def test_safe_readings_pass_and_allow_unknown_passes_only_unknowns() -> None:
    assert Guard(CKPT).refusal(SAFE) is None
    assert Guard(CKPT).refusal(SAFE._replace(free_disk=3 * CKPT, charge=21, thermal=2)) is None
    blank = Readings(free_disk=1000 * GB)  # off a Mac: every Mac probe unknown
    loose = Guard(CKPT, allow_unknown=True)
    assert loose.refusal(blank) is None
    assert loose.refusal(blank._replace(power="battery")) == "on battery power"


@pytest.mark.parametrize(
    ("trainer", "why"),
    [
        ([], "no --out among the trainer's flags"),
        (["--data", "synthetic", "--tokens", "1e6"], "no --out among the trainer's flags"),
        (["--out", "{out}", "--ckpt-every", "0"], "--ckpt-every 0 turns checkpoints off"),
        (["--out={out}", "--ckpt-every=0"], "--ckpt-every 0 turns checkpoints off"),
        (["--ckpt", "0", "--out", "{out}"], "--ckpt-every 0 turns checkpoints off"),  # a prefix
        (["--out", "{out}", "--ckpt-every", "x"], "the trainer's flags: argument --ckpt-every"),
        (["--out"], "the trainer's flags: argument --out"),
    ],
    ids=[
        "empty",
        "no-out",
        "ckpt-off",
        "ckpt-off-equals",
        "ckpt-off-prefix",
        "bad-int",
        "no-value",
    ],
)
def test_refuses_the_trainers_flags(tmp_path: Path, capsys, trainer, why) -> None:
    out = tmp_path / "run"
    argv = ["--poll", "5", "--", *(a.format(out=out) for a in trainer)]
    assert macops.main(argv, probes=FakeProbes(lambda t: SAFE)) == macops.EXIT_CONFIG
    assert f"guard: refusing to start: {why}" in capsys.readouterr().err
    assert not out.exists()


def test_refuses_while_stop_exists(tmp_path: Path, capsys) -> None:
    (tmp_path / "STOP").write_text("2026-10-21T00:00:00.000Z thermal pressure level 4\n")
    code = macops.main(["--", "--out", str(tmp_path)], probes=FakeProbes(lambda t: SAFE))
    assert code == macops.EXIT_CONFIG
    assert "STOP exists: a permanent stop; remove it to start again" in capsys.readouterr().err


def test_refuses_on_readings_and_launches_nothing(tmp_path: Path, capsys) -> None:
    out = tmp_path / "run"
    code = macops.main(["--", "--out", str(out)], probes=FakeProbes(lambda t: BATT))
    assert code == macops.EXIT_CONFIG
    assert "guard: refusing to start: on battery power" in capsys.readouterr().err
    assert not out.exists()


def test_the_start_disk_check_uses_the_newest_checkpoint(tmp_path: Path, capsys) -> None:
    # 40 GB free is under 3 checkpoints of 17 GB, but over 3 of a saved 1 KB one; the charge
    # refusal, checked after the disk, shows the disk passed.
    low = SAFE._replace(free_disk=40 * GB, charge=15)
    argv = ["--", "--out", str(tmp_path)]
    assert macops.main(argv, probes=FakeProbes(lambda t: low)) == macops.EXIT_CONFIG
    assert "40.0 GB free, below 3 checkpoints of 17.0 GB" in capsys.readouterr().err
    ckpt.save(tmp_path, 10, arrays={"dense": {"w": mx.zeros((8,))}}, state={})
    assert macops.main(argv, probes=FakeProbes(lambda t: low)) == macops.EXIT_CONFIG
    assert "charge 15%, 20% or below" in capsys.readouterr().err


@pytest.mark.parametrize("flag", [["--poll", "0"], ["--ckpt-gb", "-1"], ["--cache-limit-gb", "0"]])
def test_bad_guard_flags_are_a_usage_error(tmp_path: Path, flag: list[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        macops.main(
            [*flag, "--", "--out", str(tmp_path / "run")], probes=FakeProbes(lambda t: SAFE)
        )
    assert stop.value.code == 2
    assert not (tmp_path / "run").exists()  # refused before anything launched


def test_main_launches_under_the_guard_and_maps_a_signal_death(tmp_path, monkeypatch) -> None:
    seen: list[Runner] = []

    def guarded(self: Runner) -> int:
        seen.append(self)
        return -9

    monkeypatch.setattr(Runner, "guarded", guarded)
    out = tmp_path / "run"
    argv = ["--allow-unknown", "--", "--out", str(out)]
    assert macops.main(argv, probes=FakeProbes(lambda t: Readings(free_disk=1000 * GB))) == 137
    assert out.is_dir()
    assert seen[0].guard.allow_unknown


def test_main_hands_the_cache_cap_to_the_child_command(tmp_path, monkeypatch) -> None:
    seen: list[Runner] = []
    monkeypatch.setattr(Runner, "guarded", lambda self: seen.append(self) or 0)
    argv = ["--cache-limit-gb", "3.81", "--", "--out", str(tmp_path)]
    assert macops.main(argv, probes=FakeProbes(lambda t: SAFE)) == 0
    assert seen[0]._argv[-6:] == [
        "-c",
        macops.RUN_TRAINER,
        "--cache-limit-gb",
        "3.81",
        "--out",
        str(tmp_path),
    ]


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_the_child_is_the_trainer_itself_on_every_platform(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    argv = macops.child_argv(["--out", "runs/a"])
    assert argv == [sys.executable, "-m", "akeratos.train.entry", "--out", "runs/a"]


def test_with_a_cache_cap_the_child_runs_the_trainer_through_the_cli(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    trainer = ["--out", "runs/a"]
    argv = macops.child_argv(trainer, 3.81)
    assert argv[-7:] == [
        sys.executable,
        "-c",
        macops.RUN_TRAINER,
        "--cache-limit-gb",
        "3.81",
        *trainer,
    ]
    assert "caffeinate" not in argv  # caffeinate runs beside the trainer, not around it
    assert macops.child_argv(trainer, None) == macops.child_argv(trainer)  # no cache limit
    assert "akeratos.train" not in macops.RUN_TRAINER  # macops reaches the trainer through the CLI


def test_apply_cache_limit_sets_the_cache_and_never_the_wired_limit(monkeypatch, capsys) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: calls.append(("cache", n)) or 0)
    monkeypatch.setattr(mx, "set_wired_limit", lambda n: calls.append(("wired", n)) or 0)
    macops.apply_cache_limit(None)
    assert calls == []
    assert capsys.readouterr().err == ""
    n = int(3.81 * 2**30)
    macops.apply_cache_limit(n)
    assert calls == [("cache", n)]
    assert capsys.readouterr().err == f"macops: MLX cache limit 3.81 GB ({n:,} bytes)\n"


# ------------------------------------------------------------------ the rails


def test_battery_for_60s_stops_once_and_relaunches_after_60s_of_ac() -> None:
    g = Guard(CKPT)
    assert polls(g, BATT, 0, 50) == []
    assert g.poll(BATT, 60 * S) == [Act("term", "on battery for 60 s")]
    assert g.state == "stopping"
    assert polls(g, BATT, 70, 300) == []  # one SIGTERM per child
    assert kinds(g.exited(75, 310 * S)) == ["log"]
    assert g.state == "waiting"
    assert polls(g, BATT, 320, 400) == []
    assert polls(g, SAFE, 410, 460) == []  # 50 s of AC
    assert g.poll(SAFE, 470 * S) == [Act("relaunch", "AC has held for 60 s")]
    assert g.state == "running"
    assert polls(g, SAFE, 480, 2000) == []


def test_power_unknown_counts_as_battery_unless_allowed() -> None:
    blank = SAFE._replace(power=None)
    g = Guard(CKPT)
    assert polls(g, BATT, 0, 30) + polls(g, blank, 40, 50) == []
    assert g.poll(blank, 60 * S) == [Act("term", "the power source unknown for 60 s")]
    assert polls(Guard(CKPT, allow_unknown=True), blank, 0, 3600) == []


def test_a_flap_under_60s_stops_nothing() -> None:
    g = Guard(CKPT)
    assert polls(g, BATT, 0, 50) == []
    assert g.poll(SAFE, 55 * S) == []  # back on AC: the 60 s start again
    assert polls(g, BATT, 60, 110) == []
    assert g.poll(BATT, 120 * S) == [Act("term", "on battery for 60 s")]


def test_ac_must_hold_60s_before_a_relaunch() -> None:
    g = Guard(CKPT)
    polls(g, BATT, 0, 60)
    g.exited(75, 70 * S)
    assert polls(g, SAFE, 80, 130) + polls(g, BATT, 135, 135) == []  # 55 s of AC, then a flap
    assert polls(g, SAFE, 140, 190) == []
    assert kinds(g.poll(SAFE, 200 * S)) == ["relaunch"]


def test_thermal_level_4_stops_for_good_at_once() -> None:
    g = Guard(CKPT)
    why = "thermal pressure level 4 (sleeping)"
    assert g.poll(SAFE._replace(thermal=4), 0) == stopped(why)
    assert g.exited(75, 30 * S) == [Act("exit", code=75)]


def test_thermal_level_3_stops_after_10_min_held() -> None:
    hot, warm = SAFE._replace(thermal=3), SAFE._replace(thermal=2)
    g = Guard(CKPT)
    assert polls(g, hot, 0, 300) + polls(g, warm, 310, 310) + polls(g, hot, 320, 910) == []
    assert g.poll(hot, 920 * S) == stopped("thermal pressure level 3 (trapping) for 10 min")


def test_memory_pressure_critical_on_2_consecutive_polls() -> None:
    crit = SAFE._replace(pressure=4)
    g = Guard(CKPT)
    assert g.poll(crit, 0) + g.poll(SAFE, 10 * S) + g.poll(crit, 20 * S) == []
    assert g.poll(crit, 30 * S) == stopped("memory pressure critical on 2 consecutive polls")


@pytest.mark.parametrize(
    ("field", "name"),
    [("thermal", "thermal level"), ("pressure", "memory pressure"), ("free_disk", "free disk")],
)
def test_unknown_for_10_min_stops_after_a_warning(field: str, name: str) -> None:
    blind = SAFE._replace(**{field: None})
    g = Guard(CKPT)
    assert g.poll(blind, 0) == [Act("log", f"warning: the {name} reads unknown")]
    assert polls(g, blind, 10, 590) == []
    assert g.poll(blind, 600 * S) == stopped(f"{name} unknown for 10 min")
    loose = Guard(CKPT, allow_unknown=True)
    assert kinds(polls(loose, blind, 0, 7200)) == ["log"]  # warned, never stopped


def test_free_disk_below_2_checkpoints_stops_for_good() -> None:
    g = Guard(CKPT)
    assert g.poll(SAFE._replace(free_disk=2 * CKPT), 0) == []
    why = "34.0 GB free, below 2 checkpoints of 17.0 GB"
    assert g.poll(SAFE._replace(free_disk=2 * CKPT - 1), 10 * S) == stopped(why)
    small = Guard(CKPT)
    small.ckpt_bytes = GB  # the runner sets it from the newest checkpoint
    assert small.poll(SAFE._replace(free_disk=10 * GB), 0) == []


def test_a_charge_of_20_percent_stops_for_good_even_on_battery() -> None:
    g = Guard(CKPT)
    assert g.poll(SAFE._replace(charge=21), 0) == []
    assert g.poll(BATT._replace(charge=20), 10 * S) == stopped("charge 20%, 20% or below")
    assert g.exited(75, 20 * S) == [Act("exit", code=75)]  # no wait for AC after a permanent stop


def test_ac_drain_warns_and_marks_perf_invalid_once() -> None:
    g = Guard(CKPT)
    charges = [(0, 90), (120, 89), (240, 88)]
    assert [a for t, c in charges for a in g.poll(SAFE._replace(charge=c), t * S)] == []
    assert g.poll(SAFE._replace(charge=87), 360 * S) == [
        Act("log", "warning: on AC, the charge fell 3 points within 10 min"),
        Act("log", "perf-invalid: the charge fell on AC"),
    ]
    assert g.perf_invalid
    assert g.poll(SAFE._replace(charge=86), 480 * S) == []
    slow = Guard(CKPT)  # 3 points over 11 min is not a drain
    assert (
        slow.poll(SAFE._replace(charge=90), 0) + slow.poll(SAFE._replace(charge=87), 660 * S) == []
    )


def test_warn_only_rails_warn_once_while_they_hold_and_never_stop() -> None:
    noisy = SAFE._replace(pressure=2, swap_used=(1 << 30) + 1, display="awake")
    g = Guard(CKPT)
    assert [a.text for a in g.poll(noisy, 0)] == [
        "warning: memory pressure at warn (2)",
        "warning: swap in use 1.1 GB, over 1 GB",
        "warning: the display is awake",
    ]
    assert polls(g, noisy, 10, 7200) == []
    assert g.poll(SAFE, 7210 * S) == []
    assert kinds(g.poll(noisy, 7220 * S)) == ["log", "log", "log"]


def test_one_sigterm_per_child_and_a_later_permanent_stop_only_writes_stop() -> None:
    g = Guard(CKPT)
    polls(g, BATT, 0, 50)
    assert kinds(g.poll(BATT, 60 * S)) == ["term"]
    why = "thermal pressure level 4 (sleeping)"
    assert g.poll(BATT._replace(thermal=4), 70 * S) == [Act("stop", why)]
    assert g.poll(BATT._replace(thermal=4), 80 * S) == []
    assert g.exited(75, 90 * S) == [Act("exit", code=75)]


def test_a_relaunched_child_gets_its_own_one_sigterm() -> None:
    g = Guard(CKPT)
    assert kinds(polls(g, BATT, 0, 60)) == ["term"]
    assert kinds(g.exited(75, 65 * S) + polls(g, SAFE, 70, 130)) == ["log", "relaunch"]
    assert kinds(polls(g, BATT, 140, 400)) == ["term"]


@pytest.mark.parametrize("code", [-15, 1, 2])
def test_a_battery_stop_waits_for_ac_whatever_the_code(code: int) -> None:
    # A SIGTERM during the trainer's start-up, before `drive` installs its handler, ends it with
    # -15 or 1, not 75: still a battery stop, not a failure.
    g = Guard(CKPT)
    polls(g, BATT, 0, 60)
    why = f"the trainer exited {code} after a battery stop; relaunch on AC"
    assert g.exited(code, 70 * S) == [Act("log", why)]
    assert g.state == "waiting"
    assert polls(g, BATT, 80, 400) + polls(g, SAFE, 410, 460) == []
    assert kinds(g.poll(SAFE, 470 * S)) == ["relaunch"]
    for i in range(3):  # no relaunch was spent on it: the cap still allows 3 failures
        assert kinds(g.exited(1, (480 + i) * S)) == ["relaunch"]
    assert kinds(g.exited(1, 490 * S)) == ["stop", "exit"]


def test_no_exit_20_min_after_the_sigterm_is_logged_and_never_killed() -> None:
    hot = SAFE._replace(thermal=4)
    g = Guard(CKPT)
    assert kinds(g.poll(hot, 0)) == ["term", "stop"]
    assert polls(g, hot, 10, 1190) == []
    late = "warning: no exit 20 min after the SIGTERM; still waiting (never SIGKILL)"
    assert g.poll(hot, 1200 * S) == [Act("log", late)]
    assert polls(g, hot, 1210, 7200) == []


@pytest.mark.parametrize("code", [0, 3, 22, 78, 75, 2])
def test_exit_codes_pass_through(code: int) -> None:
    assert Guard(CKPT).exited(code, 0) == [Act("exit", code=code)]


@pytest.mark.parametrize("code", [0, 3, 22, 78])
def test_final_codes_end_the_guard_even_after_a_battery_stop(code: int) -> None:
    g = Guard(CKPT)
    polls(g, BATT, 0, 60)
    assert g.exited(code, 70 * S) == [Act("exit", code=code)]


def test_failures_relaunch_at_most_3_times_in_any_24_h() -> None:
    g = Guard(CKPT)
    hour = 3600 * S
    for i, code in enumerate((1, -9, 1)):
        assert g.exited(code, i * hour) == [
            Act("relaunch", f"the trainer failed ({code}): relaunch {i + 1} of 3 in 24 h")
        ]
    why = "the trainer failed (-11) after 3 relaunches in 24 h (--max-relaunch 3)"
    assert g.exited(-11, 3 * hour) == [Act("stop", why), Act("exit", code=-11)]
    assert kinds(g.exited(1, 24 * hour + 1)) == ["relaunch"]  # the first has aged out
    assert kinds(Guard(CKPT, max_relaunch=0).exited(1, 0)) == ["stop", "exit"]


def test_the_guards_own_signal_sends_one_sigterm_and_returns_the_childs_code() -> None:
    g = Guard(CKPT)
    assert g.signalled(0) == [Act("term", "the guard was signalled")]
    assert g.signalled(10 * S) == []
    assert polls(g, BATT, 20, 300) == []
    assert g.exited(75, 310 * S) == [Act("exit", code=75)]  # no wait for AC
    waiting = Guard(CKPT)
    polls(waiting, BATT, 0, 60)
    waiting.exited(75, 70 * S)
    assert waiting.signalled(80 * S) == [Act("exit", code=75)]


def test_a_permanent_rail_while_waiting_for_ac_writes_stop_and_exits_75() -> None:
    g = Guard(CKPT)
    polls(g, BATT, 0, 60)
    g.exited(75, 70 * S)
    assert g.poll(BATT._replace(charge=20), 80 * S) == [
        Act("stop", "charge 20%, 20% or below"),
        Act("exit", code=75),
    ]


# ------------------------------------------------------------------ the runner


def test_runner_battery_stop_then_relaunch_on_ac(tmp_path: Path) -> None:
    clock = FakeClock()
    first, second = FakeChild(clock), FakeChild(clock, exit_after=3, code=0)
    launch = Launcher(first, second)
    r = runner(tmp_path, clock, lambda t: BATT if 100 <= t < 300 else SAFE, launch=launch)
    assert r.run() == 0
    assert (first.terms, second.terms) == (1, 0)
    assert launch.argvs == [["trainer", "--out", str(tmp_path)]] * 2
    log = (tmp_path / "guard.log").read_text()
    assert "SIGTERM to the trainer: on battery for 60 s" in log
    assert "the trainer exited 75" in log
    assert "state=waiting" in log
    assert "relaunch: AC has held for 60 s" in log
    assert log.rstrip().endswith("exit 0")
    assert not (tmp_path / "STOP").exists()


def test_runner_permanent_stop_writes_stop_with_utc_time_and_reason(tmp_path: Path) -> None:
    clock = FakeClock()
    clock.set_utc(datetime(2026, 10, 21, 12, 0, tzinfo=UTC))
    child = FakeChild(clock)
    r = runner(tmp_path, clock, lambda t: SAFE._replace(thermal=4) if t >= 30 else SAFE, child)
    assert r.run() == 75
    assert child.terms == 1
    stop = (tmp_path / "STOP").read_text()
    assert stop == "2026-10-21T12:00:30.000Z thermal pressure level 4 (sleeping)\n"
    lines = (tmp_path / "guard.log").read_text().splitlines()
    assert lines[1] == (
        "2026-10-21T12:00:10.000Z poll power=ac charge=90 thermal=0 thermal_state=0 pressure=1 "
        f"swap_used=0 free_disk={1000 * GB} display=asleep quiet_s=10 state=running"
    )


def test_quiet_seconds_never_go_negative(tmp_path: Path) -> None:
    # The copier thread may stamp a line just after the runner read the clock.
    clock = FakeClock()

    def late_line(child: FakeChild) -> None:
        child.last_output_ns = clock.boot_ns() + S

    child = FakeChild(clock, exit_after=2, code=0, during_wait=late_line)
    assert runner(tmp_path, clock, lambda t: SAFE, child).run() == 0
    assert "quiet_s=0 state=running" in (tmp_path / "guard.log").read_text()


def test_runner_crash_cap_writes_stop_and_exits_with_the_childs_code(tmp_path: Path) -> None:
    clock = FakeClock()
    children = [FakeChild(clock, exit_after=1, code=1) for _ in range(4)]
    assert runner(tmp_path, clock, lambda t: SAFE, *children).run() == 1
    assert "after 3 relaunches in 24 h" in (tmp_path / "STOP").read_text()
    assert (tmp_path / "guard.log").read_text().count("relaunch: the trainer failed") == 3


def test_runner_reads_the_newest_checkpoint_for_the_disk_rail(tmp_path: Path) -> None:
    clock = FakeClock()
    ckpt.save(tmp_path, 10, arrays={"dense": {"w": mx.zeros((8,))}}, state={})
    child = FakeChild(clock, exit_after=3, code=0)
    tight = SAFE._replace(free_disk=10 * GB)  # under 2 checkpoints of 17 GB, not of 1 KB
    assert runner(tmp_path, clock, lambda t: tight, child).run() == 0
    assert child.terms == 0
    assert not (tmp_path / "STOP").exists()


def test_guarded_ignores_sighup_and_passes_sigterm_on_once(tmp_path: Path) -> None:
    def raise_signals(child: FakeChild) -> None:
        if child.waits == 2:
            os.kill(os.getpid(), signal.SIGHUP)  # ignored: the guard outlives a closed Terminal
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGINT)

    clock = FakeClock()
    child = FakeChild(clock, during_wait=raise_signals)
    before = {s: signal.getsignal(s) for s in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    assert runner(tmp_path, clock, lambda t: SAFE, child).guarded() == 75
    assert child.terms == 1
    assert {s: signal.getsignal(s) for s in before} == before
    assert "SIGTERM to the trainer: the guard was signalled" in (tmp_path / "guard.log").read_text()


@pytest.mark.parametrize(("cap", "says"), [(None, "none (MLX's default)"), (3.81, "3.81 GB")])
def test_the_start_line_names_the_cache_cap(tmp_path: Path, monkeypatch, cap, says: str) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    clock = FakeClock()
    child = FakeChild(clock, exit_after=1, code=0)
    assert runner(tmp_path, clock, lambda t: SAFE, child, cache_gb=cap).run() == 0
    start = (tmp_path / "guard.log").read_text().splitlines()[0]
    assert start.startswith("2026-01-01T00:00:00.000Z start: trainer --out ")
    assert start.endswith(f"(poll 10 s; caffeinate holding; cache cap {says})")


def test_one_caffeinate_beside_the_trainer_for_the_guards_life(tmp_path: Path, monkeypatch) -> None:
    # Battery stop, a SIGTERM in start-up (-15), a wait for AC, a relaunch, then the end: one
    # caffeinate -w <guard pid> throughout, stopped once when the guard exits.
    monkeypatch.setattr(sys, "platform", "darwin")
    clock = FakeClock()
    first, second = FakeChild(clock, on_term=-15), FakeChild(clock, exit_after=3, code=0)
    awake, launch = FakeAwake(), Launcher(first, second)

    def script(t: int) -> Readings:
        return BATT if 100 <= t < 300 else SAFE

    assert runner(tmp_path, clock, script, launch=launch, awake=awake).run() == 0
    assert awake.argvs == [["caffeinate", "-i", "-s", "-w", str(os.getpid())]]
    assert [c.terms for c in awake.started] == [1]
    assert len(launch.argvs) == 2
    assert all("caffeinate" not in argv for argv in launch.argvs)
    log = (tmp_path / "guard.log").read_text()
    assert log.splitlines()[0].endswith(
        "(poll 10 s; caffeinate holding; cache cap none (MLX's default))"
    )
    assert "the trainer exited -15 after a battery stop; relaunch on AC" in log
    assert "relaunch: AC has held for 60 s" in log
    assert "the trainer failed" not in log


def test_a_caffeinate_that_will_not_start_is_a_warning_not_a_stop(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    clock = FakeClock()
    child = FakeChild(clock, exit_after=2, code=0)
    assert runner(tmp_path, clock, lambda t: SAFE, child, awake=FakeAwake(fail=True)).run() == 0
    lines = (tmp_path / "guard.log").read_text().splitlines()
    assert "warning: caffeinate did not start, so the Mac may sleep" in lines[0]
    assert lines[1].endswith("(poll 10 s; no caffeinate; cache cap none (MLX's default))")
    assert "guard: warning: caffeinate did not start" in capsys.readouterr().err


def test_no_caffeinate_off_a_mac(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    clock = FakeClock()
    awake = FakeAwake()
    child = FakeChild(clock, exit_after=1, code=0)
    assert runner(tmp_path, clock, lambda t: SAFE, child, awake=awake).run() == 0
    assert awake.argvs == []
    start = (tmp_path / "guard.log").read_text().splitlines()[0]
    assert start.endswith("; no caffeinate; cache cap none (MLX's default))")


def test_allow_unknown_is_logged_loudly(tmp_path: Path, capsys) -> None:
    clock = FakeClock()
    child = FakeChild(clock, exit_after=1, code=0)
    loose = Guard(CKPT, allow_unknown=True)
    blank = Readings(free_disk=1000 * GB)
    assert runner(tmp_path, clock, lambda t: blank, child, guard=loose).run() == 0
    assert "warning: --allow-unknown" in (tmp_path / "guard.log").read_text()
    assert "guard: warning: --allow-unknown" in capsys.readouterr().err


STUB = (
    "import signal, sys, time\n"
    "def stop(signum, frame):\n"
    "    print('saved', flush=True)\n"
    "    sys.exit(75)\n"
    "signal.signal(signal.SIGTERM, stop)\n"
    "print('step 1', flush=True)\n"
    "time.sleep(60)\n"
)


def test_a_real_child_gets_one_sigterm_and_its_output_and_code_pass_through(
    tmp_path: Path, children
) -> None:
    clock = FakeClock()
    lines: list[bytes] = []

    def start(argv: list[str]) -> subprocess.Popen[bytes]:
        pipes = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
        }
        return children.spawn(argv, **pipes)

    def launch(argv):
        return macops.Proc(argv, clock, start=start, echo=lines.append)

    probes = FakeProbes(lambda t: SAFE._replace(thermal=4) if lines else SAFE)
    guard = Guard(CKPT)
    stub = [sys.executable, "-c", STUB]
    r = Runner(stub, tmp_path, guard, probes, clock, poll_s=0.2, launch=launch, awake=FakeAwake())
    assert r.run() == 75
    assert lines == [b"step 1\n", b"saved\n"]
    assert "thermal pressure level 4" in (tmp_path / "STOP").read_text()


def test_macops_imports_neither_mlx_nor_the_trainer(children) -> None:
    code = (
        "import sys\n"
        "import akeratos.macops\n"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'mlx' or m.startswith('akeratos.train')]\n"
        "assert not bad, bad\n"
    )
    proc = children.run([sys.executable, "-c", code], cwd=ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr


def test_system_clock_utc_now_is_aware_utc() -> None:
    now = SystemClock().utc_now()
    assert now.utcoffset() == timedelta(0)
    assert now.tzinfo is UTC
