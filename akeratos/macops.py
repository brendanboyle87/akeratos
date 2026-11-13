"""A guard that keeps a long training run safe on a laptop, and a reader of a Mac's hardware class.

The guard launches the trainer (`akeratos/train/entry.py`) as its child, copies the
child's output to its own, and polls the Mac's readings. Its rails stop the run, or warn, on
battery power, heat, memory pressure, low disk, a low or falling charge and readings that stay
unknown; after a battery stop it relaunches on AC, and the trainer resumes from its newest
checkpoint. Every poll appends one line to the `LOG` file in the trainer's `--out`, and a
permanent stop writes `STOP` there. On macOS a caffeinate beside the child keeps the Mac awake.

`Guard` decides and `Runner` does the I/O; `SystemProbes` reads the Mac. MLX is imported only
inside `apply_cache_limit` and `read_hardware`, so the guard never loads it.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import importlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from subprocess import DEVNULL, PIPE, STDOUT
from types import FrameType
from typing import Literal, NamedTuple, Protocol

from akeratos.clock import Clock, SystemClock
from akeratos.errors import ConfigError

TRAINER = "akeratos.train.entry"
RUN_TRAINER = (
    "import sys; from akeratos.cli import run_trainer; sys.exit(run_trainer(sys.argv[1:]))"
)
# The exit codes, as `akeratos/train/entry.py` names them.
EXIT_OK, EXIT_STOPPED, EXIT_DATA, EXIT_RESUMABLE, EXIT_CONFIG = 0, 3, 22, 75, 78
FINAL = frozenset({EXIT_OK, EXIT_STOPPED, EXIT_DATA, EXIT_CONFIG})  # end the guard as they are
S = 1_000_000_000  # nanoseconds in a second
BATTERY_NS = AC_NS = 60 * S
HOT_NS = BLIND_NS = DRAIN_NS = 600 * S
TERM_WAIT_NS, DAY_NS = 1_200 * S, 86_400 * S
DRAIN_POINTS, CHARGE_FLOOR = 3, 20  # percentage points, percent
TRAPPING, SLEEPING = 3, 4  # thermal pressure levels (OSThermalNotification.h)
WARN, CRITICAL, CRITICAL_POLLS = 2, 4, 2  # kern.memorystatus_vm_pressure_level; polls at 4
SWAP_WARN = 1 << 30
RUN_CKPTS, START_CKPTS = 2, 3  # free disk, in checkpoints, to keep running and to start
AC, BATTERY = "ac", "battery"
SOURCES = {"AC Power": AC, "Battery Power": BATTERY, "UPS Power": BATTERY}
UNKNOWN, LOG, STOP = "unknown", "guard.log", "STOP"
PERMANENT, BATTERY_STOP, SIGNAL_STOP = "permanent", "battery", "signal"
CKPT_FORMAT = 1  # `akeratos/ckpt.py`'s FORMAT
FRAMEWORKS = "/System/Library/Frameworks"
PMSET_BATT = ("pmset", "-g", "batt")
PRESSURE = ("sysctl", "-n", "kern.memorystatus_vm_pressure_level")
SWAPUSAGE = ("sysctl", "-n", "vm.swapusage")
IOREG_GPU = ("ioreg", "-rc", "AGXAccelerator")
HW_MODEL = ("sysctl", "-n", "hw.model")

_SOURCE = re.compile(r"Now drawing from '([^']+)'")
_GPU_CORES = re.compile(r'"gpu-core-count" = (\d+)')
_CHARGE = re.compile(r"\b(\d{1,3})%")
_SWAP = re.compile(r"\bused = (\d+(?:\.\d+)?)([KMGT])")
_UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
_CKPT = re.compile(r"ckpt-(\d+)")
_PREFIX = {"term": "SIGTERM to the trainer: ", "stop": "STOP: ", "relaunch": "relaunch: "}


class Readings(NamedTuple):
    """One poll's readings; None is `UNKNOWN`."""

    power: str | None = None  # AC or BATTERY
    charge: int | None = None  # percent
    thermal: int | None = None  # the thermal pressure level, 0 to 4
    thermal_state: int | None = None  # NSProcessInfo.thermalState, 0 to 3; recorded only
    pressure: int | None = None  # memory pressure: 1 normal, 2 warn, 4 critical
    swap_used: int | None = None  # bytes
    free_disk: int | None = None  # bytes free on --out's filesystem
    display: str | None = None  # "awake" or "asleep"; recorded only


class Act(NamedTuple):
    """One thing `Runner` does for `Guard`: a log line, a SIGTERM, STOP, a relaunch or the exit."""

    kind: Literal["log", "term", "stop", "relaunch", "exit"]
    text: str = ""
    code: int = 0


def _gb(n: int | None) -> str:
    return UNKNOWN if n is None else f"{n / 1e9:.1f} GB"


def _watched(r: Readings) -> tuple[tuple[str, object], ...]:
    """The readings that stop a run when they stay unknown for 10 min."""
    return ("thermal level", r.thermal), ("memory pressure", r.pressure), ("free disk", r.free_disk)


def _flat(r: Readings) -> tuple[bool, str]:
    """Whether the charge is at the floor, and the reason."""
    return r.charge is not None and r.charge <= CHARGE_FLOOR, f"charge {r.charge}%, 20% or below"


class Guard:
    """The rails as a pure state machine: readings or an exit code and a time in, actions out.

    Tests drive it with no sleeps and no processes. Times come from `Runner` and are the clock
    seam's `boot_ns`, which keeps counting through a system sleep: a condition seen on both sides
    of a closed lid counts as held through it, the cautious reading, and the relaunch cap's 24 h
    is elapsed time. A reading of None counts as unsafe where it gates a stop, unless
    `allow_unknown`, which lets such readings pass the refusals and the unknown rails. `ckpt` is
    a checkpoint's bytes for the disk rails until the trainer's `--out` holds a complete one;
    `Runner` then sets `ckpt_bytes` from the newest each poll.
    """

    def __init__(self, ckpt: int, *, allow_unknown: bool = False, max_relaunch: int = 3) -> None:
        self.ckpt_bytes = self.default_ckpt = ckpt
        self.allow_unknown, self.max_relaunch = allow_unknown, max_relaunch
        self.stop: str | None = None  # why this child had its one SIGTERM
        self.termed_at: int | None = None
        # waiting: for AC, after a battery stop; quit: the guard itself was signalled.
        self.waiting = self.quit = self.perf_invalid = False
        self._since: dict[str, int] = {}
        self._on: set[str] = set()
        self._charges: list[tuple[int, int]] = []
        self._relaunches: list[int] = []
        self._critical = 0

    @property
    def state(self) -> str:
        """`running`, `stopping` (the SIGTERM is out) or `waiting` (for AC)."""
        return "waiting" if self.waiting else "running" if self.stop is None else "stopping"

    def _held(self, key: str, on: bool, now: int) -> int:
        """How long condition `key` has held at `now`, or -1 when it does not hold."""
        if not on:
            self._since.pop(key, None)
            return -1
        return now - self._since.setdefault(key, now)

    def _edge(self, key: str, on: bool) -> bool:
        """True on the first poll condition `key` holds after one it did not."""
        was = key in self._on
        (self._on.add if on else self._on.discard)(key)
        return on and not was

    def _disk(self, r: Readings, ckpts: int) -> tuple[bool, str]:
        """Whether the free disk is below `ckpts` checkpoints, and the reason."""
        low = r.free_disk is not None and r.free_disk < ckpts * self.ckpt_bytes
        return low, f"{_gb(r.free_disk)} free, below {ckpts} checkpoints of {_gb(self.ckpt_bytes)}"

    def refusal(self, r: Readings) -> str | None:
        """Why the guard will not launch on readings `r`, or None."""
        unknown = [n for n, v in (("power source", r.power), *_watched(r)) if v is None]
        blind = f"{', '.join(unknown)} unknown (--allow-unknown passes it, off a Mac only)"
        checks = (
            (r.power == BATTERY, "on battery power"),
            ((r.thermal or 0) >= TRAPPING, f"thermal pressure level {r.thermal}"),
            (r.pressure == CRITICAL, "memory pressure critical"),
            self._disk(r, START_CKPTS),
            _flat(r),  # a charge the running rails would stop at once
            (bool(unknown) and not self.allow_unknown, blind),
        )
        return next((why for hit, why in checks if hit), None)

    def poll(self, r: Readings, now: int) -> list[Act]:
        """The actions for one poll's readings `r` at `now`.

        A battery stop's SIGTERM has the trainer save the step in flight; the relaunch once AC
        has held for 60 s resumes from the newest checkpoint.
        """
        acts = self._warnings(r, now)
        unsafe = r.power == BATTERY or (r.power is None and not self.allow_unknown)
        on_battery = self._held("power", unsafe, now)
        on_ac = self._held("ac", r.power == AC, now)
        why = self._permanent(r, now)
        if self.waiting:
            if why is not None:
                return [*acts, Act("stop", why), Act("exit", code=EXIT_RESUMABLE)]
            if on_ac >= AC_NS:
                self.waiting = False
                return [*acts, Act("relaunch", "AC has held for 60 s")]
            return acts
        if why is not None:
            return acts + self._stop(PERMANENT, why, now)
        if on_battery >= BATTERY_NS:
            what = "on battery" if r.power == BATTERY else "the power source unknown"
            return acts + self._stop(BATTERY_STOP, f"{what} for 60 s", now)
        return acts

    def signalled(self, now: int) -> list[Act]:
        """The guard's own SIGINT or SIGTERM: the child's one SIGTERM, then the child's code."""
        self.quit = True
        if self.waiting:
            return [Act("exit", code=EXIT_RESUMABLE)]
        return self._stop(SIGNAL_STOP, "the guard was signalled", now)

    def exited(self, code: int, now: int) -> list[Act]:
        """The actions once the child has exited with `code` (negative: killed by that signal)."""
        stop, self.stop, self.termed_at = self.stop, None, None
        self._on.discard("late")  # the next child gets its own SIGTERM and its own wait
        done = [Act("exit", code=code)]
        if self.quit or stop == PERMANENT or code in FINAL:
            return done
        if stop == BATTERY_STOP:  # whatever the code: a SIGTERM in start-up is -15 or 1
            self.waiting = True
            return [Act("log", f"the trainer exited {code} after a battery stop; relaunch on AC")]
        return self._crash(code, now) if code == 1 or code < 0 else done

    def _stop(self, kind: str, why: str, now: int) -> list[Act]:
        """One SIGTERM per child; a permanent stop also writes STOP, once."""
        acts = [Act("term", why)] if self.stop is None else []
        if kind == PERMANENT and self.stop != PERMANENT:
            acts.append(Act("stop", why))
        if self.stop is None:
            self.stop, self.termed_at = kind, now
        elif kind == PERMANENT:
            self.stop = kind
        return acts

    def _crash(self, code: int, now: int) -> list[Act]:
        """Relaunch after a failure, at most `max_relaunch` times in any 24 h; past that, STOP."""
        self._relaunches = [t for t in self._relaunches if now - t < DAY_NS]
        n, cap, failed = len(self._relaunches), self.max_relaunch, f"the trainer failed ({code})"
        if n >= cap:
            why = f"{failed} after {n} relaunches in 24 h (--max-relaunch {cap})"
            return [Act("stop", why), Act("exit", code=code)]
        self._relaunches.append(now)
        return [Act("relaunch", f"{failed}: relaunch {n + 1} of {cap} in 24 h")]

    def _permanent(self, r: Readings, now: int) -> str | None:
        """Why the run stops for good, or None."""
        hot = self._held("hot", (r.thermal or 0) >= TRAPPING, now)
        self._critical = self._critical + 1 if r.pressure == CRITICAL else 0
        strict = not self.allow_unknown
        blind = [n for n, v in _watched(r) if self._held(n, strict and v is None, now) >= BLIND_NS]
        checks = (
            (r.thermal == SLEEPING, "thermal pressure level 4 (sleeping)"),
            (hot >= HOT_NS, "thermal pressure level 3 (trapping) for 10 min"),
            (self._critical >= CRITICAL_POLLS, "memory pressure critical on 2 consecutive polls"),
            (bool(blind), f"{' and '.join(blind)} unknown for 10 min"),
            self._disk(r, RUN_CKPTS),
            _flat(r),
        )
        return next((why for hit, why in checks if hit), None)

    def _warnings(self, r: Readings, now: int) -> list[Act]:
        """The warn-only rails, each logged when it starts to hold, and the perf-invalid mark."""
        drained = self._drained(r, now)
        late = self.termed_at is not None and now - self.termed_at >= TERM_WAIT_NS
        checks = (
            ("pressure", r.pressure == WARN, "memory pressure at warn (2)"),
            ("swap", (r.swap_used or 0) > SWAP_WARN, f"swap in use {_gb(r.swap_used)}, over 1 GB"),
            ("display", r.display == "awake", "the display is awake"),
            *((n, v is None, f"the {n} reads unknown") for n, v in _watched(r)),
            ("drain", drained, "on AC, the charge fell 3 points within 10 min"),
            ("late", late, "no exit 20 min after the SIGTERM; still waiting (never SIGKILL)"),
        )
        acts = [Act("log", f"warning: {text}") for key, on, text in checks if self._edge(key, on)]
        if drained and not self.perf_invalid:
            self.perf_invalid = True
            acts.append(Act("log", "perf-invalid: the charge fell on AC"))
        return acts

    def _drained(self, r: Readings, now: int) -> bool:
        """True when, on AC, the charge has fallen 3 points or more within the last 10 min.

        The adapter is then not keeping up.
        """
        if r.power != AC or r.charge is None:
            self._charges.clear()
            return False
        self._charges = [(t, c) for t, c in self._charges if now - t <= DRAIN_NS]
        self._charges.append((now, r.charge))
        return max(c for _, c in self._charges) - r.charge >= DRAIN_POINTS


class Probes(Protocol):
    """What the guard reads each poll (`SystemProbes`; tests pass fakes).

    A read that fails is None, never an exception.
    """

    def read(self, out: Path) -> Readings:
        """This poll's readings; `out` names the filesystem whose free space counts."""
        ...


Command = Callable[[Sequence[str]], str | None]


def run_command(argv: Sequence[str]) -> str | None:
    """A command's standard output, or None when it fails or takes over 5 s."""
    try:
        return subprocess.run([*argv], capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def parse_batt(text: str | None) -> tuple[str | None, int | None]:
    """The power source (AC or BATTERY) and the charge in % from `pmset -g batt`, None if absent.

    pmset does not show IOPS's "Early" low-battery warning.
    """
    source, charge = _SOURCE.search(text or ""), _CHARGE.search(text or "")
    power = None if source is None else SOURCES.get(source[1])
    return power, None if charge is None else int(charge[1])


def parse_pressure(text: str | None) -> int | None:
    """`sysctl -n kern.memorystatus_vm_pressure_level`: 1, 2 or 4, else None."""
    value = (text or "").strip()
    return int(value) if value in ("1", "2", "4") else None


def parse_swap(text: str | None) -> int | None:
    """The bytes of swap in use from `sysctl -n vm.swapusage`, or None."""
    found = _SWAP.search(text or "")
    return None if found is None else int(float(found[1]) * _UNITS[found[2]])


@functools.cache
def _lib(path: str) -> ctypes.CDLL:
    return ctypes.CDLL(path)


def thermal_level() -> int | None:
    """The thermal pressure level from its notify key, or None.

    The levels are 0 to 4: nominal, moderate, heavy, trapping and sleeping.
    """
    token, state = ctypes.c_int(0), ctypes.c_uint64(0)
    try:
        lib = _lib("/usr/lib/libSystem.B.dylib")
        if lib.notify_register_check(b"com.apple.system.thermalpressurelevel", ctypes.byref(token)):
            return None
        failed = lib.notify_get_state(token, ctypes.byref(state))
        lib.notify_cancel(token)
    except (OSError, AttributeError):
        return None
    return None if failed or state.value > SLEEPING else state.value


def thermal_state() -> int | None:
    """`NSProcessInfo.processInfo.thermalState`, 0 (nominal) to 3 (critical), or None.

    It is read by ctypes through the Objective-C runtime, with no PyObjC.
    """
    try:
        objc = _lib("/usr/lib/libobjc.A.dylib")
        _lib(f"{FRAMEWORKS}/Foundation.framework/Foundation")  # defines NSProcessInfo
        objc.objc_getClass.restype = objc.sel_registerName.restype = ctypes.c_void_p
        # objc_msgSend takes each method's own prototype; on arm64 it must not be called variadic.
        msg = ("objc_msgSend", objc)
        send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(msg)
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(msg)
        info = send(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        state = int(send_long(info, objc.sel_registerName(b"thermalState"))) if info else -1
    except (OSError, AttributeError):
        return None
    return state if 0 <= state <= TRAPPING else None


def display() -> str | None:
    """The main display, `awake` or `asleep`, from CoreGraphics; None when it cannot be read."""
    try:
        cg = _lib(f"{FRAMEWORKS}/CoreGraphics.framework/CoreGraphics")
        main = cg.CGMainDisplayID()
        asleep = cg.CGDisplayIsAsleep(main) if main else None
    except (OSError, AttributeError):
        return None
    return None if asleep is None else "asleep" if asleep else "awake"


def free_disk(path: Path) -> int | None:
    """Bytes free on `path`'s filesystem, read at its nearest existing ancestor; or None."""
    try:
        here = next(p for p in (path, *path.absolute().parents) if p.exists())
        return shutil.disk_usage(here).free
    except (OSError, StopIteration):
        return None


class SystemProbes:
    """The real probes: the Mac's on darwin; elsewhere only the free disk, the rest unknown.

    `run` runs `pmset` and `sysctl` (`run_command`; tests pass captured output), and `mac` says
    whether to read the Mac at all.
    """

    def __init__(self, run: Command = run_command, *, mac: bool = sys.platform == "darwin") -> None:
        self._command, self._mac = run, mac

    def read(self, out: Path) -> Readings:
        """See `Probes.read`."""
        disk = free_disk(out)
        if not self._mac:
            return Readings(free_disk=disk)
        batt = parse_batt(self._command(PMSET_BATT))  # the power source and the charge
        level, swap = parse_pressure(self._command(PRESSURE)), parse_swap(self._command(SWAPUSAGE))
        # `Readings`' order: power, charge, thermal, thermal_state, pressure, swap, disk, display.
        return Readings(*batt, thermal_level(), thermal_state(), level, swap, disk, display())


def status(r: Readings) -> str:
    """One poll's readings as `key=value` words, `UNKNOWN` where a read failed."""
    return (
        f"power={r.power} charge={r.charge} thermal={r.thermal} thermal_state={r.thermal_state} "
        f"pressure={r.pressure} swap_used={r.swap_used} free_disk={r.free_disk} display={r.display}"
    ).replace("=None", f"={UNKNOWN}")


class Hardware(NamedTuple):
    """A Mac's hardware class (`key`), with its model, which the class leaves out."""

    chip: str  # MLX's device_name for the GPU, such as "Apple M4 Max"
    gpu_cores: int  # "gpu-core-count" in `ioreg -rc AGXAccelerator`
    gpu_arch: str  # MLX's architecture for the GPU, such as "applegpu_g16s"
    memory_gb: int  # MLX's memory_size in GB of 2^30 bytes, rounded
    macos: str  # platform.mac_ver()[0], such as "26.1"
    form: str  # "laptop" or "desktop"
    model: str  # `sysctl -n hw.model`, such as "Mac16,5"

    def key(self) -> str:
        """The class, such as `m4-max-40c-48gb-macos26-laptop`.

        The model stays out of it, so a class can later be split by model without new runs.
        """
        chip = re.sub(r"[^a-z0-9]+", "-", self.chip.removeprefix("Apple ").lower()).strip("-")
        major = self.macos.split(".")[0]
        return f"{chip}-{self.gpu_cores}c-{self.memory_gb}gb-macos{major}-{self.form}"

    def describe(self) -> dict[str, object]:
        """Every field and the class, as JSON-ready data for bench rows and speedrun receipts."""
        return {
            "chip": self.chip,
            "gpu_cores": self.gpu_cores,
            "gpu_arch": self.gpu_arch,
            "memory_gb": self.memory_gb,
            "macos": self.macos,
            "form": self.form,
            "model": self.model,
            "class": self.key(),
        }


def parse_hardware(
    info: Mapping[str, object], ioreg: str | None, batt: str | None, macos: str, model: str | None
) -> Hardware:
    """The hardware class from its reads; `read_hardware` makes them, and tests pass captures.

    `info` is MLX's `device_info` for the GPU, `macos` is `platform.mac_ver()[0]`, empty off a
    Mac, and the others are the outputs of `ioreg -rc AGXAccelerator`, `pmset -g batt` and
    `sysctl -n hw.model`, None where one failed.
    """
    cores, memory, batt = _GPU_CORES.search(ioreg or ""), info.get("memory_size"), batt or ""
    hw = Hardware(
        chip=str(info.get("device_name") or ""),
        gpu_cores=int(cores[1]) if cores else 0,
        gpu_arch=str(info.get("architecture") or ""),
        memory_gb=round(memory / 2**30) if isinstance(memory, int) else 0,
        macos=macos,
        # Not by the charge: a desktop on a UPS shows one too.
        form="laptop" if "InternalBattery" in batt else "desktop" if _SOURCE.search(batt) else "",
        model=(model or "").strip(),
    )
    missing = [name for name, value in zip(Hardware._fields, hw, strict=True) if not value]
    if missing:
        raise ConfigError(f"cannot read this Mac's {', '.join(missing)}")
    return hw


def read_hardware(run: Command = run_command, *, mac: bool = sys.platform == "darwin") -> Hardware:
    """This Mac's hardware class, from MLX's `device_info` for the GPU and the commands `run` runs.

    A field that cannot be read raises `ConfigError`, as does a call off a Mac.
    """
    if not mac:
        raise ConfigError("the hardware class is read only on a Mac")
    mx = importlib.import_module("mlx.core")
    info, ioreg, batt = mx.device_info(mx.gpu), run(IOREG_GPU), run(PMSET_BATT)
    return parse_hardware(info, ioreg, batt, platform.mac_ver()[0], run(HW_MODEL))


def read_conditions(run: Command = run_command) -> dict[str, object]:
    """The power source (`AC`, `BATTERY` or None) and the thermal pressure level (0-4 or None)."""
    return {"power": parse_batt(run(PMSET_BATT))[0], "thermal_level": thermal_level()}


def newest_ckpt_bytes(out: Path) -> int | None:
    """The bytes of the newest complete checkpoint in `out`, or None when it holds none.

    Complete as `akeratos/ckpt.py` judges it: `state.json` parses, names its format and its
    step, and every file it lists is there with its recorded size. The layout is read here,
    since macops does not import `ckpt`, a module of its own layer (docs/architecture.md).
    """
    found = [(int(m[1]), p) for p in sorted(out.glob("ckpt-*")) if (m := _CKPT.fullmatch(p.name))]
    sizes = (_complete_bytes(p, step) for step, p in sorted(found, reverse=True))
    return next((n for n in sizes if n is not None), None)


def _complete_bytes(path: Path, step: int) -> int | None:
    """The bytes of checkpoint `path` (its files and `state.json`), or None when incomplete."""
    state = path / "state.json"
    try:
        doc = json.loads(state.read_text(encoding="utf-8"))
        files: dict[str, int] = doc["files"]
        ok = all((path / f).is_file() and (path / f).stat().st_size == n for f, n in files.items())
        ours = doc["format"] == CKPT_FORMAT and doc["step"] == step
        return state.stat().st_size + sum(files.values()) if ok and ours else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _say(line: str) -> None:
    with contextlib.suppress(OSError, ValueError):
        sys.stderr.write(line + "\n")


def _write(path: Path, text: str, mode: str = "a") -> None:
    """Write `text` to `path`; a failure is said on stderr, never raised, so the guard guards on."""
    try:
        with path.open(mode, encoding="utf-8") as fh:
            fh.write(text)
    except OSError as e:
        _say(f"guard: cannot write {path}: {e}")


class Child(Protocol):
    """A launched process as `Runner` sees it: the trainer, or caffeinate beside it (`Proc`)."""

    last_output_ns: int  # `boot_ns` at its last line of output, or at its launch

    def wait(self, timeout_s: float) -> int | None:
        """Its exit code once it has exited, waiting up to `timeout_s`; None while it runs."""
        ...

    def terminate(self) -> None:
        """Send it one SIGTERM."""
        ...


Spawn = Callable[[list[str]], "subprocess.Popen[bytes]"]
Echo = Callable[[bytes], None]


def spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    """Start `argv` in its own session, its output and its errors on one pipe."""
    return subprocess.Popen(argv, stdin=DEVNULL, stdout=PIPE, stderr=STDOUT, start_new_session=True)


def echo_line(line: bytes) -> None:
    """Copy one line of the trainer's output to ours; with the terminal gone it is dropped."""
    with contextlib.suppress(OSError, ValueError):
        sys.stdout.buffer.write(line)
        sys.stdout.flush()


class Proc:
    """A launched process: its output copied to `echo` line by line, with the last line's time."""

    def __init__(
        self, argv: Sequence[str], clock: Clock, *, start: Spawn = spawn, echo: Echo = echo_line
    ) -> None:
        self._clock, self._echo = clock, echo
        self.last_output_ns = clock.boot_ns()
        self._popen = start(list(argv))
        self._copier = threading.Thread(target=self._copy, daemon=True)
        self._copier.start()

    def _copy(self) -> None:
        """Copy the output until the child closes it, then close our end (the thread owns it)."""
        if self._popen.stdout is None:
            return
        with self._popen.stdout as pipe:
            for line in pipe:
                self.last_output_ns = self._clock.boot_ns()
                self._echo(line)

    def wait(self, timeout_s: float) -> int | None:
        """See `Child.wait`; the rest of the output is copied before the code returns."""
        try:
            code = self._popen.wait(timeout_s)
        except subprocess.TimeoutExpired:
            return None
        self._copier.join(timeout_s)
        return code

    def terminate(self) -> None:
        """One SIGTERM to the process itself (the trainer is the child; caffeinate is beside it)."""
        self._popen.send_signal(signal.SIGTERM)


def apply_cache_limit(limit_bytes: int | None) -> None:
    """Cap MLX's buffer cache at `limit_bytes` and say so; None leaves MLX's default.

    The wired limit stays unset.
    """
    if limit_bytes is not None:
        importlib.import_module("mlx.core").set_cache_limit(limit_bytes)
        _say(f"macops: MLX cache limit {limit_bytes / 2**30:.2f} GB ({limit_bytes:,} bytes)")


def child_argv(trainer: Sequence[str], cache_gb: float | None = None) -> list[str]:
    """The trainer's command on every platform, through `cli.run_trainer` with a cache cap.

    `cache_gb` is in GB of 2^30 bytes. With it, the child runs `cli.run_trainer`, which installs
    `apply_cache_limit` as the trainer's setup hook, since macops imports neither the CLI nor the
    trainer; without it the child runs the trainer, and MLX keeps its default cache. caffeinate
    runs beside the child, not around it (`Runner`).
    """
    cap = ["-c", RUN_TRAINER, "--cache-limit-gb", str(cache_gb)]
    return [sys.executable, *(["-m", TRAINER] if cache_gb is None else cap), *trainer]


class Runner:
    """The guard's I/O: it launches the child, polls the probes and carries out `Guard`'s actions.

    `out` is the trainer's `--out`, where the `LOG` file and `STOP` go; `clock` gives `boot_ns`
    for the rails and `utc_now` for the stamps; `cache_gb`, the MLX cache cap the child applies,
    only names it in the start line; `pause` waits out a poll while no child runs; and `awake`
    starts caffeinate beside the child on darwin.
    """

    def __init__(
        self,
        argv: Sequence[str],
        out: Path,
        guard: Guard,
        probes: Probes,
        clock: Clock,
        *,
        poll_s: float = 10.0,
        cache_gb: float | None = None,
        launch: Callable[[Sequence[str]], Child] | None = None,
        pause: Callable[[float], None] = time.sleep,
        awake: Callable[[Sequence[str]], Child] | None = None,
    ) -> None:
        self.guard, self._argv, self._out, self._probes = guard, list(argv), out, probes
        self._clock, self._poll_s, self._pause, self._cache_gb = clock, poll_s, pause, cache_gb
        self._launch = launch or functools.partial(Proc, clock=clock)
        self._awake_with = awake or functools.partial(Proc, clock=clock)
        self.signalled = False

    def _stamp(self) -> str:
        return self._clock.utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _log(self, text: str, *, echo: bool = True) -> None:
        """Append `text` to the `LOG` file after the UTC time; with `echo`, say it on stderr too."""
        _write(self._out / LOG, f"{self._stamp()} {text}\n")
        if echo:
            _say(f"guard: {text}")

    def guarded(self) -> int:
        """`run`, with SIGHUP ignored and SIGINT or SIGTERM passed on to the trainer once."""

        def on_signal(signum: int, frame: FrameType | None) -> None:
            del signum, frame  # only a flag: the loop acts on it within a poll
            self.signalled = True

        previous = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}
        # With SIGHUP ignored, the guard under `nohup` outlives a closed Terminal.
        previous[signal.SIGHUP] = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            return self.run()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)

    def _awake(self) -> Child | None:
        """On darwin, one caffeinate for the guard's whole life, across relaunches, or None."""
        if sys.platform != "darwin":
            return None
        try:
            # -i -s: no idle sleep, and on AC no system sleep; -w: it ends with the guard.
            return self._awake_with(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            # One that will not start is a warning, not a stop, since the rails still guard the run.
            self._log(f"warning: caffeinate did not start, so the Mac may sleep: {e}")
            return None

    def run(self) -> int:
        """Launch the child and guard it until the guard's exit code is known."""
        awake = self._awake()
        cap = "none (MLX's default)" if self._cache_gb is None else f"{self._cache_gb:g} GB"
        held = f"{'no caffeinate' if awake is None else 'caffeinate holding'}; cache cap {cap}"
        self._log(f"start: {shlex.join(self._argv)} (poll {self._poll_s:g} s; {held})")
        if self.guard.allow_unknown:
            self._log("warning: --allow-unknown: unknown readings pass the refusals and the rails")
        child: Child | None = self._launch(self._argv)
        while True:
            code = self._wait(child)
            now = self._clock.boot_ns()
            if code is not None:
                child = None
                self._log(f"the trainer exited {code}")
                acts = self.guard.exited(code, now)
            elif self.signalled and not self.guard.quit:
                self._log("signalled: one SIGTERM to the trainer, then its exit code")
                acts = self.guard.signalled(now)
            else:
                acts = self._poll(child, now)
            for act in acts:
                if act.kind == "exit":
                    self._log(f"exit {act.code}")
                    if awake is not None:  # it would end with the guard anyway (-w)
                        awake.terminate()
                    return act.code
                child = self._apply(act, child)

    def _wait(self, child: Child | None) -> int | None:
        if child is None:
            self._pause(self._poll_s)
            return None
        return child.wait(self._poll_s)

    def _poll(self, child: Child | None, now: int) -> list[Act]:
        r = self._probes.read(self._out)
        self.guard.ckpt_bytes = newest_ckpt_bytes(self._out) or self.guard.default_ckpt
        # The copier may stamp a line after `now` was read, so the gap is at least 0.
        quiet = UNKNOWN if child is None else max(0, now - child.last_output_ns) // S
        self._log(f"poll {status(r)} quiet_s={quiet} state={self.guard.state}", echo=False)
        return self.guard.poll(r, now)

    def _apply(self, act: Act, child: Child | None) -> Child | None:
        self._log(_PREFIX.get(act.kind, "") + act.text)
        if act.kind == "term" and child is not None:
            child.terminate()
        elif act.kind == "stop":
            _write(self._out / STOP, f"{self._stamp()} {act.text}\n", "w")
        elif act.kind == "relaunch":
            return self._launch(self._argv)
        return child


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.macops",
        usage="%(prog)s [options] -- <akeratos.train.entry flags>",
    )
    p.add_argument("--poll", type=float, default=10.0, help="seconds between polls")
    p.add_argument("--ckpt-gb", type=float, default=17.0, help="checkpoint GB until one is saved")
    p.add_argument("--max-relaunch", type=int, default=3, help="relaunches on failure in 24 h")
    p.add_argument("--allow-unknown", action="store_true", help="for CPU smoke runs off a Mac")
    p.add_argument("--cache-limit-gb", type=float, help="cap MLX's buffer cache (Billet: 3.81)")
    return p


def _launchable(trainer: Sequence[str], guard: Guard, probes: Probes) -> Path | str:
    """The trainer's `--out` when the guard may launch it, else why it refuses."""
    p = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    p.add_argument("--out", type=Path)
    p.add_argument("--ckpt-every", type=int, default=10_000)  # entry's default
    try:
        a, _ = p.parse_known_args(list(trainer))
    except argparse.ArgumentError as e:
        return f"the trainer's flags: {e}"
    out: Path | None = a.out
    if out is None:
        return "no --out among the trainer's flags (after --)"
    if a.ckpt_every == 0:
        return "--ckpt-every 0 turns checkpoints off, so a stop could not save"
    if (out / STOP).exists():
        return f"{out / STOP} exists: a permanent stop; remove it to start again"
    guard.ckpt_bytes = newest_ckpt_bytes(out) or guard.default_ckpt
    why = guard.refusal(probes.read(out))
    return out if why is None else why


def main(argv: Sequence[str], probes: Probes | None = None, clock: Clock | None = None) -> int:
    """Parse the flags, then refuse or launch and guard the trainer; returns the exit code.

    As the entry point it builds the `SystemClock` and the `SystemProbes`, unless a test passes
    fakes. A reboot ends the guard with the run, so after one a human relaunches the same command.
    """
    cut = list(argv).index("--") if "--" in argv else len(argv)
    parser = _parser()
    a = parser.parse_args(argv[:cut])
    cap = a.cache_limit_gb
    if min(a.poll, a.ckpt_gb, 1 if cap is None else cap) <= 0 or a.max_relaunch < 0:
        parser.error("--poll, --ckpt-gb, --cache-limit-gb must be > 0 and --max-relaunch >= 0")
    trainer = list(argv[cut + 1 :])
    probes = SystemProbes() if probes is None else probes
    guard = Guard(int(a.ckpt_gb * 1e9), allow_unknown=a.allow_unknown, max_relaunch=a.max_relaunch)
    out = _launchable(trainer, guard, probes)
    if isinstance(out, str):
        _say(f"guard: refusing to start: {out}")
        return EXIT_CONFIG
    out.mkdir(parents=True, exist_ok=True)
    clock = SystemClock() if clock is None else clock
    argv = child_argv(trainer, cap)
    code = Runner(argv, out, guard, probes, clock, poll_s=a.poll, cache_gb=cap).guarded()
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
