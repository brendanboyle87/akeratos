"""The clock seam: the one module in the product that reads a clock.

Components that time things take a `clock: Clock` and never read time
themselves; only an entry point builds a `SystemClock`, and tests pass
`tests/fakes/clock.py`'s `FakeClock`.

Durations, tokens per second, MFU and ETA use `monotonic_ns`; `time.time` is
never used for durations. The training loop's stall timer
(`akeratos/train/saver.py`) uses `boot_ns`, so it sees a system sleep as a gap,
and the run guard (`akeratos/macops.py`) times its rails with it.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """The readings a timing component may take."""

    def monotonic_ns(self) -> int:
        """Nanoseconds for durations; stops during system sleep on macOS."""
        ...

    def boot_ns(self) -> int:
        """Nanoseconds that keep counting while the system sleeps."""
        ...

    def utc_now(self) -> datetime:
        """The time now as an aware UTC datetime, for timestamps; never for durations."""
        ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def boot_ns(self) -> int:
        """`CLOCK_MONOTONIC` on macOS, which counts sleep there; `CLOCK_BOOTTIME` on Linux."""
        if sys.platform == "darwin":
            return time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    def utc_now(self) -> datetime:
        return datetime.now(UTC)
