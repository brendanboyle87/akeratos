"""The clock seam and its fake.

`SystemClock` moves forward, and `FakeClock` is a `Clock` whose monotonic, boot and UTC
clocks move only as the test says: awake, asleep or set by hand.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from akeratos.clock import Clock, SystemClock
from tests.fakes.clock import EPOCH, FakeClock


def test_system_clock_moves_forward() -> None:
    clock: Clock = SystemClock()
    a, b = clock.monotonic_ns(), clock.boot_ns()
    assert clock.monotonic_ns() >= a
    assert clock.boot_ns() >= b


def test_fake_clock_is_a_clock() -> None:
    clock: Clock = FakeClock(start_ns=3)
    assert (clock.monotonic_ns(), clock.boot_ns()) == (3, 3)


def test_fake_advance_moves_every_clock() -> None:
    clock = FakeClock(start_ns=5)
    clock.advance(2_000_000)
    assert (clock.monotonic_ns(), clock.boot_ns()) == (2_000_005, 2_000_005)
    assert clock.utc_now() == EPOCH + timedelta(milliseconds=2)


def test_fake_system_sleep_moves_only_the_boot_clock() -> None:
    clock = FakeClock()
    clock.sleep_system(7_000_000_000)
    assert clock.monotonic_ns() == 0
    assert clock.boot_ns() - clock.monotonic_ns() == 7_000_000_000
    assert clock.utc_now() == EPOCH + timedelta(seconds=7)


def test_fake_refuses_backwards_time_and_naive_datetimes() -> None:
    clock = FakeClock()
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-1)
    with pytest.raises(ValueError, match="backwards"):
        clock.sleep_system(-1)
    with pytest.raises(ValueError, match="aware"):
        clock.set_utc(EPOCH.replace(tzinfo=None))


def test_set_utc_normalises_to_utc() -> None:
    clock = FakeClock()
    clock.set_utc(EPOCH.astimezone(timezone(timedelta(hours=2))))
    assert clock.utc_now() == EPOCH
    assert clock.utc_now().utcoffset() == timedelta(0)
