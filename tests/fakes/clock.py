"""`FakeClock`: an `akeratos.clock.Clock` the test moves by hand.

`advance(ns)` moves both clocks, as time passing while awake does;
`sleep_system(ns)` moves only the boot clock, as a system sleep does; and
`set_utc(dt)` sets the timestamp `utc_now` returns. It reads no real clock.
`TickingClock` moves one second on every monotonic read, so code that times
itself sees time pass without the test calling `advance`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class FakeClock:
    """Monotonic and boot clocks from `start_ns`, and a UTC time that moves with them."""

    def __init__(self, start_ns: int = 0) -> None:
        self._mono = start_ns
        self._boot = start_ns
        self._utc = EPOCH

    def monotonic_ns(self) -> int:
        return self._mono

    def boot_ns(self) -> int:
        return self._boot

    def utc_now(self) -> datetime:
        return self._utc

    def advance(self, ns: int) -> None:
        """Time passes while the system is awake: every clock moves by `ns`."""
        if ns < 0:
            raise ValueError("a clock does not run backwards")
        self._mono += ns
        self._boot += ns
        self._utc += timedelta(microseconds=ns // 1000)

    def sleep_system(self, ns: int) -> None:
        """The system sleeps for `ns`: the boot clock and UTC move, the monotonic clock does not."""
        if ns < 0:
            raise ValueError("a clock does not run backwards")
        self._boot += ns
        self._utc += timedelta(microseconds=ns // 1000)

    def set_utc(self, dt: datetime) -> None:
        """Set the time `utc_now` returns; it must be aware."""
        if dt.tzinfo is None:
            raise ValueError("set_utc needs an aware datetime")
        self._utc = dt.astimezone(UTC)


class TickingClock(FakeClock):
    """A fake clock that moves one second each time the monotonic clock is read."""

    def monotonic_ns(self) -> int:
        self.advance(1_000_000_000)
        return super().monotonic_ns()
