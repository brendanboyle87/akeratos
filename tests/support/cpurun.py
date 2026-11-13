"""Run the trainer's entry on MLX's CPU device, so two runs of one seed are bitwise equal.

`python -m tests.support.cpurun <flags>` sets the device, then calls `entry.main` with the flags
as `python -m akeratos.train.entry` does. MLX reads no environment variable for its device, and
on a Mac its default device is the GPU, where two runs of one seed differ.

This module imports MLX at the top, so the caller sets `MLX_ENABLE_TF32=0` (the `repro`
suite's `DETERMINISM_ENV`); otherwise `entry.main` refuses, as TF32 might already be on.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import mlx.core as mx

from akeratos.train import entry, loop, saver


def main(argv: Sequence[str]) -> int:
    mx.set_default_device(mx.cpu)
    return entry.main(list(argv), loop.Run, saver.Checkpointer)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
