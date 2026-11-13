"""A first run that kills itself with SIGKILL at a named point.

It runs `akeratos.train.entry.main` on MLX's CPU device, as `tests.support.cpurun` does,
and SIGKILLs its own process while it does optimiser step `<step>` (counted from 1), at
`<phase>`, one of `PHASES`.

The kill is self-delivered, so it lands at the exact point: a parent that tails the log and
then kills races the run, which keeps going. SIGKILL cannot be caught, so nothing in the run
gets to save or tidy up. The resumed run is the trainer's entry with the same flags, run on
MLX's CPU device through `tests.support.cpurun`.

This module imports MLX at the top, so the caller sets `MLX_ENABLE_TF32=0` (the `repro`
suite's `DETERMINISM_ENV`); otherwise `entry.main` refuses, as TF32 might already be on.
"""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx

from akeratos import ckpt
from akeratos.train import entry, loop, saver
from akeratos.train import loop as stepper


def kill() -> None:
    """SIGKILL this process; the flush first keeps the run's own stderr for the test."""
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def _between_steps(step: int) -> None:
    """Kill after the step's update and log line, before its checkpoint and the next step."""
    run_step = loop.Run.step

    def after(self: loop.Run) -> dict[str, Any]:
        rec = run_step(self)
        if self.step_no == step:
            kill()
        return rec

    loop.Run.step = after


def _mid_step(step: int) -> None:
    """Kill after the step's gradients, before its dense update."""
    step_fn = stepper.Stepper.step

    def before_update(self: stepper.Stepper, *a: Any, **kw: Any) -> stepper.StepResult:
        if self.step_no != step - 1:
            return step_fn(self, *a, **kw)
        opt = self.opt
        set_schedule = opt.set_schedule

        def dying(s: float | Mapping[str, float]) -> None:
            kill()  # set_schedule runs after the step's gradients and before its dense update
            set_schedule(s)

        opt.set_schedule = dying
        try:
            return step_fn(self, *a, **kw)
        finally:
            opt.set_schedule = set_schedule

    stepper.Stepper.step = before_update


def _mid_accumulation(step: int) -> None:
    """Kill after the step's first micro-batch, before its second."""
    grads_fn = stepper.Stepper.accumulated_grads

    def second_micro(self: stepper.Stepper, micro: Sequence[Sequence[Any]]) -> Any:
        if self.step_no == step - 1:
            if len(micro) < 2:
                raise SystemExit("mid-accumulation needs --accum 2 or more")
            add: Callable[..., Any] = self._add

            def dying(*a: Any) -> Any:
                kill()
                return add(*a)

            self._add = dying
        return grads_fn(self, micro)

    stepper.Stepper.accumulated_grads = second_micro


def _mid_checkpoint(step: int) -> None:
    """Kill inside the step's checkpoint save, once every temporary file is synced."""
    sync = ckpt._sync
    tmp_name = f"{ckpt.TMP}{ckpt.PREFIX}{step:08d}"

    def before_rename(path: Path) -> None:
        sync(path)
        if path.name == tmp_name and path.is_dir():
            kill()  # the temporary directory is the last thing synced before the rename

    ckpt._sync = before_rename


ARM: dict[str, Callable[[int], None]] = {
    "between-steps": _between_steps,
    "mid-step": _mid_step,
    "mid-accumulation": _mid_accumulation,
    "mid-checkpoint": _mid_checkpoint,
}
PHASES = tuple(ARM)


def main(argv: Sequence[str]) -> int:
    if len(argv) < 3 or argv[0] not in ARM or argv[2] != "--":
        raise SystemExit(
            f"usage: python -m tests.support.killrun {{{','.join(PHASES)}}} <step> -- <flags>"
        )
    mx.set_default_device(mx.cpu)
    ARM[argv[0]](int(argv[1]))
    return entry.main(list(argv[3:]), loop.Run, saver.Checkpointer)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
