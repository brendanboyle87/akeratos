"""What a run writes: the training log, one JSON line per step, and its checkpoints.

A log line holds the step, the tokens seen, the loss, the skip flag, s(t) and each group's rate,
the per-group gradient norms, the dense norm and the clip scale after an update, a spike flag,
the skip count, a hash of the batch's ids, the step's wall time, tokens per second, `mfu`, MLX's
allocator counters, `stall` and, with the table on, its signals under `table`. A checkpoint
holds the dense weights, each optimiser group's state, the EMA, the skip count, the configs, the
source's cursor and the log's window, and the table under `table/` when it is on. A resume
writes them back into a freshly built run before its first step, and refuses configs that
differ.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from akeratos._compat.mlx_seam import active_bytes, evaluate
from akeratos.ckpt import (
    KEEP,
    Checkpoint,
    Writer,
    checkpoints,
    disk_note,
    flatten,
    latest,
    load,
    pin,
    pinned,
    restore,
    save,
)
from akeratos.errors import ConfigError
from akeratos.flash import kernels_fit
from akeratos.table.config import TableConfig
from akeratos.table.table import Table

if TYPE_CHECKING:
    from akeratos.configs import ModelConfig
    from akeratos.train.loop import Run, StepResult


# The values come from the step's own host reads, so logging adds no device sync. `mfu` is
# always null. A spike is a finite loss above the mean plus `spike_sigma` standard deviations of
# the last `window` finite losses, once 20 are held. `stall` is the gap since the previous step
# when it is above 60 s, else null; `StallClock` reads the run's `Clock.boot_ns`, which keeps
# counting while the machine sleeps. A resumed run truncates the log to the lines before its step
# and reloads the loss window, so the log reads as one run.
MIN_SPIKE_WINDOW = 20
STALL_S = 60.0


def batch_hash(ids: NDArray[np.int32]) -> str:
    """An 8-byte BLAKE2b of the batch's ids, so two runs can be compared batch by batch."""
    return hashlib.blake2b(np.ascontiguousarray(ids).tobytes(), digest_size=8).hexdigest()


class StallClock:
    """Sleep-inclusive gaps between steps; a gap above `threshold` seconds is a stall.

    `now_ns` is a clock reading in nanoseconds: the run passes its `Clock.boot_ns`.
    """

    def __init__(self, now_ns: Callable[[], int], threshold: float = STALL_S) -> None:
        self.now_ns, self.threshold = now_ns, threshold
        self.last = now_ns()

    def mark(self) -> None:
        """Restart the gap, after work that is not a step (start-up, a save)."""
        self.last = self.now_ns()

    def stall(self) -> float | None:
        """Seconds since the last call or mark when above the threshold, else None."""
        t = self.now_ns()
        gap, self.last = (t - self.last) / 1e9, t
        return gap if gap > self.threshold else None


class Log:
    """Appends one JSON line per step to `path`."""

    def __init__(self, path: str | Path, window: int = 100, spike_sigma: float = 4.0) -> None:
        self.path = Path(path)
        self.spike_sigma = spike_sigma
        self.losses: deque[float] = deque(maxlen=window)
        self.skipped = 0

    def _spike(self, loss: float) -> bool:
        if not math.isfinite(loss):
            return False
        spike = False
        if len(self.losses) >= MIN_SPIKE_WINDOW:
            mu, sd = float(np.mean(self.losses)), float(np.std(self.losses))
            spike = loss > mu + self.spike_sigma * sd
        self.losses.append(loss)
        return spike

    def record(
        self,
        res: StepResult,
        s: float,
        peaks: Mapping[str, float],
        tokens_per_update: int,
        ids: NDArray[np.int32],
        step_s: float,
        memory: Mapping[str, int] | None = None,
        stall: float | None = None,
        table: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write step `res.step`'s line and return it."""
        rec: dict[str, Any] = {
            "step": res.step,
            "tokens": (res.step + 1) * tokens_per_update,
            "loss": res.loss,
            "skipped": res.skipped,
            "s": s,
            "lr": {g: p * s for g, p in peaks.items()},
            "grad_norm": {g: v for g, v in res.norms.items() if g != "dense"},
        }
        if res.skipped:
            self.skipped += 1
        else:
            rec |= {"dense_norm": res.norms["dense"], "clip_scale": res.clip_scale}
        rec |= {
            "spike": self._spike(res.loss),
            "skipped_total": self.skipped,
            "batch": batch_hash(ids),
            "step_s": step_s,
            "tokens_per_s": tokens_per_update / step_s if step_s > 0 else None,
            "mfu": None,
            **(dict(memory) if memory else {}),
            "stall": stall,
            **({"table": dict(table)} if table is not None else {}),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def state(self) -> dict[str, Any]:
        """The loss window and the skip count, for a checkpoint."""
        return {"losses": list(self.losses), "skipped": self.skipped}

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore what `state` returned."""
        self.losses.clear()
        self.losses.extend(float(x) for x in state["losses"])
        self.skipped = int(state["skipped"])

    def truncate(self, step: int) -> None:
        """Keep only the whole lines of steps before `step`, replacing the file atomically."""
        if not self.path.exists():
            return
        keep: list[str] = []
        for line in self.path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a line torn by a crash
            if isinstance(rec, dict) and int(cast(dict[str, Any], rec)["step"]) < step:
                keep.append(line + "\n")
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text("".join(keep))
        tmp.replace(self.path)


# `akeratos.ckpt` is the format; what a run puts in a checkpoint needs the run, so it lives
# here. Optimiser state is restored in place, since the compiled update holds it. A fresh run's
# table counts its reads per row and saves them as `table/reads.npy`; a run resumed from a
# checkpoint without them keeps none. `Checkpointer` holds the cadence, and its `writers` add a
# subdirectory to every save. With `pin_decay` it also saves at the schedule's decay start and
# pins that checkpoint outside the rotation, table included: the last stable checkpoint before
# the decay, kept for cool-down branches and post-training.
log = logging.getLogger(__name__)
TABLE = "table"  # the table's subdirectory of a checkpoint


def save_run(
    run: Run, keep: int = KEEP, dirs: Mapping[str, Writer] | None = None, keep_pinned: bool = False
) -> Path:
    """Checkpoint `run` at its current step, between steps; `dirs` adds writers such as the table's.

    `keep_pinned` keeps the checkpoint outside the rotation (`ckpt.save`).
    """
    st = run.stepper
    arrays: dict[str, Mapping[str, mx.array]] = {"dense": st.params}
    arrays |= {f"optim-{g}": flatten(o.state) for g, o in st.opt.opts.items()}
    if run.ema is not None:
        arrays["ema"] = run.ema.avg
    state = {
        "skipped": st.skipped,
        "config": run.config(),
        "source": run.source.state(),
        "log": run.log.state(),
    }
    writers = dict(dirs or {})
    if run.table is not None:  # the run's own table, under `table/`
        if TABLE in writers:
            raise ValueError(f"a {TABLE!r} writer beside the run's own table")
        state["table_event"] = run.table.event
        writers[TABLE] = run.table.write
    return save(
        run.out, st.step_no, arrays=arrays, state=state, dirs=writers, keep=keep, pin=keep_pinned
    )


def restore_run(run: Run, c: Checkpoint, config: Mapping[str, Any] | None = None) -> None:
    """Write checkpoint `c` back into a freshly built `run`, before its first step.

    The caller builds `run.source` at the saved cursor (`c.state["source"]`), since the packer
    resumes at construction. `config` is the configs the run must have, by default the saved
    ones; a cool-down branch passes the saved ones with its own changes.
    """
    want = c.state["config"] if config is None else json.loads(json.dumps(config))
    if json.loads(json.dumps(run.config())) != want:
        raise ConfigError(f"{c.path} was saved with other configs; resume with the same flags")
    st = run.stepper
    dense = c.arrays("dense")
    if set(dense) != set(st.params):
        raise ConfigError(f"{c.path} holds other weights than the model")
    st.params = {n: dense[n] for n in st.params}
    for g, o in st.opt.opts.items():
        restore(o.state, c.arrays(f"optim-{g}"))
    if run.ema is not None:
        avg = c.arrays("ema")
        run.ema.avg = {n: avg[n] for n in run.ema.avg}
    evaluate(st.params, st.opt.states(), run.ema.avg if run.ema is not None else [])
    st.step_no, st.skipped = c.step, int(c.state["skipped"])
    run.log.load_state(c.state["log"])
    if run.table is not None and run.table.event != c.state.get("table_event"):
        msg = f"{c.path}: the table is not this checkpoint's; load it with Checkpointer.table"
        raise ConfigError(msg)
    run.log.truncate(c.step)


class Checkpointer:
    """The save cadence and the resume for one run directory.

    The newest complete checkpoint in `out`, if any, is resumed; `every` 0 saves nothing at all,
    and `keep` counts complete checkpoints besides a pinned one. `pin_decay` also saves at the
    schedule's decay start (`Schedule.decay_start`) and pins that checkpoint (`ckpt.pin`). It is
    not one of the run's configs, so a resume may add or drop it. A resume at exactly that step
    pins the resumed checkpoint; one past it with no pinned checkpoint warns that nothing will be
    pinned. `start` is a checkpoint to start from when `out` holds none (`forked` is then True);
    it is only read.

    `writers` maps a subdirectory to a writer called at every save, and `resumed.path / name`
    is where that writer's files are on resume. The run's own table needs no writer: `save_run`
    writes it under `table/`, and `table(cfg)` loads it before the run is built.
    """

    def __init__(
        self,
        out: str | Path,
        every: int = 10_000,
        keep: int = KEEP,
        say: Callable[[str], None] = log.info,
        pin_decay: bool = False,
        start: str | Path | None = None,
    ) -> None:
        if pin_decay and not every:
            raise ConfigError("--pin-decay needs checkpoints on (--ckpt-every above 0)")
        self.every, self.keep, self.say = every, keep, say
        self.pin_decay = pin_decay
        self.pin_at: int | None = None
        self.writers: dict[str, Writer] = {}
        last = latest(out)
        self.forked = last is None and start is not None
        if self.forked:
            last = Path(cast(str | Path, start))
        self.resumed = load(last) if last is not None else None
        self.saved_at = -1

    def attn_path(self, cfg: ModelConfig, compute_dtype: str, seq_len: int) -> str:
        """`attn.path` for a run whose flags leave it out.

        A resume keeps its checkpoint's path, so a run started on the banded path resumes on it
        with the same flags. A fresh run is fused where `flash.kernels_fit` says the kernels run
        (the GPU, BF16, `seq_len` a multiple of the tiles, a head size up to 128), else banded,
        so a CPU run never falls back to the kernels' FP32 reference by default.
        """
        if self.resumed is not None:
            return str(self.resumed.state["config"]["model"].get("attn_path", "banded"))
        dtype = mx.bfloat16 if compute_dtype == "bfloat16" else mx.float32
        fits = kernels_fit(cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, seq_len, dtype)
        return "fused" if fits else "banded"

    def source_state(self) -> dict[str, Any] | None:
        """The saved source cursor to build the source at, or None for a fresh run."""
        return None if self.resumed is None else cast(dict[str, Any], self.resumed.state["source"])

    def table(self, cfg: TableConfig) -> Table:
        """The run's table: loaded from the resumed checkpoint's `table/`, else zero rows.

        It is built before the run, so a resume never holds a zero table and a loaded one at
        once. The write-back tripwire reads `active_bytes` (`akeratos/_compat/mlx_seam.py`). A
        saved table that is not `cfg`'s raises `ConfigError` (`Table.load`).
        """
        if self.resumed is None:
            table = Table(cfg, active_bytes)
            table.track_reads()
        else:
            table = Table.load(self.resumed.path / TABLE, cfg, active_bytes, reads=True)
        gib = table.store.nbytes() / 2**30
        self.say(f"table: {cfg.params:.3g} numbers in {len(table.store)} shards, {gib:.2f} GiB")
        if table.reads is None:
            self.say(
                "table: this run started without read counts and keeps none; rebuild them "
                "after it ends with python -m akeratos.train.reads"
            )
        return table

    def attach(self, run: Run, start_config: Mapping[str, Any] | None = None) -> None:
        """Restore the resumed checkpoint into a freshly built `run`, and plan the pin.

        `start_config` is the configs a forked run must have against `start` (`restore_run`).
        """
        if self.resumed is not None:
            restore_run(run, self.resumed, start_config if self.forked else None)
            verb = "forked" if self.forked else "resumed"
            self.say(f"{verb} from {self.resumed.path} at step {self.resumed.step:,}")
        self.saved_at = run.step_no
        if self.pin_decay:
            self._plan_pin(run)

    def _plan_pin(self, run: Run) -> None:
        at = run.sched.decay_start()
        held = [p for p in checkpoints(run.out) if pinned(p)]
        if at >= run.sched.steps:
            self.say("pin: this schedule has no decay, so nothing is pinned")
        elif held:
            self.say(f"pin: {held[-1]} is pinned")
        elif run.step_no < at:
            self.pin_at = at
            self.say(f"pin: the checkpoint at step {at:,}, before the decay, will be pinned")
        elif run.step_no == at and self.resumed is not None:
            pin(self.resumed.path)
            self.say(f"pin: pinned {self.resumed.path}, at the decay's start")
        else:
            msg = f"resumed at step {run.step_no:,}, past the decay's start at {at:,}"
            self.say(f"warning: pin: {msg}; nothing is pinned")

    def after_step(self, run: Run) -> None:
        """Save when the step count reaches a multiple of `every`, or the pin's step."""
        if (self.every and run.step_no % self.every == 0) or run.step_no == self.pin_at:
            self.save(run)

    def finish(self, run: Run) -> None:
        """Save where the run stopped (its budget, `--steps` or a signal), unless just saved."""
        if self.every and run.step_no != self.saved_at:
            self.save(run)

    def save(self, run: Run) -> None:
        """Save now, warn when the disk could not hold another, and restart the stall clock."""
        pin_now = run.step_no == self.pin_at
        path = save_run(run, self.keep, self.writers, keep_pinned=pin_now)
        self.saved_at = run.step_no
        self.say(f"saved {path}" + (", pinned: the last stable checkpoint" if pin_now else ""))
        note = disk_note(path)
        if note is not None:
            self.say(f"warning: {note}")
        run.stall_clock.mark()
