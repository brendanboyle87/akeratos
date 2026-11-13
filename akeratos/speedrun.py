"""The fixed code of the 5-minute Mac speedrun track: the driver behind `bake speedrun`.

`run` trains the model that the train file (`speedrun/train.py`) builds for `BUDGET_S` seconds
of step time on the track's training slice, scores its held-out bits per byte and writes a
receipt. Its steps, in order, are `_check_slices`, `fixed_commit`, `_load_train`, `_feed`,
`build(seed)` (untimed) and `_check_trainer`, `_train`, `_check_causal` and `bpb_of`. Then it
writes `receipt.json` beside `metrics.jsonl` in `<out>/<class>/<claim>/seed-<N>/`, a folder
that must not exist yet.

A budget other than `BUDGET_S` makes a practice run, for an agent loop: its receipt says so,
it is never a record, and its folder is `<out>/<UTC start>-seed-<N>/`, by default under
`runs/local/speedrun/`. `speedrun/records.py` holds a record to the constants below.
"""

from __future__ import annotations

import bisect
import importlib.util
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from importlib import metadata
from pathlib import Path
from typing import NamedTuple, Protocol, TextIO, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from akeratos import __version__, digest
from akeratos._compat.mlx_seam import evaluate, memory, reset_peak
from akeratos.clock import Clock
from akeratos.configs import VOCAB
from akeratos.data.packer import ExhaustedError, Packer
from akeratos.data.store import Split
from akeratos.env import tf32_off
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout import HeldOut
from akeratos.macops import Hardware, run_command
from akeratos.rng import run_seeds
from akeratos.train.source import FIRST_CONTENT_ID, PackedSource, StepRows

SCHEMA = "metalforge.speedrun-receipt/v1"
BUDGET_S = 300.0  # seconds of step time
WARMUP_STEPS = 10
WARMUP_RULE = "median"  # each warm-up step counts at the median time of the steps after it
EVAL_SEQ_LEN = 2048  # held-out rows of T + 1 tokens, one document (or window) per row
EVAL_ROWS = 4  # rows per forward pass when scoring
EVAL_TARGETS = 2**20  # whole held-out documents, in split order, until this many targets
CAUSAL_ATOL = 1e-3
CATEGORIES = ("open", "no-table")
# The slices' pins, as `speedrun/slices.py` printed them when it cut the slices.
SLICES = {
    "sha256sums": "sha256:12005f1d933f2c509bf22f7e1ca2b790a234de478ba4dd541ed8a9f851b76239",
    "store": "9545f2d205efd574",
    "dataset": "HuggingFaceFW/fineweb-edu@87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "tokenizer": "HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
}
HELDOUT = {"targets": 1048846, "docs": 986, "bytes": 4745219}  # the scored set's counts
RECORDS, PRACTICE = Path("results/speedrun"), Path("runs/local/speedrun")  # default --out
PACKAGE = Path(__file__).resolve().parent  # the akeratos/ folder this code runs from
FIXED = ("akeratos", "pyproject.toml", "uv.lock")  # what must be as committed
# A claim: the UTC date of its first run, then the person's GitHub handle. `HANDLE` copies
# `bench.HANDLE`, since the import contract keeps the two modules apart.
HANDLE = re.compile(r"(?=.{1,39}\Z)[A-Za-z0-9](?:-?[A-Za-z0-9])*")  # a GitHub user name
CLAIM = re.compile(r"(\d{4}-\d{2}-\d{2})-(.+)")


class Trainer(Protocol):
    """What the train file's `build(seed)` returns."""

    def step(self, rows: StepRows, progress: float) -> float:
        """Train on one batch and return its loss; `progress` is the timed total over the budget."""
        ...

    def arrays(self) -> object:
        """Every MLX array the model's state holds; the driver evaluates them inside the step."""
        ...

    def logits(self, ids: NDArray[np.int64]) -> mx.array:
        """Logits [B, T, 49,152] of ids [B, T], untimed: a function of the weights and `ids`."""
        ...


def fixed_commit(package: Path = PACKAGE) -> str:
    """Step 2: HEAD of the checkout whose `akeratos/` folder is `package`.

    That checkout's `FIXED` paths must be as committed at HEAD, which the receipt records. The
    train file may differ from the commit: the receipt holds its full text.
    """
    top = run_command(("git", "-C", str(package), "rev-parse", "--show-toplevel"))
    if top is None or Path(top.strip()).resolve() / "akeratos" != package.resolve():
        raise ConfigError(
            f"{package} is not the akeratos/ folder of a git checkout; bake speedrun runs from "
            "a git clone of Akeratos"
        )
    root = top.strip()
    dirty = run_command(("git", "-C", root, "status", "--porcelain", "--", *FIXED))
    head = run_command(("git", "-C", root, "rev-parse", "HEAD"))
    if dirty is None or head is None:
        raise ConfigError(f"git cannot read the state of {root}")
    if dirty:
        paths = ", ".join(line[3:] for line in dirty.splitlines())
        raise ConfigError(f"{paths}: differs from HEAD; commit or undo it (train.py may differ)")
    return head.strip()


@dataclass(frozen=True)
class Options:
    """One run. The CLI fills the fields without defaults and `budget_s`; tests change the rest."""

    data: Path  # the slices' folder
    train: Path  # the train file
    seed: int
    out: Path | None  # None: `RECORDS`, or `PRACTICE` for a practice run
    category: str  # one of `CATEGORIES`
    claim: str  # the UTC date of the claim's first run, then `person`
    person: str  # a GitHub handle
    ai_system: str  # the AI system that helped, or "none"
    rerun_of: str | None  # `<class>/<claim>` of the claim this run reruns
    hardware: Hardware
    conditions: Callable[[], Mapping[str, object]]  # the power source and the thermal level
    clock: Clock
    budget_s: float = BUDGET_S
    eval_seq_len: int = EVAL_SEQ_LEN
    eval_rows: int = EVAL_ROWS
    eval_targets: int = EVAL_TARGETS
    slices: Mapping[str, str] = field(default_factory=lambda: dict(SLICES))
    heldout: Mapping[str, int] = field(default_factory=lambda: dict(HELDOUT))
    fixed_code: Callable[[], str] = fixed_commit


class _TrainFile(NamedTuple):
    """The train file as run: its text and SHA-256, its batch shape and its `build`."""

    source: str
    sha256: str
    rows: int
    seq_len: int
    build: Callable[[int], Trainer]


class Timing:
    """Each step's time in ns, and the timed total under the warm-up rule.

    The total is the timed steps' sum plus `WARMUP_STEPS` times their median, so each warm-up
    step counts at the median time of the steps after it. It is exact in integer ns.
    """

    def __init__(self) -> None:
        self.steps: list[int] = []
        self._timed: list[int] = []  # the timed steps' times, sorted
        self._sum = 0

    def add(self, ns: int) -> None:
        """Record the next step's time."""
        if len(self.steps) >= WARMUP_STEPS:
            bisect.insort(self._timed, ns)
            self._sum += ns
        self.steps.append(ns)

    def charged_ns(self) -> int:
        """The timed total so far: 0 until the first timed step."""
        n = len(self._timed)
        if n == 0:
            return 0
        twice_median = self._timed[n // 2] + self._timed[(n - 1) // 2]
        return self._sum + WARMUP_STEPS * twice_median // 2


def _handle(claim: str) -> str | None:
    """The GitHub handle in a claim named `YYYY-MM-DD-<handle>` with a real date, else None."""
    found = CLAIM.fullmatch(claim)
    if found is None or HANDLE.fullmatch(found[2]) is None:
        return None
    try:
        date.fromisoformat(found[1])
    except ValueError:
        return None
    return found[2]


def _check(o: Options) -> None:
    """Refuse options a receipt cannot hold, before anything runs."""
    cls, _, rerun = (o.rerun_of or "").partition("/")
    other = _handle(rerun)
    bad = [
        msg
        for wrong, msg in (
            (o.category not in CATEGORIES, f"--category {o.category!r}: open or no-table"),
            (HANDLE.fullmatch(o.person) is None, f"--person {o.person!r}: a GitHub handle"),
            (
                _handle(o.claim) != o.person,
                f"--claim {o.claim!r}: YYYY-MM-DD-{o.person}, the UTC date of its first run",
            ),
            (
                o.rerun_of is not None
                and (cls != o.hardware.key() or other is None or other == o.person),
                f"--rerun-of {o.rerun_of!r}: CLASS/CLAIM, another person's claim on this class",
            ),
            (not o.ai_system.strip(), "--ai-system: the AI system that helped, or none"),
            (o.seed < 0, f"--seed {o.seed}: 0 or more"),
            (
                not (o.budget_s > 0 and math.isfinite(o.budget_s * 1e9)),
                f"--seconds {o.budget_s:g}: more than 0, and finite in nanoseconds",
            ),
        )
        if wrong
    ]
    if bad:
        raise ConfigError("; ".join(bad))


def _folder(o: Options, start: datetime) -> Path:
    """The run's folder, refused if it exists."""
    if o.budget_s == BUDGET_S:
        folder = (o.out or RECORDS) / o.hardware.key() / o.claim / f"seed-{o.seed}"
    else:
        folder = (o.out or PRACTICE) / f"{start:%Y%m%dT%H%M%SZ}-seed-{o.seed}"
    if folder.exists():
        raise ConfigError(
            f"{folder} exists; a run never replaces one: give another --seed, or --seconds N "
            "for a practice run"
        )
    return folder


def _check_slices(o: Options) -> HeldOut:
    """Step 1: the slices' files against their pins, then the scored set against its counts.

    `SHA256SUMS` must have the SHA-256 that `SLICES` pins and every file it lists the SHA-256 it
    lists, and the scored set must hold the counts `HELDOUT` pins.
    """
    sums, pin = o.data / "SHA256SUMS", o.slices["sha256sums"]
    try:
        if (got := digest.file_digest(sums)) != pin:
            raise DataMismatchError(f"{sums} has SHA-256 {got}; the track pins {pin}")
        for line in sums.read_text(encoding="utf-8").splitlines():
            sha, rel = line.split("  ", 1)
            if digest.file_digest(o.data / rel) != f"sha256:{sha}":
                raise DataMismatchError(f"{o.data / rel} differs from {sums}")
    except OSError as e:
        raise DataMismatchError(f"{e}; --data names the slices' folder") from e
    held = HeldOut(o.data / "heldout", o.eval_seq_len, o.eval_targets, doc_mask=False, vocab=VOCAB)
    counts = {"targets": held.targets, "docs": held.docs, "bytes": held.bytes}
    if counts != dict(o.heldout):
        raise DataMismatchError(f"the scored set holds {counts}; the track pins {dict(o.heldout)}")
    return held


def _count(value: object) -> int:
    """`value` when it is an integer above 0, else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _load_train(path: Path) -> _TrainFile:
    """Step 3: run the train file once, by path, and take `ROWS`, `SEQ_LEN` and `build` from it.

    `ROWS` and `SEQ_LEN` are integers, the batch for the whole run, and `build(seed)` returns a
    `Trainer`.
    """
    try:
        data = path.read_bytes()
        source = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"--train {path}: {e}") from e
    spec = importlib.util.spec_from_file_location("speedrun_train", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"--train {path}: not a Python file")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # where a dataclass in the file looks its module up
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ beside it
    try:
        spec.loader.exec_module(mod)
    except SyntaxError as e:
        raise ConfigError(f"--train {path}: {e}") from e
    finally:
        sys.dont_write_bytecode = before
    rows, seq_len = _count(getattr(mod, "ROWS", None)), _count(getattr(mod, "SEQ_LEN", None))
    build = getattr(mod, "build", None)
    if not (rows and seq_len and callable(build)):
        msg = f"--train {path} must define ROWS and SEQ_LEN, integers above 0, and build(seed)"
        raise ConfigError(msg)
    built = cast("Callable[[int], Trainer]", build)
    return _TrainFile(source, digest.digest_bytes(data), rows, seq_len, built)


def _feed(data: Path, tf: _TrainFile, seed: int) -> PackedSource:
    """Step 4: batches of the training slice from the best-fit packer, in the data seed's order."""
    split = Split(data / "train", vocab=VOCAB)
    seeds = run_seeds(seed)
    return PackedSource(
        Packer(split, tf.rows, tf.seq_len, vocab=VOCAB, data_seed=seeds["data_seed"])
    )


def _check_trainer(trainer: object, path: Path) -> None:
    """Step 5: refuse what `build` returned unless it has `Trainer`'s three methods."""
    lacks = [m for m in ("step", "arrays", "logits") if not callable(getattr(trainer, m, None))]
    if lacks:
        msg = f"--train {path}: build(seed) must return an object with step, arrays and logits"
        raise ConfigError(f"{msg}; {type(trainer).__name__} lacks {', '.join(lacks)}")


def _train(trainer: Trainer, feed: PackedSource, o: Options, log: TextIO) -> Timing:
    """Step 6: timed steps until the timed total reaches the budget, one log line each.

    Each step is timed from just before `Trainer.step` to the end of evaluating
    `Trainer.arrays()`, so its work falls inside its own interval; reading the batch is not
    timed. The first `WARMUP_STEPS` steps each count at the median time of the steps after them,
    so compilation in them is free but the training they do is not (`Timing`). The run stops
    after the step at which the timed total reaches the budget. A loss that is not finite ends
    the run with no receipt.
    """
    timing, budget = Timing(), max(1, round(o.budget_s * 1e9))
    while timing.charged_ns() < budget:
        step = len(timing.steps)
        try:
            rows = feed.next()
        except ExhaustedError as e:
            raise ConfigError(f"the training slice ran out at step {step}") from e
        progress = timing.charged_ns() / budget
        t0 = o.clock.monotonic_ns()
        loss = float(trainer.step(rows, progress))
        evaluate(trainer.arrays())
        ns = o.clock.monotonic_ns() - t0
        if not math.isfinite(loss):
            raise ConfigError(f"the loss is not finite at step {step}")
        timing.add(ns)
        line = {
            "step": step,
            "step_s": ns / 1e9,
            "warmup": step < WARMUP_STEPS,
            "loss": loss,
            "progress": progress,
        }
        log.write(json.dumps(line) + "\n")
    return timing


def _check_causal(trainer: Trainer, held: HeldOut, rows: int) -> None:
    """Step 7: logits before T/2 of the first held-out batch, when every later id changes.

    They must stay within `CAUSAL_ATOL`, and the logits must be [B, T, 49,152].
    """
    ids = next(held.batches(rows))[0]
    half = ids.shape[1] // 2
    other = ids.copy()
    span = VOCAB - FIRST_CONTENT_ID  # each later id moves to the next content id, cyclically
    other[:, half:] = FIRST_CONTENT_ID + (ids[:, half:] - FIRST_CONTENT_ID + 1) % span
    z, z2 = trainer.logits(ids), trainer.logits(other)
    if z.shape != (*ids.shape, VOCAB):
        raise ConfigError(f"train.py's logits are {z.shape}, not {(*ids.shape, VOCAB)}")
    if not mx.allclose(z[:, :half], z2[:, :half], rtol=0.0, atol=CAUSAL_ATOL, equal_nan=True):
        raise ConfigError("train.py's logits read later tokens")


def bpb_of(trainer: Trainer, held: HeldOut, rows: int) -> float:
    """Step 8: held-out bits per byte of `trainer.logits`, with `heldout.row_nats`'s nats.

    The scored set is whole documents from the start of the held-out slice, in its order, until
    they hold `EVAL_TARGETS` targets. Each document has rows of its own, of `EVAL_SEQ_LEN` + 1
    tokens, and a longer one takes several (`HeldOut` with `doc_mask=False`); each forward pass
    takes `rows` rows. It is not timed.
    """
    nats: list[NDArray[np.float64]] = []
    for ids, targets, mask in held.batches(rows):
        z = trainer.logits(ids).astype(mx.float32)
        y = mx.array(targets.astype(np.int32))
        n = mx.logsumexp(z, axis=-1) - mx.take_along_axis(z, y[..., None], axis=-1)[..., 0]
        nats.append(np.array(n).astype(np.float64)[mask > 0])
    return held.bpb(np.concatenate(nats))


def run(o: Options) -> str:
    """Check, train, score and write the receipt; return the summary line.

    Every refusal is a ConfigError, except a slice or a scored set that differs from its pin,
    which is a DataMismatchError. A score that is not finite ends the run with no receipt.
    """
    _check(o)
    tf32_off()  # refuses a process whose MLX loaded with TF32 on (the CLI turns it off first)
    start = o.clock.utc_now()
    if date.fromisoformat(o.claim[:10]) > start.date():  # `_check` made it a real date
        raise ConfigError(f"--claim {o.claim!r}: its date is after {start:%Y-%m-%d}, today in UTC")
    folder = _folder(o, start)
    held = _check_slices(o)
    commit = o.fixed_code()
    tf = _load_train(o.train)
    feed = _feed(o.data, tf, o.seed)
    begin = o.conditions()
    reset_peak()
    t0 = o.clock.monotonic_ns()
    trainer = tf.build(o.seed)
    build_ns = o.clock.monotonic_ns() - t0
    _check_trainer(trainer, o.train)
    folder.mkdir(parents=True)
    log = folder / "metrics.jsonl"
    sys.stderr.write(
        f"speedrun: training for {o.budget_s:g} s of step time, with no output until it ends; "
        f"each step is logged in {log}\n"
    )
    with log.open("w", encoding="utf-8", buffering=1) as fh:
        timing = _train(trainer, feed, o, fh)
    peak = memory()["peak_bytes"]
    sys.stderr.write(
        f"speedrun: {len(timing.steps)} steps trained; scoring the held-out set, "
        "which is not timed\n"
    )
    t0 = o.clock.monotonic_ns()
    _check_causal(trainer, held, o.eval_rows)
    bpb = bpb_of(trainer, held, o.eval_rows)
    eval_ns = o.clock.monotonic_ns() - t0
    if not math.isfinite(bpb):
        raise ConfigError("the score is not finite")
    end, steps, timed_s = o.conditions(), len(timing.steps), timing.charged_ns() / 1e9
    receipt = {
        "schema": SCHEMA,
        "category": o.category,
        "claim": o.claim,
        "kind": "record" if o.rerun_of is None else "rerun",
        "rerun_of": o.rerun_of,
        "person": o.person,
        "ai_system": o.ai_system,
        "seed": o.seed,
        "practice": o.budget_s != BUDGET_S,
        "commit": commit,
        "metalforge": __version__,
        "mlx": metadata.version("mlx"),
        "python": platform.python_version(),
        "train_py": {"sha256": tf.sha256, "source": tf.source},
        "hardware": o.hardware.describe(),
        "conditions": {k: [begin[k], end[k]] for k in ("power", "thermal_level")},
        "peak_bytes": peak,
        "data": dict(o.slices),
        "heldout": {
            "targets": held.targets,
            "docs": held.docs,
            "bytes": held.bytes,
            "seq_len": o.eval_seq_len,
            "doc_mask": False,
        },
        "rules": {"budget_s": o.budget_s, "warmup_steps": WARMUP_STEPS, "warmup_rule": WARMUP_RULE},
        "batch": {"rows": tf.rows, "seq_len": tf.seq_len},
        "steps": steps,
        "tokens": steps * tf.rows * tf.seq_len,
        "timed_s": timed_s,
        "warmup_s": sum(timing.steps[:WARMUP_STEPS]) / 1e9,
        "build_s": build_ns / 1e9,
        "eval_s": eval_ns / 1e9,
        "bpb": bpb,
        "utc_start": f"{start:%Y-%m-%dT%H:%M:%SZ}",
        "utc_end": f"{o.clock.utc_now():%Y-%m-%dT%H:%M:%SZ}",
    }
    digest.write_pretty(folder / "receipt.json", receipt)
    return f"{steps} steps, {timed_s:.3f} s timed, {bpb:.4f} bits per byte: {folder}/receipt.json"
