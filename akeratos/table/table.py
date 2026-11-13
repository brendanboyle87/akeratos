"""The n-gram table: its rows, the per-step working set, the sparse row update and saving.

Shape legend: k orders, h heads, R rows, w the table width (the model's d); a row holds w/h
numbers. The rows are zero at the start, and (order, head) slot j owns rows [j·R/(k·h),
(j+1)·R/(k·h)). Shards are contiguous row ranges of at most `max_shard_elements` (2²⁹ by
default: 17 shards for 9.0B numbers at w 256). A step reads its rows as a working set W, an
argument of the step and never a parameter, so its gradient is W-shaped, never table-shaped.

The update is a replay row-Adam (β₁ 0, β₂ 0.95 per event, ε 1e-10), per touched row j at
event e:
    v_j ← v_j·β₂^(e - 1 - stamp_j);  v_j ← β₂·v_j + (1 - β₂)·mean(g_j²);  stamp_j ← e
    u_j ← g_j/(√v_j + ε)·η·√(1 - β₂^e)
    u_j ← u_j + [u_j·r_j > 0]·r_j·decay
    r_j ← round(r_j - u_j)
This equals dense Adam (β₁ 0, row-wise v) over every row at every event. Under `table.bias`
per_row the correction is √(1 - β₂^n_j) instead, with n_j the row's own event count.
"""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from akeratos._compat.mlx_seam import evaluate
from akeratos.errors import AkeratosError, ConfigError
from akeratos.rng import derive
from akeratos.table.config import TableConfig
from akeratos.table.hasher import keys, multipliers
from akeratos.table.shards import (
    SCHEMA,
    MemoryCounter,
    ShardStore,
    decode,
    encode,
    fsync,
    round_key,
    row_fields,
    zeros,
)

READS = "reads.npy"  # the per-row read counts beside a table's shards


def hex64(x: int) -> str:
    """A 64-bit value spelled `0x` and 16 lower-case hex digits."""
    return f"0x{x:016x}"


@dataclass(frozen=True)
class Touch:
    """The rows of one shard in a working set: local row ids and their first row in W."""

    shard: int
    local: npt.NDArray[np.int32]
    offset: int


@dataclass(frozen=True)
class WorkingSet:
    """One step's rows: W [U_max, w/h] FP32 (evaluated), idx [B, T, k·h] int32, the plan."""

    W: mx.array
    idx: mx.array
    plan: tuple[Touch, ...]

    def touched(self) -> int:
        """Unique rows touched."""
        return sum(len(t.local) for t in self.plan)


class Table:
    """The lookup table. It owns its `ShardStore`, which owns the shards.

    When loading, `store` holds the shards; without one the table allocates zero shards.
    """

    def __init__(
        self,
        cfg: TableConfig,
        active_memory: MemoryCounter,
        store: ShardStore | None = None,
    ) -> None:
        self.cfg = cfg.validate()
        self.rows = cfg.rows()
        self.rps = cfg.rows_per_shard()
        self.mults = multipliers(cfg.seed, cfg.orders, cfg.heads)
        self.event = 0
        self.reads: npt.NDArray[np.int64] | None = None
        per_row = cfg.bias == "per_row"
        if store is None:
            shards = [
                zeros(self.shard_rows(s), cfg.row_width(), cfg.fmt, per_row)
                for s in range(cfg.n_shards())
            ]
            store = ShardStore(shards, active_memory)
        if len(store) != cfg.n_shards():
            raise AkeratosError(f"{len(store)} shards for a {cfg.n_shards()}-shard layout")
        self.store = store

    # ---- layout ---------------------------------------------------------------------------
    def shard_rows(self, s: int) -> int:
        """Rows in shard s; the last may be shorter."""
        return min(self.rps, self.rows - s * self.rps)

    def u_max(self, B: int, T: int) -> int:
        """Working-set rows for a B x T batch, the zero pad included."""
        return 1 + B * T * self.cfg.lookups()

    def keys(self, ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Global row ids [B, T, k·h] for ids [B, T]."""
        c = self.cfg
        return keys(ids, c.orders, self.mults, c.rows_per_slot(), c.heads)

    def track_reads(self) -> None:
        """Count every later `gather`'s lookups per row, from zero (a fresh training run).

        Each token adds k·h lookups, a skipped step's included, so the counts depend only on the
        ids, the hash seed and the data order.
        """
        self.reads = np.zeros(self.rows, np.int64)

    # ---- working set ----------------------------------------------------------------------
    def gather(self, rows: npt.NDArray[np.int64]) -> WorkingSet:
        """The working set of global row ids [B, T, k·h]; adds to `reads` when tracking.

        W holds a zero row, then the touched rows shard by shard, then zero rows up to `u_max`,
        so the compiled step always sees one shape.
        """
        B, T, k = rows.shape
        uniq, inv, n = np.unique(rows.reshape(-1), return_inverse=True, return_counts=True)
        if self.reads is not None:
            self.reads[uniq] += n
        idx = (1 + inv).reshape(B, T, k).astype(np.int32)
        edges = np.arange(len(self.store) + 1, dtype=np.int64) * self.rps
        bounds = np.searchsorted(uniq, edges)
        rw = self.cfg.row_width()
        plan: list[Touch] = []
        parts = [mx.zeros((1, rw), mx.float32)]
        off = 1
        for s in range(len(self.store)):
            a, b = int(bounds[s]), int(bounds[s + 1])
            if a == b:
                continue
            local = (uniq[a:b] - s * self.rps).astype(np.int32)
            plan.append(Touch(s, local, off))
            got = self.store.take(s, row_fields(self.cfg.fmt), mx.array(local), evaluated=False)
            parts.append(decode(got, self.cfg.fmt))
            off += b - a
        parts.append(mx.zeros((self.u_max(B, T) - off, rw), mx.float32))
        W = mx.concatenate(parts, axis=0)
        evaluate(W)
        return WorkingSet(W, mx.array(idx), tuple(plan))

    # ---- the replay row-Adam --------------------------------------------------------------
    def update(
        self,
        g_w: mx.array,
        plan: tuple[Touch, ...],
        rate: float,
        decay: float,
        step: int,
    ) -> int:
        """One update event on the rows of `plan` (a working set's, or `Accumulator.take`'s).

        `g_w` holds each touched row's gradient at its `Touch.offset`; returns the event number.
        The caller makes no call on a skipped step, so neither the event counter nor any stamp
        moves. A write-back that copies a shard raises the store's `ShardCopiedError`.
        """
        c = self.cfg
        self.event += 1
        e = self.event
        b2 = mx.array(c.beta2, mx.float32)
        step_size = rate * math.sqrt(1.0 - c.beta2**e)
        per_row = c.bias == "per_row"
        state = ("v", "stamp", "count") if per_row else ("v", "stamp")
        new: list[tuple[int, mx.array, dict[str, mx.array]]] = []
        for t in plan:
            rows = mx.array(t.local)
            got = self.store.take(t.shard, (*row_fields(c.fmt), *state), rows, evaluated=False)
            g = g_w[t.offset : t.offset + len(t.local)].astype(mx.float32)
            r = decode(got, c.fmt)
            missed = (e - 1 - got["stamp"]).astype(mx.float32)
            v = got["v"] * mx.power(b2, missed)
            v = v * b2 + (1 - c.beta2) * mx.mean(g * g, axis=1)
            size: mx.array | float = step_size
            vals: dict[str, mx.array] = {}
            if per_row:
                n = got["count"] + 1
                size = rate * mx.sqrt(1.0 - mx.power(b2, n.astype(mx.float32)))[:, None]
                vals["count"] = n
            u = g / (mx.sqrt(v)[:, None] + c.eps) * size
            if decay:
                u = u + mx.where(u * r > 0, r * decay, 0.0)
            key = round_key(c.seed, step, t.shard)
            vals |= encode(r - u, c.fmt, c.rounding == "stochastic", key)
            vals |= {"v": v, "stamp": mx.full(rows.shape, e, mx.int32)}
            new.append((t.shard, rows, vals))
        evaluate([a for _, rows, vals in new for a in (rows, *vals.values())])
        while new:
            s, rows, vals = new.pop(0)
            self.store.write_back(s, rows, vals)
        return e

    # ---- checkpoints ----------------------------------------------------------------------
    def row_starts(self) -> list[int]:
        return [s * self.rps for s in range(self.cfg.n_shards())]

    @staticmethod
    def record(cfg: TableConfig, event: int) -> dict[str, Any]:
        """`table.json` for `cfg` at `event`: geometry, row format, seeds, counter and β₂ history.

        64-bit values are spelled by `hex64`. Before the first event the rows are all zero, so
        `rows_state` is "zeros".
        """
        return {
            "schema": SCHEMA,
            "schema_minor": 0,
            "rows": cfg.rows(),
            "width": cfg.width,
            "row_format": cfg.fmt,
            "shard_elems": cfg.max_shard_elements,
            "shard_count": cfg.n_shards(),
            "mult_seed": hex64(derive(cfg.seed, "table.mult")),
            "mults": [
                [hex64(int(m)) for m in mm] for mm in multipliers(cfg.seed, cfg.orders, cfg.heads)
            ],
            "event": event,
            "beta2_history": [{"from_event": 0, "beta2": cfg.beta2}],  # β₂ is pinned
            "rows_state": "stored" if event else "zeros",
            "delta_of": None,
        }

    def meta(self) -> dict[str, Any]:
        """This table's `table.json`."""
        return self.record(self.cfg, self.event)

    def write(self, path: Path) -> list[str]:
        """Write the shards and then `table.json` into the existing directory `path`, flushed.

        This is the call a checkpoint writer that owns the directory's atomicity makes;
        it returns the files written, the record last. At event 0 no shard is written. The read
        counts, when tracked, go in the `READS` file before the record.
        """
        meta = self.meta()
        names = self.store.save(path, self.row_starts()) if meta["rows_state"] == "stored" else []
        if self.reads is not None:
            with (path / READS).open("wb") as f:
                np.save(f, self.reads, allow_pickle=False)
            fsync(path / READS)
            names.append(READS)
        record = path / "table.json"
        record.write_text(json.dumps(meta, indent=1, sort_keys=True))
        fsync(record)
        fsync(path)
        return [*names, record.name]

    def save(self, path: Path) -> None:
        """Save atomically to the directory `path`, replacing an earlier save there.

        This is for use outside a checkpoint, whose writer calls `write`.
        """
        tmp = path.with_name(path.name + ".tmp")
        old = path.with_name(path.name + ".old")
        for leftover in (tmp, old):
            if leftover.exists():
                shutil.rmtree(leftover)
        tmp.mkdir(parents=True)
        self.write(tmp)
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        fsync(path.parent)
        if old.exists():
            shutil.rmtree(old)

    @classmethod
    def load(
        cls, path: Path, cfg: TableConfig, active_memory: MemoryCounter, *, reads: bool = False
    ) -> "Table":
        """The table `write` wrote to `path`, checked against `cfg`; no zero table is built first.

        With `reads` (a training resume) the saved read counts are restored and tracked on; a
        table saved without them tracks none rather than a partial count, and
        `akeratos/train/reads.py` rebuilds them. A shard file that is missing or not its
        shard's, or with `reads` a bad `READS` file, raises `ConfigError`.
        """
        record = path / "table.json"
        if not record.exists():
            raise ConfigError(f"{path} holds no table.json")
        meta = json.loads(record.read_text())
        event = meta.get("event")
        if not isinstance(event, int) or isinstance(event, bool) or event < 0:
            raise ConfigError(f"{record}: event {event!r}")
        want = cls.record(cfg.validate(), event)
        if meta != want:
            diff = sorted(k for k in set(meta) | set(want) if meta.get(k) != want.get(k))
            raise ConfigError(f"{record}: {diff} differ from this configuration's")
        rps = cfg.rows_per_shard()
        starts = [s * rps for s in range(cfg.n_shards())]
        store = ShardStore.load(path, starts, active_memory) if event else None
        table = cls(cfg, active_memory, store)
        table.event = event
        if reads and (path / READS).exists():
            table.reads = load_reads(path / READS, table.rows)
        return table


def load_reads(path: Path, rows: int) -> npt.NDArray[np.int64]:
    """The read counts saved at `path`, one int64 per row."""
    try:
        got = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise ConfigError(f"{path}: {e}") from e
    if got.dtype != np.int64 or got.shape != (rows,):
        raise ConfigError(f"{path}: {got.dtype} {got.shape}, expected int64 ({rows},)")
    return got
