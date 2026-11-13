"""The table's gradients, summed over the micro-batches of one step before the rows update.

With gradient accumulation, one optimiser step runs K micro-batches. The table does not change
between them, so each micro-batch gathers its own working set from the same rows, and its
working-set gradient is added here, in FP32, keyed by global row, over the union of the rows
the micro-batches touched. `take` then gives one update event on that union, at the step's rate:
one event per optimiser step, so β₂ per event stays the configured one. The dense loss is
divided by the step's global N_valid, so the sum is the step's gradient.

Each micro-batch's rows are unique, so every scatter-add has distinct indices and the sum does
not depend on the backend's order of duplicate adds; the micro-batches are merged in order, so
a rerun gives the same sums. A skipped step calls `clear`, and adds nothing to the table.
"""

from itertools import pairwise

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from akeratos._compat.mlx_seam import evaluate, scatter_add
from akeratos.table.table import Touch, WorkingSet


class Accumulator:
    """The FP32 sum of working-set gradients over the global rows they touched.

    `width` is `TableConfig.row_width()`, and `rows_per_shard` its `rows_per_shard()`, which maps
    global rows to shards.
    """

    def __init__(self, width: int, rows_per_shard: int) -> None:
        self.width = width
        self.rps = rows_per_shard
        self._rows: npt.NDArray[np.int64] = np.zeros(0, np.int64)
        self._sum: mx.array | None = None

    def rows(self) -> int:
        """Unique rows in the sum so far."""
        return len(self._rows)

    def add(self, g_w: mx.array, ws: WorkingSet) -> None:
        """Add one micro-batch's working-set gradient `g_w` [U_max, w], laid out as `ws` is."""
        rows = np.concatenate(
            [t.shard * self.rps + t.local.astype(np.int64) for t in ws.plan]
            or [np.zeros(0, np.int64)]
        )
        parts = [g_w[t.offset : t.offset + len(t.local)] for t in ws.plan]
        g = mx.concatenate(parts).astype(mx.float32) if parts else mx.zeros((0, self.width))
        if self._sum is None:
            self._rows, total = rows, g
        else:
            union = np.union1d(self._rows, rows)
            total = mx.zeros((len(union), self.width), mx.float32)
            total = scatter_add(total, mx.array(np.searchsorted(union, self._rows)), self._sum)
            total = scatter_add(total, mx.array(np.searchsorted(union, rows)), g)
            self._rows = union
        evaluate(total)
        self._sum = total

    def take(self) -> tuple[mx.array, tuple[Touch, ...]]:
        """The summed gradient and its plan for `Table.update`; the sum starts again empty."""
        g = mx.zeros((0, self.width), mx.float32) if self._sum is None else self._sum
        shard = self._rows // self.rps
        edges = np.searchsorted(shard, np.arange(int(shard.max(initial=-1)) + 2))
        plan = tuple(
            Touch(s, (self._rows[a:b] - s * self.rps).astype(np.int32), int(a))
            for s, (a, b) in enumerate(pairwise(edges))
            if b > a
        )
        self.clear()
        return g, plan

    def clear(self) -> None:
        """Drop the sum, as a skipped step does."""
        self._rows, self._sum = np.zeros(0, np.int64), None
