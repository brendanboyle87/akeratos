"""Stage 1's run on a frozen table: the base checkpoint, the ids it unmasks, the frozen step.

`SFTRun` is the pretraining `Run` (`akeratos/train/loop.py`) started from a base checkpoint
instead of fresh weights, so the step, the log, `run.json`, the save and the resume
(`akeratos/train/saver.py`) are the pretraining ones. What differs:

- The dense weights are the base's raw weights or its EMA. The optimiser state is new, so
  `optim` and `recipe` are stage 1's own.
- `post.stage` unmasks ids 1 and 2. Every other reserved id stays masked, the abstention token's
  too, so abstention is by threshold.
- The table is frozen (`post.table` off): its term enters every site, but its gradient is
  dropped and it takes no event, so its shards and read counts stay the base's. It is still
  written into every checkpoint, so `eval.heldout` scores a stage-1 checkpoint and a resume
  loads it.
- `run.json` and the checkpoints carry a `post` record, so a resume with another base or
  setting is refused.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from akeratos import export
from akeratos._compat.mlx_seam import evaluate
from akeratos.ckpt import Checkpoint
from akeratos.clock import Clock
from akeratos.configs import ModelConfig
from akeratos.data.packer import segment_ids
from akeratos.errors import AkeratosError, ConfigError
from akeratos.eval.heldout import no_write_back
from akeratos.model import EMBED, Params, kind_of
from akeratos.post.data import STAGE1_IDS
from akeratos.table.accum import Accumulator
from akeratos.table.config import TableConfig
from akeratos.table.table import Table, WorkingSet
from akeratos.train.loop import DTYPES, Run, StepResult, make_loss_fn, make_table_loss_fn, prepare
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.source import Source, StepRows

TABLE_DIR = "table"  # a checkpoint's table subdirectory


def marker_rows(params: Mapping[str, mx.array], ids: Sequence[int] = STAGE1_IDS) -> Params:
    """`params` with the rows at `ids` of E and each value-embedding table at that table's mean.

    The ids never occur as pretraining inputs, so their value-embedding rows still hold their
    random initial values; the mean row is the like-for-like choice with the tied matrix. It is
    a default with no ablation.
    """
    out = dict(params)
    for name, E in params.items():
        if name != EMBED and kind_of(name) != "value_embed":
            continue
        mask = np.zeros((E.shape[0], 1), np.bool_)
        mask[list(ids)] = True
        mean = mx.mean(E, axis=0, keepdims=True)
        out[name] = mx.where(mx.array(mask.tolist(), mx.bool_), mean, E)
    return out


def base_weights(c: Checkpoint, weights: str) -> Params:
    """The base checkpoint's raw weights (`raw`) or its EMA (`ema`)."""
    if weights == "raw":
        return c.arrays("dense")
    if weights == "ema" and (c.path / "ema.safetensors").is_file():
        return c.arrays("ema")
    raise ConfigError(f"{c.path}: no {weights!r} weights; raw, or ema where the run kept one")


def load_table(c: Checkpoint, cfg: ModelConfig, *, reads: bool = False) -> Table | None:
    """The checkpoint's table, loaded read-only, or None for a model without the table term.

    With `reads` (the stage) the saved read counts load too, so its checkpoints keep the base's.
    """
    if not cfg.table:
        return None
    saved: dict[str, Any] | None = c.state["config"].get("table")
    if not saved:
        raise ConfigError(f"{c.path}: the model has the table term but the run saved no table")
    try:
        keys = dict(saved)
        keys["orders"] = tuple(keys["orders"])
        tcfg = TableConfig(**keys).validate()
        return Table.load(c.path / TABLE_DIR, tcfg, no_write_back, reads=reads)
    except (KeyError, TypeError, ValueError, AkeratosError) as e:
        raise ConfigError(f"{c.path}: the run's table: {e}") from e


def frozen_gather(table: Table, ids: NDArray[np.int32]) -> WorkingSet:
    """`table.gather` of `ids`' keys with read counting suspended, so `table.reads` is unchanged."""
    reads, table.reads = table.reads, None
    try:
        return table.gather(table.keys(ids))
    finally:
        table.reads = reads


def _drop(i: int, g_w: mx.array) -> None:
    del i, g_w


class SFTRun(Run):
    """Stage 1 from `base`'s weights on its frozen `table`.

    `post` is the stage's record for `run.json`: the stage, the base checkpoint and step, the
    weights it started from, the rate factor and the switches. `weights` picks `raw` or `ema`
    from `base`. The other arguments are `Run`'s.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        o: OptimConfig,
        r: RecipeConfig,
        source: Source,
        out: str | Path,
        *,
        base: Checkpoint,
        weights: str,
        post: Mapping[str, Any],
        table: Table | None,
        clock: Clock,
        seed: int = 0,
        init_seed: int | None = None,
        data_seed: int | None = None,
        compute_dtype: str = "bfloat16",
        compile: bool = True,
        accum: int = 1,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            cfg,
            o,
            r,
            source,
            out,
            seed=seed,
            init_seed=init_seed,
            data_seed=data_seed,
            compute_dtype=compute_dtype,
            compile=compile,
            accum=accum,
            meta=meta,
            table=table,
            clock=clock,
        )
        params = base_weights(base, weights)
        if set(params) != set(self.stepper.params):
            raise ConfigError(f"{base.path} holds other weights than the model")
        if post["stage"] == 1:  # later stages keep the marker rows their base trained
            params = marker_rows(params)
        evaluate(params)
        self.stepper.params = params
        if self.ema is not None:
            self.ema.reset(params)
        self.post = dict(post)

    def config(self) -> dict[str, Any]:
        """The pretraining run's configs and the stage's `post` record."""
        return super().config() | {"post": self.post}

    @staticmethod
    def base_checkpoint(path: Path) -> Checkpoint:
        """The checkpoint at `path`, a run directory's newest complete one, or an export.

        An export's files must be the ones its `SHA256SUMS` lists.
        """
        return export.checkpoint(path)

    @staticmethod
    def table_of(c: Checkpoint, cfg: ModelConfig) -> Table | None:
        """`load_table` with the read counts: the entry point reaches it through the run class."""
        return load_table(c, cfg, reads=True)

    def assistant_loss(self, rows: tuple[NDArray[np.int32], NDArray[np.float32]], B: int) -> float:
        """`assistant_loss` of the current raw weights, with the run's table and compute dtype."""
        params, cfg, dtype = self.stepper.params, self.cfg, self.compute_dtype
        return assistant_loss(params, cfg, rows, B, table=self.table, compute_dtype=dtype)

    def _table_step(
        self,
        table: Table,
        acc: Accumulator,
        t: int,
        s: float,
        batches: list[StepRows],
        micro: list[tuple[mx.array, ...]],
    ) -> tuple[StepResult, dict[str, Any]]:
        """A frozen-table step: each micro-batch's working set in the forward, g_w dropped.

        The entry scalars tl stay dense weights in the scalar group and train.
        """
        del acc, t

        def build(i: int) -> Callable[[], tuple[mx.array, ...]]:
            def batch() -> tuple[mx.array, ...]:
                ws = frozen_gather(table, batches[i].ids)
                return (ws.W, ws.idx, *micro[i])

            return batch

        res = self.stepper.step(
            s=s, table_micro=[build(i) for i in range(len(micro))], on_table_grad=_drop
        )
        return res, {"frozen": True, "event": table.event, "rho": res.aux}


def assistant_loss(
    params: Params,
    cfg: ModelConfig,
    rows: tuple[NDArray[np.int32], NDArray[np.float32]],
    B: int,
    *,
    table: Table | None = None,
    compute_dtype: str = "float32",
) -> float:
    """The mean loss over the answer targets of packed held-out rows (`data.pack_rows`).

    This is the sweep's criterion, with the table term when `table` is given.
    """
    ids, mask = rows
    dtype = DTYPES[compute_dtype]
    total, count = 0.0, 0.0
    plain = make_loss_fn(cfg, dtype)
    with_table = make_table_loss_fn(cfg, dtype)

    for a in range(0, len(ids), B):
        x, m = ids[a : a + B], mask[a : a + B]
        inputs = x[:, :-1].astype(np.int32)
        batch = StepRows(inputs, x[:, 1:].astype(np.int32), segment_ids(inputs), m)
        args = prepare(batch, cfg)
        n = float(args[4])
        if n == 0:
            continue
        if table is None:
            loss = plain(params, *args)
        else:
            ws = frozen_gather(table, inputs)
            loss, _ = with_table(params, ws.W, ws.idx, *args)
        total += float(loss) * n
        count += n
    if count == 0:
        raise ConfigError("the held-out conversations hold no answer target")
    return total / count
