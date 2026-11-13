"""The training loop: the loss, one optimiser step, and the run that repeats them and logs.

    loss = (Σ w·nll + z·Σ w·lse²) / N_valid     # N_valid: the step's targets with weight 1
    step: gradients → skip check → clip → AdamW per group at its peak·s(t) → table update

Step t is the schedule's step and the data position (batches read). A skipped step (non-finite)
still advances t, so it reads its batch and moves the schedule, and the run ends at its token
budget; it moves no optimiser state and no EMA. With `accum` K above 1, one step reads K
batches and takes one optimiser step on their summed gradients, so t counts optimiser steps.
With the table on, the run is given a `Table`, which takes one update event after the dense
update of each step that is not skipped.

A run writes `run.json` (`Run.manifest`) and the log `metrics.jsonl`
(`akeratos/train/saver.py`) into `out`. Any entry point that drives this loop calls
`tf32_off()` (`akeratos/env.py`) before MLX is imported.

Shape legend: B batch, T positions, N = B·T, d width, V vocabulary, C chunk tokens.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt

from akeratos._compat.mlx_seam import evaluate, memory
from akeratos.clock import Clock
from akeratos.configs import ModelConfig
from akeratos.configs import describe as describe_model
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.model import (
    EMBED,
    Params,
    finish_logits,
    head_logits,
    hidden,
    init_params,
    masked_id_mask,
    param_counts,
    residual,
    rms_norm,
    table_term,
)
from akeratos.numerics import assert_fp32
from akeratos.numerics import record as numerics_record
from akeratos.objective import depth_targets, depth_weights, mtp_loss_sum
from akeratos.rng import run_seeds, seed_hex
from akeratos.table.accum import Accumulator
from akeratos.table.rate import residual_share, table_decay, table_rate
from akeratos.table.table import Table, WorkingSet
from akeratos.train.optim import (
    DenseOptim,
    Schedule,
    clip_scale,
    ema_for,
    group_sq_norms,
    groups_for,
    schedule_for,
)
from akeratos.train.recipe import OptimConfig, RecipeConfig
from akeratos.train.recipe import describe as describe_recipe
from akeratos.train.saver import Log, StallClock
from akeratos.train.source import Source, StepRows

N_VALID_MAX = 2**24  # FP32 holds every integer up to 2**24 exactly


def prepare_targets(
    targets: npt.ArrayLike, lmask: npt.ArrayLike | None, cfg: ModelConfig
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Host guard for one micro-batch's targets [B, T] and loss mask (None reads as all ones).

    There is no ignore index: a position leaves the loss only by its weight. The FP32 weights are
    the loss mask with the masked ids removed, and every weight-0 target becomes 0, since a masked
    id's logit is -inf and inf·0 is NaN.
    """
    t = np.asarray(targets, np.int64)
    live = np.ones(t.shape, np.bool_) if lmask is None else np.asarray(lmask) > 0
    if (t[live] < 0).any() or (t[live] >= cfg.vocab_size).any():
        raise DataMismatchError(
            "a target id outside [0, vocab_size): mx.fast.cross_entropy would wrap it"
        )
    keep = live & ~np.isin(t, cfg.masked_ids)
    return np.where(keep, t, 0).astype(np.int32), keep.astype(np.float32)


def token_ce(z: mx.array, targets: mx.array) -> mx.array:
    """Per-token cross-entropy of FP32 logits z [..., V] at int32 targets [...]."""
    if z.dtype != mx.float32:  # the CPU fallback would reduce lower logits in their own dtype
        raise TypeError(f"the cross-entropy takes FP32 logits, not {z.dtype}")
    # `mx.contiguous` avoids MLX's wrong gradients for large non-contiguous logits on Metal
    # (docs/mlx-notes.md); it copies nothing when z is already row-contiguous.
    return mx.fast.cross_entropy(mx.contiguous(z), targets)


def _chunk_sum(z: mx.array, targets: mx.array, weights: mx.array, z_loss: float) -> mx.array:
    nll = token_ce(z, targets)
    s = (nll * weights).sum()
    if z_loss:  # z·lse² per token, with lse rebuilt as nll + z[y]
        lse = nll + mx.take_along_axis(z, targets[..., None], axis=-1)[..., 0]
        s = s + mx.array(z_loss, mx.float32) * (mx.square(lse) * weights).sum()
    return s


def _chunks(
    p: Params,
    cfg: ModelConfig,
    h: mx.array,
    targets: mx.array,
    weights: mx.array,
    fn: Callable[[mx.array, mx.array, mx.array], mx.array],
) -> mx.array:
    """Sum fn(z, targets, weights) over `loss.path`: the whole [B, T, V] logits, or chunks."""
    masked = masked_id_mask(cfg)
    weights = weights.astype(mx.float32)
    if cfg.loss_path == "unchunked":
        return fn(head_logits(p, cfg, h, masked), targets, weights)
    B, T, d = h.shape
    N, C = B * T, cfg.loss_chunk
    hf, tf, wf = mx.reshape(h, (N, d)), mx.reshape(targets, (N,)), mx.reshape(weights, (N,))
    total = mx.array(0.0, mx.float32)
    # Each chunk runs under `mx.checkpoint`. Its targets and weights are bound as default
    # arguments: passing them as arguments raises a VJP error, and a closure binds late. The tied
    # matrix is an argument so its gradient crosses the checkpoint.
    for c0 in range(0, N, C):

        def chunk(
            hc: mx.array,
            E: mx.array,
            tgt: mx.array = tf[c0 : c0 + C],
            w: mx.array = wf[c0 : c0 + C],
        ) -> mx.array:
            return fn(finish_logits(hc @ E.astype(hc.dtype).T, cfg, masked), tgt, w)

        total = total + mx.checkpoint(chunk)(hf[c0 : c0 + C], p[EMBED])
    return total


def loss_sum(
    p: Params, cfg: ModelConfig, h: mx.array, targets: mx.array, weights: mx.array
) -> mx.array:
    """Token-sum loss, FP32, for the final-normed state h [B, T, d], targets and weights [B, T]."""
    return _chunks(p, cfg, h, targets, weights, lambda z, t, w: _chunk_sum(z, t, w, cfg.z_loss))


class LossTerms(NamedTuple):
    """One micro-batch's FP32 sums: Σ w·nll and Σ w·lse²."""

    ce: mx.array
    lse_sq: mx.array


def _chunk_terms(z: mx.array, targets: mx.array, weights: mx.array) -> mx.array:
    nll = token_ce(z, targets)
    lse = nll + mx.take_along_axis(z, targets[..., None], axis=-1)[..., 0]
    return mx.stack([(nll * weights).sum(), (mx.square(lse) * weights).sum()])


def loss_terms(
    p: Params, cfg: ModelConfig, h: mx.array, targets: mx.array, weights: mx.array
) -> LossTerms:
    """The logged sums of one micro-batch, the unscaled z-term's included whatever `loss.z` is."""
    s = _chunks(p, cfg, h, targets, weights, _chunk_terms)
    return LossTerms(s[0], s[1])


def n_valid(weights: Iterable[npt.ArrayLike]) -> tuple[int, mx.array]:
    """The step's global valid-target count from every micro-batch's host weights.

    The count comes back as a host int and as an FP32 scalar, so a compiled step does not
    re-trace.
    """
    count = sum(int(np.count_nonzero(np.asarray(w) > 0)) for w in weights)
    if not 1 <= count <= N_VALID_MAX:
        raise DataMismatchError(f"N_valid {count}: a step needs 1 to {N_VALID_MAX} valid targets")
    return count, mx.array(count, mx.float32)


def loss_metrics(terms: Sequence[LossTerms], count: int, z_loss: float) -> dict[str, float | int]:
    """The step's loss metrics from its micro-batches' sums and the global count.

    Each micro-batch's sums are divided by the global count in FP32 and summed, as the gradient
    is, then read to host floats.
    """
    n = mx.array(count, mx.float32)
    zero = mx.array(0.0, mx.float32)
    ce = sum((t.ce / n for t in terms), zero)
    zu = sum((t.lse_sq / n for t in terms), zero)
    return {
        "train/ce_loss": float(ce),
        "train/z_loss": float(mx.array(z_loss, mx.float32) * zu),
        "train/z_loss_unscaled": float(zu),
        "train/n_valid_tokens": count,
    }


TableLossFn = Callable[..., tuple[mx.array, mx.array]]  # (loss, rho)
MTP_RANK = 3


def make_loss_fn(cfg: ModelConfig, dtype: mx.Dtype) -> LossFn:
    """The mean loss of the dense weights over one batch, computed in `dtype`.

    It is the final-normed state's token-sum loss (the multi-token one when the targets are
    [B, T, k]) divided by N_valid. An all-zero mask gives 0/0, a non-finite loss, and so a
    skipped step. `depth_w` is the multi-token weights [k] while they are in their stages and an
    empty array otherwise.
    """

    def loss_fn(
        params: Params,
        ids: mx.array,
        segment_ids: mx.array,
        targets: mx.array,
        weights: mx.array,
        n_valid: mx.array,
        depth_w: mx.array,
    ) -> mx.array:
        h = hidden(params, cfg, ids, segment_ids, dtype)
        if targets.ndim == MTP_RANK:  # the targets' rank picks the path, so each traces once
            total = mtp_loss_sum(params, cfg, h, targets, weights, depth_w)
        else:
            total = loss_sum(params, cfg, h, targets, weights)
        return total / n_valid

    return loss_fn


def prepare(
    batch: StepRows, cfg: ModelConfig, depth_w: tuple[float, ...] | None = None
) -> tuple[mx.array, ...]:
    """The step's arguments after the weights: ids, segment ids, targets, weights, N_valid, depth_w."""
    if batch.ids.min() < 0 or batch.ids.max() >= cfg.vocab_size:
        raise ValueError("an input id outside [0, vocab_size)")
    tg, w = prepare_targets(batch.targets, batch.lmask, cfg)
    n_valid = mx.array(float(w.sum()), mx.float32)
    dw = mx.array(np.zeros(0, np.float32))
    if depth_w is not None:
        tg, w = depth_targets(batch.targets, w, k=len(depth_w))
        dw = mx.array(np.asarray(depth_w, np.float32))
    return (
        mx.array(batch.ids),
        mx.array(batch.segment_ids),
        mx.array(tg),
        mx.array(w),
        n_valid,
        dw,
    )


def make_table_loss_fn(cfg: ModelConfig, dtype: mx.Dtype) -> TableLossFn:
    """Table mode: `loss_fn(params, W, idx, *batch)` returns (the mean loss, the share rho).

    W is the working set [U_max, w] FP32 and idx its row indices [B, T, k·h], from
    `Table.gather`; the table term enters every site, and rho (`akeratos/table/rate.py`) is
    read off the same forward pass, so logging it costs no second forward. Only the loss is
    differentiated, with respect to the weights and W.
    """

    def loss_fn(
        params: Params,
        W: mx.array,
        idx: mx.array,
        ids: mx.array,
        segment_ids: mx.array,
        targets: mx.array,
        weights: mx.array,
        n_valid: mx.array,
        depth_w: mx.array,
    ) -> tuple[mx.array, mx.array]:
        t = table_term(W, idx, cfg.d_model, dtype)
        x = residual(params, cfg, ids, segment_ids, dtype, t)
        h = rms_norm(x, params["final_norm.weight"], cfg.rms_norm_eps)
        if targets.ndim == MTP_RANK:
            total = mtp_loss_sum(params, cfg, h, targets, weights, depth_w)
        else:
            total = loss_sum(params, cfg, h, targets, weights)
        rho = residual_share(
            mx.stop_gradient(params["tl"]), mx.stop_gradient(t), mx.stop_gradient(x)
        )
        return total / n_valid, rho

    return loss_fn


# The scalar loss, or (loss, aux) in table mode.
LossFn = Callable[..., Any]
GradOut = tuple[mx.array, Params, mx.array, mx.array]
TableBatch = Callable[[], Sequence[mx.array]]
TableGrad = Callable[[int, mx.array], None]


@dataclass
class StepResult:
    """What one step did: the loss, the norms, the clip and whether it was skipped.

    `norms` holds each group's gradient norm by group name and, after an update,
    the global dense norm under "dense".
    """

    step: int
    loss: float
    skipped: bool
    norms: dict[str, float] = field(default_factory=dict[str, float])
    clip_scale: float = 1.0
    aux: float | None = None  # table mode: the loss function's second output, micro-batch mean


class Stepper:
    """Runs steps of `loss_fn` on the dense FP32 weights, which it owns and replaces, with `opt`.

    A step runs in a fixed order. One graph gives the loss, the dense FP32 gradients, the squared
    norm per group and the non-finite flag, evaluated before the update is built so the
    activations are freed first. A non-finite step updates nothing and comes back `skipped`.
    Then come the clip, a global FP32 norm over the dense groups only, each group's AdamW at its
    peak times s(t), and last the table update. Micro-batches' gradient graphs run one at a time
    and their FP32 gradients are summed; each loss is already divided by the whole step's N_valid.

    With `table`, `loss_fn(params, W, *batch)` returns (loss, aux) and the working set W is
    differentiated too: its gradient never reaches the norms, the clip or the dense update, and
    a non-finite one joins the skip flag. A step then takes `table_micro` alone.
    """

    def __init__(
        self,
        loss_fn: LossFn,
        params: Mapping[str, mx.array],
        opt: DenseOptim,
        clip: float,
        *,
        compile: bool = True,
        table: bool = False,
    ) -> None:
        self.params: Params = dict(params)
        self.opt, self.clip = opt, clip
        self.step_no, self.skipped = 0, 0
        self._table = table
        order = list(opt.names)
        # In table mode g_w leaves the graph beside the dense gradients in `extra`, as
        # (g_w, g_w finite, aux), and never reaches the norms, the clip or the update; its
        # finiteness joins the flag. Without the table `extra` is the empty tuple.
        vg = mx.value_and_grad(loss_fn, argnums=[0, 1] if table else 0)

        def finish(loss: mx.array, grads: Params) -> GradOut:
            sq = group_sq_norms(grads, opt.names)
            stacked = mx.stack([sq[g] for g in order])
            finite = mx.isfinite(loss) & mx.all(mx.isfinite(stacked))
            return loss, grads, stacked, finite

        def micro(params: Params, *batch: mx.array) -> tuple[mx.array, Params, tuple[Any, ...]]:
            if not table:
                loss, grads = vg(params, *batch)
                return loss, grads, ()
            (loss, aux), (grads, g_w) = vg(params, *batch)
            return loss, grads, (g_w, mx.all(mx.isfinite(g_w)), aux)

        def add(
            acc: Params, loss_acc: mx.array, params: Params, *batch: mx.array
        ) -> tuple[mx.array, Params, tuple[Any, ...]]:
            loss, grads, extra = micro(params, *batch)
            return loss_acc + loss, {k: acc[k] + grads[k] for k in acc}, extra

        def one(params: Params, *batch: mx.array) -> tuple[Any, ...]:
            loss, grads, extra = micro(params, *batch)
            return (*finish(loss, grads), extra)

        def apply(grads: Params, params: Params, scale: mx.array) -> Params:
            return opt.apply(grads, params, scale)

        if compile:
            states = opt.states()
            self._one: Callable[..., tuple[Any, ...]] = mx.compile(one)
            self._micro: Callable[..., tuple[Any, ...]] = mx.compile(micro)
            self._add: Callable[..., tuple[Any, ...]] = mx.compile(add)
            self._finish: Callable[[mx.array, Params], GradOut] = mx.compile(finish)
            self._apply: Callable[[Params, Params, mx.array], Params] = partial(
                mx.compile, inputs=states, outputs=states
            )(apply)
        else:
            self._one, self._micro, self._add = one, micro, add
            self._finish, self._apply = finish, apply

    def _micro_grads(
        self,
        n: int,
        batch: Callable[[int], Sequence[mx.array]],
        on_grad: TableGrad | None,
    ) -> tuple[GradOut, mx.array | None]:
        """Run a step's n micro-batches: summed loss and gradients, norms, flag and mean aux.

        `batch(i)` builds micro-batch i's arrays when called, so only one working set is live
        at a time. In table mode `on_grad(i, g_w)` receives micro-batch i's evaluated g_w, each
        g_w's finiteness joins the flag and the aux is the micro-batches' mean; without the
        table the aux is None.
        """
        oks: list[mx.array] = []
        auxs: list[mx.array] = []

        def keep(i: int, extra: tuple[Any, ...]) -> None:
            if not extra:
                return
            g_w, ok, aux = extra
            if on_grad is not None:
                on_grad(i, g_w)
            oks.append(ok)
            auxs.append(aux)

        if n == 1:
            loss, grads, sq, finite, extra = self._one(self.params, *batch(0))
            evaluate(loss, grads, sq, finite, extra)
            keep(0, extra)
        else:
            loss, grads, extra = self._micro(self.params, *batch(0))
            evaluate(loss, grads, extra)
            keep(0, extra)
            for i in range(1, n):
                loss, grads, extra = self._add(grads, loss, self.params, *batch(i))
                evaluate(loss, grads, extra)
                keep(i, extra)
            loss, grads, sq, finite = self._finish(loss, grads)
        aux: mx.array | None = None
        if self._table:
            finite = finite & mx.all(mx.stack(oks))
            aux = auxs[0] if n == 1 else mx.mean(mx.stack(auxs))
        evaluate(loss, grads, sq, finite, aux)
        assert_fp32(grads, "dense gradients")
        return (loss, grads, sq, finite), aux

    def grads(self, *batch: mx.array) -> GradOut:
        """Loss, FP32 gradients, squared norms per group and the finite flag, evaluated."""
        return self._micro_grads(1, lambda i: batch, None)[0]

    def accumulated_grads(self, micro: Sequence[Sequence[mx.array]]) -> GradOut:
        """As `grads`, with the losses and FP32 gradients summed over the micro-batches."""
        return self._micro_grads(len(micro), lambda i: micro[i], None)[0]

    def table_grads(
        self, micro: Sequence[TableBatch], on_grad: TableGrad
    ) -> tuple[GradOut, mx.array]:
        """Table mode: as `accumulated_grads`, plus each micro-batch's g_w and the mean aux.

        Each `micro` entry builds its batch, (W, *rest), when called, so only one working set
        is live at a time. `on_grad(i, g_w)` receives micro-batch i's evaluated g_w before the
        skip check: on a skipped step the caller drops what it gathered. The returned flag is
        false when any g_w is non-finite, so the dense and table skips agree.
        """
        out, aux = self._micro_grads(len(micro), lambda i: micro[i](), on_grad)
        return out, cast("mx.array", aux)

    def step(
        self,
        *batch: mx.array,
        s: float | Mapping[str, float] = 1.0,
        table_update: Callable[[], Any] | None = None,
        micro: Sequence[Sequence[mx.array]] | None = None,
        table_micro: Sequence[TableBatch] | None = None,
        on_table_grad: TableGrad | None = None,
    ) -> StepResult:
        """One step: gradients, the skip check, the dense update, then the table.

        `micro` replaces `batch` with K micro-batches whose gradients are summed first.
        `table_micro` and `on_table_grad` are table mode's (`table_grads`); `table_update`
        then runs only when the step was not skipped.
        """
        aux: mx.array | None = None
        if table_micro is not None:
            if batch or micro is not None or on_table_grad is None:
                raise ValueError("table mode takes table_micro and on_table_grad alone")
            if not self._table:
                raise ValueError("table_micro needs a Stepper built with table=True")
            (loss, grads, sq, finite), aux = self.table_grads(table_micro, on_table_grad)
        elif self._table:
            raise ValueError("a Stepper built with table=True takes table_micro alone")
        elif micro is not None:
            if batch:
                raise ValueError("pass a batch or micro-batches, not both")
            loss, grads, sq, finite = self.accumulated_grads(micro)
        else:
            loss, grads, sq, finite = self.grads(*batch)
        sqs = [float(x) for x in cast("list[float]", sq.tolist())]
        result = StepResult(
            step=self.step_no,
            loss=float(loss),
            skipped=not bool(finite),
            norms={g: v**0.5 for g, v in zip(self.opt.names, sqs, strict=True)},
            aux=None if aux is None else float(aux),
        )
        self.step_no += 1
        if result.skipped:
            self.skipped += 1
            return result
        result.clip_scale, result.norms["dense"] = clip_scale(sum(sqs), self.clip)
        self.opt.set_schedule(s)
        self.params = self._apply(grads, self.params, mx.array(result.clip_scale, mx.float32))
        evaluate(self.params, self.opt.states())
        del grads
        if table_update is not None:
            table_update()
        return result


N_VALID = 4  # the place of N_valid in `prepare`'s arguments
DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


def _check(cfg: ModelConfig, r: RecipeConfig, compute_dtype: str, table: Table | None) -> None:
    errs: list[str] = []
    if cfg.table != (table is not None):
        errs.append("the model's table term and the run's table must be on together")
    if table is not None and table.cfg.width != cfg.d_model:
        errs.append(f"table.width {table.cfg.width} is not the model's d {cfg.d_model}")
    if cfg.param_kind != r.param_kind:
        errs.append(
            f"param.kind is {cfg.param_kind!r} in the model, {r.param_kind!r} in the recipe"
        )
    if compute_dtype not in DTYPES:
        errs.append(f"compute dtype {compute_dtype!r}; one of {sorted(DTYPES)}")
    if errs:
        raise ConfigError("; ".join(errs))


class Run:
    """One training run of the dense model on `source`, writing to `out`.

    `clock` is required: it times each step (`akeratos/clock.py`); the entry point passes a
    `SystemClock`, tests a fake. `seed` is the master seed, and `init_seed` and `data_seed`
    derive from it unless given (`akeratos/rng.py`). `meta` adds the entry point's own fields
    to `run.json`, such as its run shape and wall-time stop. `schedule`, such as a
    `Schedule.branch`, replaces the one the recipe gives.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        o: OptimConfig,
        r: RecipeConfig,
        source: Source,
        out: str | Path,
        *,
        seed: int = 0,
        init_seed: int | None = None,
        data_seed: int | None = None,
        compute_dtype: str = "bfloat16",
        compile: bool = True,
        mtp: bool = False,
        accum: int = 1,
        meta: Mapping[str, Any] | None = None,
        table: Table | None = None,
        schedule: Schedule | None = None,
        clock: Clock | None = None,
    ) -> None:
        if clock is None:
            raise TypeError("Run needs a clock: the entry point passes clock.SystemClock()")
        _check(cfg, r, compute_dtype, table)
        if accum < 1:
            raise ConfigError(f"accumulation {accum}: at least 1 micro-batch per step")
        self.accum = accum
        self.meta = dict(meta or {})
        self.clock = clock
        self.cfg, self.o, self.r = cfg.validate(), o.validate(), r.validate()
        self.source, self.out = source, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seeds = run_seeds(seed, init_seed, data_seed)
        self.compute_dtype, self.mtp = compute_dtype, mtp
        self.stopping = False
        params = init_params(cfg, self.seeds["init_seed"])
        opt = DenseOptim(params, groups_for(params), cfg.d_model, o, r)
        self.sched = schedule_for(r, o) if schedule is None else schedule.validate()
        self.table = table
        self.acc: Accumulator | None = None
        if table is None:
            loss_fn = make_loss_fn(cfg, DTYPES[compute_dtype])
        else:
            loss_fn = make_table_loss_fn(cfg, DTYPES[compute_dtype])
            self.acc = Accumulator(table.cfg.row_width(), table.rps)
        self.stepper = Stepper(
            loss_fn, params, opt, o.clip, compile=compile, table=table is not None
        )
        self.ema = ema_for(params, params, o, r)
        self.log = Log(self.out / "metrics.jsonl")
        self.stall_clock = StallClock(clock.boot_ns)

    @property
    def step_no(self) -> int:
        """The next step's index: steps run so far, skipped ones included."""
        return self.stepper.step_no

    def config(self) -> dict[str, Any]:
        """The configs and run settings, as JSON-ready dicts."""
        return {
            "model": asdict(self.cfg),
            "optim": asdict(self.o),
            "recipe": asdict(self.r),
            **{name: seed_hex(value) for name, value in self.seeds.items()},
            "compute_dtype": self.compute_dtype,
            "mtp": self.mtp,
            "accum": self.accum,
            "table": None if self.table is None else asdict(self.table.cfg),
        }

    def manifest(self) -> dict[str, Any]:
        """Write and return `run.json`."""
        man = {
            "config": self.config(),
            "switches": describe_model(self.cfg) + describe_recipe(self.o, self.r),
            "schedule": asdict(self.sched),
            "beta2": self.o.beta2(),
            "peaks": self.stepper.opt.peak,
            "params": param_counts(self.cfg),
            "numerics": numerics_record(),
            "source": self.source.state(),
            **self.meta,
        }
        (self.out / "run.json").write_text(json.dumps(man, indent=1))
        return man

    def depth_weights(self, t: int) -> tuple[float, ...] | None:
        """Step t's multi-token weights, or None for next-token alone, as in the whole cool-down."""
        if not self.mtp or t >= self.sched.decay_start():
            return None
        return depth_weights(t / self.sched.steps)

    def step(self) -> dict[str, Any]:
        """Run one step and return its log line."""
        t = self.step_no
        batches = [self.source.next() for _ in range(self.accum)]
        size = sum(b.ids.size for b in batches)
        if size != self.o.tokens_per_update:
            msg = f"a step of {size} tokens; batch.tokens_per_update is {self.o.tokens_per_update}"
            raise ConfigError(msg)
        micro = [prepare(b, self.cfg, self.depth_weights(t)) for b in batches]
        s = self.sched(t)
        t0 = self.clock.monotonic_ns()
        # Every micro-batch's loss is divided by the whole step's N_valid.
        if self.accum > 1:
            n_valid = sum((m[N_VALID] for m in micro), mx.array(0.0, mx.float32))
            micro = [(*m[:N_VALID], n_valid, *m[N_VALID + 1 :]) for m in micro]
        table_log: dict[str, Any] | None = None
        if self.table is not None and self.acc is not None:
            res, table_log = self._table_step(self.table, self.acc, t, s, batches, micro)
        elif self.accum == 1:
            res = self.stepper.step(*micro[0], s=s)
        else:
            res = self.stepper.step(s=s, micro=micro)
        if not res.skipped and self.ema is not None:
            self.ema.update(self.stepper.params)
        dt = (self.clock.monotonic_ns() - t0) / 1e9
        # The log line's batch hash covers all K batches.
        ids = batches[0].ids if self.accum == 1 else np.concatenate([b.ids for b in batches])
        return self.log.record(
            res,
            s,
            self.stepper.opt.peak,
            self.o.tokens_per_update,
            ids,
            dt,
            memory(),
            stall=self.stall_clock.stall(),
            table=table_log,
        )

    def _table_step(
        self,
        table: Table,
        acc: Accumulator,
        t: int,
        s: float,
        batches: list[StepRows],
        micro: list[tuple[mx.array, ...]],
    ) -> tuple[StepResult, dict[str, Any]]:
        """A table-on step: a gather per micro-batch, g_w summed, one event after the dense update.

        Its log entry's `rho` is the table's residual share.
        """
        live: dict[int, WorkingSet] = {}

        # Each micro-batch's working set is gathered just before its gradient, so one is live at
        # a time.
        def build(i: int) -> Callable[[], tuple[mx.array, ...]]:
            def batch() -> tuple[mx.array, ...]:
                live[i] = ws = table.gather(table.keys(batches[i].ids))
                return (ws.W, ws.idx, *micro[i])

            return batch

        def on_grad(i: int, g_w: mx.array) -> None:
            acc.add(g_w, live.pop(i))

        emb_peak = self.stepper.opt.peak["embed"]
        # The rate is M·η_emb(t) with its floor after warm-up (`akeratos/table/rate.py`).
        rate = table_rate(table.cfg, emb_peak * s, emb_peak, after_warmup=t >= self.sched.warmup)
        decay = table_decay(table.cfg, rate, emb_peak)
        rows = 0
        res_norms: dict[str, float] = {}

        def update() -> None:
            nonlocal rows
            rows = acc.rows()
            g_w, plan = acc.take()
            # The summed g_w's norm, logged beside the dense groups' and kept out of the clip.
            res_norms["table"] = float(mx.sqrt(mx.sum(mx.square(g_w))))
            table.update(g_w, plan, rate, decay, t)

        res = self.stepper.step(
            s=s,
            table_micro=[build(i) for i in range(len(micro))],
            on_table_grad=on_grad,
            table_update=update,
        )
        res.norms.update(res_norms)
        if res.skipped:  # the sum is dropped: no event, no stamp
            acc.clear()
        return res, {
            "rate": rate,
            "decay": decay,
            "event": table.event,
            "rows": rows,
            "rho": res.aux,
        }

    def stop(self) -> None:
        """Ask `train` to stop after the step in flight; safe to call from a signal handler."""
        self.stopping = True

    def train(
        self,
        until: int | None = None,
        on_step: Callable[[Run, dict[str, Any]], None] | None = None,
    ) -> int:
        """Run to step `until` (default: the schedule's end), or until `stop`; returns the step reached.

        `on_step(run, line)` is called after each step with its log line, between steps, where
        the run's state is whole: the checkpoint saves and the stall check
        (`akeratos/train/saver.py`) hook in there. A source that runs out raises its `ExhaustedError`.
        """
        until = self.sched.steps if until is None else min(until, self.sched.steps)
        self.manifest()
        self.stall_clock.mark()
        while self.step_no < until and not self.stopping:
            rec = self.step()
            if on_step is not None:
                on_step(self, rec)
        return self.step_no
