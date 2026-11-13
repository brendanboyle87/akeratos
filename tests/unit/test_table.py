"""The lookup table (`akeratos/table/`): its hash, shards, row-wise Adam and saved files.

The hash matches golden rows from an independent implementation; replayed events equal dense
Adam in float64; the table off is bitwise the model built without it; shards stay within their
bound; the write-back happens in place, and a stale reference to a shard, in four forms, stops
the run; a gather and an update each evaluate their shard takes once; save and resume on CPU
are bitwise. float64 runs only in this file.
"""

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import numpy.typing as npt
import pytest

import akeratos.table.shards as shards_module
import akeratos.table.table as table_module
from akeratos._compat.mlx_seam import load_safetensors, save_safetensors
from akeratos.errors import ConfigError
from akeratos.model import forward, hidden, init_params, residual, rms_norm, table_term
from akeratos.rng import derive
from akeratos.table.accum import Accumulator
from akeratos.table.config import SHARD_LIMIT, SHARD_POC, TableConfig
from akeratos.table.hasher import keys, multipliers
from akeratos.table.rate import residual_share, table_decay, table_rate
from akeratos.table.shards import ShardCopiedError, ShardStore, sr_bf16, sr_int8
from akeratos.table.table import Table, Touch, WorkingSet
from tests.unit import model_ref as ref

mx.set_default_device(mx.cpu)

MiB = 2**20
active = mx.get_active_memory


def tier_mults(seed: int, orders: tuple[int, ...]) -> list[npt.NDArray[np.uint64]]:
    """The reference's multiplier draw (levels 1, heads 1), to compare the hash alone."""
    rng = np.random.default_rng(seed)
    return [rng.integers(1, 2**63, size=(1, n), dtype=np.uint64)[0] | np.uint64(1) for n in orders]


def plan_for(table: Table, rows: npt.NDArray[np.int64]) -> tuple[tuple[Touch, ...], int]:
    """A plan over sorted unique global rows, with offsets from 1 as `Table.gather` lays out."""
    rows = np.unique(rows)
    plan, off = [], 1
    for s in np.unique(rows // table.rps):
        local = (rows[rows // table.rps == s] - s * table.rps).astype(np.int32)
        plan.append(Touch(int(s), local, off))
        off += len(local)
    return tuple(plan), off


def event(table: Table, rows: npt.NDArray[np.int64], g: npt.NDArray[np.float32], **kw: float):
    plan, end = plan_for(table, rows)
    gw = np.zeros((end, g.shape[1]), np.float32)
    gw[1:] = g
    return table.update(mx.array(gw), plan, kw["rate"], kw["decay"], int(kw["step"]))


def read(table: Table, name: str) -> npt.NDArray[np.float32]:
    """Every row's field `name`, through the store, as FP32 NumPy (small tables only)."""
    out = []
    for s in range(len(table.store)):
        got = table.store.take(s, (name,), mx.arange(table.shard_rows(s)))
        out.append(np.array(got[name].astype(mx.float32)))
    return np.concatenate(out)


def bits(table: Table) -> list[dict[str, bytes]]:
    out = []
    for s in range(len(table.store)):
        names = table.store.fields(s)
        got = table.store.take(s, names, mx.arange(table.shard_rows(s)))
        out.append({k: np.array(v.view(mx.uint8)).tobytes() for k, v in got.items()})
    return out


def fill(table: Table, w0: npt.NDArray[np.float32]) -> None:
    """Set every row to w0 by writing through the store (F32 or F2)."""
    dt = mx.float32 if table.cfg.fmt == "F32" else mx.bfloat16
    for s in range(len(table.store)):
        rows = mx.arange(table.shard_rows(s))
        part = mx.array(w0[s * table.rps : s * table.rps + table.shard_rows(s)]).astype(dt)
        mx.eval(rows, part)
        table.store.write_back(s, rows, {"rows": part})


# ---- the hash ----------------------------------------------------------------------------------

# An independent reference `hash_ids` with orders (2, 3), heads 1, rows 2²⁰, levels 1, pad_id
# 65,536 and seed 7, run on 2026-10-19.
GOLDEN_IDS = [[17, 18, 19, 1000, 49_151, 17, 17, 300], [500, 501, 502, 503, 504, 505, 506, 507]]
GOLDEN = (
    [
        [360124, 998220, 483745, 233907, 168112, 548748, 662777, 582055],
        [990940, 115034, 516398, 656014, 776750, 281449, 657032, 11340],
    ],
    [
        [532387, 41918, 301646, 975134, 393305, 493571, 73441, 868619],
        [803422, 355563, 1037269, 252725, 124309, 550243, 480978, 256615],
    ],
)
# The reference on each document of [300, 301, 0, 20, 21, 22, 0, 23, 24, 25, 0, 26] separately.
SEG_IDS = [[300, 301, 0, 20, 21, 22, 0, 23, 24, 25, 0, 26]]
SEG_GOLDEN = (
    [
        547772,
        445795,
        315528,
        486084,
        772064,
        376144,
        315528,
        119792,
        1040574,
        188606,
        315528,
        862014,
    ],
    [
        1007292,
        328694,
        303294,
        430002,
        561323,
        319798,
        303294,
        293394,
        466073,
        249540,
        303294,
        789393,
    ],
)


def test_hash_equals_tier_on_golden_ids() -> None:
    rpo = 1 << 20
    got = keys(GOLDEN_IDS, (2, 3), tier_mults(7, (2, 3)), rpo)
    for o in range(2):
        assert (got[:, :, o] - o * rpo).tolist() == GOLDEN[o]


def test_segment_pads_equal_per_document_keys() -> None:
    rpo = 1 << 20
    got = keys(SEG_IDS, (2, 3), tier_mults(7, (2, 3)), rpo)
    for o in range(2):
        assert (got[0, :, o] - o * rpo).tolist() == SEG_GOLDEN[o]
    whole = keys(
        [[300, 301, 5, 20, 21, 22, 5, 23, 24, 25, 5, 26]], (2, 3), tier_mults(7, (2, 3)), rpo
    )
    assert int((whole[0, :, 0] - got[0, :, 0] != 0).sum()) > 0


def test_multipliers_come_from_the_table_mult_stream() -> None:
    a, b = multipliers(7, (2, 3), 2), multipliers(7, (2, 3), 2)
    assert [m.tolist() for m in a] == [m.tolist() for m in b]
    assert [len(m) for m in a] == [2, 2, 3, 3]
    assert all(int(x) % 2 == 1 and 0 < int(x) < 2**63 for m in a for x in m)
    assert a[0].tolist() != multipliers(8, (2, 3), 2)[0].tolist()


def test_keys_refuse_ids_outside_uint16() -> None:
    with pytest.raises(ValueError, match="65,536"):
        keys([[65_536]], (2,), multipliers(0, (2,)), 16)


def test_keys_land_in_their_slots() -> None:
    t = Table(
        TableConfig(params=64 * 4 * 300, width=64, heads=2, max_shard_elements=32 * 256), active
    )
    rows = t.keys(ref.batch(3)[0])
    rps = t.cfg.rows_per_slot()
    for j in range(4):
        assert ((rows[:, :, j] >= j * rps) & (rows[:, :, j] < (j + 1) * rps)).all()


# ---- the shard bound ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "width", "shards", "gb"), [(9.0e9, 256, 17, 18.28), (6.0e9, 384, 12, 12.12)]
)
def test_poc_layouts(params: float, width: int, shards: int, gb: float) -> None:
    c = TableConfig(params=params, width=width).validate()
    assert c.max_shard_elements == SHARD_POC == 2**29
    assert c.n_shards() == shards
    assert c.rows_per_shard() * width <= SHARD_POC
    assert abs(c.bytes() / 1e9 - gb) < 0.01


def test_a_shard_above_the_d7_cap_is_refused() -> None:
    with pytest.raises(ConfigError, match="reduces bigger arrays to the identity"):
        TableConfig(params=1e6, width=64, max_shard_elements=SHARD_LIMIT + 1).validate()
    with pytest.raises(ConfigError, match=r"table.format 'F3'"):
        TableConfig(params=1e6, width=64, fmt="F3").validate()


def test_shards_cover_the_rows_and_the_bytes() -> None:
    for fmt, bpp in (("F2", 2 + 8 / 256), ("F1", 1 + 12 / 256), ("F32", 4 + 8 / 256)):
        cfg = TableConfig(params=256 * 1000, width=256, fmt=fmt, max_shard_elements=256 * 300)
        t = Table(cfg, active)
        assert len(t.store) == 4
        assert [t.store.rows(s) for s in range(4)] == [300, 300, 300, 100]
        assert t.store.nbytes() == pytest.approx(cfg.rows() * 256 * bpp)


# ---- replay equals dense Adam ------------------------------------------------------------------


def dense_oracle(w, sched, grads, lr, beta2, eps, decay):
    w = w.astype(np.float64).copy()
    v = np.zeros(w.shape[0])
    for e, (rows, g) in enumerate(zip(sched, grads, strict=True), start=1):
        gfull = np.zeros_like(w)
        gfull[rows] = g
        v = beta2 * v + (1 - beta2) * (gfull**2).mean(1)
        u = gfull / (np.sqrt(v)[:, None] + eps) * lr * math.sqrt(1 - beta2**e)
        u = u + np.where(u * w > 0, w * decay, 0.0)
        w = w - u
    return w


@pytest.mark.parametrize("decay", [0.0, 0.00784])
def test_replay_equals_float64_dense_adam(decay: float) -> None:
    """400 events over 3 shards, rows asleep for 389 events, F32 rows: within 1.2e-6."""
    n_rows, w, E = 600, 8, 400
    rng = np.random.default_rng(0)
    sched = []
    for e in range(1, E + 1):
        r = set(rng.choice(n_rows, size=36, replace=False).tolist())
        if e in (1, E - 10):
            r |= {0, 1, 2}
        if e == E // 2:
            r |= {3}
        r -= {4}
        if e not in (1, E - 10):
            r -= {0, 1, 2}
        sched.append(np.array(sorted(r)))
    grads = [rng.normal(0, 1, (len(r), w)).astype(np.float32) for r in sched]
    w0 = rng.normal(0, 0.05, (n_rows, w)).astype(np.float32)
    t = Table(
        TableConfig(params=n_rows * w, width=w, fmt="F32", max_shard_elements=w * 200), active
    )
    assert len(t.store) == 3
    fill(t, w0)
    for e, (r, g) in enumerate(zip(sched, grads, strict=True), start=1):
        event(t, r, g, rate=0.05, decay=decay, step=e)
    got = read(t, "rows")
    want = dense_oracle(w0, sched, grads, 0.05, 0.95, 1e-10, decay)
    assert np.abs(got - want).max() / np.abs(want).max() <= 1.2e-6
    assert np.array_equal(got[4], w0[4])
    assert t.event == E


def test_first_touch_after_a_gap_moves_4_47_eta() -> None:
    t = Table(TableConfig(params=64 * 4, width=4, fmt="F32"), active)
    for e in range(1, 300):
        event(t, np.array([1]), np.ones((1, 4), np.float32), rate=0.1, decay=0.0, step=e)
    event(t, np.array([7]), np.array([[1, -1, 1, -1]], np.float32), rate=0.1, decay=0.0, step=300)
    moved = np.abs(read(t, "rows")[7]) / 0.1
    assert np.allclose(moved, 1 / math.sqrt(0.05), rtol=1e-5)


def test_per_row_bias_moves_a_first_touch_by_eta() -> None:
    t = Table(TableConfig(params=64 * 4, width=4, fmt="F32", bias="per_row"), active)
    for e in range(1, 50):
        event(t, np.array([1]), np.ones((1, 4), np.float32), rate=0.1, decay=0.0, step=e)
    event(t, np.array([7]), np.ones((1, 4), np.float32), rate=0.1, decay=0.0, step=50)
    assert np.allclose(np.abs(read(t, "rows")[7]) / 0.1, 1.0, rtol=1e-5)
    assert read(t, "count")[[1, 7, 8]].tolist() == [49, 1, 0]


def test_stochastic_rounding_is_unbiased() -> None:
    n = 1_000_000
    x = mx.full((1000, 1000), 1.0 + 2.0**-10)
    r = np.array(sr_bf16(x, mx.random.key(1)).astype(mx.float32), np.float64)
    p, spacing = 1 / 8, 2.0**-7
    z = (r.mean() - (1 + 2.0**-10)) / (spacing * math.sqrt(p * (1 - p)) / math.sqrt(n))
    xs = mx.random.uniform(-3, 3, (1000, 1000), key=mx.random.key(2))
    rs = np.array(sr_bf16(xs, mx.random.key(3)).astype(mx.float32), np.float64) - np.array(xs)
    q, sc = sr_int8(xs, mx.random.key(4))
    ri = np.array(q, np.float64) * np.array(sc)[:, None] - np.array(xs, np.float64)
    zs = rs.mean() / (rs.std() / math.sqrt(n))
    zi = ri.mean() / (ri.std() / math.sqrt(n))
    assert max(abs(z), abs(zs), abs(zi)) < 5


@pytest.mark.parametrize("fmt", ["F2", "F1"])
def test_zero_rows_stay_zero_under_a_zero_gradient(fmt: str) -> None:
    t = Table(TableConfig(params=64 * 100, width=64, fmt=fmt), active)
    event(t, np.arange(0, 100, 3), np.zeros((34, 64), np.float32), rate=0.5, decay=0.01, step=1)
    assert not read(t, "rows").any()
    assert read(t, "stamp")[[0, 1, 3]].tolist() == [1, 0, 1]


def test_rates_floor_and_decay() -> None:
    c = TableConfig(params=1e6, width=64, lr_mult=50)
    assert table_rate(c, 1e-4, 3e-3, after_warmup=False) == pytest.approx(0.005)
    late = table_rate(c, 1e-4, 3e-3, after_warmup=True)
    assert late == pytest.approx(0.045)
    assert table_decay(c, late, 3e-3) / c.d_peak == pytest.approx(0.09)
    assert table_rate(
        TableConfig(params=1e6, width=64, lr_schedule="to0"), 1e-4, 3e-3, True
    ) == pytest.approx(0.005)
    assert table_rate(
        TableConfig(params=1e6, width=64, lr_schedule="constant"), 1e-4, 3e-3, True
    ) == pytest.approx(0.15)
    assert table_decay(TableConfig(params=1e6, width=64, decay="zero"), late, 3e-3) == 0.0


# ---- the working set and the model's table term ---------------------------------------------


def test_working_set_gradient_equals_the_dense_take_gradient() -> None:
    """W's gradient, mapped back to rows, equals `take`'s gradient over the whole table."""
    cfg = TableConfig(params=64 * 2000, width=64, fmt="F32", max_shard_elements=64 * 167)
    t = Table(cfg, active)
    fill(t, np.random.default_rng(2).normal(0, 0.1, (cfg.rows(), 64)).astype(np.float32))
    ids, _ = ref.batch(6)
    rows = t.keys(ids)
    ws = t.gather(rows)
    assert ws.W.shape == (t.u_max(*ids.shape), 64)
    probe = mx.random.normal((*ids.shape, 64), key=mx.random.key(5))

    def loss(W: mx.array, idx: mx.array) -> mx.array:
        return (table_term(W, idx, 64, mx.float32) * probe).sum()

    g_w = np.array(mx.grad(loss)(ws.W, ws.idx))
    dense = mx.array(read(t, "rows"))
    g_d = np.array(mx.grad(loss)(dense, mx.array(rows.astype(np.int32))))
    mapped = np.zeros_like(g_d)
    for touch in ws.plan:
        mapped[touch.shard * t.rps + touch.local] = g_w[
            touch.offset : touch.offset + len(touch.local)
        ]
    assert len(t.store) == 12
    assert np.abs(mapped - g_d).max() <= 1e-6 * np.abs(g_d).max()
    assert not g_w[1 + ws.touched() :].any()
    assert not g_w[0].any()


def test_table_off_is_bitwise_at_zero_entry_scalars() -> None:
    """tl = 0 with non-zero rows equals the model built without the table, FP32 and BF16."""
    ids, _ = ref.batch(7)
    on, off = ref.tiny(), ref.tiny(table=False)
    t = Table(TableConfig(params=64 * 2000, width=64), active)
    fill(t, np.random.default_rng(3).normal(0, 0.1, (t.rows, 64)).astype(np.float32))
    ws = t.gather(t.keys(ids))
    p_on, p_off = init_params(on), init_params(off)
    p_on["tl"] = mx.zeros_like(p_on["tl"])
    seg = ref.seg(mx.array(ids))
    for dt in (mx.float32, mx.bfloat16):
        term = table_term(ws.W, ws.idx, 64, dt)
        a = forward(p_on, on, mx.array(ids), seg, dt, term)
        b = forward(p_off, off, mx.array(ids), seg, dt)
        assert np.array_equal(np.array(a), np.array(b))


def test_residual_share_and_the_residual_split() -> None:
    ids, _ = ref.batch(8)
    cfg = ref.tiny()
    p = init_params(cfg)
    t = Table(TableConfig(params=64 * 2000, width=64), active)
    fill(t, np.random.default_rng(4).normal(0, 0.1, (t.rows, 64)).astype(np.float32))
    ws = t.gather(t.keys(ids))
    seg = ref.seg(mx.array(ids))
    term = table_term(ws.W, ws.idx, 64, mx.float32)
    x = residual(p, cfg, mx.array(ids), seg, mx.float32, term)
    h = hidden(p, cfg, mx.array(ids), seg, mx.float32, term)
    assert np.array_equal(
        np.array(rms_norm(x, p["final_norm.weight"], cfg.rms_norm_eps)), np.array(h)
    )
    rho = float(residual_share(p["tl"], term, x))
    s = float(np.array(p["tl"], np.float64).sum()) * np.array(term, np.float64)
    rest = np.array(x, np.float64) - s
    assert rho == pytest.approx(np.sqrt((s**2).mean()) / np.sqrt((rest**2).mean()), rel=1e-5)


# ---- the write-back in place and the stale-reference guard ------------------------------------

N_ROWS, W = 1 << 18, 64  # 32 MiB BF16 shards, scaled down


def big_table() -> Table:
    cfg = TableConfig(params=3 * N_ROWS * W, width=W, max_shard_elements=N_ROWS * W)
    return Table(cfg, active)


def touch_all_shards(t: Table, k: int = 4096) -> tuple[mx.array, WorkingSet]:
    rows = np.random.default_rng(0).choice(t.rows, size=(1, k, 1), replace=False)
    ws = t.gather(rows.astype(np.int64))
    g = mx.random.normal(ws.W.shape, key=mx.random.key(1)) * 1e-3
    mx.eval(g)
    return g, ws


def test_clean_write_back_is_in_place() -> None:
    t = big_table()
    g, ws = touch_all_shards(t)
    assert len(t.store) == 3
    assert len(ws.plan) == 3
    mx.reset_peak_memory()
    base = active()
    t.update(g, ws.plan, 0.15, 0.00784, step=0)
    assert mx.get_peak_memory() - base < t.store.nbytes(0) // 4
    assert active() - base < t.store.nbytes(0) // 8


LEAKS = {
    "view": lambda s: s["rows"][:8],
    "lazy_take": lambda s: mx.take(s["rows"], mx.arange(8), axis=0),
    "numpy_view": lambda s: np.array(s["rows"].view(mx.uint16), copy=False),
    "lazy_stat": lambda s: mx.mean(s["rows"].astype(mx.float32)),
}


@pytest.mark.parametrize("leak", sorted(LEAKS))
def test_a_stale_reference_trips_the_guard(leak: str) -> None:
    """A live view of a shard, in each of four forms, stops the run at shard 1."""
    t = big_table()
    g, ws = touch_all_shards(t)
    hold = LEAKS[leak](t.store._shards[1])  # the test plants the reference
    with pytest.raises(ShardCopiedError, match="shard 1"):
        t.update(g, ws.plan, 0.15, 0.00784, step=0)
    del hold


# ---- one evaluation for a gather's or an update's shard takes ----------------------------------


def small(fmt: str = "F2", bias: str = "global") -> TableConfig:
    """3,000 rows in 3 shards (1,100, 1,100 and 800 rows)."""
    return TableConfig(
        params=64 * 3000, width=64, max_shard_elements=64 * 1100, seed=5, fmt=fmt, bias=bias
    )


@pytest.fixture
def on_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


def evaluate_every_take(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every shard take evaluated on its own, as the reference for the merged takes."""
    take = ShardStore.take

    def take_evaluated(self, s, names, rows, **_):
        return take(self, s, names, rows, evaluated=True)

    monkeypatch.setattr(ShardStore, "take", take_evaluated)


def all_shard_rows(t: Table, seed: int, k: int = 256) -> npt.NDArray[np.int64]:
    rows = np.random.default_rng(seed).choice(t.rows, size=(1, k, 1), replace=False)
    return rows.astype(np.int64)


def five_events(fmt: str, bias: str) -> tuple[list[dict[str, bytes]], list[bytes]]:
    t = Table(small(fmt, bias), active)
    ws_bytes = []
    for i in range(5):
        ws = t.gather(all_shard_rows(t, 50 + i))
        assert len(ws.plan) == 3
        ws_bytes.append(np.array(ws.W).tobytes())
        g = mx.random.normal(ws.W.shape, key=mx.random.key(60 + i)) * 1e-3
        t.update(g, ws.plan, 0.2, 0.004, step=i)
    return bits(t), ws_bytes


@pytest.mark.usefixtures("on_cpu")
@pytest.mark.parametrize("bias", ["global", "per_row"])
@pytest.mark.parametrize("fmt", ["F2", "F1"])
def test_merged_takes_are_bitwise_equal_to_per_shard_takes(
    monkeypatch: pytest.MonkeyPatch, fmt: str, bias: str
) -> None:
    """Every gather's W and the final shards match the per-shard takes, bit for bit."""
    merged = five_events(fmt, bias)
    evaluate_every_take(monkeypatch)
    assert five_events(fmt, bias) == merged


@pytest.mark.usefixtures("on_cpu")
@pytest.mark.parametrize("merged", [True, False])
def test_a_gather_and_an_update_each_evaluate_their_takes_once(
    monkeypatch: pytest.MonkeyPatch, merged: bool
) -> None:
    """On 3 touched shards a gather evaluates once (per-shard takes: 4), and an update once
    plus 3 write-backs (per-shard takes: 1 + 3 + 3)."""
    t = Table(small(), active)  # before the counters: creating the shards evaluates
    if not merged:
        evaluate_every_take(monkeypatch)
    calls = {"table": 0, "shards": 0}
    for name, module in (("table", table_module), ("shards", shards_module)):
        inner = module.evaluate

        def counted(*args, _name=name, _inner=inner, **kw):
            calls[_name] += 1
            return _inner(*args, **kw)

        monkeypatch.setattr(module, "evaluate", counted)
    ws = t.gather(all_shard_rows(t, 7))
    assert len(ws.plan) == 3
    assert calls == ({"table": 1, "shards": 0} if merged else {"table": 1, "shards": 3})
    calls.update(table=0, shards=0)
    g = mx.random.normal(ws.W.shape, key=mx.random.key(8)) * 1e-3
    mx.eval(g)
    t.update(g, ws.plan, 0.2, 0.004, step=0)
    assert calls == ({"table": 1, "shards": 3} if merged else {"table": 1, "shards": 6})


# ---- save and resume (CPU) -----------------------------------------------------------------


def test_save_and_resume_are_bitwise(tmp_path: Path) -> None:
    """F2, 3 shards, 5 events, resume after 3: rows, v, stamps and rounding draws equal."""
    rng = np.random.default_rng(4)
    cfg = TableConfig(params=64 * 3000, width=64, max_shard_elements=64 * 1100, seed=5)
    events = []
    for _ in range(5):
        r = np.unique(rng.integers(0, 3000, 400))
        events.append((r, rng.normal(0, 1, (len(r), 64)).astype(np.float32)))
    a = Table(cfg, active)
    for i, (r, g) in enumerate(events):
        if i == 3:
            a.save(tmp_path / "t")
        event(a, r, g, rate=0.2, decay=0.004, step=i)
    b = Table.load(tmp_path / "t", cfg, active)
    assert b.event == 3
    assert b.cfg == cfg
    assert len(b.store) == 3
    for i, (r, g) in enumerate(events[3:], start=3):
        event(b, r, g, rate=0.2, decay=0.004, step=i)
    assert bits(a) == bits(b)


@pytest.mark.parametrize(
    ("fmt", "fields"), [("F2", ["rows", "stamp", "v"]), ("F1", ["rows", "scale", "stamp", "v"])]
)
def test_the_saved_files_follow_px_040(tmp_path: Path, fmt: str, fields: list[str]) -> None:
    """The files' names, sorted keys, shard metadata and the hex record."""
    cfg = TableConfig(params=64 * 3000, width=64, fmt=fmt, max_shard_elements=64 * 1100, seed=5)
    t = Table(cfg, active)
    event(t, np.arange(0, 3000, 7), np.ones((429, 64), np.float32), rate=0.1, decay=0.0, step=0)
    names = t.write(tmp_path)
    assert names == [
        "shard_0000.safetensors",
        "shard_0001.safetensors",
        "shard_0002.safetensors",
        "table.json",
    ]
    for s, name in enumerate(names[:3]):
        raw = (tmp_path / name).read_bytes()
        header = json.loads(raw[8 : 8 + int.from_bytes(raw[:8], "little")])
        assert header.pop("__metadata__") == {
            "format": "mlx",
            "schema": "metalforge.table-shard/v1",
            "schema_minor": "0",
            "shard": str(s),
            "row_start": str(s * 1100),
            "row_count": str(t.shard_rows(s)),
        }
        assert sorted(header) == fields
        # MLX's byte order follows insertion; the writer inserts in sorted key order.
        arrays, meta = load_safetensors(tmp_path / name)
        again = tmp_path / f"sorted-{s}.safetensors"
        save_safetensors(again, {k: arrays[k] for k in sorted(arrays)}, meta)
        assert again.read_bytes() == raw
    record = json.loads((tmp_path / "table.json").read_text())
    hexes = [record["mult_seed"], *(m for mm in record["mults"] for m in mm)]
    assert all(re.fullmatch(r"0x[0-9a-f]{16}", h) for h in hexes)
    assert record == {
        "schema": "metalforge.table-shard/v1",
        "schema_minor": 0,
        "rows": 3000,
        "width": 64,
        "row_format": fmt,
        "shard_elems": 64 * 1100,
        "shard_count": 3,
        "mult_seed": f"0x{derive(5, 'table.mult'):016x}",
        "mults": record["mults"],
        "event": 1,
        "beta2_history": [{"from_event": 0, "beta2": 0.95}],
        "rows_state": "stored",
        "delta_of": None,
    }


def test_a_table_before_its_first_event_saves_its_record_alone(tmp_path: Path) -> None:
    cfg = TableConfig(params=64 * 300, width=64, max_shard_elements=64 * 100)
    assert Table(cfg, active).write(tmp_path) == ["table.json"]
    assert json.loads((tmp_path / "table.json").read_text())["rows_state"] == "zeros"
    loaded = Table.load(tmp_path, cfg, active)
    assert loaded.event == 0
    assert bits(loaded) == bits(Table(cfg, active))


def test_save_replaces_atomically(tmp_path: Path) -> None:
    cfg = TableConfig(params=64 * 300, width=64, max_shard_elements=64 * 100)
    t = Table(cfg, active)
    t.save(tmp_path / "t")
    event(t, np.arange(10), np.ones((10, 64), np.float32), rate=0.1, decay=0.0, step=0)
    (tmp_path / "t.tmp").mkdir()  # a crash's leftover
    t.save(tmp_path / "t")
    assert Table.load(tmp_path / "t", cfg, active).event == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t"]


def _edit(path: Path, change) -> None:
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        (lambda d: (d / "table.json").unlink(), "no table.json"),
        (
            lambda d: _edit(
                d / "table.json", lambda r: r.update(mult_seed=int(r["mult_seed"], 16))
            ),
            "mult_seed",
        ),
        (
            lambda d: _edit(
                d / "table.json",
                lambda r: r.update(mult_seed=r["mult_seed"].upper().replace("0X", "0x")),
            ),
            "mult_seed",
        ),
        (lambda d: _edit(d / "table.json", lambda r: r["mults"][0].__setitem__(0, "0x0")), "mults"),
        (lambda d: _edit(d / "table.json", lambda r: r.update(schema_minor=1)), "schema_minor"),
        (lambda d: _edit(d / "table.json", lambda r: r.update(event=0)), "rows_state"),
        (lambda d: _edit(d / "table.json", lambda r: r.update(event="1")), "event"),
        (lambda d: (d / "shard_0001.safetensors").unlink(), "shard_0001"),
        (
            lambda d: (d / "shard_0002.safetensors").write_bytes(
                (d / "shard_0000.safetensors").read_bytes()
            ),
            "metadata",
        ),
    ],
    ids=[
        "no-record",
        "int-seed",
        "upper-hex",
        "short-mult",
        "minor",
        "state",
        "event-type",
        "no-shard",
        "wrong-shard",
    ],
)
def test_load_refuses_a_record_or_shard_that_is_not_this_tables(
    tmp_path: Path, damage, match: str
) -> None:
    cfg = TableConfig(params=64 * 300, width=64, max_shard_elements=64 * 100)
    t = Table(cfg, active)
    event(t, np.arange(10), np.ones((10, 64), np.float32), rate=0.1, decay=0.0, step=0)
    t.write(tmp_path)
    damage(tmp_path)
    with pytest.raises(ConfigError, match=match):
        Table.load(tmp_path, cfg, active)


def test_load_refuses_another_seed_or_geometry(tmp_path: Path) -> None:
    cfg = TableConfig(params=64 * 300, width=64, max_shard_elements=64 * 100)
    t = Table(cfg, active)
    event(t, np.arange(10), np.ones((10, 64), np.float32), rate=0.1, decay=0.0, step=0)
    t.write(tmp_path)
    with pytest.raises(ConfigError, match="mult_seed"):
        Table.load(tmp_path, replace(cfg, seed=1), active)
    with pytest.raises(ConfigError, match="shard_count"):
        Table.load(tmp_path, replace(cfg, max_shard_elements=64 * 150), active)
    with pytest.raises(ConfigError, match="row_format"):
        Table.load(tmp_path, replace(cfg, fmt="F1"), active)


# ---- gradient accumulation ------------------------------------------------------------------


def test_accumulated_micro_batches_make_one_event_on_the_union() -> None:
    """K = 3 micro-batches summed into one event equal one event on the summed gradient."""
    cfg = TableConfig(params=64 * 2000, width=64, fmt="F32", max_shard_elements=64 * 167)
    w0 = np.random.default_rng(6).normal(0, 0.1, (cfg.rows(), 64)).astype(np.float32)
    a, b = Table(cfg, active), Table(cfg, active)
    fill(a, w0)
    fill(b, w0)
    acc = Accumulator(cfg.row_width(), cfg.rows_per_shard())
    dense = np.zeros((cfg.rows(), 64), np.float32)
    for m in range(3):
        ws = a.gather(a.keys(ref.batch(10 + m)[0]))
        g = mx.random.normal(ws.W.shape, key=mx.random.key(20 + m))
        acc.add(g, ws)
        gn = np.array(g)
        for t in ws.plan:
            dense[t.shard * a.rps + t.local] += gn[t.offset : t.offset + len(t.local)]
    touched = np.flatnonzero(np.abs(dense).sum(1))
    assert acc.rows() == len(touched)
    g_sum, plan = acc.take()
    assert acc.rows() == 0
    a.update(g_sum, plan, 0.05, 0.004, step=1)
    event(b, touched, dense[touched], rate=0.05, decay=0.004, step=1)
    assert a.event == b.event == 1
    np.testing.assert_allclose(read(a, "rows"), read(b, "rows"), rtol=0, atol=1e-6)
    assert np.array_equal(read(a, "stamp"), read(b, "stamp"))


def test_a_cleared_accumulator_adds_nothing() -> None:
    cfg = TableConfig(params=64 * 300, width=64, max_shard_elements=64 * 100)
    t = Table(cfg, active)
    acc = Accumulator(cfg.row_width(), cfg.rows_per_shard())
    ws = t.gather(t.keys(ref.batch(1)[0]))
    acc.add(mx.ones(ws.W.shape), ws)
    acc.clear()
    g, plan = acc.take()
    assert g.shape == (0, 64)
    assert plan == ()
