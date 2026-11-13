"""The readouts script for Billet and its twin, scripts/analysis/analysis.py.

Synthetic data only: a two-shard held-out split, held-out records as the scorer writes them, mc
records whose metrics come from akeratos.eval.tasks.aggregate, training logs and a probe
summary. No real run's file is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from akeratos.eval.tasks import aggregate

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analysis.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("billet_analysis", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["billet_analysis"] = mod
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ beside it
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod


an = _load()

V, T, N = 64, 2048, 24_415
LN2 = math.log(2.0)
PLANTED = -0.004  # bpb: Billet's planted held-out Δ
FINGERPRINT = "5e7c0de5" * 8  # the synthetic stores' fingerprint; the fixture makes it the real one
TB = np.random.default_rng(7).integers(1, 5, V).astype("<u2")
TB[:17] = 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- synthetic inputs


def _docs(seed: int, n: int) -> list[np.ndarray]:
    """Documents' ids: short ones, one empty, and three that need windows 0-2."""
    g = np.random.default_rng(seed)
    docs = [g.integers(17, V, int(g.integers(1, 300))) for _ in range(n)]
    return [*docs, np.zeros(0, np.int64), *(g.integers(17, V, k) for k in (T, T + 1, 2 * T + 7))]


def _write_split(root: Path, docs: list[np.ndarray]) -> Path:
    """A two-shard held-out split, token_bytes.u16 and the root manifest's SHA-256."""
    split = root / "heldout"
    split.mkdir(parents=True)
    names, tokens = [], []
    for s, part in enumerate((docs[: len(docs) // 2], docs[len(docs) // 2 :])):
        body = np.concatenate([np.concatenate([[0], d]) for d in part]).astype("<u2")
        idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in part])]).astype("<u8")
        body.tofile(split / f"{s:06d}.bin")
        idx.tofile(split / f"{s:06d}.idx")
        names.append(f"{s:06d}.bin")
        tokens.append(len(body))
    m = {"dtype": "uint16", "eos": 0, "vocab": V, "shards": names, "tokens": tokens}
    (split / "manifest.json").write_text(json.dumps(m))
    (root / "token_bytes.u16").write_bytes(TB.tobytes())
    sha = hashlib.sha256(TB.tobytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"outputs": {"token_bytes_sha256": sha}}))
    (root / "fingerprint.json").write_text(json.dumps({"fingerprint": FINGERPRINT}))
    return split


def _write_scored(
    path: Path, table: str, step: int, docs: list[np.ndarray], sets: dict[str, np.ndarray]
) -> None:
    """heldout.json and its records as the scorer's `_write_records` writes them."""
    lengths = np.array([len(d) for d in docs], np.int64)
    doc = np.repeat(np.arange(len(docs), dtype=np.int32), lengths)
    first = np.repeat(np.cumsum(lengths) - lengths, lengths)
    window = ((np.arange(len(doc)) - first) // T).astype(np.int32)
    nbytes = np.array([int(TB[d].sum()) for d in docs], np.int64)
    per_doc: dict[str, np.ndarray] = {"targets": lengths, "bytes": nbytes}
    per_target: dict[str, np.ndarray] = {"document": doc, "window": window}
    rec: dict[str, Any] = {"ckpt": str(path.parent), "step": step, "table": table}
    rec |= {"dtype": "mlx.core.float32", "data": "heldout", "seq_len": T, "docs": len(docs)}
    rec |= {"targets": len(doc), "bytes": int(nbytes.sum())}
    for name, nats32 in sets.items():
        nats = nats32.astype(np.float32).astype(np.float64)
        per_doc[name] = np.bincount(doc, weights=nats, minlength=len(docs))
        per_target[name] = nats.astype(np.float32)
        bpb = float(np.sum(nats, dtype=np.float64) / (LN2 * rec["bytes"]))
        base, _, off = name.partition(".")
        if off:
            rec[base]["table_off"] = {"bpb": bpb}
        else:
            rec[name] = {"bpb": bpb, "nats_per_target": float(nats.mean())}
    out = path.parent / f"{path.stem}.records"
    out.mkdir(parents=True)
    for f, cols in (("documents.npy", per_doc), ("targets.npy", per_target)):
        np.save(out / f, np.rec.fromarrays(list(cols.values()), names=list(cols)))
    files = ("documents.npy", "targets.npy")
    rec["records"] = {"dir": str(out), "sha256": {f: _sha(out / f) for f in files}}
    path.write_text(json.dumps(rec, sort_keys=True) + "\n")


def _arm_nats(seed: int, docs: list[np.ndarray], shift: float | None) -> dict[str, np.ndarray]:
    """Per-target nats: the twin's (shift None), or Billet's with Δ planted as shift bpb.

    Billet's EMA Δ is planted at twice the raw one, and its records also hold its table-off
    sets, as the scorer writes them.
    """
    g = np.random.default_rng(seed)
    tbytes = np.concatenate([TB[d] for d in docs]).astype(np.float64)
    base = np.random.default_rng(99).gamma(4.0, 0.6, len(tbytes))
    names = ("raw", "ema") if shift is None else ("raw", "raw.table_off", "ema", "ema.table_off")
    sets = {}
    for name in names:
        planted = 0.0 if shift is None or name.endswith("table_off") else shift
        planted *= 2 if name == "ema" else 1
        noise = g.normal(0.0, 0.01, len(tbytes))
        sets[name] = (base + planted * LN2 * tbytes + noise).astype(np.float32)
    return sets


def _write_set(root: Path) -> dict[str, list[Any]]:
    """A prepared set of 30 items a task, in tasks.py's item format."""
    g = np.random.default_rng(11)
    items: dict[str, list[Any]] = {}
    rows = {}
    for task in an.TASKS:
        its: list[Any] = []
        for i in range(30):
            if task == "lambada_openai":
                reqs = [{"context": f"the quick brown {i}", "continuation": " fox"}]
            elif task == "winogrande":
                reqs = [{"context": f"Sam met {o}", "continuation": " and left."} for o in "AB"]
            else:
                k = 2 if task == "piqa" else 4
                ctx = f"Question: q{i}\nAnswer:"
                ends = ("é" * int(g.integers(1, 9)) for _ in range(k))
                reqs = [{"context": ctx, "continuation": " " + e} for e in ends]
            gold = int(g.integers(0, len(reqs)))
            its.append({"task": task, "doc_id": str(i), "requests": reqs, "gold": gold})
        items[task] = its
        path = root / f"{task}.jsonl"
        path.write_text("".join(json.dumps(it) + "\n" for it in its))
        rows[task] = {"sha256": _sha(path), "n": len(its), "revision": "0" * 40}
    m = {"schema": "metalforge.eval.mc-set/v1", "set_id": "synthetic", "tasks": rows}
    (root / "manifest.json").write_text(json.dumps(m))
    return items


def _write_mc(root: Path, items: dict[str, list[Any]], table: str, seed: int) -> None:
    """mc.json and its samples as mc.py writes them; three items a task dropped."""
    g = np.random.default_rng(seed)
    rec: dict[str, Any] = {"step": N, "set_id": "synthetic", "dtype": "mlx.core.float32"}
    rec |= {"table": table, "limit": None, "raw": {}, "tokenizer_sha256": "c" * 64, "seq_len": T}
    rec["decontam"] = {"dir": "/decontam"} | {
        t: {"sha256": "a" * 64, "dropped": 3, "kept": len(its) - 3} for t, its in items.items()
    }
    terms = ("table", "table_off") if table == "on" else ("off",)
    samples = []
    for t, its in items.items():
        kept = its[3:]
        for weights in ("raw", "ema"):
            for term in terms:
                res = [
                    [(float(g.normal(-8.0, 2.0)), bool(g.random() < 0.5)) for _ in it["requests"]]
                    for it in kept
                ]
                if weights == "raw" and term == terms[0]:
                    rec["raw"][t] = aggregate(t, kept, res)
                for it, r in zip(kept, res, strict=True):
                    row = {"task": t, "doc_id": it["doc_id"], "weights": weights, "term": term}
                    samples.append(row | {"ll": [x[0] for x in r], "greedy": [x[1] for x in r]})
    (root / f"mc-{table}.json").write_text(json.dumps(rec))
    (root / f"mc-{table}.items.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))


def _write_log(path: Path, loss: np.ndarray, batch: list[str], last: int = N) -> None:
    lines = [
        json.dumps({"step": s, "loss": float(loss[s]), "skipped": False, "batch": batch[s]})
        for s in range(last)
    ]
    path.write_text("\n".join(lines) + "\n")


def _probe_result(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The crossover estimate, worked here with NumPy, independent of the script."""
    used = [p for p in pairs if not p["void"]]
    first = np.log([p["ratio"] for p in used if p["table_first"]])
    second = np.log([p["ratio"] for p in used if not p["table_first"]])
    est = (first.mean() + second.mean()) / 2
    df = len(used) - 2
    ss = ((first - first.mean()) ** 2).sum() + ((second - second.mean()) ** 2).sum()
    se = math.sqrt(ss / df / 4 * (1 / len(first) + 1 / len(second)))
    t = {5: 2.571, 6: 2.447}[df]
    ci = [math.exp(est - t * se), math.exp(est + t * se)]
    return {"pairs_used": len(used), "ratio": math.exp(est), "ci95": ci, "df": df}


def _write_probe(path: Path, ratios: list[float]) -> None:
    """A probe summary: eight pairs in ABBABAABBAABABBA's order, the third void."""
    order = "ABBABAABBAABABBA"
    pairs = []
    for i, r in enumerate(ratios):
        off = 15.0 + 0.1 * i
        p = {"blocks": [2 * i + 1, 2 * i + 2], "table_first": order[2 * i] == "A"}
        pairs.append(p | {"on_mean_s": off * r, "off_mean_s": off, "void": i == 2})
        pairs[-1]["ratio"] = pairs[-1]["on_mean_s"] / off
    design = {"order": order, "minutes": 6.0, "warmup_minutes": 5.0, "drop": 3, "data": "made up"}
    summary = {
        "design": design,
        "pairs": pairs,
        "result": _probe_result(pairs),
        "checks": {"every_block_exit_3": True},
    }
    path.write_text(json.dumps(summary))


def _curve(x: np.ndarray) -> np.ndarray:
    return 2.5 + 3.0 / (1.0 + x / 2000.0)


def _curves(stretch: float) -> tuple[np.ndarray, np.ndarray]:
    """Billet's loss f(s) and a twin that reaches f(s) at step stretch times s."""
    s = np.arange(N, dtype=np.float64)
    return _curve(s), _curve(s / stretch)


def _world(
    root: Path, shift: float = PLANTED, stretch: float = 1.08, offset: float = 0.0
) -> list[str]:
    """Every input of a full run, synthetic; returns the command line.

    Billet's held-out Δ is planted as `shift`; the twin's training loss reaches Billet's at
    `stretch` times the step, and Billet's loss sits `offset` above that curve.
    """
    docs = _docs(1, 40)
    split = _write_split(root / "store", docs)
    for name, step, planted in (("final", N, shift), ("pinned", 19_532, shift / 2)):
        _write_scored(root / f"b-{name}.json", "on", step, docs, _arm_nats(2, docs, planted))
        _write_scored(root / f"t-{name}.json", "off", step, docs, _arm_nats(3, docs, None))
    targets = sum(len(d) for d in docs)
    np.save(root / "reads.npy", np.random.default_rng(4).integers(0, 50, targets))
    o = np.zeros(len(docs), dtype=[("distinct", "<i8"), ("found", "<i8")])
    o["distinct"] = [max(len(d) - 12, 0) for d in docs]
    o["found"] = np.random.default_rng(5).integers(0, 3, len(docs)) * o["distinct"] // 4
    np.save(root / "overlap.npy", o)
    (root / "set").mkdir()
    items = _write_set(root / "set")
    _write_mc(root, items, "on", 6)
    _write_mc(root, items, "off", 7)
    lb, lt = _curves(stretch)
    batch = [f"{s:016x}" for s in range(N)]
    twin_batch = [*batch[:100], *("f" * 16 for _ in range(3)), *batch[103:]]  # steps 100-102 differ
    lb[2_000] = math.nan  # a non-finite loss in the 10% window
    _write_log(root / "b-metrics.jsonl", lb + offset, batch)
    _write_log(root / "t-metrics.jsonl", lt, twin_batch)
    _write_probe(root / "summary.json", [1.04, 1.03, 2.0, 1.06, 1.04, 1.05, 1.05, 1.04])
    s6 = _docs(8, 30)
    _write_split(root / "s6store", s6)
    _write_scored(root / "b-s6.json", "on", N, s6, _arm_nats(9, s6, shift))
    _write_scored(root / "t-s6.json", "off", N, s6, _arm_nats(10, s6, None))
    np.save(root / "thirds.npy", np.arange(len(s6)) % 3)

    def two(name: str) -> list[str]:
        return [str(root / f"b-{name}"), str(root / f"t-{name}")]

    argv = ["--out", str(root / "report.json"), "--split", str(split)]
    argv += ["--heldout", *two("final.json"), "--pinned", *two("pinned.json")]
    argv += ["--reads", str(root / "reads.npy"), "--overlap", str(root / "overlap.npy")]
    argv += [
        "--mc",
        str(root / "mc-on.json"),
        str(root / "mc-off.json"),
        "--set",
        str(root / "set"),
    ]
    argv += ["--samples", str(root / "mc-on.items.jsonl"), str(root / "mc-off.items.jsonl")]
    argv += ["--metrics", *two("metrics.jsonl"), "--probe", str(root / "summary.json")]
    return [*argv, "--s6", *two("s6.json"), "--thirds", str(root / "thirds.npy")]


def _gold_bpb(root: Path, samples: str, term: str, task: str) -> float:
    """A task's gold-answer bpb worked from the files: the gold bits over the gold bytes."""
    items = {}
    for line in (root / "set" / f"{task}.jsonl").read_text().splitlines():
        it = json.loads(line)
        items[it["doc_id"]] = it
    bits = nbytes = 0.0
    for line in (root / samples).read_text().splitlines():
        row = json.loads(line)
        if (row["task"], row["weights"], row["term"]) == (task, "raw", term):
            it = items[row["doc_id"]]
            bits -= row["ll"][it["gold"]] / LN2
            nbytes += len(it["requests"][it["gold"]]["continuation"].encode())
    return bits / nbytes


@pytest.fixture
def synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic store and held-out counts in place of the real ones, and fewer resamples."""
    docs = _docs(1, 40)
    counts = {"docs": len(docs), "targets": sum(len(d) for d in docs)}
    counts["bytes"] = int(sum(int(TB[d].sum()) for d in docs))
    for k, v in counts.items():
        monkeypatch.setitem(an.EXPECT, k, v)
    monkeypatch.setattr(an, "STORE", FINGERPRINT[:16])
    monkeypatch.setattr(an, "RESAMPLES", 2_000)


def _report(root: Path, argv: list[str]) -> dict[str, Any]:
    assert an.main(argv) == 0
    return json.loads((root / "report.json").read_text())


def _records(root: Path, arm: str, name: str = "documents.npy") -> np.ndarray:
    return np.load(root / f"{arm}-final.records" / name)


def _ratio_interval(cols: list[np.ndarray], gen: np.random.Generator) -> list[float]:
    """The documented draws, one integers(0, n, n) per resample, each giving Σ(b - t) ÷ (ln 2 · Σy)."""
    n = len(cols[0])
    reps = np.empty(an.RESAMPLES)
    for r in range(an.RESAMPLES):
        i = gen.integers(0, n, n)
        reps[r] = (cols[0][i].sum() - cols[1][i].sum()) / (LN2 * cols[2][i].sum())
    return [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def _delta(nb: np.ndarray, nt: np.ndarray, y: np.ndarray) -> float:
    return float((nb.sum() - nt.sum()) / (LN2 * y.sum()))


def _lower(x: np.ndarray, k: int) -> np.ndarray:
    """Bins by hand: edge j is the ⌈jn/k⌉-th smallest value; a value at an edge goes below."""
    edges = [np.sort(x)[math.ceil(j * len(x) / k) - 1] for j in range(1, k)]
    return np.array([sum(e < v for e in edges) for v in x])


def _strict(token: str) -> None:
    raise AssertionError(f"{token} is not JSON")


# ---------------------------------------------------------------- the readouts end to end


@pytest.mark.usefixtures("synthetic")
def test_a_full_report_on_synthetic_inputs(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    p = rep["primary"]
    b, t = (np.load(tmp_path / f"{a}-final.records" / "documents.npy") for a in "bt")
    want = (b["raw"].sum() - t["raw"].sum()) / (LN2 * b["bytes"].sum())
    assert p["delta"] == pytest.approx(want, abs=1e-12)
    assert p["delta"] == pytest.approx(PLANTED, abs=5e-4)
    assert p["ci95"][0] <= PLANTED <= p["ci95"][1]
    assert p["band"] == "no measurable difference"
    assert p["equivalence"]["claimed"] is (abs(p["delta"]) < 0.0018)
    assert rep["bar"] == an.bar(an.EXPECT["targets"], an.EXPECT["bytes"])
    assert rep["ema"]["delta"] == pytest.approx(2 * PLANTED, abs=5e-4)
    assert rep["ema"]["band"] == "the table helps"  # -0.008 against the synthetic set's δ, 0.0055
    assert "equivalence" not in rep["ema"]
    assert rep["pinned"]["delta"] == pytest.approx(PLANTED / 2, abs=5e-4)
    assert set(rep["splits"]) == {
        *(f"read-count decile {j}" for j in range(1, 11)),
        "first windows",
        *(f"13-gram overlap: {g}" for g in ("none", "any")),
        *(f"13-gram overlap: any, quartile {j}" for j in range(1, 5)),
    }
    for task in an.TASKS:
        z = rep["zero_shot"]["tasks"][task]
        assert z["items"] == 27
        assert z["p holm"] >= z["p"]
        assert z["delta_acc"] == pytest.approx(z["billet_acc"] - z["twin_acc"])
        only = z["right for Billet only"] - z["right for the twin only"]
        assert only == round(z["delta_acc"] * 27)
    assert rep["zero_shot"]["tasks"]["piqa"]["gold bpb billet"] == pytest.approx(
        _gold_bpb(tmp_path, "mc-on.items.jsonl", "table", "piqa")
    )
    assert rep["pinned"]["status"] == "described, not tested"
    m = rep["memorisation"]
    assert m["held-out delta"] == p["delta"]
    assert [m[k]["documents"] for k in ("first third", "middle third", "last third", "all")] == [
        12,
        11,
        11,
        34,
    ]
    assert m["all"]["delta"] == pytest.approx(PLANTED, abs=5e-4)
    assert m["all"]["gap delta"] == pytest.approx(m["all"]["delta"] - p["delta"], abs=1e-12)
    assert m["all"]["gap delta ci95"][0] <= m["all"]["gap delta"] <= m["all"]["gap delta ci95"][1]
    assert m["exploratory"].endswith("M1 does not name them")
    assert rep["per_mac_hour"]["reading"] == "the table leads per Mac-hour in the hold"
    labels = [v["interval"] for v in (p, rep["ema"], rep["pinned"], *rep["splits"].values())]
    labels += [rep["training_loss"]["interval"], rep["memorisation"]["interval"]]
    labels += [z["interval"] for z in rep["zero_shot"]["tasks"].values()]
    assert all(lab.startswith("evaluation noise only") for lab in labels)
    assert rep["versions"] == {"python": platform.python_version(), "numpy": np.__version__}
    distinct = np.load(tmp_path / "overlap.npy")["distinct"]
    nothing = rep["bins"]["13-gram overlap"]["documents in none with no 13-gram to find"]
    assert nothing == int((distinct == 0).sum())
    assert 0 < nothing < rep["bins"]["13-gram overlap"]["documents"]["none"]


@pytest.mark.usefixtures("synthetic")
def test_the_first_window_split_against_its_records(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    b, t = (np.load(tmp_path / f"{a}-final.records" / "targets.npy") for a in "bt")
    first = b["window"] == 0
    assert int(first.sum()) < len(first)  # the long documents have later windows
    docs = _docs(1, 40)
    tbytes = np.concatenate([TB[d] for d in docs]).astype(np.float64)
    nb, nt = (x["raw"].astype(np.float64) for x in (b, t))
    want = (nb[first].sum() - nt[first].sum()) / (LN2 * tbytes[first].sum())
    assert rep["splits"]["first windows"]["delta"] == pytest.approx(want, abs=1e-12)
    assert rep["bins"]["first window targets"] == int(first.sum())


@pytest.mark.usefixtures("synthetic")
def test_training_loss_windows_and_left_out_steps(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path, stretch=1.0, offset=0.01))
    w = rep["training_loss"]["windows"]
    assert [w[k]["steps"] for k in ("10%", "25%", "50%", "80%", "100%")] == [
        [1441, 2440],
        [5103, 6102],
        [11207, 12206],
        [18532, 19531],
        [23415, 24414],
    ]
    assert w["10%"]["left out, loss not finite"] == 1
    assert w["10%"]["kept"] == 999
    assert rep["training_loss"]["all steps"]["left out, batch hashes differ"] == 3
    for v in w.values():
        assert v["delta"] == pytest.approx(0.01, abs=1e-9)  # Billet sat 0.01 above the twin


@pytest.mark.usefixtures("synthetic")
def test_readouts_without_inputs_are_reported_as_not_run(tmp_path: Path) -> None:
    argv = _world(tmp_path)
    keep = ("--out", "--split", "--heldout")
    short = [a for i, a in enumerate(argv) if a in keep or (i and argv[i - 1] in keep)]
    short += [argv[argv.index("--heldout") + 2]]
    rep = _report(tmp_path, short)
    for name in ("pinned", "zero_shot", "training_loss", "per_mac_hour", "memorisation"):
        assert rep[name] == "not run: inputs not given"
    assert rep["splits"]["read-count deciles"] == "not run: inputs not given"
    assert rep["splits"]["13-gram overlap"] == "not run: inputs not given"
    assert rep["splits"]["first windows"]["status"] == "described, not tested"


@pytest.mark.usefixtures("synthetic")
def test_a_relative_records_dir_is_found_as_heldout_writes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _world(tmp_path)
    rec = json.loads((tmp_path / "b-final.json").read_text())
    rec["records"]["dir"] = f"{tmp_path.name}/b-final.records"  # relative to the scorer's cwd
    (tmp_path / "b-final.json").write_text(json.dumps(rec))
    monkeypatch.chdir(tmp_path.parent)
    assert _report(tmp_path, argv)["primary"]["delta"] == pytest.approx(PLANTED, abs=5e-4)
    rec["records"]["dir"] = "b-final.records"  # a run folder moved with its records
    (tmp_path / "b-final.json").write_text(json.dumps(rec))
    monkeypatch.chdir(tmp_path / "set")
    assert _report(tmp_path, argv)["primary"]["delta"] == pytest.approx(PLANTED, abs=5e-4)


@pytest.mark.usefixtures("synthetic")
def test_the_primary_interval_is_a_paired_document_bootstrap(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    b, t = _records(tmp_path, "b"), _records(tmp_path, "t")
    want = _ratio_interval([b["raw"], t["raw"], b["bytes"].astype(float)], an.rng("documents"))
    assert rep["primary"]["ci95"] == pytest.approx(want, rel=1e-9)


@pytest.mark.usefixtures("synthetic")
def test_the_pinned_interval_is_a_paired_document_bootstrap(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    b, t = (np.load(tmp_path / f"{a}-pinned.records" / "documents.npy") for a in "bt")
    want = _ratio_interval([b["raw"], t["raw"], b["bytes"].astype(float)], an.rng("pinned"))
    assert rep["pinned"]["ci95"] == pytest.approx(want, rel=1e-9)


@pytest.mark.usefixtures("synthetic")
def test_every_held_out_figure_follows_the_primarys_paired_draws(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    b, t = _records(tmp_path, "b"), _records(tmp_path, "t")
    bt, tt = _records(tmp_path, "b", "targets.npy"), _records(tmp_path, "t", "targets.npy")
    n, y = len(b), b["bytes"].astype(np.float64)
    doc = bt["document"].astype(np.int64)
    tbytes = np.concatenate([TB[d] for d in _docs(1, 40)]).astype(np.float64)
    per_target = [x["raw"].astype(np.float64) for x in (bt, tt)] + [tbytes]

    def by_target(m: np.ndarray) -> list[np.ndarray]:  # each document's sums over its targets in m
        return [np.bincount(doc, weights=w * m, minlength=n) for w in per_target]

    o = np.load(tmp_path / "overlap.npy")
    hit = o["found"] > 0
    q = _lower(o["found"][hit] / o["distinct"][hit], 4)
    groups = {"none": ~hit, "any": hit}
    groups |= {
        f"any, quartile {j + 1}": np.isin(np.arange(n), np.flatnonzero(hit)[q == j])
        for j in range(4)
    }
    dec = _lower(np.load(tmp_path / "reads.npy"), 10)
    figures = {"ema": [b["ema"], t["ema"], y], "first windows": by_target(bt["window"] == 0)}
    figures |= {f"read-count decile {j + 1}": by_target(dec == j) for j in range(10)}
    figures |= {
        f"13-gram overlap: {g}": [b["raw"] * m, t["raw"] * m, y * m] for g, m in groups.items()
    }
    gen = an.rng("documents")  # every held-out figure takes the primary's draws
    draws = [gen.integers(0, n, n) for _ in range(an.RESAMPLES)]
    for name, cols in figures.items():
        got = rep["ema"] if name == "ema" else rep["splits"][name]
        assert got["delta"] == pytest.approx(_delta(*cols), abs=1e-12), name
        sums = np.array([[c[i].sum() for c in cols] for i in draws])
        has = sums[:, 2] > 0
        reps = (sums[has, 0] - sums[has, 1]) / (LN2 * sums[has, 2])
        assert got["resamples left out, no bytes"] == int((~has).sum()), name
        want = np.percentile(reps, [2.5, 97.5]).tolist()
        assert got["ci95"] == pytest.approx(want, rel=1e-9, abs=1e-12), name


@pytest.mark.usefixtures("synthetic")
def test_memorisation_intervals_follow_the_documented_draws(tmp_path: Path) -> None:
    m = _report(tmp_path, _world(tmp_path))["memorisation"]
    sb, st = (np.load(tmp_path / f"{a}-s6.records" / "documents.npy") for a in "bt")
    hb, ht = _records(tmp_path, "b"), _records(tmp_path, "t")
    thirds = np.load(tmp_path / "thirds.npy")
    gen, n = an.rng("documents"), len(hb)  # the held-out replicates are the primary's draws
    held = np.empty((an.RESAMPLES, 2))
    for r in range(an.RESAMPLES):
        i = gen.integers(0, n, n)
        y = LN2 * hb["bytes"][i].sum()
        held[r] = hb["raw"][i].sum() / y, ht["raw"][i].sum() / y
    sums = {}
    for j, name in enumerate(("first third", "middle third", "last third")):
        docs = np.flatnonzero(thirds == j)  # resampled within its third, on its own stream
        gen, k = an.rng("training documents", j), len(docs)
        s = np.empty((an.RESAMPLES, 3))
        for r in range(an.RESAMPLES):
            i = docs[gen.integers(0, k, k)]
            s[r] = sb["raw"][i].sum(), st["raw"][i].sum(), sb["bytes"][i].sum()
        sums[name] = s
    sums["all"] = sums["first third"] + sums["middle third"] + sums["last third"]
    for name, s in sums.items():
        rb, rt = s[:, 0] / (LN2 * s[:, 2]), s[:, 1] / (LN2 * s[:, 2])
        gb, gt = rb - held[:, 0], rt - held[:, 1]
        for key, reps in (
            ("ci95", rb - rt),
            ("gap billet ci95", gb),
            ("gap twin ci95", gt),
            ("gap delta ci95", gb - gt),
        ):
            want = np.percentile(reps, [2.5, 97.5]).tolist()
            assert m[name][key] == pytest.approx(want, abs=1e-12)


@pytest.mark.usefixtures("synthetic")
def test_the_overlap_and_read_count_splits_by_hand(tmp_path: Path) -> None:
    s = _report(tmp_path, _world(tmp_path))["splits"]
    b, t = _records(tmp_path, "b"), _records(tmp_path, "t")
    y = b["bytes"].astype(float)
    o = np.load(tmp_path / "overlap.npy")
    hit = o["found"] > 0
    for name, m in (("none", ~hit), ("any", hit)):
        want = _delta(b["raw"][m], t["raw"][m], y[m])
        assert s[f"13-gram overlap: {name}"]["delta"] == pytest.approx(want, abs=1e-12)
    q = _lower(o["found"][hit] / o["distinct"][hit], 4)
    for j in range(4):
        m = np.zeros(len(hit), bool)
        m[np.flatnonzero(hit)[q == j]] = True
        got = s[f"13-gram overlap: any, quartile {j + 1}"]
        assert got["bytes"] == int(y[m].sum())
        if m.any():
            want = _delta(b["raw"][m], t["raw"][m], y[m])
            assert got["delta"] == pytest.approx(want, abs=1e-12)
    bt, tt = _records(tmp_path, "b", "targets.npy"), _records(tmp_path, "t", "targets.npy")
    tbytes = np.concatenate([TB[d] for d in _docs(1, 40)]).astype(float)
    dec = _lower(np.load(tmp_path / "reads.npy"), 10)
    for j in range(10):
        m = dec == j
        got = s[f"read-count decile {j + 1}"]
        assert got["bytes"] == int(tbytes[m].sum())
        want = _delta(bt["raw"][m].astype(float), tt["raw"][m].astype(float), tbytes[m])
        assert got["delta"] == pytest.approx(want, abs=1e-9)


@pytest.mark.usefixtures("synthetic")
def test_an_empty_bin_has_null_figures_and_says_so(tmp_path: Path) -> None:
    argv = _world(tmp_path)
    np.save(tmp_path / "reads.npy", np.zeros_like(np.load(tmp_path / "reads.npy")))
    o = np.load(tmp_path / "overlap.npy")
    o["found"] = o["distinct"]  # every share is 1, so quartiles 2-4 are empty
    np.save(tmp_path / "overlap.npy", o)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no 0 ÷ 0 on the way
        assert an.main(argv) == 0
    rep = json.loads((tmp_path / "report.json").read_text(), parse_constant=_strict)
    s = rep["splits"]
    empty = [f"read-count decile {j}" for j in range(2, 11)]
    empty += [f"13-gram overlap: any, quartile {j}" for j in range(2, 5)]
    keys = ("bytes", "billet_bpb", "twin_bpb", "delta", "ci95", "resamples left out, no bytes")
    for name in empty:
        assert [s[name][k] for k in keys] == [0, None, None, None, None, an.RESAMPLES]
        assert s[name]["status"] == "empty bin: no bytes, so no bpb"
    assert s["read-count decile 1"]["status"] == "described, not tested"
    assert s["read-count decile 1"]["delta"] == pytest.approx(rep["primary"]["delta"], abs=1e-12)
    whole = s["13-gram overlap: any"]["delta"]
    assert s["13-gram overlap: any, quartile 1"]["delta"] == pytest.approx(whole, abs=1e-12)


@pytest.mark.usefixtures("synthetic")
def test_a_resample_with_no_bytes_in_a_bin_is_left_out_and_counted(tmp_path: Path) -> None:
    s = _report(tmp_path, _world(tmp_path))["splits"]
    b, t = _records(tmp_path, "b"), _records(tmp_path, "t")
    y = b["bytes"].astype(float)
    o = np.load(tmp_path / "overlap.npy")
    hit = o["found"] > 0
    q = _lower(o["found"][hit] / o["distinct"][hit], 4)
    gen, n = an.rng("documents"), len(y)  # every split takes the primary's draws
    draws = [gen.integers(0, n, n) for _ in range(an.RESAMPLES)]
    left = []
    for j in range(4):
        m = np.zeros(n, bool)
        m[np.flatnonzero(hit)[q == j]] = True
        reps = []
        for i in draws:
            k = i[m[i]]  # the drawn documents in the bin
            if y[k].sum() > 0:
                reps.append((b["raw"][k].sum() - t["raw"][k].sum()) / (LN2 * y[k].sum()))
        got = s[f"13-gram overlap: any, quartile {j + 1}"]
        left.append(got["resamples left out, no bytes"])
        assert left[-1] == an.RESAMPLES - len(reps)
        assert got["ci95"] == pytest.approx(np.percentile(reps, [2.5, 97.5]).tolist(), rel=1e-9)
    assert max(left) > 0  # the quartiles here are small enough for some resamples to miss one


@pytest.mark.usefixtures("synthetic")
def test_a_bin_of_one_document_has_no_interval_and_says_why(tmp_path: Path) -> None:
    argv = _world(tmp_path)
    o = np.load(tmp_path / "overlap.npy")
    hit = np.flatnonzero(o["distinct"] >= 12)[:5]
    o["found"] = 0
    o["found"][hit] = [d * (k + 1) // 6 for k, d in enumerate(o["distinct"][hit])]  # 5 shares
    np.save(tmp_path / "overlap.npy", o)
    rep = _report(tmp_path, argv)
    held = rep["bins"]["13-gram overlap"]["documents"]
    assert [held[f"any, quartile {j}"] for j in range(1, 5)] == [2, 1, 1, 1]
    s = rep["splits"]
    b, t = _records(tmp_path, "b"), _records(tmp_path, "t")
    y = b["bytes"].astype(np.float64)
    for j in (2, 3, 4):  # quartile j holds hit[j] alone
        got, k = s[f"13-gram overlap: any, quartile {j}"], hit[j : j + 1]
        assert (got["ci95"], got["resamples left out, no bytes"]) == (None, None)
        assert "one document" in got["interval"]
        assert got["delta"] == pytest.approx(_delta(b["raw"][k], t["raw"][k], y[k]), abs=1e-12)
    two = s["13-gram overlap: any, quartile 1"]  # hit[0] and hit[1]: round 2's rule
    m = np.isin(np.arange(len(y)), hit[:2])
    gen, n = an.rng("documents"), len(y)
    reps = []
    for _ in range(an.RESAMPLES):
        k = (i := gen.integers(0, n, n))[m[i]]  # the drawn documents in the bin
        if len(k):
            reps.append((b["raw"][k].sum() - t["raw"][k].sum()) / (LN2 * y[k].sum()))
    assert two["resamples left out, no bytes"] == an.RESAMPLES - len(reps) > 0
    assert two["ci95"] == pytest.approx(np.percentile(reps, [2.5, 97.5]).tolist(), rel=1e-9)
    assert two["interval"] == an.noise("held-out documents")


@pytest.mark.usefixtures("synthetic")
def test_steps_with_other_batches_are_left_out_of_their_window(tmp_path: Path) -> None:
    argv = _world(tmp_path, stretch=1.0, offset=0.01)
    path = tmp_path / "t-metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for s in (1_500, 1_501, 1_502):  # inside the 10% window, steps 1,441-2,440
        rows[s]["batch"], rows[s]["loss"] = "e" * 16, rows[s]["loss"] + 5.0
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    w = _report(tmp_path, argv)["training_loss"]["windows"]["10%"]
    assert w["left out, batch hashes differ"] == 3
    assert w["kept"] == 996
    assert w["delta"] == pytest.approx(0.01, abs=1e-9)


# ---------------------------------------------------------------- held-out Δ and its interval


def test_a_planted_delta_is_recovered_inside_its_interval() -> None:
    g = np.random.default_rng(3)
    y = g.integers(50, 4000, 400).astype(np.float64)
    nt = y * g.normal(0.8, 0.1, 400)
    nb = nt + PLANTED * LN2 * y + g.normal(0.0, 0.05, 400) * np.sqrt(y)
    r = an.Ratios()
    r.add("x", nb, nt, y)
    fig, _ = r.figures(an.rng("documents"))
    assert fig["x"]["delta"] == pytest.approx(PLANTED, abs=2e-4)
    assert fig["x"]["ci95"][0] <= PLANTED <= fig["x"]["ci95"][1]
    assert fig["x"]["interval"].startswith(
        "evaluation noise only: 95% percentile interval of 10,000"
    )


def test_document_intervals_cover_a_planted_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(an, "RESAMPLES", 1_000)
    g = np.random.default_rng(4)
    hits = 0
    for rep in range(100):
        y = g.integers(50, 4000, 200).astype(np.float64)
        nt = y * g.normal(0.8, 0.1, 200)
        nb = nt + PLANTED * LN2 * y + g.normal(0.0, 0.05, 200) * np.sqrt(y)
        r = an.Ratios()
        r.add("x", nb, nt, y)
        lo, hi = r.figures(an.rng("documents", rep))[0]["x"]["ci95"]
        hits += lo <= PLANTED <= hi
    assert 88 <= hits <= 99


def test_the_bar_at_m1s_counts() -> None:
    bar = an.bar(7_636_111, 34_765_120)
    assert bar["0.01 nats per target"] == pytest.approx(0.0031689, abs=1e-7)
    assert bar["delta"] == bar["4 sigma"] == 4 * 0.00108


def test_bands_and_the_equivalence_edge() -> None:
    edge = 4 * an.SIGMA
    assert an.band(-edge, edge) == "the table helps"
    assert an.band(-edge + 1e-12, edge) == "no measurable difference"
    assert an.band(edge, edge) == "the table hurts"
    assert an.band(0.0, edge) == "no measurable difference"
    assert an.EQUIVALENCE == 0.0018


def test_the_interval_is_the_2_5_and_97_5_percentiles() -> None:
    assert an.interval(np.arange(1001.0)) == [25.0, 975.0]


def _scored(nats: np.ndarray, y: np.ndarray, table: str) -> Any:
    """A record of len(y) documents of 100 targets each, with these nats and bytes; EMA = raw."""
    per = 100
    docs = np.rec.fromarrays(
        [np.full(len(y), per), y, nats, nats], names=["targets", "bytes", "raw", "ema"]
    )
    doc = np.repeat(np.arange(len(y)), per)
    nats32 = (nats / per)[doc].astype(np.float32)
    window = np.zeros(len(doc), np.int32)
    tg = np.rec.fromarrays(
        [doc, window, nats32, nats32], names=["document", "window", "raw", "ema"]
    )
    return an.Scored(Path(f"{table}.json"), {"table": table}, docs, tg)


def test_the_bands_compare_delta_with_delta_unrounded() -> None:
    edge = an.bar(7_636_111, 34_765_120)["delta"]
    y, nt = np.full(10, 300), np.full(10, 600.0)
    bins = an.Bins(np.ones(1_000, bool), None, None, None, None)
    cases = (
        (-0.00431, "no measurable difference"),
        (-0.004321, "the table helps"),
        (0.00431, "no measurable difference"),
    )
    for planted, want in cases:
        b, t = _scored(nt + planted * LN2 * y, y, "on"), _scored(nt, y, "off")
        out, _ = an.heldout(b, t, bins, np.full(1_000, 3), edge)
        assert out["primary"]["band"] == want
        assert out["ema"]["band"] == want


def test_equivalence_is_claimed_only_inside_its_edge() -> None:
    edge = an.EQUIVALENCE * LN2 * 2_048  # nats on one 2,048-byte document: |Δ| = 0.0018 exactly
    inside = float(np.nextafter(edge, 0.0))
    y, bins = np.array([2_048]), an.Bins(np.ones(100, bool), None, None, None, None)
    for nb, nt, claimed in (
        (edge, 0.0, False),
        (0.0, edge, False),
        (inside, 0.0, True),
        (0.0, inside, True),
    ):
        b, t = _scored(np.array([nb]), y, "on"), _scored(np.array([nt]), y, "off")
        p = an.heldout(b, t, bins, np.full(100, 3), 4 * an.SIGMA)[0]["primary"]
        assert (abs(p["delta"]) == an.EQUIVALENCE) is not claimed
        assert p["equivalence"]["claimed"] is claimed


# ---------------------------------------------------------------- bins come before scores


@pytest.mark.usefixtures("synthetic")
def test_bins_are_fixed_before_any_score_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = _world(tmp_path)
    opened: list[str] = []
    for name in ("_json", "_lines", "_npy", "sha256"):
        real = getattr(an, name)
        monkeypatch.setattr(an, name, lambda p, real=real: opened.append(Path(p).name) or real(p))
    fix = an.fix_bins

    def fixed(*a: Any) -> Any:
        bins = fix(*a)
        opened.append("BINS FIXED")
        return bins

    monkeypatch.setattr(an, "fix_bins", fixed)
    _report(tmp_path, argv)
    before = opened[: opened.index("BINS FIXED")]
    score_free = {
        "manifest.json",
        "fingerprint.json",
        "token_bytes.u16",
        "reads.npy",
        "overlap.npy",
    }
    assert set(before) <= score_free
    assert "b-final.json" in opened
    assert "documents.npy" in opened[opened.index("BINS FIXED") :]


@pytest.mark.usefixtures("synthetic")
def test_bins_do_not_change_with_the_scores(tmp_path: Path) -> None:
    a = _report(tmp_path / "a", _world(tmp_path / "a", shift=PLANTED))
    b = _report(tmp_path / "b", _world(tmp_path / "b", shift=-0.03))
    assert a["primary"]["delta"] != b["primary"]["delta"]
    assert a["bins"] == b["bins"]


def test_lower_bins_put_a_tied_value_below_its_edge() -> None:
    bins, edges = an.lower_bins(np.array([3, 1, 2, 2, 2, 5, 7, 7, 9, 10]), 4)
    assert edges.tolist() == [2, 3, 7]
    assert bins.tolist() == [1, 0, 0, 0, 0, 2, 2, 2, 3, 3]


# ---------------------------------------------------------------- zero-shot


def test_holm_matches_a_hand_worked_case() -> None:
    # Sorted: d 0.005 x 5 = 0.025, a 0.01 x 4 = 0.04, c 0.03 x 3 = 0.09, b 0.04 x 2 = 0.08,
    # raised to the running 0.09, e 0.20 x 1 = 0.20. At 5%, d and a are rejected, then it stops.
    adj = an.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005, "e": 0.20})
    assert adj == pytest.approx({"a": 0.04, "b": 0.09, "c": 0.09, "d": 0.025, "e": 0.20})
    assert sorted(k for k, v in adj.items() if v <= an.ALPHA) == ["a", "d"]


def test_mcnemar_exact_matches_hand_values() -> None:
    assert an.mcnemar(0, 5) == 2 / 32
    assert an.mcnemar(1, 9) == 2 * (1 + 10) / 1024
    assert an.mcnemar(3, 3) == 1.0
    assert an.mcnemar(0, 0) == 1.0


def test_gold_bytes_count_the_leading_space_and_moved_whitespace() -> None:
    assert an.scored_continuation("Answer:", " café") == " café"
    assert an.scored_continuation("Answer:  ", "fox") == "  fox"
    reqs = [{"context": "q", "continuation": " ab"}, {"context": "q", "continuation": " abcd"}]
    item = {"doc_id": "0", "gold": 1, "requests": reqs}
    row = {"ll": [-2.0, -3.0], "greedy": [False, False]}
    right, bits, nbytes, margin = an.item_scores("hellaswag", item, row)
    assert (nbytes, bits) == (5, pytest.approx(3.0 / LN2))
    assert margin == pytest.approx(-3.0 / 4 - (-2.0 / 2))  # per character: acc_norm's units
    assert right  # -0.75 > -1.0 per character
    right, _, _, margin = an.item_scores("piqa", item | {"task": "piqa"}, row)
    assert not right
    assert margin == pytest.approx(-1.0)


def test_acc_norm_divides_by_characters_not_bytes() -> None:
    reqs = [{"context": "q", "continuation": " éé"}, {"context": "q", "continuation": " abc"}]
    item = {"doc_id": "0", "gold": 0, "requests": reqs}
    row = {"ll": [-5.0, -6.0], "greedy": [False, False]}
    right, _, nbytes, margin = an.item_scores("hellaswag", item, row)
    # Per character the gold scores -2.5 against -2.0 and loses; per byte it would win.
    assert not right
    assert margin == pytest.approx(-5.0 / 2 - (-6.0 / 3))
    assert nbytes == 5  # the leading space and two 2-byte characters


@pytest.mark.usefixtures("synthetic")
def test_zero_shot_counts_and_gold_bpb_by_hand(tmp_path: Path) -> None:
    rep = _report(tmp_path, _world(tmp_path))
    for i, task in enumerate(an.TASKS):
        items = {}
        for line in (tmp_path / "set" / f"{task}.jsonl").read_text().splitlines():
            it = json.loads(line)
            items[it["doc_id"]] = it
        arm: dict[str, dict[str, tuple[bool, float, int]]] = {}
        for name, term in (("on", "table"), ("off", "off")):
            arm[name] = {}
            for line in (tmp_path / f"mc-{name}.items.jsonl").read_text().splitlines():
                row = json.loads(line)
                if (row["task"], row["weights"], row["term"]) != (task, "raw", term):
                    continue
                it = items[row["doc_id"]]
                ll = np.array(row["ll"])
                if task == "lambada_openai":
                    right = bool(row["greedy"][0])
                elif task == "hellaswag":  # acc_norm: per character after the delimiter
                    chars = [len(q["continuation"]) - 1 for q in it["requests"]]
                    right = int(np.argmax(ll / chars)) == it["gold"]
                else:
                    right = int(np.argmax(ll)) == it["gold"]
                gold = it["requests"][it["gold"]]["continuation"]
                bits = float(-ll[it["gold"]] / LN2)
                arm[name][row["doc_id"]] = (right, bits, len(gold.encode()))
        ids = sorted(arm["on"])
        b = sum(arm["on"][d][0] and not arm["off"][d][0] for d in ids)
        c = sum(arm["off"][d][0] and not arm["on"][d][0] for d in ids)
        z = rep["zero_shot"]["tasks"][task]
        assert (z["right for Billet only"], z["right for the twin only"]) == (b, c)
        n = b + c
        assert z["p"] == min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2**n)
        bits_b = np.array([arm["on"][d][1] for d in ids])
        bits_t = np.array([arm["off"][d][1] for d in ids])
        y = np.array([float(arm["on"][d][2]) for d in ids])
        want = (bits_b.sum() - bits_t.sum()) / y.sum()
        assert z["gold bpb delta"] == pytest.approx(want, abs=1e-12)
        # The item bootstrap of Δ gold bpb: the same draws, over the items in the set's order.
        order = [d for d in items if d in arm["on"]]
        cols = [np.array([arm[a][d][1] for d in order]) * LN2 for a in ("on", "off")]
        cols.append(np.array([float(arm["on"][d][2]) for d in order]))
        want_ci = _ratio_interval(cols, an.rng("items", i))
        assert z["gold bpb delta ci95"] == pytest.approx(want_ci, rel=1e-9)


# Discordant counts (right for Billet only, right for the twin only) planted in each task. By
# hand, p = 2/2^9, 2/2^7, 20/2^9, 1 and 1; Holm's step-down: piqa 5 x 2^-8 = 0.01953125, LAMBADA
# max(that, 4 x 2^-6) = 0.0625, ARC-Easy 3 x 20/512 = 0.1171875, then 1 and 1. Only piqa is
# claimed, though LAMBADA's and ARC-Easy's unadjusted p are under 5%.
PLANT = {
    "piqa": (9, 0),
    "lambada_openai": (7, 0),
    "arc_easy": (8, 1),
    "hellaswag": (3, 3),
    "winogrande": (2, 2),
}
HOLM = {
    "piqa": 0.01953125,
    "lambada_openai": 0.0625,
    "arc_easy": 0.1171875,
    "hellaswag": 1.0,
    "winogrande": 1.0,
}


def _plant(root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rewrite both arms' raw samples and records so each task's discordant counts are PLANT's.

    Returns each task's (Billet, twin) correctness over its kept items, in the set's order.
    """
    items = {}
    for task in an.TASKS:
        lines = (root / "set" / f"{task}.jsonl").read_text().splitlines()
        items[task] = [json.loads(line) for line in lines][3:]  # _write_mc drops the first three
    flags = {}
    for task, (b, c) in PLANT.items():  # four both wrong, Billet only, twin only, both right
        i = np.arange(len(items[task]))
        flags[task] = (((i >= 4) & (i < 4 + b)) | (i >= 4 + b + c), i >= 4 + b)
    for arm, (table, term) in enumerate((("on", "table"), ("off", "off"))):
        path = root / f"mc-{table}.items.jsonl"
        rows = _rows(path)
        got: dict[tuple[str, str], list[tuple[float, bool]]] = {}
        where = {t: {it["doc_id"]: j for j, it in enumerate(its)} for t, its in items.items()}
        for row in rows:
            if (row["weights"], row["term"]) != ("raw", term):
                continue
            task, j = row["task"], where[row["task"]][row["doc_id"]]
            right, gold = bool(flags[task][arm][j]), items[task][j]["gold"]
            n = len(items[task][j]["requests"])
            row["ll"] = [-1.0 if (q == gold) == right else -30.0 for q in range(n)]
            row["greedy"] = [right] * n
            got[(task, row["doc_id"])] = list(zip(row["ll"], row["greedy"], strict=True))
        _write_rows(path, rows)
        rec = json.loads((root / f"mc-{table}.json").read_text())
        for task, its in items.items():
            rec["raw"][task] = aggregate(task, its, [got[(task, it["doc_id"])] for it in its])
        (root / f"mc-{table}.json").write_text(json.dumps(rec))
    return flags


@pytest.mark.usefixtures("synthetic")
def test_zero_shot_claims_only_by_holm_over_the_five(tmp_path: Path) -> None:
    argv = _world(tmp_path)
    flags = _plant(tmp_path)
    z = _report(tmp_path, argv)["zero_shot"]["tasks"]
    for i, task in enumerate(an.TASKS):
        cb, ct = (f.astype(np.float64) for f in flags[task])
        b, c = PLANT[task]
        assert (z[task]["right for Billet only"], z[task]["right for the twin only"]) == (b, c)
        assert z[task]["p holm"] == pytest.approx(HOLM[task], rel=1e-12)
        assert z[task]["difference claimed"] is (task == "piqa")
        n = len(cb)
        assert z[task]["delta_acc"] == pytest.approx((b - c) / n, abs=1e-15)
        gen = an.rng("items", i)  # Δ accuracy's interval: the documented draws over items
        reps = [float((cb - ct)[gen.integers(0, n, n)].mean()) for _ in range(an.RESAMPLES)]
        want = np.percentile(reps, [2.5, 97.5]).tolist()
        assert z[task]["delta_acc ci95"] == pytest.approx(want, abs=1e-12)


# ---------------------------------------------------------------- training loss and per Mac-hour


def test_moving_blocks_leave_out_steps_and_hold_a_constant() -> None:
    d = np.full(1_000, 0.25)
    kept = np.ones(1_000, bool)
    d[100:140], kept[100:140] = 1e6, False
    reps = an.block_means(d, kept, an.rng("steps"))
    assert an.interval(reps) == [0.25, 0.25]


def test_moving_blocks_cover_a_planted_mean_under_correlated_noise() -> None:
    g = np.random.default_rng(12)
    e = np.zeros(1_000)
    for i in range(1, 1_000):
        e[i] = 0.9 * e[i - 1] + g.normal(0.0, 0.01)
    lo, hi = an.interval(an.block_means(0.02 + e, np.ones(1_000, bool), an.rng("steps")))
    assert lo <= 0.02 <= hi


def test_moving_blocks_follow_the_documented_draws() -> None:
    assert an.BLOCK == 50
    d = np.random.default_rng(5).normal(0.01, 0.02, 1_000)
    kept = np.ones(1_000, bool)
    kept[[3, 50, 51, 400, 999]] = False
    got = an.block_means(d, kept, an.rng("steps", 7))
    gen = an.rng("steps", 7)
    want = np.empty(an.RESAMPLES)
    for r in range(an.RESAMPLES):
        idx = np.concatenate([np.arange(a, a + 50) for a in gen.integers(0, 951, 20)])
        want[r] = d[idx][kept[idx]].mean()
    assert got == pytest.approx(want, rel=1e-12)


def test_training_loss_windows_follow_the_documented_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(an, "RESAMPLES", 2_000)
    x = np.arange(N, dtype=np.float64)
    lt = _curve(x)
    lb = lt + 0.01 + 0.004 * np.sin(x / 37.0) + 0.003 * (x / N) ** 2  # Δ varies within windows
    lb[[2_000, 5_500]] = math.nan
    hashes = np.array([f"{s:x}" for s in range(N)], dtype=object)
    other = hashes.copy()
    other[[1_600, 1_601, 1_602, 12_000]] = "f" * 16
    present = np.ones(N, bool)
    got = an.training_loss(an.Log(lb, present, hashes), an.Log(lt, present, other))
    kept = np.isfinite(lb) & (hashes == other)
    d = np.where(kept, lb - lt, 0.0)
    for i, pct in enumerate((10, 25, 50, 80, 100)):
        end = N * pct // 100
        dw, kw = d[end - 1_000 : end], kept[end - 1_000 : end]
        w = got["windows"][f"{pct}%"]
        assert w["steps"] == [end - 1_000, end - 1]
        assert w["delta"] == pytest.approx(float(dw[kw].mean()), abs=1e-12)
        gen = an.rng("steps", i)
        reps = np.empty(an.RESAMPLES)
        for r in range(an.RESAMPLES):
            idx = np.concatenate([np.arange(a, a + 50) for a in gen.integers(0, 951, 20)])
            reps[r] = dw[idx][kw[idx]].mean()
        want = np.percentile(reps, [2.5, 97.5]).tolist()
        assert w["ci95"] == pytest.approx(want, abs=1e-12)


def _log(loss: np.ndarray) -> Any:
    return an.Log(loss, np.ones(N, bool), np.array([f"{s:x}" for s in range(N)], dtype=object))


def test_k_recovers_a_planted_stretch_and_reads_it_against_r() -> None:
    lb, lt = _curves(1.08)
    k = an.multiplier(_log(lb), _log(lt))
    assert k["k"] == pytest.approx(1.08, abs=2e-3)
    assert (k["s from"], k["s to"]) == (1_500, 19_031)
    assert k["s used"] + k["s left out (no s')"] == 17_532
    lb2, lt2 = _curves(0.95)  # the twin is ahead: it reaches Billet's loss sooner
    assert an.multiplier(_log(lb2), _log(lt2))["k"] == pytest.approx(0.95, abs=2e-3)


def test_k_counts_reaching_as_at_or_below() -> None:
    same, _ = _curves(1.0)
    k = an.multiplier(_log(same), _log(same.copy()))
    assert k["k"] == 1.0
    # Only s = 1,500 is left out: the twin is at Billet's loss already at s' = 1,500.
    assert k["s left out (no s')"] == 1
    assert k["left out, the twin at or below already at s' = 1,500"] == 1


def test_k_is_the_median_of_k_s() -> None:
    lb, _ = _curves(1.0)
    lt = _curve(np.arange(N, dtype=np.float64) - 1_000)  # the twin is 1,000 steps behind
    k = an.multiplier(_log(lb), _log(lt))
    # k(s) = (s + 1,000)/s for s from 1,500 to 18,031, whose middle is 9,765.5; the mean of
    # k(s) would be about 1.150.
    assert k["k"] == pytest.approx(1 + 1_000 / 9_765.5, abs=1e-4)
    assert k["s left out (no s')"] == pytest.approx(1_000, abs=1)


@pytest.mark.parametrize("stretch", [1.06, 0.93])
def test_k_matches_a_plain_search_on_noisy_curves(stretch: float) -> None:
    g = np.random.default_rng(21)
    x = np.arange(N, dtype=np.float64)
    lb = _curve(x) + g.normal(0.0, 0.05, N)
    lt = _curve(x / stretch) + g.normal(0.0, 0.05, N) + 0.03 * np.sin(x / 700.0)
    got = an.multiplier(_log(lb), _log(lt))
    s = np.arange(1_500, 19_032)
    mb = np.array([lb[v - 500 : v + 501].mean() for v in s])
    mt = np.array([lt[v - 500 : v + 501].mean() for v in s])
    assert (np.diff(mt) > 0).any()  # the twin's smoothed curve is not monotone
    ks: list[float] = []
    before = after = 0
    for i, v in enumerate(s):
        hit = np.flatnonzero(mt <= mb[i])
        if not len(hit):
            after += 1
        elif hit[0] == 0:  # at or below already at s' = 1,500: the range hides when it got there
            before += 1
        else:
            ks.append(float(s[hit[0]] / v))
    assert got["s used"] == len(ks)
    assert got["left out, the twin at or below already at s' = 1,500"] == before
    assert got["left out, the twin not at or below by s' = 19,031"] == after
    assert (before if stretch < 1 else after) > 0  # a twin ahead fills one end, behind the other
    assert got["k"] == pytest.approx(float(np.median(ks)), rel=1e-12)


def test_k_smooths_the_finite_losses_only() -> None:
    g = np.random.default_rng(22)
    x = np.arange(N, dtype=np.float64)
    lb = _curve(x) + g.normal(0.0, 0.05, N)
    lt = _curve(x / 1.06) + g.normal(0.0, 0.05, N)
    lb[g.integers(1_000, 19_532, 60)] = np.nan
    lt[g.integers(1_000, 19_532, 60)] = np.inf
    lt[g.integers(1_000, 19_532, 60)] = np.nan
    got = an.multiplier(_log(lb), _log(lt))
    s = np.arange(1_500, 19_032)

    def mean(v: np.ndarray, c: int) -> float:  # the finite losses of steps c - 500 to c + 500
        w = v[c - 500 : c + 501]
        return float(w[np.isfinite(w)].mean())

    mb = np.array([mean(lb, c) for c in s.tolist()])
    mt = np.array([mean(lt, c) for c in s.tolist()])
    ks = []
    for i, c in enumerate(s):
        hit = np.flatnonzero(mt <= mb[i])
        if len(hit) and hit[0] > 0:
            ks.append(float(s[hit[0]] / c))
    assert got["s used"] == len(ks)
    assert got["k"] == pytest.approx(float(np.median(ks)), rel=1e-12)


def test_k_with_undefined_smoothed_losses() -> None:
    x = np.arange(N, dtype=np.float64)
    lb, lt = _curve(x), _curve(x / 1.06)
    lb[3_000:4_100] = np.nan  # Billet's smoothed loss is undefined for s = 3,500 to 3,599
    lt[8_000:9_100] = np.inf  # the twin's for s' = 8,500 to 8,599
    got = an.multiplier(_log(lb), _log(lt))
    s = np.arange(1_500, 19_032)

    def mean(v: np.ndarray, c: int) -> float:  # NaN when the window holds no finite loss
        w = v[c - 500 : c + 501]
        w = w[np.isfinite(w)]
        return float(w.mean()) if len(w) else math.nan

    mb = np.array([mean(lb, c) for c in s.tolist()])
    mt = np.array([mean(lt, c) for c in s.tolist()])
    ks: list[float] = []
    for i, c in enumerate(s):
        hit = np.flatnonzero(mt <= mb[i])  # an undefined twin never matches; nor does Billet
        if len(hit) and hit[0] > 0:
            ks.append(float(s[hit[0]] / c))
    assert got["left out, Billet's smoothed loss undefined"] == int(np.isnan(mb).sum()) == 100
    assert got["s used"] == len(ks)
    assert got["k"] == pytest.approx(float(np.median(ks)), rel=1e-12)


@pytest.mark.parametrize(
    ("ratios", "reading"),
    [
        (
            [1.04, 1.03, 2.0, 1.06, 1.04, 1.05, 1.05, 1.04],
            "the table leads per Mac-hour in the hold",
        ),
        ([1.07, 1.08, 2.0, 1.09, 1.08, 1.07, 1.09, 1.08], "neither leads per Mac-hour in the hold"),
        (
            [1.12, 1.11, 2.0, 1.13, 1.12, 1.12, 1.11, 1.13],
            "the table trails per Mac-hour in the hold",
        ),
    ],
)
def test_per_mac_hour_reading(tmp_path: Path, ratios: list[float], reading: str) -> None:
    _write_probe(tmp_path / "summary.json", ratios)
    lb, lt = _curves(1.08)
    out = an.per_mac_hour(_log(lb), _log(lt), tmp_path / "summary.json")
    assert out["r"]["pairs_used"] == 7  # the void pair is left out
    assert out["r"]["df"] == 5
    assert out["reading"] == reading


@pytest.mark.parametrize(
    ("ratios", "side"),
    [
        ([1.07, 1.06, 2.0, 1.08, 1.07, 1.09, 1.06, 1.08], 1),  # r 1.0725 (1.0609-1.0841)
        ([1.08, 1.07, 2.0, 1.09, 1.08, 1.10, 1.07, 1.09], -1),  # r 1.0825 (1.0709-1.0941)
    ],
)
def test_k_inside_rs_interval_reads_neither_on_either_side_of_r(
    tmp_path: Path, ratios: list[float], side: int
) -> None:
    _write_probe(tmp_path / "summary.json", ratios)
    lb, lt = _curves(1.08)
    out = an.per_mac_hour(_log(lb), _log(lt), tmp_path / "summary.json")
    k, (lo, hi) = out["k"]["k"], out["r"]["ci95"]
    assert lo < k < hi
    assert (k - out["r"]["ratio"]) * side > 0  # k above r's estimate, then below it
    assert out["reading"] == "neither leads per Mac-hour in the hold"


def test_the_probe_summary_must_follow_its_rule(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    _write_probe(path, [1.04, 1.03, 2.0, 1.06, 1.04, 1.05, 1.05, 1.04])
    s = json.loads(path.read_text())
    s["result"]["ratio"] *= 1.001
    path.write_text(json.dumps(s))
    lb, lt = _curves(1.08)
    with pytest.raises(an.Refused, match="differs from its pairs"):
        an.per_mac_hour(_log(lb), _log(lt), path)


# ---------------------------------------------------------------- refusals


def _rewrite(path: Path, **changes: Any) -> None:
    rec = json.loads(path.read_text()) | changes
    path.write_text(json.dumps(rec))


@pytest.mark.usefixtures("synthetic")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dtype": "mlx.core.bfloat16"}, "not mlx.core.float32"),
        ({"mask": "base"}, "--mask record"),
        ({"step": 24_000}, "step 24000"),
        ({"bytes": 1}, "counts differ"),
    ],
)
def test_a_held_out_record_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, Any], message: str
) -> None:
    argv = _world(tmp_path)
    _rewrite(tmp_path / "t-final.json", **change)
    assert an.main(argv) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


@pytest.mark.usefixtures("synthetic")
def test_records_that_changed_after_scoring_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _world(tmp_path)
    f = tmp_path / "b-final.records" / "targets.npy"
    a = np.load(f)
    a["raw"][0] += 1.0
    np.save(f, a)
    assert an.main(argv) == 2
    assert "SHA-256 differs" in capsys.readouterr().err


@pytest.mark.usefixtures("synthetic")
def test_records_of_another_split_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _world(tmp_path)
    for flag in ("--reads", "--overlap"):  # their bins would refuse the other split first
        del argv[argv.index(flag) : argv.index(flag) + 2]
    argv[argv.index("--split") + 1] = str(tmp_path / "s6store" / "heldout")
    assert an.main(argv) == 2
    assert "differ from the split's" in capsys.readouterr().err


@pytest.mark.usefixtures("synthetic")
def test_a_pinned_pair_of_another_split_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _world(tmp_path)
    other = _docs(8, 30)
    for arm, table, shift in (("b", "on", PLANTED), ("t", "off", None)):
        _write_scored(
            tmp_path / "x" / f"{arm}.json", table, 19_532, other, _arm_nats(2, other, shift)
        )
    i = argv.index("--pinned")
    argv[i + 1 : i + 3] = [str(tmp_path / "x" / "b.json"), str(tmp_path / "x" / "t.json")]
    assert an.main(argv) == 2
    assert "differ from the split's" in capsys.readouterr().err


@pytest.mark.usefixtures("synthetic")
def test_a_twin_log_short_of_its_last_step_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _world(tmp_path)
    _, lt = _curves(1.08)
    _write_log(tmp_path / "t-metrics.jsonl", lt, [f"{s:016x}" for s in range(N)], last=20_000)
    assert an.main(argv) == 2
    assert "has not ended" in capsys.readouterr().err


@pytest.mark.usefixtures("synthetic")
@pytest.mark.parametrize(
    ("change", "message"),
    [({"limit": 5}, "no --limit"), ({"decontam": None}, "not decontaminated")],
)
def test_an_mc_record_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, Any], message: str
) -> None:
    argv = _world(tmp_path)
    _rewrite(tmp_path / "mc-off.json", **change)
    assert an.main(argv) == 2
    assert message in capsys.readouterr().err


def _swap(argv: list[str], flag: str) -> None:
    i = argv.index(flag)
    argv[i + 1], argv[i + 2] = argv[i + 2], argv[i + 1]


def _drop_flag(argv: list[str], flag: str) -> None:
    i = argv.index(flag)
    n = 3 if flag in ("--heldout", "--pinned", "--mc", "--samples", "--metrics", "--s6") else 2
    del argv[i : i + n]


def _probe_nowhere(root: Path, argv: list[str]) -> None:
    argv[argv.index("--probe") + 1] = str(root / "nowhere.json")


# One command-line change: a function of the world's root and its argv.
ARGV: dict[str, Any] = {
    "heldout arms swapped": lambda r, a: _swap(a, "--heldout"),
    "mc arms swapped": lambda r, a: _swap(a, "--mc"),
    "probe missing": _probe_nowhere,
    "mc without samples": lambda r, a: _drop_flag(a, "--samples"),
    "s6 without thirds": lambda r, a: _drop_flag(a, "--thirds"),
    "probe without metrics": lambda r, a: _drop_flag(a, "--metrics"),
}

# One JSON field changed: (file, keys to the field, its new value from the old one, or None to
# remove the field).
FIELD: dict[str, tuple[str, tuple[Any, ...], Any]] = {
    "record bpb": ("t-final.json", ("raw", "bpb"), lambda v: v + 1e-6),
    "no records block": ("t-final.json", ("records",), None),
    "no ema set": ("t-final.json", ("ema",), None),
    "records seq_len": ("t-final.json", ("seq_len",), lambda v: v // 2),
    "another store": ("store/fingerprint.json", ("fingerprint",), lambda _: "0" * 64),
    "token bytes": ("store/manifest.json", ("outputs", "token_bytes_sha256"), lambda _: "0" * 64),
    "split eos": ("store/heldout/manifest.json", ("eos",), lambda _: 1),
    "shard tokens": ("store/heldout/manifest.json", ("tokens", 0), lambda v: v - 1),
    "split manifest no shards": ("store/heldout/manifest.json", ("shards",), None),
    "mc bfloat16": ("mc-off.json", ("dtype",), lambda _: "mlx.core.bfloat16"),
    "mc another step": ("mc-off.json", ("step",), lambda _: 24_000),
    "mc headline": ("mc-off.json", ("raw", "piqa", "headline"), lambda _: "acc_norm"),
    "mc drops differ": ("mc-off.json", ("decontam", "piqa", "sha256"), lambda _: "b" * 64),
    "mc tokenizer": ("mc-off.json", ("tokenizer_sha256",), lambda _: "d" * 64),
    "mc seq_len": ("mc-off.json", ("seq_len",), lambda v: v // 2),
    "mc no seq_len": ("mc-on.json", ("seq_len",), None),
    "mc accuracy": ("mc-off.json", ("raw", "piqa", "acc"), lambda v: v + 1 / 27),  # one item
    "mc set id": ("mc-off.json", ("set_id",), lambda _: "other"),
    "set file": ("set/manifest.json", ("tasks", "piqa", "sha256"), lambda _: "0" * 64),
    "set count": ("set/manifest.json", ("tasks", "piqa", "n"), lambda v: v + 1),
    "pair ratio": ("summary.json", ("pairs", 0, "ratio"), lambda v: v * 1.01),  # not on ÷ off
    "probe design": ("summary.json", ("design", "drop"), lambda _: 50),
    "probe no design": ("summary.json", ("design",), None),
}


def _edit_json(path: Path, keys: tuple[Any, ...], change: Any) -> None:
    doc = json.loads(path.read_text())
    node = doc
    for k in keys[:-1]:
        node = node[k]
    if change is None:
        del node[keys[-1]]
    else:
        node[keys[-1]] = change(node[keys[-1]])
    path.write_text(json.dumps(doc))


def _edit_npy(path: Path, change: Any) -> None:
    np.save(path, change(np.load(path)))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _rehash(root: Path, name: str) -> None:
    """The twin's record takes the new SHA-256 of its changed records file `name`."""
    rec = json.loads((root / "t-final.json").read_text())
    rec["records"]["sha256"][name] = _sha(root / "t-final.records" / name)
    (root / "t-final.json").write_text(json.dumps(rec))


def _nats_do_not_sum(root: Path) -> None:  # the hash follows, so only the sum check is left
    a = np.load(root / "t-final.records" / "targets.npy")
    a["raw"][0] += 1.0
    np.save(root / "t-final.records" / "targets.npy", a)
    _rehash(root, "targets.npy")


def _other_documents(root: Path) -> None:  # a byte moves between two of the twin's documents
    a = np.load(root / "t-final.records" / "documents.npy")
    i, j = np.flatnonzero(a["bytes"] > 0)[:2]
    a["bytes"][i] += 1
    a["bytes"][j] -= 1
    np.save(root / "t-final.records" / "documents.npy", a)
    _rehash(root, "documents.npy")


def _no_tokenizer(root: Path) -> None:  # neither mc record names its tokenizer
    for table in ("on", "off"):
        rec = json.loads((root / f"mc-{table}.json").read_text())
        del rec["tokenizer_sha256"]
        (root / f"mc-{table}.json").write_text(json.dumps(rec))


def _both_seq_len(root: Path) -> None:  # both mc records at seq_len 4,096, so they agree
    for table in ("on", "off"):
        _edit_json(root / f"mc-{table}.json", ("seq_len",), lambda v: v * 2)


def _drop_a_kept_item(root: Path) -> None:
    rows = _rows(root / "mc-off.items.jsonl")
    first = next(k for k, r in enumerate(rows) if (r["task"], r["weights"]) == ("piqa", "raw"))
    _write_rows(root / "mc-off.items.jsonl", rows[:first] + rows[first + 1 :])


def _repeat_a_log_step(root: Path) -> None:
    rows = _rows(root / "t-metrics.jsonl")
    _write_rows(root / "t-metrics.jsonl", [*rows, rows[5_000]])


def _null_batch(root: Path) -> None:
    rows = _rows(root / "t-metrics.jsonl")
    rows[7]["batch"] = None
    _write_rows(root / "t-metrics.jsonl", rows)


def _two_thirds(root: Path) -> None:  # thirds 0 and 1 only
    np.save(root / "thirds.npy", np.arange(len(np.load(root / "thirds.npy"))) % 2)


def _open_with_17(root: Path) -> None:  # the second document of shard 0 opens with id 17
    body = np.fromfile(root / "store" / "heldout" / "000000.bin", dtype="<u2")
    idx = np.fromfile(root / "store" / "heldout" / "000000.idx", dtype="<u8")
    body[idx[1]] = 17
    body.tofile(root / "store" / "heldout" / "000000.bin")


def _bytes_on_id_3(root: Path) -> None:  # id 3 gets a byte; the store manifest follows
    tb = TB.copy()
    tb[3] = 1
    (root / "store" / "token_bytes.u16").write_bytes(tb.tobytes())
    sha = hashlib.sha256(tb.tobytes()).hexdigest()
    _edit_json(root / "store" / "manifest.json", ("outputs", "token_bytes_sha256"), lambda _: sha)


def _twin_window(root: Path) -> None:  # one of the twin's targets moves to another window
    a = np.load(root / "t-final.records" / "targets.npy")
    a["window"][np.flatnonzero(a["window"] == 1)[0]] = 2
    np.save(root / "t-final.records" / "targets.npy", a)
    _rehash(root, "targets.npy")


def _target_total(root: Path) -> None:  # the twin's documents count one target more than it has
    a = np.load(root / "t-final.records" / "documents.npy")
    a["targets"][0] += 1
    np.save(root / "t-final.records" / "documents.npy", a)
    _rehash(root, "documents.npy")


def _nats_one_step(root: Path) -> None:  # one target's nats move one float32 step; hash follows
    a = np.load(root / "t-final.records" / "targets.npy")
    a["raw"][0] = np.nextafter(a["raw"][0], np.float32(np.inf))
    np.save(root / "t-final.records" / "targets.npy", a)
    _rehash(root, "targets.npy")


def _sample_twice(root: Path) -> None:
    rows = _rows(root / "mc-off.items.jsonl")
    first = next(r for r in rows if (r["task"], r["weights"]) == ("piqa", "raw"))
    _write_rows(root / "mc-off.items.jsonl", [*rows, first])


def _sample_short(root: Path) -> None:  # one item's samples lack a choice
    rows = _rows(root / "mc-off.items.jsonl")
    first = next(r for r in rows if (r["task"], r["weights"]) == ("piqa", "raw"))
    first["ll"], first["greedy"] = first["ll"][:-1], first["greedy"][:-1]
    _write_rows(root / "mc-off.items.jsonl", rows)


def _log_row(root: Path, row: dict[str, Any], at: int | None) -> None:
    rows = _rows(root / "t-metrics.jsonl")
    if at is None:
        rows.append(row)
    else:
        rows[at] = row
    _write_rows(root / "t-metrics.jsonl", rows)


def _found_past_distinct(root: Path) -> None:  # one document finds more 13-grams than it has
    o = np.load(root / "overlap.npy")
    o["found"][1] = o["distinct"][1] + 1
    np.save(root / "overlap.npy", o)


# Other changes to the world's files: a function of its root.
SPECIAL: dict[str, Any] = {
    "nats do not sum": _nats_do_not_sum,
    "nats one float32 step": _nats_one_step,
    "arms score other documents": _other_documents,
    "twin windows": _twin_window,
    "target total": _target_total,
    "bytes on ids 0-16": _bytes_on_id_3,
    "document opens": _open_with_17,
    "reads short": lambda r: _edit_npy(r / "reads.npy", lambda x: x[:-1]),
    "reads negative": lambda r: _edit_npy(
        r / "reads.npy", lambda x: np.where(np.arange(len(x)) == 5, -1, x)
    ),
    "overlap short": lambda r: _edit_npy(r / "overlap.npy", lambda x: x[:-1]),
    "overlap found": _found_past_distinct,
    "mc no tokenizer": _no_tokenizer,
    "mc pair seq_len": _both_seq_len,
    "mc kept items": _drop_a_kept_item,
    "mc not json": lambda r: (r / "mc-off.json").write_text("{"),
    "sample twice": _sample_twice,
    "sample short": _sample_short,
    "log step repeated": _repeat_a_log_step,
    "log batch hash": _null_batch,
    "log line": lambda r: _log_row(r, {"step": 5, "batch": "5"}, 5),
    "log step past N": lambda r: _log_row(r, {"step": N, "loss": 1.0, "batch": "x"}, None),
    "metrics not json": lambda r: (r / "t-metrics.jsonl").write_text("{\n"),
    "thirds": _two_thirds,
    "thirds not npy": lambda r: (r / "thirds.npy").write_bytes(b"not an array"),
}


def _break(root: Path, argv: list[str], case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Change one input of a synthetic world as `case` names."""
    if case in FIELD:
        name, keys, change = FIELD[case]
        _edit_json(root / name, keys, change)
    elif case in SPECIAL:
        SPECIAL[case](root)
    elif case in ARGV:
        ARGV[case](root, argv)
    elif case == "held-out count":  # the real count, one more than the records'
        monkeypatch.setitem(an.EXPECT, "docs", an.EXPECT["docs"] + 1)


@pytest.mark.usefixtures("synthetic")
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("heldout arms swapped", "(want 'on')"),
        ("record bpb", "documents give bpb"),
        ("no records block", "no records block"),
        ("no ema set", "no 'ema' set"),
        ("records seq_len", "T 1024"),
        ("nats do not sum", "do not sum to their documents'"),
        ("nats one float32 step", "do not sum to their documents'"),
        ("arms score other documents", "score different documents or targets"),
        ("twin windows", "score different documents or targets"),
        ("target total", "its counts differ from its records'"),
        ("held-out count", "held-out documents M1 names"),
        ("another store", f"not store {FINGERPRINT[:16]}"),
        ("token bytes", "differs from the store manifest's"),
        ("split eos", "not an id-0-first uint16 split"),
        ("bytes on ids 0-16", "no bytes on ids 0-16"),
        ("shard tokens", "offsets do not match its tokens"),
        ("document opens", "does not open with id 0"),
        ("split manifest no shards", "KeyError('shards')"),
        ("reads short", "not one non-negative integer per held-out target"),
        ("reads negative", "not one non-negative integer per held-out target"),
        ("overlap short", "not one (distinct, found) row per held-out document"),
        ("overlap found", "found must lie in 0..distinct"),
        ("mc arms swapped", "with the table on"),
        ("mc bfloat16", "not float32"),
        ("mc another step", "or not step"),
        ("mc headline", "headline is not acc"),
        ("mc drops differ", "drop different items"),
        ("mc tokenizer", "differ in tokenizer_sha256"),
        ("mc no tokenizer", "tokenizer_sha256, or lack it"),
        ("mc seq_len", "mc-off.json: seq_len 1024; want 2048"),
        ("mc no seq_len", "mc-on.json: seq_len None; want 2048"),
        ("mc pair seq_len", "mc-on.json: seq_len 4096; want 2048"),
        ("mc kept items", "the arms scored different items"),
        ("mc accuracy", "accuracy differs from its record's"),
        ("mc set id", "another set"),
        ("mc not json", "mc-off.json: Expecting"),
        ("mc without samples", "--mc, --samples and --set go together"),
        ("set file", "differs from the manifest's"),
        ("set count", "distinct items, the manifest's n"),
        ("sample twice", "appears twice"),
        ("sample short", "its samples do not fit its requests"),
        ("log step repeated", "or repeated"),
        ("log batch hash", "has no batch hash"),
        ("log line", "a line that is not a step record"),
        ("log step past N", f"step {N} is outside 0..{N - 1}"),
        ("metrics not json", "t-metrics.jsonl: Expecting"),
        ("pair ratio", "a pair's ratio"),
        ("probe design", "is not the rule's"),
        ("probe no design", "KeyError('design')"),
        ("probe missing", "nowhere.json: missing"),
        ("probe without metrics", "--probe needs --metrics"),
        ("thirds", "thirds: not 0, 1 or 2"),
        ("thirds not npy", "thirds.npy: "),
        ("s6 without thirds", "--s6 needs --thirds"),
    ],
)
def test_more_inputs_are_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    argv = _world(tmp_path)
    _break(tmp_path, argv, case, monkeypatch)
    assert an.main(argv) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()
