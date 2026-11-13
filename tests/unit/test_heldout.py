"""Held-out bits per byte over a checkpoint: `akeratos/eval/heldout.py`.

The readout is checked against a hand computation, packing, byte counts, the training cross
entropy and the table term, then through its entry point over training's checkpoints, with the
reserved-id refusal and the per-document records. Everything runs at tiny sizes on MLX's CPU
device; the shapes are fixtures, not the readout's.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from akeratos import ckpt
from akeratos.configs import ModelConfig
from akeratos.data.packer import segment_ids
from akeratos.errors import AkeratosError, DataMismatchError
from akeratos.eval import heldout
from akeratos.eval.heldout import LN2, HeldOut, score
from akeratos.model import EMBED, head_logits, hidden, init_params, masked_id_mask
from akeratos.post.data import stage_config
from akeratos.table.config import TableConfig
from akeratos.table.table import Table
from akeratos.train import entry
from akeratos.train.loop import Run, token_ce
from akeratos.train.saver import Checkpointer

mx.set_default_device(mx.cpu)

V, T, B = 512, 32, 2
TINY = ModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=2,
    head_dim=32,
    n_kv_heads=1,
    ffn=128,
    vocab_size=V,
    window=16,
    table=False,
    param_kind="sp",
)
TINY_KEYS = [
    f"{k}={json.dumps(v)}" for k, v in asdict(TINY).items() if v != getattr(ModelConfig(), k)
]


def token_table(seed: int = 0) -> np.ndarray:
    tb = np.random.default_rng(seed).integers(1, 5, size=V).astype("<u2")
    tb[:17] = 0
    return tb


def docs_for(seed: int, n: int, long: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [rng.integers(17, V, size=int(rng.integers(3, 20))) for _ in range(n)]
    out += [rng.integers(17, V, size=3 * T + 5) for _ in range(long)]
    return out


def write_split(path: Path, docs: list[np.ndarray], tb: np.ndarray | None = None) -> None:
    """A one-shard store split; with `tb`, the held-out split's `.bytes` sidecar too."""
    path.mkdir(parents=True)
    body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
    np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8").tofile(
        path / "000000.idx"
    )
    body.tofile(path / "000000.bin")
    m = {"dtype": "uint16", "eos": 0, "vocab": V, "shards": ["000000.bin"], "tokens": [len(body)]}
    m |= {"documents": [len(docs)], "sep_position": "start"}
    if tb is not None:
        np.array([int(tb[d].sum()) for d in docs], "<u8").tofile(path / "000000.bytes")
        m["bytes"] = ["000000.bytes"]
    (path / "manifest.json").write_text(json.dumps(m))


def write_store(root: Path, docs: list[np.ndarray], train: bool = False) -> Path:
    """A data store with its held-out split, `token_bytes.u16` and the root manifest's SHA-256."""
    tb = token_table()
    write_split(root / "heldout", docs, tb)
    if train:
        write_split(root / "train", docs_for(9, 200))
    (root / "token_bytes.u16").write_bytes(tb.tobytes())
    sha = hashlib.sha256(tb.tobytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({"outputs": {"token_bytes_sha256": sha}}))
    return root / "heldout"


def test_bpb_against_a_hand_computation(tmp_path: Path) -> None:
    """With the tied matrix at zero every target costs ln(V - 16) nats."""
    docs = docs_for(1, 30, long=2)
    held = HeldOut(write_store(tmp_path, docs), T)
    tb = token_table()
    assert held.docs == len(docs)
    assert held.targets == sum(len(d) for d in docs)
    assert held.bytes == sum(int(tb[d].sum()) for d in docs)
    p = init_params(TINY, 0)
    p[EMBED] = mx.zeros_like(p[EMBED])
    nats = score(p, TINY, held, B)
    hand = held.targets * math.log(V - 16) / (LN2 * held.bytes)
    assert len(nats) == held.targets
    assert abs(held.bpb(nats) - hand) / hand < 1e-6
    capped = HeldOut(tmp_path / "heldout", T, tokens=40)
    assert capped.targets >= 40
    assert capped.targets - (len(docs[capped.docs - 1])) < 40


def test_long_documents_are_windowed_once(tmp_path: Path) -> None:
    docs = docs_for(2, 4, long=1)
    held = HeldOut(write_store(tmp_path, docs), T)
    long = docs[-1]
    windows = math.ceil(len(long) / T)
    assert int(held.mask.sum()) == held.targets
    assert np.array_equal(held.ids[-windows][: T + 1], np.concatenate([[0], long])[: T + 1])
    assert np.array_equal(held.ids[-windows + 1][0], long[T - 1])


def test_packed_equals_each_document_alone(tmp_path: Path) -> None:
    """Packing under the document mask changes no document's nats beyond rounding."""
    split = write_store(tmp_path, docs_for(3, 25, long=1))
    packed, alone = HeldOut(split, T), HeldOut(split, T, doc_mask=False)
    assert len(packed.ids) < len(alone.ids) == 26 + 3
    p = init_params(TINY, 1)
    a, b = score(p, TINY, packed, B), score(p, TINY, alone, B)
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < 1e-4


@pytest.mark.parametrize(
    ("damage", "match"),
    [
        ("sidecar", "`.bytes` entry"),
        ("token_bytes", "`.bytes` entry"),
        ("reserved_bytes", "ids 0-16"),
        ("sha", "SHA-256"),
        ("no_sidecar", "no `.bytes` sidecar"),
        ("short_sidecar", "byte counts for"),
        ("reserved_id", "reserved id"),
        ("no_token_bytes", "missing"),
        ("no_sidecar_file", "no `.bytes` sidecar"),
        ("vocab", "vocabulary"),
    ],
)
def test_bad_byte_counts_are_refused(tmp_path: Path, damage: str, match: str) -> None:
    """A store whose byte counts disagree is refused before anything is scored."""
    docs = docs_for(4, 10)
    if damage == "reserved_id":
        docs[3][1] = 5
    split = write_store(tmp_path, docs)
    sidecar, tb_path = split / "000000.bytes", tmp_path / "token_bytes.u16"
    nb = np.fromfile(sidecar, "<u8")
    tb = np.frombuffer(tb_path.read_bytes(), "<u2").copy()
    if damage == "sidecar":
        nb[2] += 1
        nb.tofile(sidecar)
    elif damage == "short_sidecar":
        nb[:-1].tofile(sidecar)
    elif damage in ("token_bytes", "reserved_bytes"):
        tb[int(docs[3][0]) if damage == "token_bytes" else 5] += 1
        tb_path.write_bytes(tb.tobytes())
        (tmp_path / "manifest.json").unlink()
    elif damage == "sha":
        (tmp_path / "manifest.json").write_text(
            json.dumps({"outputs": {"token_bytes_sha256": "0"}})
        )
    elif damage == "no_sidecar":
        m = json.loads((split / "manifest.json").read_text())
        del m["bytes"]
        (split / "manifest.json").write_text(json.dumps(m))
    elif damage == "no_token_bytes":
        tb_path.unlink()
    elif damage == "no_sidecar_file":
        sidecar.unlink()
    with pytest.raises(DataMismatchError, match=match):
        HeldOut(split, T, vocab=V + 1 if damage == "vocab" else V)


def test_nats_equal_the_training_cross_entropy(tmp_path: Path) -> None:
    """The readout's nats equal `akeratos.train.loop.token_ce` on the same FP32 logits."""
    held = HeldOut(write_store(tmp_path, docs_for(12, 10)), T)
    p = init_params(TINY, 6)
    nats = score(p, TINY, held, B)
    want: list[np.ndarray] = []
    for ids, tg, m in held.batches(B):
        x, seg = mx.array(ids.astype(np.int32)), mx.array(segment_ids(ids.astype(np.int32)))
        z = head_logits(p, TINY, hidden(p, TINY, x, seg, mx.float32), masked_id_mask(TINY))
        n = token_ce(z, mx.array(tg.astype(np.int32)))
        want.append(np.array(n).astype(np.float64)[m > 0])
    assert np.max(np.abs(nats - np.concatenate(want))) < 1e-5


def test_table_off_equals_every_tl_zero(tmp_path: Path) -> None:
    """No table term equals tl = 0 bitwise; the term matters once tl is not zero."""
    held = HeldOut(write_store(tmp_path, docs_for(5, 12)), T)
    cfg = replace(TINY, table=True)
    p = init_params(cfg, 2)
    rng = np.random.default_rng(0)

    def term(ids: np.ndarray) -> mx.array:
        return mx.array(rng.normal(size=(*ids.shape, cfg.d_model)).astype(np.float32))

    on = score(p, cfg, held, B, term=term)
    zero = score(p | {"tl": mx.zeros_like(p["tl"])}, cfg, held, B, term=term)
    off = score(p, cfg, held, B)
    assert np.array_equal(off, zero)
    assert not np.array_equal(on, off)


def save(out: Path, cfg: ModelConfig, ema: bool) -> Path:
    p = init_params(cfg, 3)
    arrays = {"dense": p} | ({"ema": init_params(cfg, 4)} if ema else {})
    state = {"config": {"model": asdict(cfg), "compute_dtype": "float32"}}
    return ckpt.save(out, 5, arrays=arrays, state=state)


def test_entry_point_scores_raw_and_ema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    split = write_store(tmp_path / "store", docs_for(6, 20))
    c = save(tmp_path / "run", TINY, ema=True)
    rc = heldout.main(["--ckpt", str(tmp_path / "run"), "--data", str(split), "--seq-len", str(T)])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert (rec["ckpt"], rec["step"], rec["table"]) == (str(c), 5, "off")
    held = HeldOut(split, T)
    want = held.bpb(score(ckpt.load(c).arrays("ema"), TINY, held, 4))
    assert rec["ema"]["bpb"] == want
    assert rec["raw"]["bpb"] != want


def test_entry_point_refusals(tmp_path: Path) -> None:
    split = write_store(tmp_path / "store", docs_for(7, 8))
    args = ["--data", str(split), "--seq-len", str(T)]
    assert heldout.main(["--ckpt", str(tmp_path / "none"), *args]) == heldout.EXIT_CONFIG
    save(tmp_path / "table", replace(TINY, table=True), ema=False)
    assert heldout.main(["--ckpt", str(tmp_path / "table"), *args]) == heldout.EXIT_CONFIG
    out = tmp_path / "new" / "deeper" / "rec.json"  # neither folder exists yet
    tail = ["--table", "off", "--out", str(out)]
    assert heldout.main(["--ckpt", str(tmp_path / "table"), *args, *tail]) == 0
    rec = json.loads(out.read_text())
    assert rec["table"] == "off"
    assert "ema" not in rec
    assert math.isfinite(rec["raw"]["bpb"])
    small = replace(TINY, table=True, vocab_size=V - 1)
    save(tmp_path / "small", small, ema=False)
    small_args = ["--ckpt", str(tmp_path / "small"), *args, "--table", "off"]
    assert heldout.main(small_args) == heldout.EXIT_DATA
    (split / "000000.bytes").write_bytes(b"")
    assert (
        heldout.main(["--ckpt", str(tmp_path / "table"), *args, "--table", "off"])
        == heldout.EXIT_DATA
    )


def test_reads_a_trainer_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The readout reads the checkpoint training saves, EMA included."""
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    split = write_store(tmp_path / "store", docs_for(8, 20), train=True)
    run = tmp_path / "run"
    args = ["--data", str(tmp_path / "store" / "train"), "--tokens", str(3 * B * T)]
    args += ["--rows", str(B), "--seq-len", str(T), "--out", str(run), "--compute-dtype", "float32"]
    args += [f"--model={k}" for k in TINY_KEYS] + [
        "--recipe=lr.warmup=1",
        "--recipe=param.kind=sp",
        "--recipe=opt.decay_ind=0.0",
    ]
    assert entry.main([*args, "--print-every", "0", "--buffer", "16"], Run, Checkpointer) == 0
    capsys.readouterr()
    assert heldout.main(["--ckpt", str(run), "--data", str(split), "--seq-len", str(T)]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["step"] == 3
    assert math.isfinite(rec["raw"]["bpb"])
    assert math.isfinite(rec["ema"]["bpb"])


# ---- the table-on readout (the table in the checkpoint) ---------------------------------------

TABLE_KEYS = [f"params={4096 * 64}", f"max_shard_elements={1024 * 64}"]


def table_run(out: Path, monkeypatch: pytest.MonkeyPatch, steps: int = 3) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    keys = [k for k in TINY_KEYS if not k.startswith("table=")]
    args = ["--data", "synthetic", "--tokens", str(steps * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--out", str(out), "--compute-dtype", "float32"]
    args += [f"--model={k}" for k in keys] + [f"--table={k}" for k in TABLE_KEYS]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    assert entry.main([*args, "--print-every", "0"], Run, Checkpointer) == 0


def test_zero_table_term_equals_table_off(tmp_path: Path) -> None:
    """Absent equals off: a never-trained table's term leaves every target's nats bitwise."""
    held = HeldOut(write_store(tmp_path, docs_for(10, 12)), T)
    cfg = replace(TINY, table=True)
    tcfg = TableConfig(params=4096 * 64, width=64, max_shard_elements=1024 * 64)
    term = heldout.table_term_of(Table(tcfg, mx.get_active_memory), cfg.d_model, mx.float32)
    p = init_params(cfg, 5)
    p["tl"] = mx.ones_like(p["tl"])
    assert np.array_equal(score(p, cfg, held, B, term=term), score(p, cfg, held, B))


def test_reads_a_table_on_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trained table is loaded from the checkpoint and moves the readout; table_off is kept."""
    split = write_store(tmp_path / "store", docs_for(11, 20))
    run = tmp_path / "run"
    table_run(run, monkeypatch)
    capsys.readouterr()
    args = ["--ckpt", str(run), "--data", str(split), "--seq-len", str(T)]
    assert heldout.main(args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert (rec["table"], rec["step"], rec["table_event"]) == ("on", 3, 3)
    for name in ("raw", "ema"):
        assert math.isfinite(rec[name]["bpb"])
        assert rec[name]["bpb"] != rec[name]["table_off"]["bpb"]
    assert heldout.main([*args, "--table", "off"]) == 0
    off = json.loads(capsys.readouterr().out)
    assert off["table"] == "off"
    assert "table_event" not in off
    assert off["raw"]["bpb"] == rec["raw"]["table_off"]["bpb"]
    records = tmp_path / "records"
    assert heldout.main([*args, "--records", str(records)]) == 0
    capsys.readouterr()
    per_target = np.load(records / "targets.npy", allow_pickle=False)
    sets = ("raw", "raw.table_off", "ema", "ema.table_off")
    assert per_target.dtype.names == ("document", "window", *sets)
    for name in ("raw", "ema"):
        nats = per_target[f"{name}.table_off"].astype(np.float64)
        assert float(np.sum(nats) / (LN2 * rec["bytes"])) == rec[name]["table_off"]["bpb"]
    newest = ckpt.latest(run)
    assert newest is not None
    record = newest / heldout.TABLE_DIR / "table.json"
    text: str = record.read_text()
    record.write_text(text.replace("table-shard/v1", "table-shard/v0"))
    assert heldout.main(args) == heldout.EXIT_CONFIG
    record.write_text("{")
    assert heldout.main(args) == heldout.EXIT_CONFIG


def test_the_readout_never_writes_the_table_back() -> None:
    """The store's counter is read only around a write-back; the readout's refuses one."""
    with pytest.raises(AkeratosError, match="never writes"):
        heldout.no_write_back()


# ---- per-document and per-target records, and the base model's mask -------------------------


def readout(args: list[str], out: Path) -> dict[str, Any]:
    """The entry point's record, written to `out`."""
    assert heldout.main([*args, "--out", str(out)]) == 0
    record: dict[str, Any] = json.loads(out.read_text())
    return record


def scored_records(
    tmp_path: Path, docs: list[np.ndarray], *extra: str
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    """The record of TINY over `docs` with --records and `extra`, its two files, and the args."""
    split = write_store(tmp_path / "store", docs)
    save(tmp_path / "run", TINY, ema=True)
    args = ["--ckpt", str(tmp_path / "run"), "--data", str(split), "--seq-len", str(T), *extra]
    rec = readout([*args, "--records", str(tmp_path / "records")], tmp_path / "rec.json")
    load = [np.load(tmp_path / "records" / f, allow_pickle=False) for f in heldout.RECORDS]
    return rec, load[0], load[1], args


def test_without_the_options_the_record_is_as_before(tmp_path: Path) -> None:
    """Without the options the record keeps its keys; they add `records` and `mask` only."""
    split = write_store(tmp_path / "store", docs_for(13, 20, long=1))
    c = save(tmp_path / "run", TINY, ema=True)
    args = ["--ckpt", str(tmp_path / "run"), "--data", str(split), "--seq-len", str(T)]
    out = tmp_path / "rec.json"
    readout(args, out)
    held = HeldOut(split, T)
    want: dict[str, Any] = {"ckpt": str(c), "step": 5, "table": "off", "dtype": str(mx.float32)}
    want |= {"data": str(split), "seq_len": T}
    want |= {"docs": held.docs, "targets": held.targets, "bytes": held.bytes}
    for name, arrays in (("raw", "dense"), ("ema", "ema")):
        nats = score(ckpt.load(c).arrays(arrays), TINY, held, 4)
        want[name] = {"bpb": held.bpb(nats), "nats_per_target": float(nats.mean(dtype=np.float64))}
    assert out.read_text() == json.dumps(want, sort_keys=True) + "\n"
    records = tmp_path / "records"
    flagged = [*args, "--records", str(records), "--mask", "base"]
    rec = readout(flagged, tmp_path / "flagged.json")
    assert rec.pop("mask") == "base"  # TINY is a base model: its own mask is the base mask
    assert rec.pop("records")["dir"] == str(records)
    assert json.dumps(rec, sort_keys=True) + "\n" == out.read_text()


@pytest.mark.parametrize("tokens", [None, 60])
def test_records_sum_to_the_record(tmp_path: Path, tokens: int | None) -> None:
    """Per-document sums give bits per byte and bytes; per-target nats give bits per byte exactly.

    Under --tokens the records hold a row for each document read, and none for the rest.
    """
    docs = docs_for(14, 30, long=2)
    extra = [] if tokens is None else ["--tokens", str(tokens)]
    rec, per_doc, per_target, args = scored_records(tmp_path, docs, *extra)
    assert (rec["docs"] < len(docs)) == (tokens is not None)
    assert per_doc.dtype.names == ("targets", "bytes", "raw", "ema")
    assert per_target.dtype.names == ("document", "window", "raw", "ema")
    assert per_doc["targets"].tolist() == [len(d) for d in docs[: rec["docs"]]]
    assert int(per_doc["bytes"].sum()) == rec["bytes"]
    assert len(per_target) == rec["targets"]
    for name in ("raw", "ema"):
        bpb = per_doc[name].sum() / (LN2 * per_doc["bytes"].sum())
        assert math.isclose(bpb, rec[name]["bpb"], rel_tol=1e-12)
        nats = per_target[name].astype(np.float64)
        assert float(np.sum(nats) / (LN2 * rec["bytes"])) == rec[name]["bpb"]
        assert np.array_equal(np.bincount(per_target["document"], weights=nats), per_doc[name])
    files = tmp_path / "records"
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files.iterdir()}
    assert sorted(sha) == sorted(heldout.RECORDS)
    assert rec["records"] == {"dir": str(files), "sha256": sha}
    assert heldout.main([*args, "--records", str(files)]) == heldout.EXIT_CONFIG


def test_records_are_refused_before_the_checkpoint_is_opened(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DIR holding either file, or a file as DIR, is refused before anything is scored."""
    split = write_store(tmp_path / "store", docs_for(19, 8))
    args = ["--ckpt", str(tmp_path / "run"), "--data", str(split), "--seq-len", str(T)]

    def never(*_: object) -> None:
        raise AssertionError("the checkpoint was opened before --records was refused")

    monkeypatch.setattr(heldout, "open_ckpt", never)
    for file in heldout.RECORDS:
        taken = tmp_path / Path(file).stem
        taken.mkdir()
        (taken / file).write_bytes(b"")
        assert heldout.main([*args, "--records", str(taken)]) == heldout.EXIT_CONFIG
        assert "holds records already" in capsys.readouterr().err
    a_file = tmp_path / "a-file"
    a_file.write_text("")
    assert heldout.main([*args, "--records", str(a_file)]) == heldout.EXIT_CONFIG
    assert str(a_file) in capsys.readouterr().err


def test_window_ids_follow_the_rows(tmp_path: Path) -> None:
    """Targets are read document by document, and a long document's k-th row is its window k."""
    docs = docs_for(15, 10, long=2)
    held = HeldOut(write_store(tmp_path, docs), T)
    doc, window = held.windows()
    assert np.array_equal(held.ids[:, 1:][held.mask > 0], np.concatenate(docs))
    assert np.array_equal(doc, np.repeat(np.arange(len(docs)), [len(d) for d in docs]))
    row = np.nonzero(held.mask)[0]
    assert np.array_equal(window, row - row[np.searchsorted(doc, doc)])
    assert int(window.max()) == math.ceil(len(docs[-1]) / T) - 1


def test_a_document_scored_alone_equals_its_record(tmp_path: Path) -> None:
    """A document packed among others, and a long one, scored alone give their records."""
    docs = docs_for(16, 20, long=1)
    _, per_doc, per_target, _ = scored_records(tmp_path, docs)
    held = HeldOut(tmp_path / "store" / "heldout", T)
    column = np.nonzero(held.mask)[1][np.searchsorted(held.windows()[0], 5)]
    assert column > 0  # document 5 starts after another in its row
    for i in (5, len(docs) - 1):
        alone = tmp_path / f"alone-{i}"
        args = ["--ckpt", str(tmp_path / "run"), "--data", str(write_store(alone, [docs[i]]))]
        readout([*args, "--seq-len", str(T), "--records", str(alone / "rec")], alone / "rec.json")
        one = np.load(alone / "rec" / "documents.npy", allow_pickle=False)
        assert len(one) == 1
        assert (one["targets"][0], one["bytes"][0]) == (per_doc["targets"][i], per_doc["bytes"][i])
        mine = per_target[per_target["document"] == i]
        alone_t = np.load(alone / "rec" / "targets.npy", allow_pickle=False)
        assert np.array_equal(alone_t["window"], mine["window"])
        for name in ("raw", "ema"):
            assert np.max(np.abs(alone_t[name] - mine[name])) < 1e-4
            assert math.isclose(one[name][0], per_doc[name][i], rel_tol=1e-5)


def test_a_long_document_is_the_sum_of_its_windows(tmp_path: Path) -> None:
    """Each window of a long document, scored as its own row, gives its targets' records."""
    docs = docs_for(17, 6, long=1)
    _, per_doc, per_target, _ = scored_records(tmp_path, docs)
    newest = ckpt.latest(tmp_path / "run")
    assert newest is not None
    p = ckpt.load(newest).arrays("dense")
    i, full = len(docs) - 1, np.concatenate([[0], docs[-1]])
    mine = per_target[per_target["document"] == i]
    assert mine["window"].tolist() == [j // T for j in range(len(docs[-1]))]
    total = 0.0
    for k in range(math.ceil(len(docs[-1]) / T)):
        piece = full[k * T : k * T + T + 1]
        row = np.zeros((1, T + 1), np.int64)
        row[0, : len(piece)] = piece
        nats = heldout.row_nats(p, TINY, row[:, :-1], row[:, 1:])[0, : len(piece) - 1]
        assert np.max(np.abs(nats - mine["raw"][mine["window"] == k])) < 1e-5
        total += float(nats.sum())
    assert math.isclose(per_doc["raw"][i], total, rel_tol=1e-6)


def test_mask_base_scores_a_fine_tune_over_its_base_softmax(tmp_path: Path) -> None:
    """Stage 1 unmasks ids 1 and 2; --mask base scores its weights as the base head would."""
    split = write_store(tmp_path / "store", docs_for(18, 15))
    tuned = stage_config(TINY, 1)
    assert tuned.masked_ids == tuple(range(3, 17))
    c = save(tmp_path / "sft", tuned, ema=False)
    args = ["--ckpt", str(tmp_path / "sft"), "--data", str(split), "--seq-len", str(T)]
    own = readout(args, tmp_path / "own.json")
    base = readout([*args, "--mask", "base"], tmp_path / "base.json")
    held, p = HeldOut(split, T), ckpt.load(c).arrays("dense")
    assert own["raw"]["bpb"] == held.bpb(score(p, tuned, held, 4))
    assert base["raw"]["bpb"] == held.bpb(score(p, TINY, held, 4))
    assert own["raw"]["bpb"] > base["raw"]["bpb"]  # two more ids share the softmax
    assert (base["mask"], "mask" in own) == ("base", False)
    other = replace(tuned, masked_ids=(*tuned.masked_ids, 100))
    assert heldout._base_mask(other).masked_ids == (*range(1, 17), 100)  # added to its own mask


def test_mask_base_gives_the_base_record_with_the_table_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table-on checkpoint with an EMA relabelled as stage 1: --mask base gives its base record.

    Under its own mask every set's bits per byte is higher: raw, EMA and both without the table.
    """
    split = write_store(tmp_path / "store", docs_for(20, 20))
    run = tmp_path / "run"
    table_run(run, monkeypatch)
    capsys.readouterr()
    args = ["--ckpt", str(run), "--data", str(split), "--seq-len", str(T)]
    base = readout(args, tmp_path / "base.json")
    newest = ckpt.latest(run)
    assert newest is not None
    state = json.loads((newest / ckpt.STATE).read_text())
    model = state["state"]["config"]["model"]
    assert model["masked_ids"] == list(range(1, 17))
    model["masked_ids"] = list(range(3, 17))  # stage 1's mask (stage_config): ids 1 and 2 unmasked
    (newest / ckpt.STATE).write_text(json.dumps(state))
    own = readout(args, tmp_path / "own.json")
    masked = readout([*args, "--mask", "base"], tmp_path / "masked.json")
    assert masked.pop("mask") == "base"
    assert masked == base
    for name in ("raw", "ema"):
        assert own[name]["bpb"] > base[name]["bpb"]
        assert own[name]["table_off"]["bpb"] > base[name]["table_off"]["bpb"]
