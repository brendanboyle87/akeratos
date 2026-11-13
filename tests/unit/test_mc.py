"""Zero-shot multiple-choice scores over a checkpoint.

The scorer's Σ log p and greedy flags are held to a direct, unbatched and unpadded forward pass,
the store's tokenizer to its manifest, and the entry point to checkpoints with and without a table.
Everything runs at tiny sizes on MLX's CPU device; the tokenizer is a byte-level BPE with no merges.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from akeratos import ckpt
from akeratos.configs import ModelConfig
from akeratos.data.packer import segment_ids
from akeratos.errors import DataMismatchError
from akeratos.eval import mc, tasks
from akeratos.eval.tasks import TASKS, Item
from akeratos.model import EMBED, head_logits, hidden, init_params, masked_id_mask
from akeratos.train import entry
from akeratos.train.loop import Run
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


def byte_chars() -> list[str]:
    """GPT-2's byte-to-unicode table, in byte order."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, cs, strict=True))
    return [chr(table[b]) for b in range(256)]


def tokenizer_json() -> bytes:
    """Ids 0-16 special, then the 256 byte tokens at 17-272: every string encodes, byte by byte."""
    vocab = {f"<|s{i}|>": i for i in range(17)} | {c: 17 + i for i, c in enumerate(byte_chars())}
    tok = Tokenizer(models.BPE(vocab, []))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return tok.to_str().encode()


def write_store(root: Path, cut: int | None = None) -> Path:
    raw = tokenizer_json()
    name = "tokenizer.json" if cut is None else f"tokenizer.prefix-{cut}.json"
    (root / "tokenizer").mkdir(parents=True)
    (root / "tokenizer" / name).write_bytes(raw)
    files = [{"path": name, "sha256": hashlib.sha256(raw).hexdigest()}]
    (root / "manifest.json").write_text(
        json.dumps({"tokenizer": {"files": files, "merge_prefix": cut}})
    )
    return root


def items_for(n: int = 3) -> dict[str, list[Item]]:
    rows = {
        "piqa": lambda i: {
            "goal": f"Open jar {i}",
            "sol1": "Twist.",
            "sol2": f"Freeze it {i}.",
            "label": i % 2,
        },
        "winogrande": lambda i: {
            "sentence": f"A{i} hit B because _ was mad.",
            "option1": "A",
            "option2": "B",
            "answer": "1",
        },
        "lambada_openai": lambda i: {"text": f"the cat {i} sat on the mat"},
    }
    return {t: [tasks.render(t, f(i), i) for i in range(n)] for t, f in rows.items()}


def write_set(root: Path, items: dict[str, list[Item]]) -> Path:
    src = {t: {"repo": TASKS[t].repo, "revision": "ab" * 20} for t in items}
    set_id, files = tasks.set_files(src, items, "test")
    (root / set_id).mkdir(parents=True)
    for name, body in files.items():
        (root / set_id / name).write_bytes(body)
    return root / set_id


def direct(p: dict[str, mx.array], cfg: ModelConfig, ids: list[int], n: int) -> tuple[float, bool]:
    x = np.asarray(ids[:-1], np.int32)[None]
    cfg = replace(
        cfg, attn_path="masked"
    )  # any length; the scorer bands at multiples of the window
    h = hidden(p, cfg, mx.array(x), mx.array(segment_ids(x)), mx.float32)
    z = np.array(head_logits(p, cfg, h, masked_id_mask(cfg)))[0].astype(np.float64)
    y = np.asarray(ids[1:])
    lp = z[np.arange(len(y)), y] - np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1)) - z.max(-1)
    return float(lp[-n:].sum()), bool((z.argmax(-1) == y)[-n:].all())


def test_batched_scores_equal_a_direct_forward_pass(tmp_path: Path) -> None:
    encode, _ = mc.encoder(write_store(tmp_path))
    its = items_for(4)
    seqs = [r for t in sorted(its) for r in mc.requests_of(its[t], encode, 3 * T, V)]
    assert all(r[0][0] == 0 for r in seqs)
    p = init_params(TINY, 1)
    got = mc.loglikelihoods(p, TINY, seqs, B)
    for (ids, n), (ll, greedy) in zip(seqs, got, strict=True):
        want_ll, want_greedy = direct(p, TINY, ids, n)
        assert abs(ll - want_ll) < 1e-3
        assert greedy == want_greedy


def test_a_flat_model_costs_ln_v_minus_16_per_token(tmp_path: Path) -> None:
    """With the tied matrix at zero, every unmasked id is equally likely."""
    encode, _ = mc.encoder(write_store(tmp_path))
    seqs = mc.requests_of(items_for(2)["piqa"], encode, T, V)
    p = init_params(TINY, 0)
    p[EMBED] = mx.zeros_like(p[EMBED])
    for (_, n), (ll, _) in zip(seqs, mc.loglikelihoods(p, TINY, seqs, B), strict=True):
        assert ll == pytest.approx(-n * math.log(V - 16), rel=1e-5)


def test_long_requests_are_cut_from_the_left(tmp_path: Path) -> None:
    encode, _ = mc.encoder(write_store(tmp_path))
    item = tasks.render("lambada_openai", {"text": "word " * 40 + "end"}, 0)
    [(ids, n)] = mc.requests_of([item], encode, T, V)
    assert len(ids) == T + 1
    assert n == len(" end")
    assert ids[0] != 0


def test_requests_refuse_reserved_and_out_of_range_ids(tmp_path: Path) -> None:
    encode, _ = mc.encoder(write_store(tmp_path))
    item = items_for(1)["piqa"][0]
    with pytest.raises(DataMismatchError, match="an id 0-16"):
        mc.requests_of([item], encode, T, 100)
    with pytest.raises(DataMismatchError, match="an id 0-16"):
        mc.requests_of([item], lambda s: [5] * len(s), T, V)
    with pytest.raises(DataMismatchError, match="exceeds"):
        mc.requests_of([item], encode, 4, V)


def test_the_encoder_is_the_stores_own(tmp_path: Path) -> None:
    encode, sha = mc.encoder(write_store(tmp_path / "a"))
    assert sha == hashlib.sha256(tokenizer_json()).hexdigest()
    assert encode(" é!") == [17 + b for b in " é!".encode()]
    assert encode("<|s1|>") == [17 + b for b in b"<|s1|>"]  # special-token text is plain bytes
    cut, _ = mc.encoder(write_store(tmp_path / "b", cut=8192))
    assert cut("ab") == encode("ab")
    bad = write_store(tmp_path / "c")
    (bad / "tokenizer" / "tokenizer.json").write_bytes(tokenizer_json() + b" ")
    with pytest.raises(DataMismatchError, match="SHA-256"):
        mc.encoder(bad)
    with pytest.raises(DataMismatchError, match="no store tokenizer"):
        mc.encoder(tmp_path / "none")


def save(out: Path, cfg: ModelConfig, ema: bool) -> Path:
    p = init_params(cfg, 3)
    arrays = {"dense": p} | ({"ema": init_params(cfg, 4)} if ema else {})
    state = {"config": {"model": asdict(cfg), "compute_dtype": "float32"}}
    return ckpt.save(out, 5, arrays=arrays, state=state)


def test_entry_point_scores_every_task(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store, items = write_store(tmp_path / "store"), items_for(3)
    s = write_set(tmp_path / "evals", items)
    c = save(tmp_path / "run", TINY, ema=True)
    # Neither folder exists yet.
    out, samples = tmp_path / "a" / "mc.json", tmp_path / "b" / "samples.jsonl"
    args = ["--ckpt", str(tmp_path / "run"), "--set", str(s), "--store", str(store)]
    files = ["--out", str(out), "--samples", str(samples)]
    assert mc.main([*args, "--seq-len", str(T), "--rows", "3", *files]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text()) == rec
    assert (rec["ckpt"], rec["step"], rec["table"], rec["limit"]) == (str(c), 5, "off", None)
    assert rec["set_id"] == s.name
    assert rec["prefix"] == [0]
    encode, _ = mc.encoder(store)
    p = ckpt.load(c).arrays("ema")
    for t, its in items.items():
        assert rec["ema"][t]["n"] == 3
        res = mc.loglikelihoods(p, TINY, mc.requests_of(its, encode, T, V), 4)
        k, per = 0, []
        for it in its:
            per.append(res[k : k + len(it["requests"])])
            k += len(it["requests"])
        assert rec["ema"][t] == pytest.approx(tasks.aggregate(t, its, per), nan_ok=True)
    lines = [json.loads(x) for x in samples.read_text().splitlines()]
    assert len(lines) == 2 * 9
    assert {(x["weights"], x["term"]) for x in lines} == {("raw", "off"), ("ema", "off")}
    assert mc.main([*args, "--seq-len", str(T), "--limit", "1", "--tasks", "piqa"]) == 0
    one = json.loads(capsys.readouterr().out)
    assert one["limit"] == 1
    assert list(one["raw"]) == ["piqa"]
    assert one["raw"]["piqa"]["n"] == 1
    assert rec["decontam"] is None
    dec = tmp_path / "decontam"
    dec.mkdir()
    sha = hashlib.sha256((s / "piqa.jsonl").read_bytes()).hexdigest()
    # The fixture store has no fingerprint.json, so a matching record names none.
    tok, store_meta = rec["tokenizer_sha256"], {"fingerprint": None}
    report: dict[str, object] = {"complete": True, "tokenizer_sha256": tok, "store": store_meta}
    d: dict[str, object] = {"schema": tasks.DECONTAM_SCHEMA, "task_sha256": sha, "drop": ["0"]}
    (dec / "piqa.decontam.json").write_text(json.dumps(d | {"report": report}))
    one_task = [*args, "--seq-len", str(T), "--tasks", "piqa", "--decontam", str(dec)]
    assert mc.main(one_task) == 0
    clean = json.loads(capsys.readouterr().out)
    assert clean["raw"]["piqa"]["n"] == 2
    assert clean["decontam"]["piqa"]["dropped"] == 1
    assert mc.main([*args, "--seq-len", str(T), "--decontam", str(dec)]) == mc.EXIT_DATA
    (store / "fingerprint.json").write_text(json.dumps({"fingerprint": "9545f2d205efd574"}))
    assert mc.main(one_task) == mc.EXIT_DATA  # the record was made against another store


def test_a_fused_checkpoint_is_scored_on_the_banded_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store, s = write_store(tmp_path / "store"), write_set(tmp_path / "evals", items_for(2))
    recs: list[object] = []
    for path in ("banded", "fused"):
        save(tmp_path / path, replace(TINY, attn_path=path), ema=False)
        args = ["--ckpt", str(tmp_path / path), "--set", str(s), "--store", str(store)]
        assert mc.main([*args, "--seq-len", str(T)]) == 0
        recs.append(json.loads(capsys.readouterr().out)["raw"])
    assert recs[0] == recs[1]  # same bits: the fused path is never used to score


def test_entry_point_refusals(tmp_path: Path) -> None:
    store, s = write_store(tmp_path / "store"), write_set(tmp_path / "evals", items_for(2))
    save(tmp_path / "run", TINY, ema=False)
    args = ["--set", str(s), "--store", str(store), "--seq-len", str(T)]
    assert mc.main(["--ckpt", str(tmp_path / "none"), *args]) == mc.EXIT_CONFIG
    good = ["--ckpt", str(tmp_path / "run"), *args]
    assert mc.main([*good, "--tasks", "hellaswag"]) == mc.EXIT_DATA
    small = replace(TINY, vocab_size=100)
    save(tmp_path / "small", small, ema=False)
    assert mc.main(["--ckpt", str(tmp_path / "small"), *args]) == mc.EXIT_DATA
    (s / "piqa.jsonl").write_bytes(b"")
    assert mc.main(good) == mc.EXIT_DATA


TABLE_KEYS = [f"params={4096 * 64}", f"max_shard_elements={1024 * 64}"]
TINY_KEYS = [
    f"{k}={json.dumps(v)}" for k, v in asdict(TINY).items() if v != getattr(ModelConfig(), k)
]


def test_a_table_on_checkpoint_is_scored_with_and_without_its_term(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    run = tmp_path / "run"
    keys = [k for k in TINY_KEYS if not k.startswith("table=")]
    a = ["--data", "synthetic", "--tokens", str(3 * B * T), "--rows", str(B), "--seq-len", str(T)]
    a += ["--out", str(run), "--compute-dtype", "float32", "--print-every", "0"]
    a += [f"--model={k}" for k in keys] + [f"--table={k}" for k in TABLE_KEYS]
    a += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    assert entry.main(a, Run, Checkpointer) == 0
    capsys.readouterr()
    store, s = write_store(tmp_path / "store"), write_set(tmp_path / "evals", items_for(3))
    args = ["--ckpt", str(run), "--set", str(s), "--store", str(store), "--seq-len", str(T)]
    assert mc.main(args) == 0
    rec = json.loads(capsys.readouterr().out)
    assert (rec["table"], rec["table_event"]) == ("on", 3)
    for name in ("raw", "ema"):
        assert set(rec[name]) == {"piqa", "winogrande", "lambada_openai", "table_off"}
        assert math.isfinite(rec[name]["lambada_openai"]["perplexity"])
        on, off = rec[name]["lambada_openai"], rec[name]["table_off"]["lambada_openai"]
        assert on["perplexity"] != off["perplexity"]
    assert mc.main([*args, "--table", "off"]) == 0
    off_rec = json.loads(capsys.readouterr().out)
    assert off_rec["table"] == "off"
    assert off_rec["raw"]["piqa"] == rec["raw"]["table_off"]["piqa"]
