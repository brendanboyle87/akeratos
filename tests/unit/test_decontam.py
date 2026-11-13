"""Decontamination: token 13-gram overlap of eval and SFT items with the training shards.

A planted overlap is found and dropped, a near miss is not, and an n-gram never crosses a
document boundary. The rest pin the record format the SFT data preparation and the
multiple-choice scorer read, the chunked scan, and the Mac script that tokenises the sets (run
here on a word-level fixture tokenizer, never SmolLM2's).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from akeratos.data import decontam
from akeratos.data.decontam import (
    ItemSet,
    build_index,
    read_items,
    records,
    scan,
    sft_against_eval,
    store_meta,
)
from akeratos.data.store import Split
from akeratos.errors import DataMismatchError

V = 512
FP = "f" * 64
SHA = "a" * 64
SCRIPT = Path("scripts/data/decontam_items.py")


def rand_doc(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(17, V, size=n).astype("<u2")


def write_train(root: Path, shards: list[list[np.ndarray]]) -> Path:
    """A store with a train split of one shard per list of documents, and its fingerprint."""
    path = root / "train"
    path.mkdir(parents=True)
    names, tokens, counts = [], [], []
    for s, docs in enumerate(shards):
        body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
        idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
        body.tofile(path / f"{s:06d}.bin")
        idx.tofile(path / f"{s:06d}.idx")
        names.append(f"{s:06d}.bin")
        tokens.append(len(body))
        counts.append(len(docs))
    m = {"dtype": "uint16", "eos": 0, "vocab": V, "shards": names, "tokens": tokens}
    m |= {"documents": counts, "sep_position": "start"}
    (path / "manifest.json").write_text(json.dumps(m))
    (root / "fingerprint.json").write_text(json.dumps({"fingerprint": FP}))
    return root


def item_set(name: str, items: dict[Any, list[np.ndarray]], kind: str = "eval") -> ItemSet:
    segs = [[np.asarray(s, "<u2") for s in v] for v in items.values()]
    return ItemSet(name, kind, f"{name}.jsonl", SHA, SHA, list(items), segs)


def run(store: Path, sets: list[ItemSet], **kw: Any) -> list[dict[str, Any]]:
    split = Split(store / "train")
    idx = build_index(sets)
    sc = scan(idx, split, **kw)
    return records(sets, idx, sc, store_meta(store, split, sc), sft_against_eval(sets))


def test_planted_overlap_is_found_logged_and_dropped(tmp_path: Path) -> None:
    """An eval item planted in shard 1, document 2 is dropped."""
    item = rand_doc(1, 20)
    host = np.concatenate([rand_doc(2, 30), item[3:18], rand_doc(3, 10)])
    store = write_train(tmp_path, [[rand_doc(4, 50)], [rand_doc(5, 40), rand_doc(6, 9), host]])
    rec = run(store, [item_set("hellaswag", {7: [item], 8: [rand_doc(9, 30)]})])[0]
    assert rec["schema"] == "metalforge.eval-decontam/v1"
    assert rec["task_sha256"] == SHA
    assert rec["drop"] == [7]
    (o,) = rec["report"]["overlaps"]
    assert o["id"] == 7
    assert o["hits"] == 3  # 15 shared ids hold three 13-grams
    m = o["matches"][0]
    assert (m["shard"], m["document"]) == ("000001.bin", 2)
    assert m["ngram"] == item[3:16].tolist()
    assert rec["report"]["dropped"] == 1
    assert rec["report"]["complete"] is True


def test_no_false_drop(tmp_path: Path) -> None:
    """No 13-gram in the shards, so nothing is dropped.

    Twelve shared ids is one short of an n-gram.
    """
    item = rand_doc(1, 20)
    host = np.concatenate([rand_doc(2, 30), item[:12], rand_doc(3, 30)])
    store = write_train(tmp_path, [[host, rand_doc(4, 100)]])
    rec = run(store, [item_set("piqa", {"a": [item], "b": [rand_doc(5, 40)]})])[0]
    assert rec["drop"] == []
    assert rec["report"]["overlaps"] == []
    assert rec["report"]["items"] == 2


def test_an_ngram_across_a_document_boundary_does_not_count(tmp_path: Path) -> None:
    """The item's 13 ids sit in the shard split by a document's id 0."""
    item = rand_doc(1, 13)
    store = write_train(tmp_path, [[np.concatenate([rand_doc(2, 9), item[:6]]), item[6:]]])
    assert np.isin(np.fromfile(store / "train/000000.bin", "<u2"), 0).sum() == 2
    rec = run(store, [item_set("lambada", {0: [item]})])[0]
    assert rec["drop"] == []


def test_an_ngram_across_two_segments_does_not_count(tmp_path: Path) -> None:
    """Each message is a segment: an n-gram made of two turns' ends is never formed."""
    seq = rand_doc(1, 13)
    store = write_train(tmp_path, [[np.concatenate([rand_doc(2, 5), seq, rand_doc(3, 5)])]])
    rec = run(store, [item_set("sft", {"c1": [seq[:7], seq[7:]]}, kind="sft")])[0]
    assert rec["drop"] == []
    assert rec["report"]["unchecked"] == 1


def test_sft_record_is_the_drop_list_prepare_reads(tmp_path: Path) -> None:
    """The SFT record holds schema, conversations_sha256, drop and report."""
    turn = rand_doc(1, 40)
    store = write_train(tmp_path, [[np.concatenate([rand_doc(2, 5), turn[10:30]])]])
    sets = [
        item_set(
            "sft", {"oasst2:x": [rand_doc(3, 20), turn], "oasst2:y": [rand_doc(4, 30)]}, "sft"
        ),
        item_set("arc_easy", {1: [turn[5:25]], 2: [rand_doc(5, 25)]}),
    ]
    sft, arc = run(store, sets)
    assert set(sft) == {"schema", "conversations_sha256", "drop", "report"}
    assert sft["schema"] == "metalforge.sft-decontam/v1"
    assert sft["drop"] == ["oasst2:x"]
    assert arc["drop"] == [1]
    assert sft["report"]["n"] == 13
    assert sft["report"]["store"]["fingerprint"] == FP


def test_an_sft_conversation_quoting_an_eval_item_is_dropped(tmp_path: Path) -> None:
    """The SFT text trains the post-trained model, so a turn that quotes an eval item drops it.

    The store holds neither; the eval item itself is kept, so both models score the same set.
    """
    gold = rand_doc(1, 30)
    store = write_train(tmp_path, [[rand_doc(2, 200)]])
    quote = np.concatenate([rand_doc(3, 6), gold[4:20], rand_doc(4, 6)])
    sft = {"c1": [rand_doc(5, 20)], "c2": [rand_doc(6, 15), quote], "c3": [gold[:12], gold[12:24]]}
    sets = [
        item_set("sft", sft, "sft"),
        item_set("hellaswag", {"h0": [gold], "h1": [rand_doc(7, 30)]}),
    ]
    sft_rec, hs = run(store, sets)
    assert sft_rec["drop"] == ["c2"]
    assert sft_rec["report"]["dropped"] == 1
    (o,) = sft_rec["report"]["eval_overlaps"]
    assert o["id"] == "c2"
    assert [(e["set"], e["id"]) for e in o["eval"]] == [("hellaswag", "h0")]
    assert o["eval"][0]["ngram"] == gold[4:17].tolist()
    assert sft_rec["report"]["overlaps"] == []
    assert hs["drop"] == []
    assert "eval_overlaps" not in hs["report"]


def test_short_items_are_counted_unchecked_and_kept(tmp_path: Path) -> None:
    store = write_train(tmp_path, [[rand_doc(1, 50)]])
    rec = run(store, [item_set("piqa", {0: [rand_doc(1, 12)], 1: [rand_doc(2, 13)]})])[0]
    assert rec["report"]["unchecked"] == 1
    assert rec["drop"] == []


@pytest.mark.parametrize("chunk", [1, 7, 13, 64, 1 << 20])
def test_the_chunked_scan_finds_every_hit_once(tmp_path: Path, chunk: int) -> None:
    item = rand_doc(1, 30)
    docs = [np.concatenate([rand_doc(s, 11), item, rand_doc(s + 50, 3)]) for s in range(2, 6)]
    store = write_train(tmp_path, [docs])
    rec = run(store, [item_set("hellaswag", {0: [item]})], chunk=chunk)[0]
    (o,) = rec["report"]["overlaps"]
    assert o["hits"] == 4 * 18
    assert o["matches"][0]["document"] == 0


def test_repeated_ngrams_share_one_key_and_count_per_item(tmp_path: Path) -> None:
    """Two items holding the same 13 ids are both dropped; three matches are shown at most."""
    shared = rand_doc(1, 20)
    store = write_train(tmp_path, [[np.concatenate([rand_doc(2, 4), shared])]])
    sets = [item_set("winogrande", {"a": [shared], "b": [shared, shared]})]
    idx = build_index(sets)
    assert len(idx.hashes) == 8
    rec = run(store, sets)[0]
    assert rec["drop"] == ["a", "b"]
    assert [len(o["matches"]) for o in rec["report"]["overlaps"]] == [3, 3]
    assert [o["hits"] for o in rec["report"]["overlaps"]] == [8, 8]


def test_a_partial_scan_says_so(tmp_path: Path) -> None:
    item = rand_doc(1, 20)
    store = write_train(tmp_path, [[rand_doc(2, 40)], [item]])
    rec = run(store, [item_set("piqa", {0: [item]})], shards=1)[0]
    assert rec["report"]["complete"] is False
    assert rec["report"]["store"]["shards_scanned"] == ["000000.bin"]
    assert rec["drop"] == []


def write_items(path: Path, header: dict[str, Any], items: list[dict[str, Any]]) -> Path:
    lines = [json.dumps(header)] + [json.dumps(i) for i in items]
    path.write_text("\n".join(lines) + "\n")
    return path


def header(**kw: Any) -> dict[str, Any]:
    h = {"schema": "metalforge.decontam-items/v1", "set": "piqa", "kind": "eval"}
    h |= {"source": "piqa.jsonl", "source_sha256": SHA, "tokenizer_sha256": SHA}
    return h | {"store_fingerprint": FP, "items": 1} | kw


@pytest.mark.parametrize(
    ("head", "items", "what"),
    [
        (header(store_fingerprint="0" * 64), [{"id": 0, "segments": []}], "tokenised for store"),
        (header(items=2), [{"id": 0, "segments": []}] * 2, "repeated ids"),
        (header(), [{"id": 0, "segments": [[17, V]]}], "outside"),
        (header(), [{"id": True, "segments": []}], "not a string"),
        (header(kind="train"), [{"id": 0, "segments": []}], "bad kind"),
        (header(set="../x"), [{"id": 0, "segments": []}], "bad kind"),
        (header(source_sha256="x"), [{"id": 0, "segments": []}], "SHA-256"),
        (header(schema="v0"), [], "header"),
        (header(), [{"id": 0}], "not an item line"),
        (header(), [{"id": 0, "segments": [[17], "x"]}], "not a list of ids"),
    ],
)
def test_read_items_refuses_a_bad_file(
    tmp_path: Path, head: dict[str, Any], items: list[dict[str, Any]], what: str
) -> None:
    path = write_items(tmp_path / "x.items.jsonl", head, items)
    with pytest.raises(DataMismatchError, match=what):
        read_items(path, V, FP)


def test_read_items_refuses_text_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "x.items.jsonl"
    path.write_text(json.dumps(header()) + "\n{not json\n")
    with pytest.raises(DataMismatchError, match=r"x\.items\.jsonl"):
        read_items(path, V, FP)


def test_main_writes_one_record_per_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    item = rand_doc(1, 20)
    store = write_train(tmp_path / "store", [[np.concatenate([rand_doc(2, 5), item])]])
    a = write_items(
        tmp_path / "a.items.jsonl",
        header(items=2),
        [{"id": 0, "segments": [item.tolist()]}, {"id": 1, "segments": [rand_doc(3, 20).tolist()]}],
    )
    b = write_items(
        tmp_path / "b.items.jsonl", header(set="sft", kind="sft"), [{"id": "c", "segments": []}]
    )
    out = tmp_path / "out"
    argv = ["--store", str(store), "--items", str(a), str(b), "--out", str(out)]
    assert decontam.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "complete": True,
        "shards_scanned": 1,
        "sets": {"piqa": [2, 1], "sft": [1, 0]},
    }
    assert json.loads((out / "piqa.decontam.json").read_text())["drop"] == [0]
    assert json.loads((out / "sft.decontam.json").read_text())["conversations_sha256"] == SHA
    assert decontam.main([*argv[:4], str(a), "--out", str(out), "--n", "0"]) == decontam.EXIT_CONFIG
    assert decontam.main([*argv[:3], str(a), str(a), "--out", str(out)]) == decontam.EXIT_DATA


def fixture_tokenizer(store: Path, words: list[str]) -> dict[str, int]:
    """A word-level tokenizer.json in the store, with ids 0-16 special and words from 17."""
    vocab = {f"<|s{i}|>": i for i in range(17)} | {w: 17 + i for i, w in enumerate(words)}
    tok = Tokenizer(WordLevel(vocab, "<|s3|>"))
    tok.pre_tokenizer = WhitespaceSplit()
    data = tok.to_str().encode()
    (store / "tokenizer").mkdir()
    (store / "tokenizer/tokenizer.json").write_bytes(data)
    files = [{"path": "tokenizer.json", "sha256": hashlib.sha256(data).hexdigest()}]
    (store / "manifest.json").write_text(json.dumps({"tokenizer": {"files": files}}))
    return vocab


def test_the_mac_script_and_the_check_end_to_end(tmp_path: Path) -> None:
    """The Mac script tokenises the sets with the store's tokenizer; the check reads its output."""
    words = [f"w{i}" for i in range(200)]
    store = tmp_path / "store"
    store.mkdir()
    vocab = fixture_tokenizer(store, words)
    answer = " ".join(words[100:120])
    ids = np.array([vocab[w] for w in answer.split()], "<u2")
    write_train(store, [[rand_doc(1, 30), np.concatenate([np.full(5, 17, "<u2"), ids])]])
    convs = tmp_path / "conversations.jsonl"
    msgs = [
        {"role": "user", "content": " ".join(words[:15])},
        {"role": "assistant", "content": answer},
    ]
    rows = [{"id": "oasst2:1", "set": "train", "messages": msgs}]
    rows.append({"id": "oasst2:2", "set": "train", "messages": msgs[:1]})
    convs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    task = tmp_path / "arc_easy.jsonl"
    reqs = [{"context": " ".join(words[100:110]), "continuation": " " + " ".join(words[110:120])}]
    reqs.insert(0, {"context": reqs[0]["context"], "continuation": " w1 w2 w3"})
    lines = [{"task": "arc_easy", "doc_id": 5, "requests": reqs, "gold": 1}]
    lines.append({"task": "arc_easy", "doc_id": 6, "requests": reqs, "gold": 0})
    task.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    items = tmp_path / "items"
    cmd = [sys.executable, "-I", "-B", str(SCRIPT.resolve()), "--store", str(store)]
    cmd += ["--conversations", str(convs), "--tasks", str(task), "--out", str(items)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    sft = read_items(items / "sft.items.jsonl", V, FP)
    assert sft.ids == ["oasst2:1", "oasst2:2"]
    assert sft.segments[0][1].tolist() == ids.tolist()
    out = tmp_path / "out"
    argv = ["--store", str(store), "--out", str(out), "--items"]
    assert (
        decontam.main([*argv, str(items / "sft.items.jsonl"), str(items / "arc_easy.items.jsonl")])
        == 0
    )
    sft_rec = json.loads((out / "sft.decontam.json").read_text())
    assert sft_rec["drop"] == ["oasst2:1"]
    assert sft_rec["conversations_sha256"] == hashlib.sha256(convs.read_bytes()).hexdigest()
    arc_rec = json.loads((out / "arc_easy.decontam.json").read_text())
    assert arc_rec["drop"] == [5]
    assert arc_rec["task_sha256"] == hashlib.sha256(task.read_bytes()).hexdigest()


def test_a_line_separator_inside_a_message_does_not_split_the_line(tmp_path: Path) -> None:
    """A raw U+2028 inside content stays in its JSON string; only "\\n" ends a line."""
    store = tmp_path / "store"
    store.mkdir()
    fixture_tokenizer(store, ["w0", "w1"])
    write_train(store, [[rand_doc(1, 30)]])
    msgs = [{"role": "user", "content": "w0\u2028w1 w0"}, {"role": "assistant", "content": "w1"}]
    rows = [{"id": f"wc:{i}", "set": "train", "messages": msgs} for i in range(2)]
    convs = tmp_path / "conversations.jsonl"
    convs.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    assert "\u2028" in convs.read_text(encoding="utf-8")
    items = tmp_path / "items"
    cmd = [sys.executable, "-I", "-B", str(SCRIPT.resolve()), "--store", str(store)]
    cmd += ["--conversations", str(convs), "--out", str(items)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    assert read_items(items / "sft.items.jsonl", V, FP).ids == ["wc:0", "wc:1"]
