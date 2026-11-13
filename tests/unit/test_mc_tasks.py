"""The shared multiple-choice task module: rendering, set files, split, metrics.

The expected strings are written by hand from lm-eval v0.4.13's zero-shot templates (the YAMLs and
`hellaswag/utils.py`, `winogrande/preprocess_winogrande.py` at the tag), and the expected metrics
from its `process_results`, so a drift in `akeratos/eval/tasks.py` shows here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from akeratos.errors import DataMismatchError
from akeratos.eval import tasks
from akeratos.eval.tasks import TASKS, Item, aggregate, encode_pair, read_set, render, set_files

ROWS: dict[str, dict[str, Any]] = {
    "hellaswag": {
        "activity_label": "Removing ice from car",
        "ctx_a": "Then, the man writes over the snow covering the window of a car.",
        "ctx_b": "a woman wearing winter clothes smiles.",
        "endings": [
            " then, the man adds wax to the windshield.",
            "Wipe it off [title] Done [step] now.",
        ],
        "label": "1",
    },
    "arc_easy": {
        "question": "Which is a gas?",
        "choices": {"text": ["ice", "steam", "rock"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    },
    "piqa": {"goal": "Open a jar", "sol1": "Twist the lid.", "sol2": "Freeze it.", "label": 0},
    "winogrande": {
        "sentence": "The cup fell off the table because the _ was wobbly.",
        "option1": "cup",
        "option2": "table",
        "answer": "2",
    },
    "lambada_openai": {"text": "he opened the door and saw his dog"},
}

WANT: dict[str, Item] = {
    "hellaswag": {
        "task": "hellaswag",
        "doc_id": "7",
        "requests": [
            {
                "context": "Removing ice from car: Then, the man writes over the snow covering the "
                "window of a car. A woman wearing winter clothes smiles.",
                "continuation": " then, the man adds wax to the windshield.",
            },
            {
                "context": "Removing ice from car: Then, the man writes over the snow covering the "
                "window of a car. A woman wearing winter clothes smiles.",
                "continuation": " Wipe it off. Done now.",
            },
        ],
        "gold": 1,
    },
    "arc_easy": {
        "task": "arc_easy",
        "doc_id": "7",
        "requests": [
            {"context": "Question: Which is a gas?\nAnswer:", "continuation": " ice"},
            {"context": "Question: Which is a gas?\nAnswer:", "continuation": " steam"},
            {"context": "Question: Which is a gas?\nAnswer:", "continuation": " rock"},
        ],
        "gold": 1,
    },
    "piqa": {
        "task": "piqa",
        "doc_id": "7",
        "requests": [
            {"context": "Question: Open a jar\nAnswer:", "continuation": " Twist the lid."},
            {"context": "Question: Open a jar\nAnswer:", "continuation": " Freeze it."},
        ],
        "gold": 0,
    },
    "winogrande": {
        "task": "winogrande",
        "doc_id": "7",
        "requests": [
            {
                "context": "The cup fell off the table because the cup",
                "continuation": " was wobbly.",
            },
            {
                "context": "The cup fell off the table because the table",
                "continuation": " was wobbly.",
            },
        ],
        "gold": 1,
    },
    "lambada_openai": {
        "task": "lambada_openai",
        "doc_id": "7",
        "requests": [{"context": "he opened the door and saw his", "continuation": " dog"}],
        "gold": 0,
    },
}


@pytest.mark.parametrize("task", sorted(ROWS))
def test_render_matches_lm_eval_templates(task: str) -> None:
    assert render(task, ROWS[task], 7) == WANT[task]


def test_the_task_table_is_a82s_five() -> None:
    assert sorted(TASKS) == ["arc_easy", "hellaswag", "lambada_openai", "piqa", "winogrande"]
    assert "sciq" not in TASKS  # CC-BY-NC: non-commercial data is out
    assert TASKS["winogrande"].metrics() == ("acc",)
    assert TASKS["lambada_openai"].metrics() == ("acc", "perplexity")
    assert TASKS["hellaswag"].metrics() == ("acc", "acc_norm", "acc_bytes")
    assert {t: s.headline for t, s in TASKS.items()} == {  # each task's headline metric
        "arc_easy": "acc",
        "hellaswag": "acc_norm",
        "lambada_openai": "acc",
        "piqa": "acc",
        "winogrande": "acc",
    }
    for spec in TASKS.values():
        assert spec.headline in spec.metrics()
        for f in spec.files:
            re.compile(f)


@pytest.mark.parametrize(
    ("task", "change"),
    [
        ("hellaswag", {"endings": "not a list"}),
        ("hellaswag", {"label": "x"}),
        ("hellaswag", {"label": "4"}),
        ("arc_easy", {"answerKey": "D"}),
        ("arc_easy", {"choices": {"text": ["a"], "label": ["A", "B"]}}),
        ("piqa", {"sol2": None}),
        ("winogrande", {"sentence": "no blank here"}),
        ("winogrande", {"answer": "3"}),
        ("lambada_openai", {"text": "oneword"}),
    ],
)
def test_render_refuses_bad_rows(task: str, change: dict[str, Any]) -> None:
    with pytest.raises(DataMismatchError):
        render(task, ROWS[task] | change, 0)


def test_check_item_refuses_bad_shapes() -> None:
    good = WANT["arc_easy"]
    assert tasks.check_item(json.loads(json.dumps(good))) == good
    bad: list[object] = [
        good | {"task": "sciq"},
        good | {"gold": 3},
        good | {"gold": True},
        good | {"doc_id": 7},
        good | {"extra": 1},
        good | {"requests": [{"context": "q", "continuation": "no delimiter"}]},
        good | {"requests": [{"context": "", "continuation": " a"}]},
        WANT["lambada_openai"] | {"requests": WANT["lambada_openai"]["requests"] * 2},
        [good],
    ]
    for b in bad:
        with pytest.raises(DataMismatchError):
            tasks.check_item(b)


def words(s: str) -> list[int]:
    """A merge-sensitive toy encoder: each ' word' or run of spaces is one token."""
    return [sum(map(ord, w)) % 997 + 17 for w in re.findall(r" ?\S+| +", s)]


def test_encode_pair_moves_trailing_spaces_and_splits_the_whole() -> None:
    ids, n = encode_pair(words, "the cat ", "sat", prefix=[0])
    assert ids == [0, *words("the cat sat")]
    assert n == 1
    assert ids[-1] == words(" sat")[0]
    ids, n = encode_pair(words, "Answer:", " Twist the lid.")
    assert ids == words("Answer: Twist the lid.")
    assert n == 3
    ids, n = encode_pair(lambda s: [ord(c) for c in s], "ab", "cd", prefix=[])  # code points as ids
    assert (ids, n) == ([ord(c) for c in "abcd"], 2)


def test_encode_pair_cuts_from_the_left_and_keeps_the_continuation() -> None:
    ctx = " ".join(f"w{i}" for i in range(50))
    full, n = encode_pair(words, ctx, " end", prefix=[0])
    cut, m = encode_pair(words, ctx, " end", prefix=[0], max_len=10)
    assert m == n == 1
    assert cut == full[-11:]
    assert len(cut) == 11
    short, _ = encode_pair(words, "a", " b", prefix=[0], max_len=10)
    assert short == [0, *words("a b")]
    with pytest.raises(ValueError, match="exceeds"):
        encode_pair(words, "a", " b c d", max_len=2)
    with pytest.raises(ValueError, match="prefix"):
        encode_pair(words, "", " b")
    assert encode_pair(words, "", " b", prefix=[0]) == ([0, *words(" b")], 1)


def mc_item(conts: list[str], gold: int, task: str = "hellaswag") -> Item:
    return {
        "task": task,
        "doc_id": "0",
        "requests": [{"context": "q", "continuation": c} for c in conts],
        "gold": gold,
    }


def test_aggregate_follows_lm_eval_process_results() -> None:
    """acc on Σ log p; acc_norm on Σ log p / len(choice); acc_bytes on its UTF-8 bytes."""
    a = mc_item([" ab", " abcdefgh"], 1)  # norm flips the raw argmax to the long choice
    b = mc_item([" x", " y"], 0)  # a tie goes to the first choice
    c = mc_item([" é", " ee"], 0)  # 1 character but 2 bytes against 2 and 2
    res = [
        [(-2.0, False), (-3.0, False)],
        [(-1.0, True), (-1.0, True)],
        [(-1.5, False), (-2.0, False)],
    ]
    got = aggregate("hellaswag", [a, b, c], res)
    acc, norm, nbytes = np.array([0.0, 1, 1]), np.array([1.0, 1, 0]), np.array([1.0, 1, 1])
    assert got["n"] == 3
    assert got["headline"] == "acc_norm"
    for k, v in (("acc", acc), ("acc_norm", norm), ("acc_bytes", nbytes)):
        assert got[k] == pytest.approx(v.mean())
        sd = math.sqrt(sum((x - v.mean()) ** 2 for x in v) / (len(v) - 1))  # lm-eval mean_stderr
        assert got[k + "_stderr"] == pytest.approx(sd / math.sqrt(len(v)))
    w = aggregate(
        "winogrande", [mc_item([" x", " x"], 1, "winogrande")], [[(-2.0, False), (-1.0, False)]]
    )
    assert w["acc"] == 1.0
    assert "acc_norm" not in w
    assert math.isnan(w["acc_stderr"])


def test_aggregate_cloze_reports_greedy_acc_and_perplexity() -> None:
    items: list[Item] = [WANT["lambada_openai"]] * 3
    ll = [-1.0, -2.0, -4.0]
    got = aggregate("lambada_openai", items, [[(x, x > -1.5)] for x in ll])
    assert got["acc"] == pytest.approx(1 / 3)
    ppl = math.exp(-np.mean(ll))  # lm-eval's perplexity aggregation
    assert got["perplexity"] == pytest.approx(ppl)
    assert got["perplexity_stderr"] == pytest.approx(ppl * np.std(ll, ddof=1) / math.sqrt(3))
    with pytest.raises(ValueError, match="0 results"):
        aggregate("lambada_openai", items, [])


def sources() -> dict[str, dict[str, object]]:
    return {
        t: {
            "repo": s.repo,
            "revision": "ab" * 20,
            "config": s.config,
            "split": s.split,
            "source": {"path": "x.parquet", "size": 1, "sha256": "00" * 32},
            "licence_tag": s.licence,
            "licence_override": False,
        }
        for t, s in TASKS.items()
    }


def write(root: Path, files: dict[str, bytes]) -> None:
    root.mkdir(parents=True)
    for name, body in files.items():
        (root / name).write_bytes(body)


def test_a_set_round_trips_and_its_id_is_its_inputs(tmp_path: Path) -> None:
    items = {t: [WANT[t], WANT[t] | {"doc_id": "8"}] for t in TASKS}
    set_id, files = set_files(sources(), items, "test")
    again, _ = set_files(sources(), items, "another commit")
    assert again == set_id
    assert re.fullmatch(r"[0-9a-f]{16}", set_id)
    write(tmp_path / set_id, files)
    m, got = read_set(tmp_path / set_id)
    assert got == items
    assert m["set_id"] == set_id
    sums = (tmp_path / set_id / "SHA256SUMS").read_text().splitlines()
    for line in sums:
        h, name = line.split("  ")
        assert hashlib.sha256((tmp_path / set_id / name).read_bytes()).hexdigest() == h
    _, some = read_set(tmp_path / set_id, ["piqa"])
    assert list(some) == ["piqa"]
    changed = sources()
    changed["piqa"]["revision"] = "cd" * 20
    assert set_files(changed, items, "test")[0] != set_id


def test_read_set_refuses_a_changed_set(tmp_path: Path) -> None:
    items = {"piqa": [WANT["piqa"], WANT["piqa"] | {"doc_id": "8"}]}
    _, files = set_files(sources(), items, "test")
    cases: dict[str, dict[str, bytes]] = {
        "tampered": files | {"piqa.jsonl": files["piqa.jsonl"].replace(b"Freeze", b"Boil")},
        "missing": {k: v for k, v in files.items() if k != "piqa.jsonl"},
        "schema": files
        | {"manifest.json": files["manifest.json"].replace(b"mc-set/v1", b"mc-set/v0")},
    }
    dup_items = {"piqa": [WANT["piqa"], WANT["piqa"]]}
    cases["repeated"] = set_files(sources(), dup_items, "test")[1]
    for name, fs in cases.items():
        write(tmp_path / name, fs)
        with pytest.raises(DataMismatchError):
            read_set(tmp_path / name)
    write(tmp_path / "ok", files)
    with pytest.raises(DataMismatchError, match="no manifest row"):
        read_set(tmp_path / "ok", ["hellaswag"])


def test_decontam_files_drop_their_doc_ids_and_refuse_another_set(tmp_path: Path) -> None:
    items: dict[str, list[Item]] = {"piqa": [{**WANT["piqa"], "doc_id": str(i)} for i in range(3)]}
    set_id, files = set_files(sources(), items, "test")
    write(tmp_path / set_id, files)
    m, got = read_set(tmp_path / set_id)
    sha, tok, fp = hashlib.sha256(files["piqa.jsonl"]).hexdigest(), "ab" * 32, "9545f2d205efd574"
    report = {"complete": True, "tokenizer_sha256": tok, "store": {"fingerprint": fp}}

    def decontam(name: str, **over: object) -> Path:
        d: dict[str, object] = {"schema": tasks.DECONTAM_SCHEMA, "task_sha256": sha, "drop": ["1"]}
        (tmp_path / name).mkdir()
        (tmp_path / name / "piqa.decontam.json").write_text(
            json.dumps(d | {"report": report} | over)
        )
        return tmp_path / name

    ok = decontam("ok")
    (ok / "sft.decontam.json").write_text("not read")  # the SFT drop list shares the directory
    kept, rec = tasks.read_decontam(ok, m, got, tok, fp)
    assert [i["doc_id"] for i in kept["piqa"]] == ["0", "2"]
    body = (tmp_path / "ok" / "piqa.decontam.json").read_bytes()
    want = {"sha256": hashlib.sha256(body).hexdigest(), "dropped": 1, "kept": 2}
    assert rec == {"dir": str(tmp_path / "ok"), "piqa": want}
    bad: dict[str, tuple[dict[str, object], str]] = {
        "stale": ({"task_sha256": "0" * 64}, "task_sha256"),
        "schema": ({"schema": "metalforge.eval-decontam/v0"}, "schema"),
        "unknown": ({"drop": ["9"]}, "not in piqa"),
        "repeated": ({"drop": ["1", "1"]}, "repeats"),
        "not-strings": ({"drop": [1]}, "not a list of strings"),
        "sample": ({"report": report | {"complete": False}}, "complete"),
        "no-report": ({"report": None}, "report"),
        "tokenizer": ({"report": report | {"tokenizer_sha256": "cd" * 32}}, "tokenizer"),
        "store": ({"report": report | {"store": {"fingerprint": "0" * 16}}}, "store"),
    }
    for name, (over, why) in bad.items():
        with pytest.raises(DataMismatchError, match=why):
            tasks.read_decontam(decontam(name, **over), m, got, tok, fp)
    with pytest.raises(DataMismatchError, match="not readable"):
        tasks.read_decontam(tmp_path / "none", m, got, tok, fp)
