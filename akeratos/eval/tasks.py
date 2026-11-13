"""The multiple-choice eval sets: task table, rendering, set files, token split and metrics.

This module is the one copy of the code that Akeratos's scorer (`akeratos/eval/mc.py`) and
the peer baselines (which run outside `akeratos/`) share, so both score byte-identical strings
with the same split and the same metrics. It imports only the standard library and NumPy: no
MLX, no model, no tokenizer library.

The five tasks are HellaSwag, ARC-Easy, PIQA, LAMBADA (OpenAI's version) and Winogrande, each
zero-shot, in lm-eval v0.4.13's form. SciQ is left out because its CC-BY-NC licence is not
permissive. A set is rendered once, when it is prepared, so no template code runs at score time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from akeratos.errors import DataMismatchError

MANIFEST_SCHEMA = "metalforge.eval.mc-set/v1"
DECONTAM_SCHEMA = "metalforge.eval-decontam/v1"
DELIM = " "  # lm-eval's default target_delimiter
REV = re.compile(r"[0-9a-f]{40}")


class Request(TypedDict):
    """One scored string pair."""

    context: str
    continuation: str


class Item(TypedDict):
    """One task item: its requests and the index of the gold one.

    `doc_id` is the row's index in the pinned source file, as a string. The decontamination scan
    reads an item's text as its first request's context plus every continuation.
    """

    task: str
    doc_id: str
    requests: list[Request]
    gold: int


@dataclass(frozen=True)
class TaskSpec:
    """One task's source and scoring.

    `files` are regular expressions over repository paths, tried in order at the pinned revision:
    the first that matches any file must match exactly one. `licence` is the card's expected tag;
    `n` the published item count (a check prints a difference, it does not stop on one).
    """

    name: str
    repo: str
    config: str | None
    split: str
    kind: str  # "mc", "partial" (Winogrande) or "cloze" (LAMBADA)
    files: tuple[str, ...]
    licence: str
    n: int
    headline: str

    def metrics(self) -> tuple[str, ...]:
        """The metrics `aggregate` reports for this task."""
        return {"mc": ("acc", "acc_norm", "acc_bytes"), "partial": ("acc",)}.get(
            self.kind, ("acc", "perplexity")
        )


# Repositories, configs and splits are lm-eval v0.4.13's. The licence tags and file layouts here
# are unverified ❓; `scripts/data/prepare_evals.py` reads both at the pinned revision and stops
# on a difference. Counts are the published split sizes ❓.
TASKS: dict[str, TaskSpec] = {
    t.name: t
    for t in (
        TaskSpec(
            "hellaswag",
            "Rowan/hellaswag",
            None,
            "validation",
            "mc",
            (
                r"data/validation-[0-9]{5}-of-[0-9]{5}\.parquet",
                r"default/validation/[0-9]{4}\.parquet",
            ),
            "mit",
            10_042,
            "acc_norm",
        ),
        TaskSpec(
            "arc_easy",
            "allenai/ai2_arc",
            "ARC-Easy",
            "test",
            "mc",
            (r"ARC-Easy/test-[0-9]{5}-of-[0-9]{5}\.parquet",),
            "cc-by-sa-4.0",
            2_376,
            "acc",
        ),
        TaskSpec(
            "piqa",
            "baber/piqa",
            None,
            "validation",
            "mc",
            (
                r"piqa_validation\.parquet",  # the root layout of the repository's main branch
                r"data/validation-[0-9]{5}-of-[0-9]{5}\.parquet",
                r"default/validation/[0-9]{4}\.parquet",
            ),
            "afl-3.0",
            1_838,
            "acc",
        ),
        TaskSpec(
            "lambada_openai",
            "EleutherAI/lambada_openai",
            "default",
            "test",
            "cloze",
            (
                r"default/test-[0-9]{5}-of-[0-9]{5}\.parquet",
                r"default/test/[0-9]{4}\.parquet",
                r"default/test/default\.parquet",
                r"data/lambada_test\.jsonl",
            ),
            "mit",
            5_153,
            "acc",
        ),
        TaskSpec(
            "winogrande",
            "allenai/winogrande",
            "winogrande_xl",
            "validation",
            "partial",
            (r"winogrande_xl/validation-[0-9]{5}-of-[0-9]{5}\.parquet",),
            "cc-by",
            1_267,
            "acc",
        ),
    )
}


def _hellaswag_text(text: str) -> str:
    """lm-eval's HellaSwag `preprocess`."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def _str(row: Mapping[str, Any], key: str) -> str:
    v = row.get(key)
    if not isinstance(v, str):
        raise DataMismatchError(f"field {key!r} is {type(v).__name__}, not a string")
    return v


def _seq(v: object, what: str) -> list[object]:
    if not isinstance(v, list | tuple):
        raise DataMismatchError(f"{what} is not a list")
    return list(cast("Sequence[object]", v))


def _strs(v: object, what: str) -> list[str]:
    seq = _seq(v, what)
    out = [s for s in seq if isinstance(s, str)]
    if len(out) != len(seq):
        raise DataMismatchError(f"{what} is not a list of strings")
    return out


def _map(v: object, what: str) -> dict[str, object]:
    if not isinstance(v, Mapping):
        raise DataMismatchError(f"{what} is not a mapping")
    return dict(cast("Mapping[str, object]", v))


def _mc(context: str, choices: Sequence[str]) -> list[Request]:
    return [{"context": context, "continuation": DELIM + c} for c in choices]


def render(task: str, row: Mapping[str, Any], doc_id: int) -> Item:
    """One source row as an item, in lm-eval v0.4.13's zero-shot form.

    - Multiple choice: one request per choice, context = doc_to_text, continuation = `DELIM` +
      choice.
    - Winogrande (lm-eval's partial scoring): the context is the sentence up to the blank plus the
      option, and the continuation, " " + the rest of the sentence, is shared.
    - LAMBADA: one request, the text up to its last space and " " + its last word; no delimiter is
      added to a loglikelihood target.
    """
    spec = TASKS[task]
    if task == "hellaswag":
        ctx = _str(row, "ctx_a") + " " + _str(row, "ctx_b").capitalize()
        context = _hellaswag_text(_str(row, "activity_label") + ": " + ctx)
        choices = [_hellaswag_text(e) for e in _strs(row.get("endings"), "endings")]
        requests, gold = _mc(context, choices), _label(row.get("label"))
    elif task == "arc_easy":
        ch = _map(row.get("choices"), "field 'choices'")
        texts, labels = (
            _strs(ch.get("text"), "choices.text"),
            _strs(ch.get("label"), "choices.label"),
        )
        key = _str(row, "answerKey")
        if len(texts) != len(labels) or key not in labels:
            raise DataMismatchError(f"answerKey {key!r} is not one of the labels {labels}")
        requests = _mc("Question: " + _str(row, "question") + "\nAnswer:", texts)
        gold = labels.index(key)
    elif task == "piqa":
        requests = _mc(
            "Question: " + _str(row, "goal") + "\nAnswer:", [_str(row, "sol1"), _str(row, "sol2")]
        )
        gold = _label(row.get("label"))
    elif task == "winogrande":
        s = _str(row, "sentence")
        if "_" not in s:
            raise DataMismatchError("the sentence has no blank")
        i = s.index("_")
        rest = DELIM + s[i + 1 :].strip()
        options = [_str(row, "option1"), _str(row, "option2")]
        requests = [{"context": s[:i] + o, "continuation": rest} for o in options]
        answer = _str(row, "answer")
        if answer not in ("1", "2"):
            raise DataMismatchError(f"answer {answer!r} is not '1' or '2'")
        gold = int(answer) - 1
    elif task == "lambada_openai":
        words = _str(row, "text").split(" ")
        requests = [{"context": " ".join(words[:-1]), "continuation": " " + words[-1]}]
        gold = 0
    else:
        raise KeyError(task)
    return check_item(
        {"task": spec.name, "doc_id": str(doc_id), "requests": requests, "gold": gold}
    )


def _label(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int | str):
        raise DataMismatchError(f"label {v!r} is not an integer")
    try:
        return int(v)
    except ValueError:
        raise DataMismatchError(f"label {v!r} is not an integer") from None


def check_item(item: object) -> Item:
    """An item's shape, returned typed: fields, strings, a gold index in range, no empty request."""
    m = _map(item, "an item")
    task, doc_id, gold = m.get("task"), m.get("doc_id"), m.get("gold")
    if not isinstance(task, str) or task not in TASKS or set(m) != set(Item.__annotations__):
        raise DataMismatchError(f"not an item of a known task: keys {sorted(m)}")
    reqs = _seq(m["requests"], f"{task} item requests")
    if not isinstance(doc_id, str) or not isinstance(gold, int) or isinstance(gold, bool):
        raise DataMismatchError(f"{task} item {doc_id!r}: bad doc_id or gold")
    if not 0 <= gold < len(reqs):
        raise DataMismatchError(f"{task} item {doc_id}: gold {gold} for {len(reqs)} requests")
    cloze = TASKS[task].kind == "cloze"
    if cloze != (len(reqs) == 1):
        raise DataMismatchError(f"{task} item {doc_id}: {len(reqs)} requests")
    out: list[Request] = []
    for r in reqs:
        q = _map(r, f"{task} item {doc_id}: a request")
        c, k = q.get("context"), q.get("continuation")
        if (
            set(q) != {"context", "continuation"}
            or not isinstance(c, str)
            or not isinstance(k, str)
        ):
            raise DataMismatchError(
                f"{task} item {doc_id}: a request is not context and continuation"
            )
        if not c or not k or (not cloze and not k.startswith(DELIM)):
            raise DataMismatchError(f"{task} item {doc_id}: an empty request, or no delimiter")
        out.append({"context": c, "continuation": k})
    return {"task": task, "doc_id": doc_id, "requests": out, "gold": gold}


def item_line(item: Item) -> bytes:
    """An item's JSONL line: sorted keys, UTF-8, a trailing newline."""
    return (json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n").encode()


def set_files(
    sources: Mapping[str, Mapping[str, object]], items: Mapping[str, Sequence[Item]], made_by: str
) -> tuple[str, dict[str, bytes]]:
    """(set_id, {file name: bytes}): a set's files, for the prep script to write.

    The set is a directory `<root>/<set_id>/` of `manifest.json` (schema `MANIFEST_SCHEMA`), one
    `<task>.jsonl` per task with one item per line in the source file's row order, and
    `SHA256SUMS` with every other file's SHA-256.
    `sources[task]` is the task's manifest row before the counts and hashes this adds: repo,
    revision, config, split, source, licence_tag, licence_override. The set id is the first 16 hex
    of the SHA-256 of the canonical tasks block, so the same pinned inputs give the same folder.
    `made_by` (the script and its commit) is recorded but not hashed.
    """
    files: dict[str, bytes] = {}
    rows: dict[str, object] = {}
    for t in sorted(items):
        body = b"".join(item_line(check_item(i)) for i in items[t])
        files[f"{t}.jsonl"] = body
        spec = TASKS[t]
        rows[t] = dict(sources[t]) | {
            "kind": spec.kind,
            "metrics": list(spec.metrics()),
            "headline": spec.headline,
            "licence_expected": spec.licence,
            "n": len(items[t]),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    canon = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    set_id = hashlib.sha256(canon).hexdigest()[:16]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "set_id": set_id,
        "templates": "lm-eval v0.4.13 zero-shot",
        "made_by": made_by,
        "tasks": rows,
    }
    files["manifest.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in sorted(files.items()))
    files["SHA256SUMS"] = sums.encode()
    return set_id, files


def read_set(
    path: str | Path, tasks: Iterable[str] | None = None
) -> tuple[dict[str, object], dict[str, list[Item]]]:
    """The manifest and the items of `tasks` (default: every task in the set), checked.

    Each task file's SHA-256 and item count must equal the manifest's, and each line must be an
    item of that task with a doc_id unique in the file.
    """
    root = Path(path)
    try:
        m = _map(json.loads((root / "manifest.json").read_text()), f"{root}/manifest.json")
    except (OSError, ValueError) as e:
        raise DataMismatchError(f"{root}: no readable manifest.json ({e})") from None
    if m.get("schema") != MANIFEST_SCHEMA:
        raise DataMismatchError(
            f"{root}: manifest schema {m.get('schema')!r}, not {MANIFEST_SCHEMA}"
        )
    rows = _map(m.get("tasks"), f"{root}: the manifest's tasks")
    out: dict[str, list[Item]] = {}
    for t in list(rows) if tasks is None else list(tasks):
        if t not in TASKS or t not in rows:
            raise DataMismatchError(f"{root}: no manifest row for task {t!r}")
        row = _map(rows[t], f"{root}: the manifest row for {t!r}")
        if not REV.fullmatch(str(row.get("revision"))):
            raise DataMismatchError(f"{root}: task {t!r} has no 40-hex revision")
        try:
            raw = (root / f"{t}.jsonl").read_bytes()
        except OSError as e:
            raise DataMismatchError(f"{root}/{t}.jsonl: {e}") from None
        if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            raise DataMismatchError(f"{root}/{t}.jsonl: SHA-256 differs from the manifest's")
        items = [check_item(json.loads(line)) for line in raw.decode().splitlines()]
        if any(i["task"] != t for i in items):
            raise DataMismatchError(f"{root}/{t}.jsonl: an item of another task")
        if len(items) != row.get("n") or len({i["doc_id"] for i in items}) != len(items):
            msg = f"{root}/{t}.jsonl: {len(items)} items or repeated doc_ids, manifest n {row.get('n')}"
            raise DataMismatchError(msg)
        out[t] = items
    return m, out


def _check_report(f: Path, report: object, tokenizer_sha256: str, fingerprint: str | None) -> None:
    """A drop record's report: a scan of the whole store, at this tokenizer and this store."""
    r = _map(report, f"{f}: report")
    if r.get("complete") is not True:
        raise DataMismatchError(f"{f}: report.complete is not true (a sample scan?)")
    if r.get("tokenizer_sha256") != tokenizer_sha256:
        raise DataMismatchError(f"{f}: made with tokenizer {r.get('tokenizer_sha256')!r}")
    store = _map(r.get("store"), f"{f}: report.store")
    if store.get("fingerprint") != fingerprint:
        raise DataMismatchError(f"{f}: made against store {store.get('fingerprint')!r}")


def read_decontam(
    path: str | Path,
    manifest: Mapping[str, object],
    items: Mapping[str, Sequence[Item]],
    tokenizer_sha256: str,
    fingerprint: str | None,
) -> tuple[dict[str, list[Item]], dict[str, object]]:
    """(the items left, a record): `items` without the doc_ids each `<task>.decontam.json` drops.

    `manifest` is `read_set`'s; `tokenizer_sha256` and `fingerprint` are the scored store's (its
    tokenizer file's SHA-256 and `fingerprint.json`'s value, None when it has none). Every task in
    `items` needs a file under `path`. The record has, per task, the file's SHA-256 and the dropped
    and kept counts.

    A file is `{"schema": DECONTAM_SCHEMA, "task_sha256", "drop": [doc_id, ...], "report"}`. The
    files sit in a directory of their own, never the set's, whose `SHA256SUMS` lists every file.
    Each must name the scored task file's SHA-256 and report a complete scan made with this
    tokenizer and store; other files in `path` are never read.
    """
    root = Path(path)
    rows = _map(manifest.get("tasks"), "the manifest's tasks")
    kept: dict[str, list[Item]] = {}
    rec: dict[str, object] = {"dir": str(root)}
    for t, its in items.items():
        f = root / f"{t}.decontam.json"
        try:
            raw = f.read_bytes()
            d = _map(json.loads(raw), str(f))
        except (OSError, ValueError) as e:
            raise DataMismatchError(f"{f}: not readable ({e})") from None
        if d.get("schema") != DECONTAM_SCHEMA:
            raise DataMismatchError(f"{f}: schema {d.get('schema')!r}, not {DECONTAM_SCHEMA}")
        if d.get("task_sha256") != _map(rows.get(t), f"the manifest row for {t!r}").get("sha256"):
            raise DataMismatchError(f"{f}: task_sha256 is not the scored {t}.jsonl's")
        _check_report(f, d.get("report"), tokenizer_sha256, fingerprint)
        drop = set(_strs(d.get("drop"), f"{f}: drop"))
        if len(drop) != len(_seq(d.get("drop"), f"{f}: drop")) or not drop <= {
            i["doc_id"] for i in its
        }:
            raise DataMismatchError(f"{f}: drop repeats a doc_id or names one not in {t}")
        kept[t] = [i for i in its if i["doc_id"] not in drop]
        rec[t] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dropped": len(drop),
            "kept": len(kept[t]),
        }
    return kept, rec


def encode_pair(
    encode: Callable[[str], Sequence[int]],
    context: str,
    continuation: str,
    prefix: Sequence[int] = (),
    max_len: int | None = None,
) -> tuple[list[int], int]:
    """(ids, n): the scored sequence and how many of its last tokens are the continuation.

    Inputs are ids[:-1] and targets ids[1:]; the continuation is the last n targets. The split is
    lm-eval's `TemplateLM._encode_pair` at v0.4.13: trailing whitespace moves from the context
    into the continuation, and the continuation's tokens are those that encoding the whole adds to
    the context's. Akeratos passes the prefix [0] on every request, because every training
    document opens with id 0; an HF baseline passes [] or lm-eval's prefix token for an empty
    context. The sequence is cut from the left to its last max_len + 1 tokens, as lm-eval's
    `_loglikelihood_tokens` does.
    """
    if not context and not prefix:  # lm-eval asserts a non-empty context
        raise ValueError("an empty context needs a prefix token")
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    ctx = list(encode(context)) if context else []
    whole = list(encode(context + continuation))
    cont = whole[len(ctx) :]
    if not cont:
        raise ValueError(f"the continuation {continuation!r} encodes to no tokens")
    ids = list(prefix) + ctx + cont
    if max_len is not None:
        if len(cont) > max_len:
            raise ValueError(f"a continuation of {len(cont)} tokens exceeds max_len {max_len}")
        ids = ids[-(max_len + 1) :]
    return ids, len(cont)


def _stderr(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def aggregate(
    task: str, items: Sequence[Item], results: Sequence[Sequence[tuple[float, bool]]]
) -> dict[str, Any]:
    """The task's metrics over items and their per-request (Σ log p, greedy) results.

    The metrics are lm-eval's `process_results`. acc is the argmax of Σ log p over the requests;
    acc_norm divides Σ log p by len(choice) in characters, choice being the continuation without
    its delimiter, and acc_bytes by len(choice.encode()). Winogrande reports acc only; LAMBADA
    reports acc (every continuation token greedy) and perplexity = exp(-mean Σ log p). Ties go to
    the first choice (`np.argmax`). Each score carries its standard error: lm-eval's `mean_stderr`
    (sample standard deviation, ddof 1, over √n) for the accuracies, and for perplexity the delta
    method, ppl · stderr(Σ log p). lm-eval bootstraps perplexity's standard error, so that one
    value differs from its output.

    Returns {"n", metric: value, metric + "_stderr": value, "headline": name}.
    """
    spec = TASKS[task]
    if len(items) != len(results) or not items:
        raise ValueError(f"{task}: {len(items)} items, {len(results)} results")
    rec: dict[str, Any] = {"n": len(items), "headline": spec.headline}
    if spec.kind == "cloze":
        ll = np.array([r[0][0] for r in results], np.float64)
        acc = np.array([float(r[0][1]) for r in results])
        ppl = math.exp(-float(ll.mean()))
        rec |= {"acc": float(acc.mean()), "acc_stderr": _stderr(acc)}
        rec |= {"perplexity": ppl, "perplexity_stderr": ppl * _stderr(ll)}
        return rec
    cols: dict[str, list[float]] = {m: [] for m in spec.metrics()}
    for it, res in zip(items, results, strict=True):
        if len(res) != len(it["requests"]):
            raise ValueError(
                f"{task} item {it['doc_id']}: {len(res)} results for {len(it['requests'])} requests"
            )
        ll = np.array([r[0] for r in res], np.float64)
        cols["acc"].append(float(int(np.argmax(ll)) == it["gold"]))
        if spec.kind == "mc":
            choices = [q["continuation"][len(DELIM) :] for q in it["requests"]]
            chars = np.array([float(len(c)) for c in choices])
            nbytes = np.array([float(len(c.encode())) for c in choices])
            cols["acc_norm"].append(float(int(np.argmax(ll / chars)) == it["gold"]))
            cols["acc_bytes"].append(float(int(np.argmax(ll / nbytes)) == it["gold"]))
    for m, v in cols.items():
        a = np.array(v)
        rec |= {m: float(a.mean()), m + "_stderr": _stderr(a)}
    return rec
