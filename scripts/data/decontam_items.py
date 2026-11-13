"""Tokenise the eval and SFT sets for decontamination: one items file per set.

It reads only local files and never downloads, and runs in the environment `requirements.txt` beside it
locks. The tokenizer is the store's own copy, `BPE.from_store`, checked against the store's manifest, so the
ids are the ones the shards hold. `akeratos/data/decontam.py` then matches the items against the store's
`train/` split and holds the matching rule.

Each set is written to `<out>/<set>.items.jsonl`: an `ITEMS_SCHEMA` header line, then one
`{"id", "segments"}` line per item, in the input's order, each segment a list of ids. The header binds the
input file's SHA-256, which the drop record carries and the post-training preparation and the scorer
check. The SFT set is named `sft`; an eval set is named after its task.

It needs the standard library and, through `bpe.py`, NumPy and `tokenizers`.
"""

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # under `python -I` the script's own directory is not on sys.path

ITEMS_SCHEMA = "metalforge.decontam-items/v1"
SET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class InputError(ValueError):
    pass


def read_jsonl(path):
    """The file's SHA-256 and its lines as objects."""
    with open(path, "rb") as f:
        data = f.read()
    # Split on "\n" alone: str.splitlines also breaks at U+2028 and other separators that a
    # JSON string may hold unescaped (WildChat prompts carry U+2028).
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise InputError(f"{path}: empty, or a line that is not a JSON object")
    return hashlib.sha256(data).hexdigest(), rows


def sft_items(rows, where):
    """(id, texts) per line of a `conversations.jsonl`: one text per message's `content`.

    Each text is encoded alone, so no n-gram spans two turns. Role headers and ChatML ids are not text and
    are left out.
    """
    out = []
    for i, r in enumerate(rows, start=1):
        msgs = r.get("messages")
        if not isinstance(r.get("id"), str) or not isinstance(msgs, list) or not msgs:
            raise InputError(f"{where}:{i}: a conversation needs a string id and messages")
        if not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in msgs):
            raise InputError(f"{where}:{i}: a message without string content")
        out.append((r["id"], [m["content"] for m in msgs]))
    return out


def task_items(rows, where):
    """The task's name and (doc_id, [gold context + continuation]) per line of one task's file.

    A line is `{"task", "doc_id", "requests": [{"context", "continuation"}, ...], "gold"}`. The gold request's
    context and continuation are encoded as one string: the gold answer in its context, as the scorer sees it.
    """
    names = {r.get("task") for r in rows}
    if len(names) != 1 or not SET_NAME.fullmatch(str(next(iter(names)))):
        raise InputError(f"{where}: every line must name the same task, a file-name-safe name")
    out = []
    for i, r in enumerate(rows, start=1):
        reqs, gold = r.get("requests"), r.get("gold")
        doc = r.get("doc_id")
        if not isinstance(doc, (str, int)) or isinstance(doc, bool):
            raise InputError(f"{where}:{i}: doc_id is not a string or an integer")
        if not isinstance(reqs, list) or not isinstance(gold, int) or not 0 <= gold < len(reqs):
            raise InputError(f"{where}:{i}: gold does not index requests")
        req = reqs[gold]
        if not isinstance(req, dict) or not all(isinstance(req.get(k), str) for k in ("context", "continuation")):
            raise InputError(f"{where}:{i}: the gold request needs string context and continuation")
        out.append((doc, [req["context"] + req["continuation"]]))
    return str(next(iter(names))), out


def write_items(out, name, kind, source, sha, tok, fingerprint, items):
    """Encode every text and write `<out>/<name>.items.jsonl`; returns its path."""
    texts = [t for _, ts in items for t in ts]
    ids = iter(tok.encode(texts))
    lines = [json.dumps({"schema": ITEMS_SCHEMA, "set": name, "kind": kind, "source": os.path.basename(source),
                         "source_sha256": sha, "tokenizer_sha256": tok.sha256, "store_fingerprint": fingerprint,
                         "items": len(items)}, sort_keys=True)]
    for item_id, ts in items:
        lines.append(json.dumps({"id": item_id, "segments": [next(ids).tolist() for _ in ts]}))
    if len({i for i, _ in items}) != len(items):
        raise InputError(f"{source}: repeated item ids")
    path = os.path.join(out, f"{name}.items.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--store", required=True, help="the data store whose tokenizer and fingerprint to use")
    p.add_argument("--conversations", help="the SFT set's conversations.jsonl")
    p.add_argument("--tasks", nargs="*", default=[], help="the eval sets' task files, one per task")
    p.add_argument("--out", required=True, help="the folder for the items files")
    a = p.parse_args(argv)
    from bpe import BPE, TokenizerError

    try:
        tok = BPE.from_store(a.store)
        with open(os.path.join(a.store, "fingerprint.json")) as f:
            fingerprint = json.load(f)["fingerprint"]
        os.makedirs(a.out, exist_ok=True)
        jobs = []
        if a.conversations:
            sha, rows = read_jsonl(a.conversations)
            jobs.append(("sft", "sft", a.conversations, sha, sft_items(rows, a.conversations)))
        for path in a.tasks:
            sha, rows = read_jsonl(path)
            name, items = task_items(rows, path)
            jobs.append((name, "eval", path, sha, items))
        if not jobs or len({j[0] for j in jobs}) != len(jobs):
            raise InputError("no sets, or two inputs with the same set name")
        for name, kind, source, sha, items in jobs:
            path = write_items(a.out, name, kind, source, sha, tok, fingerprint, items)
            print(f"{name}: {len(items)} items -> {path}")
    except (InputError, TokenizerError, OSError, KeyError, ValueError) as e:
        sys.stderr.write(f"decontam_items: {e}\n")
        return 65
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
