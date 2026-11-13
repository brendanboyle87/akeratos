"""The eval sets' Mac preparation script, against a fake Hub.

`scripts/data/prepare_evals.py` runs in a fresh `python -I -B` process
through `prep_evals_entry.py`, as it runs on the Mac, over crafted Parquet and JSONL files.
Nothing here reaches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset

from akeratos.eval import tasks
from akeratos.eval.tasks import TASKS

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "data" / "prepare_evals.py"
ENTRY = Path(__file__).with_name("prep_evals_entry.py")
REV = {t: hashlib.sha256(t.encode()).hexdigest()[:40] for t in TASKS}
MAIN = {t: hashlib.sha256((t + "main").encode()).hexdigest()[:40] for t in TASKS}
Change = Callable[[dict[str, Any]], None]

ROWS: dict[str, list[dict[str, Any]]] = {
    "hellaswag": [
        {
            "activity_label": "Cooking",
            "ctx_a": "A man cuts.",
            "ctx_b": "he then",
            "endings": ["a", "b", "c", "d"],
            "label": str(i % 4),
        }
        for i in range(5)
    ],
    "arc_easy": [
        {
            "question": f"Q{i}?",
            "choices": {"text": ["x", "y"], "label": ["A", "B"]},
            "answerKey": "B",
        }
        for i in range(4)
    ],
    "piqa": [{"goal": f"G{i}", "sol1": "s1", "sol2": "s2", "label": i % 2} for i in range(3)],
    "winogrande": [
        {"sentence": f"P{i} saw _ there.", "option1": "a", "option2": "b", "answer": "2"}
        for i in range(3)
    ],
    "lambada_openai": [{"text": f"the end of story {i}"} for i in range(6)],
}
FILES = {
    "hellaswag": "data/validation-00000-of-00001.parquet",
    "arc_easy": "ARC-Easy/test-00000-of-00001.parquet",
    "piqa": "piqa_validation.parquet",  # baber/piqa's live main keeps its files at the root
    "winogrande": "winogrande_xl/validation-00000-of-00001.parquet",
    "lambada_openai": "data/lambada_test.jsonl",
}


def git_blob(data: bytes) -> str:
    """git's blob id, which the Hub lists for a file outside LFS (not a security hash)."""
    head = b"blob " + str(len(data)).encode() + b"\0"
    return hashlib.sha1(head + data, usedforsecurity=False).hexdigest()


def entry(path: str, data: bytes, lfs: bool) -> dict[str, Any]:
    return {
        "type": "file",
        "path": path,
        "size": len(data),
        "blob_id": git_blob(data),
        "lfs": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} if lfs else None,
        "xet_hash": None,
        "security": {"safe": True, "status": "safe", "av_scan": None, "pickle_import_scan": None},
    }


def make_world(tmp: Path, **change: Change) -> Path:
    """A fake Hub holding the five tasks' files; each `change` alters one task's repository."""
    src = tmp / "hub"
    repos: dict[str, Any] = {}
    for t, spec in TASKS.items():
        f = src / t / FILES[t]
        f.parent.mkdir(parents=True, exist_ok=True)
        if FILES[t].endswith(".jsonl"):
            f.write_text("".join(json.dumps(r) + "\n" for r in ROWS[t]))
        else:
            Dataset.from_list(ROWS[t]).to_parquet(str(f))  # datasets is declared; pyarrow is not
        card = src / t / "README.md"
        card.write_text(f"---\nlicense: {spec.licence}\n---\n# {t}\n")
        tree = [
            entry(FILES[t], f.read_bytes(), lfs=True),
            entry("README.md", card.read_bytes(), lfs=False),
        ]
        if "/" in FILES[t]:
            tree.append({"type": "folder", "path": FILES[t].rsplit("/", 1)[0]})
        info = {"gated": False, "private": False, "disabled": False, "license": spec.licence}
        repos[spec.repo] = {
            "info": {REV[t]: info | {"sha": REV[t]}, "main": info | {"sha": MAIN[t]}},
            "tree": tree,
            "files": {FILES[t]: str(f), "README.md": str(card)},
        }
    for t, fn in change.items():
        fn(repos[TASKS[t].repo])
    world = tmp / "world.json"
    world.write_text(json.dumps({"repos": repos, "log": str(tmp / "hub.log")}))
    return world


def run(world: Path, *args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-I", "-B", str(ENTRY), str(SCRIPT), str(world), *args]
    # check and fetch refuse a set HF_HUB_OFFLINE; the fake Hub never opens a socket
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False, env=env)


def pins(rev: dict[str, str] | None = None) -> list[str]:
    return ["--confirm-pins", *(f"{t}={r}" for t, r in sorted((rev or REV).items()))]


def downloads(tmp: Path) -> int:
    log = tmp / "hub.log"
    return log.read_text().count('"download"') if log.exists() else 0


def the_set(out: Path) -> Path:
    found = [p for p in out.iterdir() if p.name != "raw"]
    assert len(found) == 1
    return found[0]


def test_check_prints_the_live_mains_and_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "evals"
    r = run(make_world(tmp_path), "check", "--out", str(out))
    assert r.returncode == 0, r.stderr
    for t in TASKS:
        assert f"--confirm-pins {t}={MAIN[t]}" in r.stdout
        assert FILES[t] in r.stdout
    assert not out.exists()
    assert downloads(tmp_path) == 0


def test_check_lists_every_layout_miss_and_still_prints_the_rest(tmp_path: Path) -> None:
    world = make_world(tmp_path, piqa=renamed, winogrande=renamed)
    r = run(world, "check", "--out", str(tmp_path / "evals"))
    assert r.returncode == 78, r.stderr
    misses = [x for x in r.stdout.splitlines() if x.startswith("LAYOUT: ")]
    assert [m.split(":")[1].strip() for m in misses] == ["piqa", "winogrande"]
    assert all("other/validation.parquet" in m for m in misses)
    for t in ("arc_easy", "hellaswag", "lambada_openai"):
        assert f"--confirm-pins {t}={MAIN[t]}" in r.stdout
    assert downloads(tmp_path) == 0


def test_fetch_prepare_verify_build_the_rendered_set(tmp_path: Path) -> None:
    world, out = make_world(tmp_path), tmp_path / "evals"
    r = run(world, "fetch", "--out", str(out), *pins())
    assert r.returncode == 0, r.stderr
    rec = json.loads((out / "raw" / "eval-piqa" / REV["piqa"] / "fetch.json").read_text())
    assert (rec["revision"], rec["licence_tag"], rec["token_sent"]) == (
        REV["piqa"],
        "afl-3.0",
        False,
    )
    r = run(world, "prepare", "--out", str(out), *pins())
    assert r.returncode == 0, r.stderr
    set_dir = the_set(out)
    m, items = tasks.read_set(set_dir)
    assert m["set_id"] == set_dir.name
    rows: Any = m["tasks"]
    for t in TASKS:
        assert items[t] == [tasks.render(t, row, i) for i, row in enumerate(ROWS[t])]
        assert rows[t]["revision"] == REV[t]
        assert rows[t]["n"] == len(ROWS[t])
        assert rows[t]["source"]["path"] == FILES[t]
        assert rows[t]["licence_override"] is None
    assert "published count" in r.stdout
    assert run(world, "verify", "--set", str(set_dir)).returncode == 0
    again = run(world, "prepare", "--out", str(out), *pins())
    assert again.returncode == 0
    assert "already complete" in again.stdout
    (set_dir / "piqa.jsonl").write_bytes(b"")
    assert run(world, "verify", "--set", str(set_dir)).returncode == 1
    raw = out / "raw" / "eval-arc_easy" / REV["arc_easy"] / FILES["arc_easy"]
    raw.chmod(0o644)
    raw.write_bytes(raw.read_bytes() + b"\0")
    assert run(world, "prepare", "--out", str(out), *pins()).returncode == 65


def test_fetch_refuses_missing_and_unresolved_pins(tmp_path: Path) -> None:
    world, out = make_world(tmp_path), str(tmp_path / "evals")
    some = ["--confirm-pins", f"piqa={REV['piqa']}"]
    assert run(world, "fetch", "--out", out, *some).returncode == 78
    assert run(world, "fetch", "--out", out, *pins(REV | {"piqa": "f" * 40})).returncode == 78
    assert run(world, "fetch", "--out", out, *pins(REV | {"piqa": "abc"})).returncode == 78
    assert run(world, "prepare", "--out", out, *pins()).returncode == 73
    assert downloads(tmp_path) == 0


def no_licence(repo: dict[str, Any]) -> None:
    for v in repo["info"].values():
        v["license"] = None


def test_a_missing_licence_tag_stops_until_it_is_accepted(tmp_path: Path) -> None:
    world, out = make_world(tmp_path, hellaswag=no_licence), str(tmp_path / "evals")
    r = run(world, "check", "--out", out)
    assert r.returncode == 77
    assert "LICENCE MISMATCH" in r.stdout
    assert run(world, "fetch", "--out", out, *pins()).returncode == 77
    assert (
        run(world, "fetch", "--out", out, *pins(), "--accept-licence", "sciq=mit").returncode == 78
    )
    assert downloads(tmp_path) == 0
    accept = ["--accept-licence", "hellaswag=none"]
    assert run(world, "fetch", "--out", out, *pins(), *accept).returncode == 0
    assert run(world, "prepare", "--out", out, *pins()).returncode == 0
    m, _ = tasks.read_set(the_set(Path(out)))
    rows: Any = m["tasks"]
    assert rows["hellaswag"]["licence_tag"] is None
    assert rows["hellaswag"]["licence_override"] == "none"


def two_licences(repo: dict[str, Any]) -> None:
    for v in repo["info"].values():
        v["license"] = ["mit", "cc-by-4.0"]


def test_an_accepted_licence_is_recorded(tmp_path: Path) -> None:
    world, out = make_world(tmp_path, piqa=two_licences), tmp_path / "evals"
    assert run(world, "fetch", "--out", str(out), *pins()).returncode == 77
    accept = ["--accept-licence", "piqa=mit"]
    assert run(world, "fetch", "--out", str(out), *pins(), *accept).returncode == 0
    assert run(world, "prepare", "--out", str(out), *pins()).returncode == 0
    m, _ = tasks.read_set(the_set(out))
    rows: Any = m["tasks"]
    row = rows["piqa"]
    assert row["licence_override"] == "mit"
    assert row["licence_tag"] == ["mit", "cc-by-4.0"]


def two_files(repo: dict[str, Any]) -> None:
    e = dict(repo["tree"][0])
    e["path"] = "data/validation-00001-of-00002.parquet"
    repo["tree"].append(e)


def renamed(repo: dict[str, Any]) -> None:
    repo["tree"][0]["path"] = "other/validation.parquet"


@pytest.mark.parametrize("change", [two_files, renamed])
def test_an_unexpected_layout_stops_before_any_download(tmp_path: Path, change: Change) -> None:
    world = make_world(tmp_path, hellaswag=change)
    r = run(world, "fetch", "--out", str(tmp_path / "evals"), *pins())
    assert r.returncode == 78
    assert "hellaswag" in r.stderr
    assert downloads(tmp_path) == 0


def flagged(repo: dict[str, Any]) -> None:
    repo["tree"][0]["security"] = {
        "safe": False,
        "status": "unsafe",
        "av_scan": None,
        "pickle_import_scan": None,
    }


def test_a_flagged_file_needs_its_override(tmp_path: Path) -> None:
    world, out = make_world(tmp_path, winogrande=flagged), tmp_path / "evals"
    assert run(world, "fetch", "--out", str(out), *pins()).returncode == 65
    allow = ["--allow-unsafe-file", FILES["winogrande"]]
    assert run(world, "fetch", "--out", str(out), *pins(), *allow).returncode == 0
    rec = json.loads(
        (out / "raw" / "eval-winogrande" / REV["winogrande"] / "fetch.json").read_text()
    )
    assert rec["files"][0]["security_override"] is True


def test_the_output_root_must_be_outside_the_clone_and_hold_only_sets(tmp_path: Path) -> None:
    world = make_world(tmp_path)
    clone = ROOT
    assert run(world, "check", "--out", str(clone / "evals-here")).returncode == 73
    store_root = tmp_path / "mf-data"
    (store_root / "not-a-set").mkdir(parents=True)
    assert run(world, "check", "--out", str(store_root)).returncode == 73
