"""The held-out 13-gram scan, scripts/analysis/overlap.py.

Synthetic stores only: a two-shard training split and a two-shard held-out split whose documents
copy training text whole, in part, across a training document boundary, or not at all; two of
them share one training 13-gram and one new one. Every count is checked against a brute-force
count over Python tuples. No real store is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from akeratos.data import decontam
from tests.support.proc import Children

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "overlap.py"
FIRSTNAME = SCRIPT.with_name("firstname.py")
V, N = 512, 13
FP = "9545f2d205efd574" + "f" * 48  # Billet's store's fingerprint starts 9545f2d205efd574
OTHER = "0" * 64  # another store's
STORE_FILES = ("fingerprint.json", "manifest.json", "heldout/manifest.json", "train/manifest.json")


def _load(path: Path = SCRIPT, name: str = "billet_overlap") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ beside it
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod


_load(FIRSTNAME, "billet_firstname").alias()
ov = _load()


def _ids(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(17, V, size=n).astype("<u2")


A, B, C, D = _ids(1, 30), _ids(2, 20), _ids(3, 25), _ids(4, 40)
X = D[5:18]  # one 13-gram of D, which the last training document repeats
Y = _ids(10, 13)  # a 13-gram no training document holds
TRAIN = [[A, B], [C, D, np.concatenate([_ids(9, 5), X])]]
HELDOUT = [
    A[5:25],  # training text: all 8 of its 13-grams found
    _ids(5, 30),  # new text: none found
    _ids(6, 12),  # 12 ids: no 13-gram
    np.concatenate([_ids(7, 8), C[:13], _ids(8, 8)]),  # 1 of its 17 found
    np.concatenate([A[-6:], B[:7]]),  # A's end and B's start: not a training 13-gram
    np.concatenate([X, X]),  # 14 windows, 13 distinct, X found
    np.concatenate([B[2:15], _ids(11, 4), Y]),  # 1 of its 18 found: B[2:15]
    np.concatenate([Y, _ids(12, 4), B[2:15]]),  # the same two 13-grams as the one before
]


def _write_split(path: Path, shards: list[list[np.ndarray]]) -> None:
    """A split in the store's format, with each shard file's SHA-256 in its manifest."""
    path.mkdir(parents=True)
    m: dict[str, Any] = {"dtype": "uint16", "eos": 0, "sep_position": "start", "vocab": V}
    for key in ("shards", "tokens", "documents", "sha256_bin", "sha256_idx"):
        m[key] = []
    for s, docs in enumerate(shards):
        body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
        idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
        body.tofile(path / f"{s:06d}.bin")
        idx.tofile(path / f"{s:06d}.idx")
        m["shards"].append(f"{s:06d}.bin")
        m["tokens"].append(len(body))
        m["documents"].append(len(docs))
        m["sha256_bin"].append(hashlib.sha256(body.tobytes()).hexdigest())
        m["sha256_idx"].append(hashlib.sha256(idx.tobytes()).hexdigest())
    (path / "manifest.json").write_text(json.dumps(m))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stamp(root: Path, fingerprint: str = FP, named: str | None = None) -> None:
    """fingerprint.json, and the store's manifest.json: the fingerprint it names (`named`, by
    default fingerprint.json's) and both split manifests' SHA-256s."""
    (root / "fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}))
    out = {f"{s}_manifest_sha256": _sha(root / s / "manifest.json") for s in ("heldout", "train")}
    out["fingerprint"] = named or fingerprint
    (root / "manifest.json").write_text(json.dumps({"outputs": out}))


def _store(root: Path, heldout: list[list[np.ndarray]]) -> Path:
    _write_split(root / "train", TRAIN)
    _write_split(root / "heldout", heldout)
    _stamp(root)
    return root


def _brute(train: list[list[np.ndarray]], docs: list[np.ndarray]) -> list[tuple[int, ...]]:
    """(tokens, distinct, found, hits) per document, from sets and counts of id tuples."""
    seen = Counter(
        tuple(d[i : i + N].tolist())
        for shard in train
        for d in shard
        for i in range(len(d) - N + 1)
    )
    rows = []
    for d in docs:
        grams = {tuple(d[i : i + N].tolist()) for i in range(len(d) - N + 1)}
        rows.append(
            (len(d), len(grams), sum(g in seen for g in grams), sum(seen[g] for g in grams))
        )
    return rows


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _store(tmp_path / "store", [HELDOUT[:4], HELDOUT[4:]])


def test_the_scan_matches_a_brute_force_count(store: Path, tmp_path: Path) -> None:
    """Run as the README says, through firstname.py: every row equals the brute-force count."""
    out = tmp_path / "out"
    args = ["--store", str(store), "--out", str(out)]
    cmd = [sys.executable, "-I", "-B", str(FIRSTNAME), str(SCRIPT), *args]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    rows = np.load(out / "overlap.npy")
    assert rows.dtype == ov.FIELDS
    assert rows.tolist() == _brute(TRAIN, HELDOUT)
    assert rows["distinct"].tolist() == [8, 18, 0, 17, 1, 13, 18, 18]
    assert rows["found"].tolist() == [8, 0, 0, 1, 0, 1, 1, 1]
    assert rows["hits"].tolist() == [8, 0, 0, 1, 0, 2, 1, 1]  # X occurs twice in training
    rec = json.loads((out / "overlap.json").read_text())
    assert rec["complete"] is True
    assert rec["train"] == {"shards": 2, "shards_scanned": 2, "tokens": 138}
    assert rec["heldout"] == {"shards": 2, "documents": 8, "tokens": 198}
    assert rec["outputs"]["overlap.npy"] == _sha(out / "overlap.npy")
    assert rec["command"][-4:] == ["--store", str(store), "--out", str(out)]
    res = rec["result"]
    assert (res["no_13gram"], res["none_found"], res["any_found"], res["all_found"]) == (1, 3, 5, 1)
    assert (res["distinct"], res["found"], res["any_found_share"]) == (93, 12, 0.625)
    assert res["quartile_edges"] == [1 / 18, 1 / 17, 1 / 13]  # the 2nd, 3rd and 4th of 5 shares
    assert res["quartile_documents"] == [2, 1, 1, 1]
    assert res["share_min_max"] == [1 / 18, 1.0]
    assert json.loads(done.stdout) == res


def test_the_record_names_the_code_and_the_store(
    store: Path, tmp_path: Path, children: Children
) -> None:
    """Run from a copy committed alone in a new repository, whose commit and changes are known."""
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(SCRIPT, repo / "overlap.py")
    who = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
    for args in (["init", "-q"], ["add", "overlap.py"], [*who, "commit", "-q", "-m", "Scan"]):
        assert children.run(["git", *args], cwd=repo).returncode == 0
    head = children.run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    copy = _load(repo / "overlap.py", "billet_overlap_copy")
    out = tmp_path / "out"
    assert copy.main(["--store", str(store), "--out", str(out)]) == 0
    rec = json.loads((out / "overlap.json").read_text())
    data = Path(decontam.__file__).parent
    assert rec["code"] == {
        "commit": head,
        "uncommitted": [],
        "overlap.py": _sha(SCRIPT),
        **{name: _sha(data / name) for name in ("decontam.py", "store.py", "packer.py")},
    }
    assert rec["store"] == {
        "path": str(store),
        "fingerprint": FP,
        **{name: _sha(store / name) for name in STORE_FILES},
    }
    with (repo / "overlap.py").open("a") as f:
        f.write("# an uncommitted change\n")
    assert copy.code()["uncommitted"] == [" M overlap.py"]


def test_quartile_edges_take_the_lower_value_and_ties_fall_below() -> None:
    """Edge j of 4 is the ⌈jn/4⌉-th smallest share; a share equal to an edge falls below it."""
    edges, per = ov.quartiles(np.array([0.5, 0.1, 1.0, 0.2, 0.2]))
    assert edges.tolist() == [0.2, 0.2, 0.5]
    assert per.tolist() == [3, 0, 1, 1]
    edges, per = ov.quartiles(np.array([0.3, 0.1, 0.2]))  # NumPy's "lower" ends at 0.2
    assert edges.tolist() == [0.1, 0.2, 0.3]
    assert per.tolist() == [1, 1, 1, 0]
    edges, per = ov.quartiles(np.arange(1, 9) / 8)
    assert edges.tolist() == [2 / 8, 4 / 8, 6 / 8]
    assert per.tolist() == [2, 2, 2, 2]
    edges, per = ov.quartiles(np.zeros(0))
    assert edges.tolist() == []
    assert per.tolist() == [0, 0, 0, 0]


def test_a_sample_of_shards_says_it_is_incomplete(store: Path, tmp_path: Path) -> None:
    """--shards 1 reads shard 0 alone: C and D's 13-grams go unfound, and the record says so."""
    out = tmp_path / "out"
    assert ov.main(["--store", str(store), "--out", str(out), "--shards", "1"]) == 0
    rec = json.loads((out / "overlap.json").read_text())
    assert rec["complete"] is False
    assert rec["train"]["shards_scanned"] == 1
    assert np.load(out / "overlap.npy")["found"].tolist() == [8, 0, 0, 0, 0, 0, 1, 1]


def _flip(path: Path, byte: int) -> None:
    data = bytearray(path.read_bytes())
    data[byte] ^= 1
    path.write_bytes(bytes(data))


def _split(store: Path, split: str, **changes: Any) -> None:
    """Change keys of a split's manifest (None removes one); the store's manifest.json follows."""
    path = store / split / "manifest.json"
    m = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            del m[key]
        else:
            m[key] = value
    path.write_text(json.dumps(m))
    _stamp(store)


def _append(path: Path, data: bytes) -> None:
    with path.open("ab") as f:
        f.write(data)


def _cut(path: Path) -> None:
    path.write_bytes(path.read_bytes()[:-2])


SHARD = "a shard file differs from its manifest's SHA-256"
NOT_BILLETS = "not store 9545f2d205efd574"
UNNAMED = "not the one the store's manifest.json names"
REFUSED: dict[str, tuple[Callable[[Path], object], str]] = {
    "held-out .bin": (lambda s: _flip(s / "heldout" / "000001.bin", 10), SHARD),
    "training .bin": (lambda s: _flip(s / "train" / "000001.bin", 10), SHARD),
    "training .idx": (lambda s: _flip(s / "train" / "000000.idx", 8), SHARD),  # 31 to 30
    "another store": (lambda s: _stamp(s, OTHER), NOT_BILLETS),
    "store manifest of another store": (lambda s: _stamp(s, named=OTHER), NOT_BILLETS),
    "held-out manifest": (lambda s: _append(s / "heldout" / "manifest.json", b"\n"), UNNAMED),
    "training manifest": (lambda s: _append(s / "train" / "manifest.json", b"\n"), UNNAMED),
    "vocab": (lambda s: _split(s, "train", vocab=V + 1), "vocab"),
    "no shard SHA-256s": (
        lambda s: _split(s, "heldout", sha256_idx=None),
        "no sha256_bin and sha256_idx for every shard",
    ),
    "layout": (lambda s: _cut(s / "train" / "000001.bin"), "bytes, expected"),
    "missing key": (lambda s: (s / "fingerprint.json").write_text("{}"), "'fingerprint'"),
}


@pytest.mark.parametrize(("change", "words"), REFUSED.values(), ids=list(REFUSED))
def test_inputs_that_fail_a_check_are_refused(
    store: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: Callable[[Path], object],
    words: str,
) -> None:
    """Each change is refused: the refusal names what failed, and nothing is written.

    A flipped bit leaves the layout valid (the .idx offset 31 becomes 30), so only the SHA-256s
    can refuse it; "another store" has both its fingerprint.json and its manifest.json name it.
    """
    change(store)
    out = tmp_path / "out"
    assert ov.main(["--store", str(store), "--out", str(out)]) == ov.EXIT_REFUSED
    assert words in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("opens", [True, False])
def test_held_out_ids_0_that_are_not_document_starts_are_refused(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], opens: bool
) -> None:
    """Id 0 inside document 1, which opens with it or not: the documents would be cut wrong."""
    path = store / "heldout"
    body = np.fromfile(path / "000000.bin", dtype="<u2")
    start = len(HELDOUT[0]) + 1  # document 1's leading 0
    body[start + 3] = 0
    if not opens:
        body[start] = 17  # one id 0 per document still, but not at document 1's start
    body.tofile(path / "000000.bin")
    m = json.loads((path / "manifest.json").read_text())
    _split(store, "heldout", sha256_bin=[_sha(path / "000000.bin"), *m["sha256_bin"][1:]])
    out = tmp_path / "out"
    assert ov.main(["--store", str(store), "--out", str(out)]) == ov.EXIT_REFUSED
    assert "id 0 does not open a document, or occurs inside one" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("name", ["overlap.npy", "overlap.json"])
def test_an_out_folder_holding_either_file_is_refused_before_the_store_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """So a sample cannot replace a full scan's files. The store here does not even exist."""
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text("earlier")
    assert ov.main(["--store", str(tmp_path / "none"), "--out", str(out)]) == ov.EXIT_REFUSED
    assert "already holds overlap.npy or overlap.json" in capsys.readouterr().err
    assert [p.name for p in out.iterdir()] == [name]
    assert (out / name).read_text() == "earlier"


def test_shards_below_1_is_a_usage_error(store: Path, tmp_path: Path) -> None:
    """--shards 0 would scan nothing and find no 13-gram."""
    with pytest.raises(SystemExit) as e:
        ov.main(["--store", str(store), "--out", str(tmp_path / "out"), "--shards", "0"])
    assert e.value.code == 2
    assert not (tmp_path / "out").exists()
