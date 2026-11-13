"""The memorisation split that scripts/analysis/s6.py builds from Billet's training data.

Synthetic stores and checkpoints only, at a small training shape with Billet's data seed: 2 rows of
T 16, 3 micro-batches a step, a buffer of 8 and 10 steps, so thirds of 4, 3 and 3 steps, and 4
documents picked a third. The two-shard training split holds documents shorter and longer than a row, so
rows crop some, one of them to its leading 0 alone. The order is checked against a best-fit
packer written here from packer.py's docstring, the selection against the docstring's rule, and
the split is read back by the scorer's own reader. No real store or run is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from platform import python_version
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from akeratos import ckpt
from akeratos.data.packer import Packer
from akeratos.data.store import Split
from akeratos.eval.heldout import HeldOut
from akeratos.rng import derive, seed_hex
from akeratos.train.source import PackedSource

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "s6.py"
V = 512
FP = "9545f2d205efd574" + "f" * 48  # Billet's store's fingerprint starts 9545f2d205efd574
OTHER = "0" * 64  # another store's
SEED = 20_261_101  # the analysis seed, which s6.py's docstring keys the selection by
DATA_SEED = 0x84D17AD2E2342315  # Billet's data seed
SMALL = {"rows": 2, "seq_len": 16, "accum": 3, "buffer": 8, "steps": 10, "data_seed": DATA_SEED}
PICK = 4
REUSED = (
    "ckpt.py",
    "rng.py",
    "data/packer.py",
    "data/store.py",
    "eval/heldout.py",
    "train/source.py",
)
OUTPUTS = (
    "heldout/000000.bin",
    "heldout/000000.idx",
    "heldout/000000.bytes",
    "heldout/manifest.json",
    "token_bytes.u16",
    "manifest.json",
    "thirds.npy",
)


def _load(path: Path = SCRIPT, name: str = "billet_s6") -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # no __pycache__ in scripts/
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = before
    return mod


s6 = _load()

TB = np.random.default_rng(1).integers(1, 5, size=V).astype("<u2")
TB[:17] = 0


def _docs(seed: int, n: int) -> list[np.ndarray]:
    """n documents' ids after the leading 0: 1 to 39 ids, so 2 to 40 tokens against rows of 17."""
    rng = np.random.default_rng(seed)
    return [rng.integers(17, V, size=int(rng.integers(1, 40))).astype("<u2") for _ in range(n)]


TRAIN = [_docs(2, 100), _docs(3, 100)]
DOCS = [d for shard in TRAIN for d in shard]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(path: Path, shards: list[list[np.ndarray]]) -> None:
    """A training split in the store's format, with each shard file's SHA-256 in its manifest."""
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
        m["sha256_bin"].append(_sha(path / f"{s:06d}.bin"))
        m["sha256_idx"].append(_sha(path / f"{s:06d}.idx"))
    (path / "manifest.json").write_text(json.dumps(m))


def _stamp(root: Path, fingerprint: str = FP, named: str | None = None) -> None:
    """fingerprint.json, and the store's manifest.json: the fingerprint it names (`named`, by
    default fingerprint.json's), the training manifest's SHA-256 and token_bytes.u16's."""
    (root / "fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}))
    out = {
        "fingerprint": named or fingerprint,
        "train_manifest_sha256": _sha(root / "train" / "manifest.json"),
        "token_bytes_sha256": _sha(root / "token_bytes.u16"),
    }
    (root / "manifest.json").write_text(json.dumps({"outputs": out}))


def _store(root: Path, train: list[list[np.ndarray]] = TRAIN) -> Path:
    root.mkdir(parents=True)
    (root / "token_bytes.u16").write_bytes(TB.tobytes())
    _write_split(root / "train", train)
    _stamp(root)
    return root


def _trained(root: Path, buffer: int = SMALL["buffer"]) -> dict[str, Any]:
    """The data cursor a SMALL run saves at its end: the stand-in for Billet's checkpoint."""
    o = SMALL
    split = Split(root / "train", vocab=V)
    packer = Packer(split, o["rows"], o["seq_len"], vocab=V, data_seed=DATA_SEED, buffer=buffer)
    source = PackedSource(packer)
    for _ in range(o["steps"] * o["accum"]):
        source.next()
    return source.state()


def _ckpt(run: Path, source: dict[str, Any], step: int = SMALL["steps"], **config: Any) -> Path:
    """A complete checkpoint at `step` with `source` as its data cursor; `config` overrides.

    Its run had a table and document masking, as Billet's did; only the keys s6.py reads are here.
    """
    cfg = {
        "model": {"vocab_size": V, "doc_mask": True},
        "data_seed": seed_hex(DATA_SEED),
        "accum": 3,
        "table": {"orders": [2, 3]},
    }
    return ckpt.save(run, step, arrays={}, state={"config": cfg | config, "source": source})


def _small(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in SMALL.items():
        monkeypatch.setitem(s6.ORDER, k, v)
    monkeypatch.setattr(s6, "PER_THIRD", PICK)


def _best_fit(lengths: list[int]) -> list[tuple[int, int, int]]:
    """(document, tokens taken, row) of each document a SMALL run places, by packer.py's
    docstring: the epoch's Philox permutation keyed by derive(data seed, "data", 0) fills a buffer
    before every pick; the longest buffered document that fits the row's space goes in, the
    earliest buffered among equal lengths; when none fits, the shortest, the earliest among ties,
    is cropped to fill the row."""
    o = SMALL
    gen = np.random.Generator(np.random.Philox(key=derive(DATA_SEED, "data", 0)))
    order = gen.permutation(len(lengths)).tolist()
    buf: list[tuple[int, int]] = []  # (arrival, document)
    pulled, out = 0, []
    for row in range(o["steps"] * o["accum"] * o["rows"]):
        left = o["seq_len"] + 1
        while left > 0:
            while len(buf) < o["buffer"]:
                buf.append((pulled, order[pulled]))
                pulled += 1
            fits = [e for e in buf if lengths[e[1]] <= left]
            if fits:
                e = max(fits, key=lambda e: (lengths[e[1]], -e[0]))
                take = lengths[e[1]]
            else:
                e = min(buf, key=lambda e: (lengths[e[1]], e[0]))
                take = left
            buf.remove(e)
            out.append((e[1], take, row))
            left -= take
    return out


LOG = _best_fit([len(d) + 1 for d in DOCS])


def _third(row: int) -> int:
    o = SMALL
    return 3 * (row // o["rows"] // o["accum"]) // o["steps"]


def _picks() -> tuple[list[int], list[int]]:
    """The log positions the docstring's rule picks, in training order, and each third's count."""
    picks, candidates = [], []
    for j in range(3):
        cand = [p for p, (_, taken, row) in enumerate(LOG) if _third(row) == j and taken >= 2]
        candidates.append(len(cand))
        gen = np.random.Generator(np.random.Philox(key=derive(SEED, "s6", j)))
        picks += sorted(gen.choice(np.array(cand), size=PICK, replace=False).tolist())
    return picks, candidates


PICKS, CANDIDATES = _picks()
PARTS = [np.concatenate([[0], DOCS[LOG[p][0]]])[: LOG[p][1]] for p in PICKS]


def _argv(root: Path, run: Path, out: Path) -> list[str]:
    return ["--store", str(root), "--ckpt", str(run), "--out", str(out)]


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    _small(monkeypatch)
    root = _store(tmp_path / "store")
    return root, _ckpt(tmp_path / "run", _trained(root))


def test_the_fixture_crops_and_leaves_a_row_end_with_no_target() -> None:
    """Rows crop documents, and one is cropped to its leading 0 alone, which is never picked."""
    assert any(taken < len(DOCS[d]) + 1 for d, taken, _ in LOG)
    assert any(taken == 1 for _, taken, _ in LOG)
    assert all(LOG[p][1] >= 2 for p in PICKS)


def test_the_split_holds_the_trained_heads_of_the_picked_documents(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Each document is its row's part, in training order, with its third."""
    root, run = inputs
    out = tmp_path / "out"
    assert s6.main(_argv(root, run, out)) == 0
    split = Split(out / "heldout", vocab=V)
    got = [split.read(i, int(n)).tolist() for i, n in enumerate(split.lengths)]
    assert got == [p.tolist() for p in PARTS]
    thirds = np.load(out / "thirds.npy")
    assert thirds.dtype == np.int64
    assert thirds.tolist() == [_third(LOG[p][2]) for p in PICKS]
    assert thirds.tolist() == [0] * PICK + [1] * PICK + [2] * PICK


def test_the_scorer_reads_each_document_whole_from_its_leading_0(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    """The scorer's reader takes the split as written: window 0, the part's targets and bytes."""
    root, run = inputs
    out = tmp_path / "out"
    assert s6.main(_argv(root, run, out)) == 0
    held = HeldOut(out / "heldout", 2048, None, True, vocab=V)
    assert held.docs == 3 * PICK
    assert held.doc_targets.tolist() == [len(p) - 1 for p in PARTS]
    assert held.doc_bytes.tolist() == [int(TB[p[1:]].sum()) for p in PARTS]
    assert not held.windows()[1].any()
    assert (out / "token_bytes.u16").read_bytes() == TB.tobytes()


def test_the_record_names_every_input_count_and_output(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each file read and written, by SHA-256; the run's time and versions; the order, the replay,
    the selection and the split."""
    root, run = inputs
    out = tmp_path / "out"
    before = datetime.now(UTC).replace(microsecond=0)
    assert s6.main(_argv(root, run, out)) == 0
    rec = json.loads((out / "split.json").read_text())
    run_fields = ["what", "command", "cwd", "started", "seconds", "code", "python", "numpy"]
    rest = ["inputs", "store", "checkpoint", "order", "replay", "selection", "split", "outputs"]
    assert list(rec) == run_fields + rest
    assert (
        rec["what"]
        == "S6: training documents Billet read in each third of training, as a held-out split"
    )
    assert before <= datetime.fromisoformat(rec["started"]) <= datetime.now(UTC)
    assert list(rec["seconds"]) == ["checks", "replay", "write"]
    assert all(0 <= v < 30 for v in rec["seconds"].values())
    assert (rec["cwd"], rec["python"], rec["numpy"]) == (
        str(Path.cwd()),
        python_version(),
        np.__version__,
    )
    read = [root / n for n in ("fingerprint.json", "manifest.json", "train/manifest.json")]
    read += [root / "train" / n for n in ("000000.bin", "000000.idx", "000001.bin", "000001.idx")]
    read += [root / "token_bytes.u16", run / "state.json"]
    assert rec["inputs"] == {str(p): _sha(p) for p in read}
    assert rec["outputs"] == {n: _sha(out / n) for n in OUTPUTS}
    held = out / "heldout"
    assert json.loads((held / "manifest.json").read_text()) == {
        "dtype": "uint16",
        "eos": 0,
        "sep_position": "start",
        "vocab": V,
        "shards": ["000000.bin"],
        "tokens": [sum(len(p) for p in PARTS)],
        "documents": [3 * PICK],
        "bytes": ["000000.bytes"],
        **{f"sha256_{k}": [_sha(held / f"000000.{k}")] for k in ("bin", "idx", "bytes")},
    }
    assert json.loads((out / "manifest.json").read_text())["outputs"] == {
        "token_bytes_sha256": _sha(out / "token_bytes.u16"),
        "heldout_manifest_sha256": _sha(out / "heldout" / "manifest.json"),
    }
    state = json.loads((run / "state.json").read_text())["state"]["source"]
    assert rec["order"] == SMALL | {
        "data_seed": "0x84d17ad2e2342315",
        "epochs": 1,
        "fingerprint": state["fingerprint"],
    }
    assert rec["replay"] == {
        "micro_batches": 30,
        "documents": len(LOG),
        "cropped": sum(taken < len(DOCS[d]) + 1 for d, taken, _ in LOG),
        "end": {k: state[k] for k in ("epoch", "cursor", "read", "used")},
    }
    assert rec["selection"] == {
        "seed": SEED,
        "stream": "s6",
        "steps": [[0, 3], [4, 6], [7, 9]],
        "candidates": CANDIDATES,
        "per_third": PICK,
    }
    assert rec["split"] == {
        "documents": 3 * PICK,
        "targets": sum(len(p) - 1 for p in PARTS),
        "bytes": sum(int(TB[p[1:]].sum()) for p in PARTS),
        "cropped": sum(
            len(p) < len(DOCS[LOG[q][0]]) + 1 for p, q in zip(PARTS, PICKS, strict=True)
        ),
    }
    assert rec["store"] == {"path": str(root), "vocab": V}
    assert rec["checkpoint"] == {"path": str(run), "step": 10}
    assert rec["command"] == sys.orig_argv  # in-process here: the test runner's command
    assert json.loads(capsys.readouterr().out) == rec["split"]


def test_billets_shape_is_the_one_its_command_gives() -> None:
    """The module's order is Billet's run, and its picks the held-out split's count in thirds."""
    assert {
        "rows": 2,
        "seq_len": 2048,
        "accum": 32,
        "buffer": 1000,
        "steps": 24_415,
        "data_seed": derive(0, "data"),
    } == s6.ORDER
    assert 3 * s6.PER_THIRD == 7_314
    third = 3 * np.arange(24_415) // 24_415  # the docstring's thirds of Billet's steps
    ends = [np.flatnonzero(third == j)[[0, -1]].tolist() for j in range(3)]
    assert ends == [[0, 8_138], [8_139, 16_276], [16_277, 24_414]]


def test_the_record_names_the_code(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run from a copy committed alone in a new repository, whose commit and changes are known."""
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(SCRIPT, repo / "s6.py")
    who = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
    for args in (["init", "-q"], ["add", "s6.py"], [*who, "commit", "-q", "-m", "Split"]):
        assert (
            subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False).returncode
            == 0
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    copy = _load(repo / "s6.py", "billet_s6_copy")
    for k, v in SMALL.items():
        monkeypatch.setitem(copy.ORDER, k, v)
    monkeypatch.setattr(copy, "PER_THIRD", PICK)
    root, run = inputs
    out = tmp_path / "out"
    assert copy.main(_argv(root, run, out)) == 0
    pkg = Path(ckpt.__file__).resolve().parent
    assert json.loads((out / "split.json").read_text())["code"] == {
        "commit": git.stdout.strip(),
        "uncommitted": [],
        "s6.py": _sha(SCRIPT),
        **{f"akeratos/{m}": _sha(pkg / m) for m in REUSED},
    }
    with (repo / "s6.py").open("a") as f:
        f.write("# an uncommitted change\n")
    assert copy.code()["uncommitted"] == [" M s6.py"]


def _flip(path: Path, byte: int) -> None:
    data = bytearray(path.read_bytes())
    data[byte] ^= 1
    path.write_bytes(bytes(data))


def _append(path: Path, data: bytes) -> None:
    with path.open("ab") as f:
        f.write(data)


def _manifest(root: Path, **changes: Any) -> None:
    """Change keys of the training manifest (None removes one); the store's manifest follows."""
    path = root / "train" / "manifest.json"
    m = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            del m[key]
        else:
            m[key] = value
    path.write_text(json.dumps(m))
    _stamp(root)


SHARD = "differs from its manifest's SHA-256"
NOT_BILLETS = "not store 9545f2d205efd574"
NOT_FINAL = "source, not Billet's final"
STORE_REFUSED: dict[str, tuple[Callable[[Path], object], str]] = {
    "another store": (lambda s: _stamp(s, OTHER), NOT_BILLETS),
    "store manifest of another store": (lambda s: _stamp(s, named=OTHER), NOT_BILLETS),
    "training manifest": (
        lambda s: _append(s / "train" / "manifest.json", b"\n"),
        "not the one the store's manifest.json names",
    ),
    "training .bin": (lambda s: _flip(s / "train" / "000001.bin", 10), SHARD),
    "training .idx": (lambda s: _flip(s / "train" / "000000.idx", 8), SHARD),
    "no shard SHA-256s": (
        lambda s: _manifest(s, sha256_bin=None),
        "no sha256_bin and sha256_idx for every shard",
    ),
    "token bytes": (
        lambda s: _flip(s / "token_bytes.u16", 40),
        "SHA-256 differs from the root manifest's",
    ),
    "a training split of another vocabulary": (
        lambda s: _manifest(s, vocab=V + 1),
        "the split's vocabulary is 513, the model's 512",
    ),
}


@pytest.mark.parametrize(("edit", "words"), STORE_REFUSED.values(), ids=list(STORE_REFUSED))
def test_a_store_that_fails_a_check_is_refused(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    edit: Callable[[Path], object],
    words: str,
) -> None:
    """Each change is refused, the refusal names what failed, and nothing is written.

    A flipped bit leaves the layout valid (an .idx offset becomes another), so only the
    SHA-256s can refuse it; the token bytes are refused by the scorer's own check.
    """
    root, run = inputs
    edit(root)
    out = tmp_path / "out"
    assert s6.main(_argv(root, run, out)) == s6.EXIT_REFUSED
    assert words in capsys.readouterr().err
    assert not out.exists()


# Checkpoints that are not Billet's final one: the step, the source state's keys and the run's
# configs that differ, and the buffer its data cursor was made with.
CKPT_REFUSED: dict[str, tuple[dict[str, Any], str]] = {
    "not the final step": ({"step": 8}, NOT_FINAL),
    "a mixture's cursor": ({"source": {"kind": "mix"}}, NOT_FINAL),
    "another data seed": ({"data_seed": seed_hex(1)}, "not Billet's"),
    "another accumulation": ({"accum": 4}, "not Billet's"),
    "the twin's: no table": ({"table": None}, "table off, document mask True: not Billet's"),
    "no document mask": (
        {"model": {"vocab_size": V, "doc_mask": False}},
        "mask False: not Billet's",
    ),
    "vocab": (
        {"model": {"vocab_size": V + 1, "doc_mask": True}},
        "entries for a vocabulary of 513",
    ),
    "another buffer": (
        {"buffer": 9},
        "fingerprint (lengths, data seed, buffer, rows, T) is not the checkpoint's",
    ),
    "another cursor": (
        {"source": {"cursor": 1_000}},
        "does not end at the checkpoint's data cursor",
    ),
}


@pytest.mark.parametrize(("kw", "words"), CKPT_REFUSED.values(), ids=list(CKPT_REFUSED))
def test_a_checkpoint_that_is_not_billets_final_one_is_refused(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kw: dict[str, Any],
    words: str,
) -> None:
    """Each change is refused, the refusal names what failed, and nothing is written."""
    root, _ = inputs
    config = dict(kw)
    step, buffer = config.pop("step", SMALL["steps"]), config.pop("buffer", SMALL["buffer"])
    run = _ckpt(tmp_path / "c", _trained(root, buffer) | config.pop("source", {}), step, **config)
    out = tmp_path / "out"
    assert s6.main(_argv(root, run, out)) == s6.EXIT_REFUSED
    assert words in capsys.readouterr().err
    assert not out.exists()


def test_run_as_the_mac_runs_it_a_small_checkpoint_is_refused(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    """In its own interpreter, without the tests' smaller order: step 10 is not Billet's final."""
    root, run = inputs
    out = tmp_path / "out"
    cmd = [sys.executable, "-I", "-B", str(SCRIPT), *_argv(root, run, out)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == s6.EXIT_REFUSED
    assert "step 10 of a 'packed' source, not Billet's final" in done.stderr
    assert not out.exists()


def test_a_checkpoint_whose_state_is_not_json_is_refused(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ckpt.load finds no complete checkpoint there."""
    root, run = inputs
    (run / "state.json").write_text("{")
    assert s6.main(_argv(root, run, tmp_path / "out")) == s6.EXIT_REFUSED
    assert "incomplete checkpoint" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_run_longer_than_the_data_is_refused(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A replay that runs out of documents is refused, as the run would have stopped."""
    root, _ = inputs
    monkeypatch.setitem(s6.ORDER, "steps", 400)
    run = _ckpt(tmp_path / "c", _trained(root), step=400)
    assert s6.main(_argv(root, run, tmp_path / "out")) == s6.EXIT_REFUSED
    assert "every document has been read 1 time(s)" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_document_that_breaks_the_id_rules_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reserved id 5 first in every document: the packer's own check refuses one it reads."""
    _small(monkeypatch)
    root = _store(tmp_path / "store", [[np.concatenate([[5], d[1:]]) for d in s] for s in TRAIN])
    good = _store(tmp_path / "good")
    run = _ckpt(tmp_path / "run", _trained(good))  # the same lengths, so the same fingerprint
    assert s6.main(_argv(root, run, tmp_path / "out")) == s6.EXIT_REFUSED
    assert "breaks the format: a reserved id not in allow_reserved" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_third_with_too_few_documents_is_refused(
    inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fewer documents with a target than the split takes from a third: nothing is picked."""
    root, run = inputs
    monkeypatch.setattr(s6, "PER_THIRD", CANDIDATES[0] + 1)
    assert s6.main(_argv(root, run, tmp_path / "out")) == s6.EXIT_REFUSED
    assert f"third 0: {CANDIDATES[0]} documents with a target" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_an_existing_out_folder_is_refused_before_any_input_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """So a run cannot mix its files with an earlier one's. The inputs here do not even exist."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "thirds.npy").write_text("earlier")
    none = tmp_path / "none"
    assert s6.main(_argv(none, none, out)) == s6.EXIT_REFUSED
    assert "already exists" in capsys.readouterr().err
    assert [p.name for p in out.iterdir()] == ["thirds.npy"]
    assert (out / "thirds.npy").read_text() == "earlier"
