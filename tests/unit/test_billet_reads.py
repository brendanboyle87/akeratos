"""The per-target read counts, scripts/analysis/reads.py.

Synthetic stores and checkpoints only: a two-shard held-out split whose documents are packed
together, fill a row exactly, or are cut into two and three windows, and a checkpoint at Billet's
final step with a small table whose made-up read counts sum as Billet's must. Every count is
checked against a brute-force count that hashes each n-gram in Python integers, as hasher.py's
docstring states the hash. No real store or run is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from platform import python_version
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from akeratos import ckpt
from akeratos.configs import ModelConfig
from akeratos.table.config import TableConfig
from akeratos.table.hasher import multipliers
from akeratos.table.table import Table

BILLET = Path(__file__).resolve().parents[2] / "scripts" / "analysis"
SCRIPT = BILLET / "reads.py"
V, T = 512, 2048
FP = "9545f2d205efd574" + "f" * 48  # Billet's store's fingerprint starts 9545f2d205efd574
OTHER = "0" * 64  # another store's
PAD, GOLD, MASK = 65_536, 0x9E3779B97F4A7C15, (1 << 64) - 1
REUSED = (
    "ckpt.py",
    "rng.py",
    "data/packer.py",
    "data/store.py",
    "eval/heldout.py",
    "configs.py",
    "table/config.py",
    "table/hasher.py",
    "table/table.py",
)
RUN_FIELDS = ["what", "command", "cwd", "started", "seconds", "code", "python", "numpy"]


def _load(path: Path = SCRIPT, name: str = "billet_reads") -> ModuleType:
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


rd = _load()

TCFG = TableConfig(params=4096 * 64, width=64, max_shard_elements=1024 * 64)
MODEL = ModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=2,
    head_dim=32,
    n_kv_heads=1,
    ffn=128,
    vocab_size=V,
    window=16,
    table=True,
    param_kind="sp",
)
TOTAL = 24_415 * 131_072 * 2  # Billet's steps x tokens a step x 2 lookups a token
COUNTS = np.random.default_rng(0).integers(0, 1_000_000, size=TCFG.rows()).astype(np.int64)
COUNTS[-1] = TOTAL - COUNTS[:-1].sum()
TB = np.random.default_rng(1).integers(1, 5, size=V).astype("<u2")
TB[:17] = 0


def _ids(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(17, V, size=n).astype("<u2")


# Each document's ids after its leading 0. Shard 0: two packed together, one of T + 1 tokens that
# fills a row alone, one more. Shard 1: one of T + 2 tokens (two windows, the second of 1 target),
# one of 2T + 301 (three windows), and two packed together. 9 rows in all.
SHARDS = [
    [_ids(2, 30), _ids(3, 5), _ids(4, T), _ids(5, 12)],
    [_ids(6, T + 1), _ids(7, 2 * T + 300), _ids(8, 1), _ids(9, 40)],
]
DOCS = [d for shard in SHARDS for d in shard]
ROWS = 9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_split(path: Path, shards: list[list[np.ndarray]]) -> None:
    """A held-out split in the store's format, with .bytes sidecars and each shard's SHA-256s."""
    path.mkdir(parents=True)
    m: dict[str, Any] = {"dtype": "uint16", "eos": 0, "sep_position": "start", "vocab": V}
    for key in ("shards", "tokens", "documents", "bytes", "sha256_bin", "sha256_idx"):
        m[key] = []
    for s, docs in enumerate(shards):
        body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
        idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
        body.tofile(path / f"{s:06d}.bin")
        idx.tofile(path / f"{s:06d}.idx")
        np.array([int(TB[d].sum()) for d in docs], "<u8").tofile(path / f"{s:06d}.bytes")
        m["shards"].append(f"{s:06d}.bin")
        m["tokens"].append(len(body))
        m["documents"].append(len(docs))
        m["bytes"].append(f"{s:06d}.bytes")
        m["sha256_bin"].append(_sha(path / f"{s:06d}.bin"))
        m["sha256_idx"].append(_sha(path / f"{s:06d}.idx"))
    (path / "manifest.json").write_text(json.dumps(m))


def _stamp(root: Path, fingerprint: str = FP, named: str | None = None) -> None:
    """fingerprint.json, and the store's manifest.json: the fingerprint it names (`named`, by
    default fingerprint.json's), the held-out manifest's SHA-256 and token_bytes.u16's."""
    (root / "fingerprint.json").write_text(json.dumps({"fingerprint": fingerprint}))
    out = {
        "fingerprint": named or fingerprint,
        "heldout_manifest_sha256": _sha(root / "heldout" / "manifest.json"),
        "token_bytes_sha256": _sha(root / "token_bytes.u16"),
    }
    (root / "manifest.json").write_text(json.dumps({"outputs": out}))


def _store(root: Path) -> Path:
    """A store with the held-out split; returns the split."""
    root.mkdir(parents=True)
    (root / "token_bytes.u16").write_bytes(TB.tobytes())
    _write_split(root / "heldout", SHARDS)
    _stamp(root)
    return root / "heldout"


def _ckpt(
    run: Path,
    step: int = 24_415,
    config: dict[str, Any] | None = None,
    record: dict[str, Any] | None = None,
    counts: np.ndarray | None = COUNTS,
    vocab: int = V,
) -> Path:
    """A complete checkpoint at `step`: its run's configs (`config` overrides them) and a
    `table/` of `record` as table.json (default: TCFG's at event 7) and `counts` as reads.npy."""

    def write(path: Path) -> None:
        meta = Table.record(TCFG, 7) if record is None else record
        (path / "table.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        if counts is not None:
            np.save(path / "reads.npy", counts)

    model = asdict(replace(MODEL, vocab_size=vocab))
    state = {"config": {"model": model, "table": asdict(TCFG)} | (config or {})}
    return ckpt.save(run, step, arrays={}, state=state, dirs={"table": write})


def _row(grams: list[int], mult: np.ndarray, slot: int, rps: int) -> int:
    """One n-gram's global row: x = id + 1; acc = (acc ^ x·mult[i])·0x9E3779B97F4A7C15 mod 2⁶⁴;
    acc ^= acc >> 29; row = acc mod rows_per_slot, in slot `slot`."""
    acc = 0
    for tok, m in zip(grams, mult.tolist(), strict=True):
        acc = ((acc ^ ((tok + 1) * m & MASK)) * GOLD) & MASK
    acc ^= acc >> 29
    return acc % rps + slot * rps


def _brute(docs: list[np.ndarray], counts: np.ndarray) -> list[int]:
    """Each target's count, document by document: the counts of the bigram and trigram ending at
    the input before it, with a pad for every position before the target's window starts."""
    mults = multipliers(TCFG.seed, TCFG.orders)
    rps = TCFG.rows_per_slot()
    out = []
    for ids in docs:
        d = [0, *ids.tolist()]
        for j in range(1, len(d)):
            start = (j - 1) // T * T  # the first position of target j's window
            total = 0
            for slot, n in enumerate(TCFG.orders):
                grams = [d[i] if i >= start else PAD for i in range(j - n, j)]
                total += int(counts[_row(grams, mults[slot], slot, rps)])
            out.append(total)
    return out


def _argv(split: Path, run: Path, out: Path) -> list[str]:
    return ["--ckpt", str(run), "--split", str(split), "--out", str(out)]


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    return _store(tmp_path / "store"), _ckpt(tmp_path / "run")


def test_the_counts_match_a_brute_force_count(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    """Run as the Mac runs it (`python -I -B`): each target's count equals the brute-force one."""
    split, run = inputs
    out = tmp_path / "out"
    cmd = [sys.executable, "-I", "-B", str(SCRIPT), *_argv(split, run, out)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
    assert done.returncode == 0, done.stderr
    reads = np.load(out / "reads.npy")
    assert reads.dtype == np.int64
    assert reads.tolist() == _brute(DOCS, COUNTS)
    rec = json.loads((out / "reads.json").read_text())
    assert rec["result"] == {
        "targets": len(reads),
        "min": int(reads.min()),
        "max": int(reads.max()),
    }
    assert json.loads(done.stdout) == rec["result"]
    assert rec["outputs"] == {"reads.npy": _sha(out / "reads.npy")}
    assert rec["command"] == cmd


def test_the_record_names_every_input_and_count(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    """Each file read, by SHA-256; the run's time and versions; the checkpoint; the counts."""
    split, run = inputs
    out = tmp_path / "out"
    before = datetime.now(UTC).replace(microsecond=0)
    assert rd.main(_argv(split, run, out)) == 0
    rec = json.loads((out / "reads.json").read_text())
    assert list(rec) == [*RUN_FIELDS, "inputs", "checkpoint", "heldout", "result", "outputs"]
    assert rec["what"] == "S3: each held-out target's training reads of its bigram and trigram rows"
    assert before <= datetime.fromisoformat(rec["started"]) <= datetime.now(UTC)
    assert 0 <= rec["seconds"] < 30
    assert (rec["cwd"], rec["python"], rec["numpy"]) == (
        str(Path.cwd()),
        python_version(),
        np.__version__,
    )
    root, table = split.parent, run / "table"
    read = [root / n for n in ("fingerprint.json", "manifest.json", "token_bytes.u16")]
    read += [split / n for n in ("manifest.json", "000000.bin", "000000.idx", "000001.bin")]
    read += [split / n for n in ("000001.idx", "000000.bytes", "000001.bytes")]
    read += [run / "state.json", table / "table.json", table / "reads.npy"]
    assert rec["inputs"] == {str(p): _sha(p) for p in read}
    assert rec["checkpoint"] == {
        "path": str(run),
        "step": 24_415,
        "table_event": 7,
        "rows": 4096,
        "orders": [2, 3],
        "heads": 1,
        "mult_seed": Table.record(TCFG, 7)["mult_seed"],
        "reads": TOTAL,
    }
    assert rec["heldout"] == {
        "split": str(split),
        "seq_len": T,
        "doc_mask": True,
        "documents": len(DOCS),
        "targets": sum(len(d) for d in DOCS),
        "bytes": sum(int(TB[d].sum()) for d in DOCS),
        "rows": ROWS,
    }


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_rows_keyed_a_few_at_a_time_give_the_same_counts(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk: int
) -> None:
    """Rows are keyed CHUNK at a time; any chunk gives each row its own keys and order."""
    monkeypatch.setattr(rd, "CHUNK", chunk)
    split, run = inputs
    assert rd.main(_argv(split, run, tmp_path / "out")) == 0
    assert np.load(tmp_path / "out" / "reads.npy").tolist() == _brute(DOCS, COUNTS)


def test_analysis_bins_the_counts_by_the_split(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    """analysis.py reads reads.npy as one non-negative integer per target of the split."""
    spec = importlib.util.spec_from_file_location("billet_analysis_reads", BILLET / "analysis.py")
    assert spec is not None
    assert spec.loader is not None
    an = importlib.util.module_from_spec(spec)
    sys.modules["billet_analysis_reads"] = an
    before, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(an)
    finally:
        sys.dont_write_bytecode = before
    split, run = inputs
    assert rd.main(_argv(split, run, tmp_path / "out")) == 0
    bins = an.fix_bins(an.read_split(split), tmp_path / "out" / "reads.npy", None)
    assert bins.reads is not None
    assert np.bincount(bins.reads, minlength=10).sum() == sum(len(d) for d in DOCS)


def test_the_record_names_the_code(inputs: tuple[Path, Path], tmp_path: Path) -> None:
    """Run from a copy committed alone in a new repository, whose commit and changes are known."""
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(SCRIPT, repo / "reads.py")
    who = ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
    for args in (["init", "-q"], ["add", "reads.py"], [*who, "commit", "-q", "-m", "Count"]):
        assert (
            subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False).returncode
            == 0
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    copy = _load(repo / "reads.py", "billet_reads_copy")
    split, run = inputs
    out = tmp_path / "out"
    assert copy.main(_argv(split, run, out)) == 0
    pkg = Path(ckpt.__file__).resolve().parent
    assert json.loads((out / "reads.json").read_text())["code"] == {
        "commit": git.stdout.strip(),
        "uncommitted": [],
        "reads.py": _sha(SCRIPT),
        **{f"akeratos/{m}": _sha(pkg / m) for m in REUSED},
    }
    with (repo / "reads.py").open("a") as f:
        f.write("# an uncommitted change\n")
    assert copy.code()["uncommitted"] == [" M reads.py"]


def _flip(path: Path, byte: int) -> None:
    data = bytearray(path.read_bytes())
    data[byte] ^= 1
    path.write_bytes(bytes(data))


def _append(path: Path, data: bytes) -> None:
    with path.open("ab") as f:
        f.write(data)


def _manifest(root: Path, **changes: Any) -> None:
    """Change keys of the held-out manifest (None removes one); the store's manifest follows."""
    path = root / "heldout" / "manifest.json"
    m = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            del m[key]
        else:
            m[key] = value
    path.write_text(json.dumps(m))
    _stamp(root)


def _negative() -> np.ndarray:
    c = COUNTS.copy()
    c[1] += c[0] + 1
    c[0] = -1
    return c


SHARD = "differs from its manifest's SHA-256"
NOT_BILLETS = "not store 9545f2d205efd574"
NOT_ALL = "non-negative reads in all"
Change = Callable[[Path], object] | None
REFUSED: dict[str, tuple[Change, dict[str, Any], str]] = {
    "another store": (lambda s: _stamp(s, OTHER), {}, NOT_BILLETS),
    "store manifest of another store": (lambda s: _stamp(s, named=OTHER), {}, NOT_BILLETS),
    "held-out manifest": (
        lambda s: _append(s / "heldout" / "manifest.json", b"\n"),
        {},
        "not the one the store's manifest.json names",
    ),
    "held-out .bin": (lambda s: _flip(s / "heldout" / "000001.bin", 10), {}, SHARD),
    "held-out .idx": (lambda s: _flip(s / "heldout" / "000000.idx", 8), {}, SHARD),
    "no shard SHA-256s": (
        lambda s: _manifest(s, sha256_idx=None),
        {},
        "no sha256_bin and sha256_idx for every shard",
    ),
    "byte counts": (lambda s: _flip(s / "heldout" / "000000.bytes", 0), {}, "`.bytes` entry is"),
    "vocab": (None, {"vocab": V + 1}, "the split's vocabulary is 512, the model's 513"),
    "not the final step": (None, {"step": 24_414}, "not Billet's final step 24415"),
    "no table": (None, {"config": {"table": None}}, "its run had no table"),
    "bigrams only": (
        None,
        {"config": {"table": asdict(replace(TCFG, orders=(2,)))}},
        "orders (2,), 1 heads; Billet's: (2, 3), 1",
    ),
    "two heads": (
        None,
        {"config": {"table": asdict(replace(TCFG, heads=2))}},
        "orders (2, 3), 2 heads; Billet's: (2, 3), 1",
    ),
    "another table": (
        None,
        {"record": Table.record(replace(TCFG, seed=1), 7)},
        "not the table its configuration gives",
    ),
    "event not a count": (
        None,
        {"record": Table.record(TCFG, 7) | {"event": "7"}},
        "not the table its configuration gives",
    ),
    "event a bool": (
        None,
        {"record": Table.record(TCFG, 7) | {"event": True}},
        "not the table its configuration gives",
    ),
    "no reads.npy": (None, {"counts": None}, "No such file"),
    "int32 counts": (None, {"counts": COUNTS.astype(np.int32)}, "expected int64 (4096,)"),
    "a row short": (None, {"counts": COUNTS[:-1]}, "expected int64 (4096,)"),
    "a read short": (None, {"counts": COUNTS - (np.arange(4096) == 0)}, NOT_ALL),
    "a negative count": (None, {"counts": _negative()}, NOT_ALL),
}


@pytest.mark.parametrize(("change", "kw", "words"), REFUSED.values(), ids=list(REFUSED))
def test_inputs_that_fail_a_check_are_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: Change,
    kw: dict[str, Any],
    words: str,
) -> None:
    """Each change is refused: the refusal names what failed, and nothing is written.

    A flipped bit leaves the layout valid (an .idx offset 31 becomes 30), so only the SHA-256s
    can refuse it; a flipped .bytes entry is refused by the scorer's own reader.
    """
    split = _store(tmp_path / "store")
    if change is not None:
        change(split.parent)
    run = _ckpt(tmp_path / "run", **kw)
    out = tmp_path / "out"
    assert rd.main(_argv(split, run, out)) == rd.EXIT_REFUSED
    assert words in capsys.readouterr().err
    assert not out.exists()


def test_an_incomplete_checkpoint_is_refused(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file whose size differs from state.json's makes the checkpoint incomplete."""
    split, run = inputs
    _append(run / "table" / "table.json", b" ")
    assert rd.main(_argv(split, run, tmp_path / "out")) == rd.EXIT_REFUSED
    assert "incomplete checkpoint" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["reads.npy", "reads.json"])
def test_an_out_folder_holding_either_file_is_refused_before_any_input_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    """So a run cannot replace earlier counts. The inputs here do not even exist."""
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text("earlier")
    none = tmp_path / "none"
    assert rd.main(_argv(none, none, out)) == rd.EXIT_REFUSED
    assert "already holds reads.npy or reads.json" in capsys.readouterr().err
    assert [p.name for p in out.iterdir()] == [name]
    assert (out / name).read_text() == "earlier"


def test_an_out_path_that_is_a_file_is_refused(
    inputs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The counts would have nowhere to go, so nothing is read or written."""
    split, run = inputs
    out = tmp_path / "out"
    out.write_text("a file")
    assert rd.main(_argv(split, run, out)) == rd.EXIT_REFUSED
    assert "a file, or already holds" in capsys.readouterr().err
    assert out.read_text() == "a file"
