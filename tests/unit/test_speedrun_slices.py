"""speedrun/slices.py: the 5-minute track's slices, cut from a data store.

The store is a fixture in the data store's layout with its own pin
(tests/support/slice_store.py).
`speedrun/slices.py` is loaded by path, through tests/support/byfile.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from akeratos import digest
from akeratos.data.store import Split
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout import HeldOut
from akeratos.speedrun import EVAL_SEQ_LEN, EVAL_TARGETS
from tests.support.byfile import load
from tests.support.slice_store import (
    ATTRIBUTION,
    DATASET,
    FINGERPRINT,
    SPDX,
    TOKENIZER,
    VOCAB,
    Store,
    sha256,
    write_store,
)

COPIED = [
    "heldout/000000.bin",
    "heldout/000000.bytes",
    "heldout/000000.idx",
    "heldout/manifest.json",
    "token_bytes.u16",
    "train/000000.bin",
    "train/000000.idx",
]
WRITTEN = ["README.md", "SHA256SUMS", "manifest.json", "train/manifest.json"]


slices = load("speedrun/slices.py", "speedrun_slices")


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return write_store(tmp_path / "store")


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def test_a_cut_copies_the_store_s_files_and_writes_the_rest(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    cut = slices.cut(store.path, out, store.pin)
    assert _files(out) == sorted(COPIED + WRITTEN)
    for rel in COPIED:
        assert (out / rel).read_bytes() == (store.path / rel).read_bytes(), rel
    assert cut.sizes == {rel: (out / rel).stat().st_size for rel in _files(out)}


def test_sha256sums_lists_every_file_but_itself_and_the_card(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    cut = slices.cut(store.path, out, store.pin)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    assert all(re.fullmatch(r"[0-9a-f]{64}  \S+", line) for line in lines)
    pairs = [line.split("  ", 1) for line in lines]
    assert [rel for _, rel in pairs] == sorted(set(COPIED + WRITTEN) - {"README.md", "SHA256SUMS"})
    assert all(sha256((out / rel).read_bytes()) == sha for sha, rel in pairs)
    assert cut.slices == {
        "sha256sums": digest.file_digest(out / "SHA256SUMS"),
        "store": FINGERPRINT[:16],
        "dataset": "@".join(DATASET),
        "tokenizer": "@".join(TOKENIZER),
    }


def test_the_train_manifest_keeps_the_first_shard_and_its_hashes(
    store: Store, tmp_path: Path
) -> None:
    out = tmp_path / "slices"
    slices.cut(store.path, out, store.pin)
    source = json.loads((store.path / "train" / "manifest.json").read_bytes())
    lists = {k for k, v in source.items() if isinstance(v, list) and k != "reserved_ids"}
    assert len(source["shards"]) == 2
    got = json.loads((out / "train" / "manifest.json").read_bytes())
    assert got == {k: v[:1] if k in lists else v for k, v in source.items()}
    assert Split(out / "train", vocab=VOCAB).tokens == source["tokens"][0]


def test_the_root_manifest_names_the_source_and_its_licence(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    slices.cut(store.path, out, store.pin)
    assert json.loads((out / "manifest.json").read_bytes()) == {
        "schema": "metalforge.speedrun-slices/v1",
        "source": {"store": FINGERPRINT[:16], "manifest_sha256": store.pin},
        "dataset": "@".join(DATASET),
        "tokenizer": "@".join(TOKENIZER),
        "spdx": SPDX,
        "attribution": ATTRIBUTION,
        "outputs": {"token_bytes_sha256": sha256(store.tb.tobytes())},
    }


def test_the_card(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    slices.cut(store.path, out, store.pin)
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: odc-by\n---\n")
    assert f"https://huggingface.co/datasets/{DATASET[0]}/tree/{DATASET[1]}" in card
    assert f"https://huggingface.co/{TOKENIZER[0]}/tree/{TOKENIZER[1]}" in card
    assert "shasum -a 256 -c SHA256SUMS" in card
    assert store.pin in card
    assert card.rstrip().endswith(ATTRIBUTION)


def test_the_scored_set_is_the_whole_split_when_it_is_smaller(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    cut = slices.cut(store.path, out, store.pin)
    targets = sum(len(d) for d in store.heldout)
    assert targets < EVAL_TARGETS
    assert cut.heldout == {
        "targets": targets,
        "docs": len(store.heldout),
        "bytes": sum(int(store.tb[d].sum()) for d in store.heldout),
    }
    held = HeldOut(out / "heldout", EVAL_SEQ_LEN, EVAL_TARGETS, doc_mask=False, vocab=VOCAB)
    assert cut.heldout == {"targets": held.targets, "docs": held.docs, "bytes": held.bytes}


def test_the_scored_set_stops_after_eval_targets(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 150
    monkeypatch.setattr(slices, "EVAL_TARGETS", cap)
    cut = slices.cut(store.path, tmp_path / "slices", store.pin)
    n = next(k for k in range(len(store.heldout)) if sum(map(len, store.heldout[:k])) >= cap)
    docs = store.heldout[:n]
    assert cut.heldout == {
        "targets": sum(len(d) for d in docs),
        "docs": n,
        "bytes": sum(int(store.tb[d].sum()) for d in docs),
    }


def test_the_same_store_cuts_the_same_bytes(store: Store, tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    slices.cut(store.path, a, store.pin)
    slices.cut(store.path, b, store.pin)
    assert _files(a) == _files(b)
    assert all((a / rel).read_bytes() == (b / rel).read_bytes() for rel in _files(a))


@pytest.mark.parametrize(
    "rel",
    [
        "manifest.json",
        "train/manifest.json",
        "heldout/manifest.json",
        "token_bytes.u16",
        "train/000000.bin",
        "train/000000.idx",
        "heldout/000000.bin",
        "heldout/000000.idx",
        "heldout/000000.bytes",
    ],
)
def test_a_file_that_differs_from_its_pin_or_record_is_refused(
    store: Store, tmp_path: Path, rel: str
) -> None:
    path = store.path / rel
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    out = tmp_path / "slices"
    with pytest.raises(DataMismatchError, match=re.escape(f"{path}: SHA-256")):
        slices.cut(store.path, out, store.pin)
    assert not out.exists()


def test_a_bytes_file_the_store_does_not_list_is_refused(store: Store, tmp_path: Path) -> None:
    sums = store.path / "SHA256SUMS"
    lines = [ln for ln in sums.read_text().splitlines() if not ln.endswith("heldout/000000.bytes")]
    sums.write_text("".join(f"{ln}\n" for ln in lines))
    out = tmp_path / "slices"
    with pytest.raises(DataMismatchError, match=re.escape("does not list heldout/000000.bytes")):
        slices.cut(store.path, out, store.pin)
    assert not out.exists()


def test_an_out_that_exists_is_refused(store: Store, tmp_path: Path) -> None:
    out = tmp_path / "slices"
    out.mkdir()
    with pytest.raises(ConfigError, match="exists"):
        slices.cut(store.path, out, store.pin)
    assert not any(out.iterdir())


def test_main_prints_the_sizes_and_the_pins(
    store: Store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(slices, "STORE_MANIFEST_SHA256", store.pin)
    out = tmp_path / "slices"
    assert slices.main(["--store", str(store.path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    head, pins = text.split("SLICES = ", 1)
    pin, counts = pins.split("HELDOUT = ", 1)
    sizes = {rel: (out / rel).stat().st_size for rel in _files(out)}
    total = f"{len(sizes)} files, {sum(sizes.values())} bytes"
    assert head.splitlines() == [f"{rel} {size}" for rel, size in sizes.items()] + [total]
    assert json.loads(pin)["sha256sums"] == digest.file_digest(out / "SHA256SUMS")
    held = HeldOut(out / "heldout", EVAL_SEQ_LEN, EVAL_TARGETS, doc_mask=False, vocab=VOCAB)
    assert json.loads(counts) == {"targets": held.targets, "docs": held.docs, "bytes": held.bytes}
    assert str(tmp_path) not in text


def test_main_exit_codes(store: Store, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "slices"
    argv = ["--store", str(store.path), "--out", str(out)]
    assert slices.main(argv) == 22  # the fixture is not the pinned store
    assert f"{store.path / 'manifest.json'}: SHA-256" in capsys.readouterr().err
    assert not out.exists()
    out.mkdir()
    assert slices.main(argv) == 78
    assert "exists" in capsys.readouterr().err
    assert slices.main(["--store", str(tmp_path / "none"), "--out", str(tmp_path / "new")]) == 78
    assert not (tmp_path / "new").exists()
