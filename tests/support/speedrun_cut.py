"""Slices cut from the fixture store, and a run's options over them, for the speedrun's tests.

`cut` writes a fixture store in scripts/data/prepare_store.py's layout and cuts it with
`speedrun/slices.py`, so the slices are the real cutter's output with the fixture's own pins.
The held-out pins are for the tests' small evaluation values. `options` fills a run's options
for those slices, with a stub of the fixed-code check that returns a commit on main; the run's
train file is `<root>/train.py` and its `--out` `<root>/out`.
"""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import pytest

from akeratos import speedrun
from akeratos.clock import Clock
from akeratos.eval.heldout import HeldOut
from akeratos.macops import Hardware
from tests.support.byfile import load
from tests.support.slice_store import VOCAB, write_store

COMMIT = "d92b3d39fae05becf33a6e09b45f5748955c25a4"  # a commit on main
OTHER_MAIN = "2a6596e0a2252ad55960c342089621b54a62751a"  # another commit on main
HARDWARE = Hardware("Apple M4 Max", 40, "applegpu_g16s", 48, "26.1", "laptop", "Mac16,5")
TRAIN_DOCS = 1_200  # the packer buffers 1,000 documents before it fills a row

slices = load("speedrun/slices.py", "speedrun_slices")


def fixture_commits_on_main(records: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the loaded `records` module count COMMIT and OTHER_MAIN as on main.

    Git knows those two commits only in a clone of Akeratos with its history, so without this
    the checker would refuse the fixture receipts in any other copy of the tree. Every other
    commit still goes to git.
    """
    on_main = records._on_main

    def fixture_or_git(repo: Path, main: str, commit: str) -> bool:
        if commit in (COMMIT, OTHER_MAIN):
            return True
        return on_main(repo, main, commit)

    monkeypatch.setattr(records, "_on_main", fixture_or_git)


class Cut(NamedTuple):
    """Cut slices: their folder, their `SLICES` pin and their scored set's counts."""

    path: Path
    pins: dict[str, str]
    heldout: dict[str, int]


def cut(root: Path, seq_len: int, targets: int) -> Cut:
    """Slices cut under `root` from a new fixture store, pinned for `seq_len` and `targets`."""
    store = write_store(root / "store", train_docs=TRAIN_DOCS)
    pins = slices.cut(store.path, root / "slices", store.pin).slices
    held = HeldOut(root / "slices" / "heldout", seq_len, targets, doc_mask=False, vocab=VOCAB)
    counts = {"targets": held.targets, "docs": held.docs, "bytes": held.bytes}
    return Cut(root / "slices", pins, counts)


def _conditions() -> Mapping[str, object]:
    return {"power": "ac", "thermal_level": 0}


def options(
    c: Cut, clock: Clock, root: Path, seq_len: int, targets: int, **changes: Any
) -> speedrun.Options:
    """A record run's options over `c`, its evaluation values the tests'; `changes` replace any."""
    o = speedrun.Options(
        data=c.path,
        train=root / "train.py",
        seed=0,
        out=root / "out",
        category="no-table",
        claim="2026-01-01-BhodiSea",  # the UTC day of a new `FakeClock`
        person="BhodiSea",
        ai_system="Claude Code",
        rerun_of=None,
        hardware=HARDWARE,
        conditions=_conditions,
        clock=clock,
        eval_seq_len=seq_len,
        eval_rows=2,
        eval_targets=targets,
        slices=c.pins,
        heldout=c.heldout,
        fixed_code=lambda: COMMIT,
    )
    return replace(o, **changes)
