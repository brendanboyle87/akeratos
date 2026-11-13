"""The speedrun's receipt checker and records page in `speedrun/records.py`.

records.py is loaded by path. tests/fixtures/speedrun/ holds one valid claim as `bake speedrun`
writes it: five runs of a made-up training file on the M4 Max laptop class, at a commit on main,
with timed steps of about 25 s so that each log is 12 lines. `check` accepts it and refuses each
broken copy of it, made in a temporary folder, for its fault. The commit rule also runs on a
temporary repository. `bake speedrun`'s option check takes and refuses the same claims as
`check`. K, m and t follow from s, the choice of K and the rounding up included. A test that
depends on the rules' numbers sets them, so setting s changes
none of these tests: `unset` sets the numbers records.py holds while s is not set (K = 5, with
m, t and the reference unset), and the page's ranking and verification run with numbers set
here, at m and t exactly too. A claim of more runs than K scores its seeds 0 to K - 1.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from akeratos import digest, speedrun
from akeratos.errors import ConfigError
from akeratos.macops import Hardware
from tests.fakes.clock import FakeClock
from tests.support.byfile import ROOT, load
from tests.support.speedrun_cut import OTHER_MAIN, fixture_commits_on_main

records = load("speedrun/records.py", "speedrun_records_claims")

FIXTURES = ROOT / "tests" / "fixtures" / "speedrun"
CLASS = "m4-max-40c-48gb-macos26-laptop"
CLAIM = f"{CLASS}/2026-11-08-BhodiSea"  # its mean is 1.27378
SEED2 = f"{CLAIM}/seed-2"
UNKNOWN = "0123456789abcdef0123456789abcdef01234567"  # no commit of this repository
STUDIO = Hardware("Apple M4 Max", 40, "applegpu_g16s", 48, "26.6.2", "desktop", "Mac16,9")
STALE = "not the page the {} give; write it with uv run python speedrun/records.py --write"
CLAIM_PROBLEMS = ("person is not a GitHub handle", "claim is not YYYY-MM-DD-<person>")

Change = Callable[[dict[str, Any]], None]


@pytest.fixture(autouse=True)
def _commits_on_main(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_commits_on_main(records, monkeypatch)


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """records.py's numbers while s is not set: K = 5, with m, t and the reference unset."""
    for name, value in (("K", 5), ("MARGIN", None), ("TOLERANCE", None), ("REFERENCE", None)):
        monkeypatch.setattr(records, name, value)


def copy(tmp_path: Path) -> Path:
    """The fixture claim, copied into a results folder of its own."""
    root = tmp_path / "speedrun"
    shutil.copytree(FIXTURES, root)
    return root


def setting(where: str, value: object) -> Change:
    """A change of the receipt field at the dotted path `where`."""
    *parents, leaf = where.split(".")

    def apply(r: dict[str, Any]) -> None:
        for key in parents:
            r = r[key]
        r[leaf] = value

    return apply


def edit(root: Path, claim: str, change: Change, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    """Apply `change` to the receipts of `claim`'s `seeds`, written back in the pretty profile."""
    for seed in seeds:
        path = root / claim / f"seed-{seed}" / "receipt.json"
        r = json.loads(path.read_text(encoding="utf-8"))
        change(r)
        path.write_text(digest.pretty(r), encoding="utf-8")


def add_claim(
    root: Path,
    person: str,
    day: str,
    rerun_of: str | None = None,
    hardware: Hardware | None = None,
    runs_on: str | None = None,
    **fields: object,
) -> str:
    """A copy of the fixture claim made another person's claim, or rerun; `fields` replace any.

    Its runs move to the UTC day `runs_on`, by default `day`, at the fixture's times of day,
    unless that day is no date.
    """
    cls = CLASS if hardware is None else hardware.key()
    key = f"{cls}/{day}-{person}"
    shutil.copytree(FIXTURES / CLAIM, root / key)
    runs_on = day if runs_on is None else runs_on

    def change(r: dict[str, Any]) -> None:
        try:
            date.fromisoformat(runs_on)
        except ValueError:
            pass  # a claim that names no date is refused for its name
        else:
            r |= {k: runs_on + r[k][10:] for k in ("utc_start", "utc_end")}
        r |= {"person": person, "claim": f"{day}-{person}", "rerun_of": rerun_of}
        r |= {"kind": "record" if rerun_of is None else "rerun", **fields}
        if hardware is not None:
            r["hardware"] = hardware.describe()

    edit(root, key, change)
    return key


def on_seed_2(where: str, value: object) -> Callable[[Path], None]:
    """A change of one field of seed 2's receipt."""
    return lambda root: edit(root, CLAIM, setting(where, value), (2,))


def a_step_after_the_budget(root: Path) -> None:
    """Seed 2's log goes on for a step after its timed total reached the budget."""
    with (root / SEED2 / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"step": 12, "step_s": 25.2, "warmup": false, "loss": 6.7, "progress": 1.0}\n')


def a_log_without_its_last_step(root: Path) -> None:
    """Seed 2's log stops after step 10, at 277.2 s, under the budget."""
    path = root / SEED2 / "metrics.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")


def a_step_of_a_day(root: Path) -> None:
    """Seed 2's first step takes a day."""
    path = root / SEED2 / "metrics.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[0] = json.dumps(json.loads(lines[0]) | {"step_s": 86_400.0}) + "\n"
    path.write_text("".join(lines), encoding="utf-8")


def a_log_out_of_order(root: Path) -> None:
    """Seed 2's log has its steps 3 and 4 swapped."""
    path = root / SEED2 / "metrics.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[3], lines[4] = lines[4], lines[3]
    path.write_text("".join(lines), encoding="utf-8")


def not_pretty(root: Path) -> None:
    """Seed 2's receipt written as compact JSON."""
    path = root / SEED2 / "receipt.json"
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8"))), encoding="utf-8")


def a_lone_surrogate(root: Path) -> None:
    """Seed 2's AI system holds a lone surrogate, which only a JSON escape can write."""
    path = root / SEED2 / "receipt.json"
    r = json.loads(path.read_text(encoding="utf-8"))
    r["ai_system"] = "Claude Code \ud800"
    path.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n", encoding="utf-8")


R, M = f"{SEED2}/receipt.json", f"{SEED2}/metrics.jsonl"
SEEDS = "not 0 to J - 1 with J at least 5"
STEP = "a step_s above 0 and under a day"
FIRST_RUN = "the UTC date of its first run"
FAULTS: dict[str, tuple[Callable[[Path], object], list[str]]] = {
    "wrong pins": (
        on_seed_2("data.sha256sums", "sha256:" + "0" * 64),
        [f"{R}: data is not the track's slices"],
    ),
    "other held-out counts": (
        on_seed_2("heldout.targets", 1_048_847),
        [f"{R}: heldout is not the track's scored set"],
    ),
    "a 3 s budget": (
        on_seed_2("rules.budget_s", 3.0),
        [
            f"{R}: rules are not the track's",
            f"{M}: the timed total reached the budget before step 11",
        ],
    ),
    "a practice run": (
        on_seed_2("practice", True),
        [f"{R}: practice is not false: a practice run is never a record"],
    ),
    "source and hash disagree": (
        on_seed_2("train_py.source", "ROWS = 8\n"),
        [f"{R}: train_py.sha256 is not the SHA-256 of train_py.source"],
    ),
    "timed_s over its log": (
        on_seed_2("timed_s", 310.0),
        [f"{M}: timed_s is not the timed total its log gives"],
    ),
    "a step after the budget": (
        a_step_after_the_budget,
        [f"{M}: the timed total reached the budget before step 12"],
    ),
    "a log that stops short of the budget": (
        a_log_without_its_last_step,
        [
            f"{M}: steps is not the number of lines in metrics.jsonl",
            f"{M}: the timed total in metrics.jsonl is under the budget",
            f"{M}: timed_s is not the timed total its log gives",
        ],
    ),
    "warmup_s not its log's": (
        on_seed_2("warmup_s", 8.0),
        [f"{M}: warmup_s is not the warm-up steps' time in metrics.jsonl"],
    ),
    "a step of a day": (a_step_of_a_day, [f"{M}: line 1 is not step 0, warmup true, {STEP}"]),
    "a score of 0": (on_seed_2("bpb", 0.0), [f"{R}: bpb is not above 0"]),
    "a mean that is not finite": (
        lambda root: edit(root, CLAIM, setting("bpb", 1e308)),
        [f"{CLAIM}: its mean bits per byte is not finite"],
    ),
    "steps not its log's": (
        lambda root: edit(root, CLAIM, setting("steps", 11), (2,)),
        [
            f"{M}: steps is not the number of lines in metrics.jsonl",
            f"{M}: tokens is not steps x rows x seq_len",
        ],
    ),
    "a log out of order": (
        a_log_out_of_order,
        [f"{M}: line 4 is not step 3, warmup true, {STEP}"],
    ),
    "a missing seed": (
        lambda root: shutil.rmtree(root / SEED2),
        [f"{CLAIM}: its seeds are [0, 1, 3, 4], {SEEDS}"],
    ),
    "fewer than K seeds": (
        lambda root: shutil.rmtree(root / CLAIM / "seed-4"),
        [f"{CLAIM}: its seeds are [0, 1, 2, 3], {SEEDS}"],
    ),
    "mixed commits": (on_seed_2("commit", OTHER_MAIN), [f"{CLAIM}: its runs differ in commit"]),
    "mixed AI systems": (
        on_seed_2("ai_system", "none"),
        [f"{CLAIM}: its runs differ in ai_system"],
    ),
    "a commit not on main": (
        lambda root: edit(root, CLAIM, setting("commit", UNKNOWN)),
        [f"{CLAIM}/seed-{s}/receipt.json: commit is not on origin/main" for s in range(5)],
    ),
    "a short commit": (on_seed_2("commit", "d92b3d39"), [f"{R}: commit is not 40 hex digits"]),
    "a rerun by the same person": (
        lambda root: add_claim(root, "BhodiSea", "2026-11-12", CLAIM),
        [f"{CLASS}/2026-11-12-BhodiSea: a rerun is by another person than the claim it reruns"],
    ),
    "a rerun of other code": (
        lambda root: add_claim(root, "rerunner", "2026-11-12", CLAIM, commit=OTHER_MAIN),
        [
            f"{CLASS}/2026-11-12-rerunner: a rerun runs the commit and training file of the claim it reruns"
        ],
    ),
    "a rerun of no claim": (
        lambda root: add_claim(root, "rerunner", "2026-11-12", f"{CLASS}/2026-11-01-nobody"),
        [f"{CLASS}/2026-11-12-rerunner: rerun_of names no record claim of its class"],
    ),
    "a rerun of another class": (
        lambda root: add_claim(root, "rerunner", "2026-11-12", CLAIM, STUDIO),
        [f"{STUDIO.key()}/2026-11-12-rerunner: rerun_of names no record claim of its class"],
    ),
    "a rerun without kind": (
        on_seed_2("rerun_of", CLAIM),
        [f"{R}: kind is not rerun"],
    ),
    "a claim dated before its first run": (
        lambda root: add_claim(root, "runner-a", "2026-11-07", runs_on="2026-11-08"),
        [f"{CLASS}/2026-11-07-runner-a: its date is not 2026-11-08, {FIRST_RUN}"],
    ),
    "a claim dated after its first run": (  # such as the date east of UTC
        lambda root: add_claim(root, "runner-a", "2026-11-09", runs_on="2026-11-08"),
        [f"{CLASS}/2026-11-09-runner-a: its date is not 2026-11-08, {FIRST_RUN}"],
    ),
    "a run of seed 2 before the claim's date": (
        on_seed_2("utc_start", "2026-11-07T23:50:00Z"),
        [f"{CLAIM}: its date is not 2026-11-07, {FIRST_RUN}"],
    ),
    "a claim of another person": (
        on_seed_2("person", "rerunner"),
        [f"{R}: claim is not YYYY-MM-DD-<person>"],
    ),
    "a name, not a handle": (
        lambda root: add_claim(root, "Thomas_N", "2026-11-12"),
        [
            f"{CLASS}/2026-11-12-Thomas_N/seed-{s}/receipt.json: person is not a GitHub handle"
            for s in range(5)
        ],
    ),
    "local time": (
        on_seed_2("utc_end", "2026-11-08T12:52:02+08:00"),
        [f"{R}: utc_start or utc_end is not in UTC"],
    ),
    "a class not its key": (
        on_seed_2("hardware.memory_gb", 64),
        [
            f"{R}: hardware.class is not m4-max-40c-64gb-macos26-laptop",
        ],
    ),
    "another seed's folder": (
        on_seed_2("seed", 7),
        [f"{R}: its folder is not {CLAIM}/seed-7"],
    ),
    "an unknown field": (on_seed_2("notes", "x"), [f"{R}: the receipt has an unknown field notes"]),
    "not pretty": (
        not_pretty,
        [f"{R}: not in the pretty profile that bake speedrun writes"],
    ),
    "a lone surrogate": (
        a_lone_surrogate,
        [f"{R}: not in the pretty profile that bake speedrun writes"],
    ),
    "no log": (
        lambda root: (root / M).unlink(),
        [f"{M}: missing"],
    ),
    "a stray file": (
        lambda root: (root / SEED2 / "notes.txt").write_text("x", encoding="utf-8"),
        [f"{SEED2}/notes.txt: only receipt.json and metrics.jsonl belong here"],
    ),
    "a stray file beside the claims": (
        lambda root: (root / CLASS / "notes.txt").write_text("x", encoding="utf-8"),
        [f"{CLASS}/notes.txt: only folders belong here"],
    ),
}


def test_the_fixture_claim_checks_out() -> None:
    assert records.check(FIXTURES, page=False) == []


@pytest.mark.usefixtures("unset")  # "a missing seed" and "fewer than K seeds" need K = 5
@pytest.mark.parametrize("fault", list(FAULTS))
def test_a_claim_with_one_fault_is_refused_for_it(tmp_path: Path, fault: str) -> None:
    make, problems = FAULTS[fault]
    root = copy(tmp_path)
    make(root)
    assert records.check(root, page=False) == problems


def test_a_rerun_by_another_person_at_the_same_code_checks_out(tmp_path: Path) -> None:
    root = copy(tmp_path)
    add_claim(root, "rerunner", "2026-11-12", CLAIM, bpb=1.2801)
    assert records.check(root, page=False) == []


def _driver_takes(claim: str, person: str) -> bool:
    """Whether `bake speedrun`'s option check takes `--claim` and `--person`; the rest are valid."""
    o = speedrun.Options(
        data=Path("slices"),
        train=Path("speedrun/train.py"),
        seed=0,
        out=None,
        category="no-table",
        claim=claim,
        person=person,
        ai_system="Claude Code",
        rerun_of=None,
        hardware=STUDIO,
        conditions=lambda: {},
        clock=FakeClock(),
    )
    try:
        speedrun._check(o)
    except ConfigError:
        return False
    return True


@pytest.mark.parametrize(
    ("day", "person", "takes"),
    [
        ("2026-11-12", "rerunner", True),
        ("2026-11-12", "a", True),
        ("2026-11-12", "a-" * 19 + "b", True),  # 39 characters
        ("2028-02-29", "Bhodi-Sea", True),  # a leap day
        ("2026-11-12", "Bhodi--Sea", False),  # two hyphens in a row
        ("2026-11-12", "-BhodiSea", False),
        ("2026-11-12", "BhodiSea-", False),
        ("2026-11-12", "a" * 40, False),
        ("2026-02-30", "BhodiSea", False),  # no such date
        ("2026-02-29", "BhodiSea", False),  # not a leap year
        ("2026-13-01", "BhodiSea", False),
        ("2026-11-12", "Bhodi_Sea", False),  # not a GitHub handle
        ("2026-11-12", "@BhodiSea", False),
    ],
)
def test_bake_speedrun_takes_the_claims_the_checker_takes(
    tmp_path: Path, day: str, person: str, takes: bool
) -> None:
    # The driver copies bench.HANDLE and checks the date itself, since it does not import bench.
    # If the two differed, the checker could refuse a claim only after its runs.
    root = tmp_path / "speedrun"
    add_claim(root, person, day)
    problems = records.check(root, page=False)
    assert all(p.endswith(CLAIM_PROBLEMS) for p in problems)
    assert (_driver_takes(f"{day}-{person}", person), problems == []) == (takes, takes)


def _git(repo: Path, *args: str) -> str:
    env = os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    who = ("-c", "user.name=test", "-c", "user.email=test@example.invalid")
    proc = subprocess.run(
        ["git", "-C", str(repo), *who, *args], capture_output=True, text=True, check=True, env=env
    )
    return proc.stdout.strip()


def test_a_commit_must_be_an_ancestor_of_origin_main(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "on main")
    on_main = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", on_main)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "after main")
    after = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(records, "REPO", repo)
    root = copy(tmp_path)
    edit(root, CLAIM, setting("commit", on_main))
    assert records.check(root, page=False) == []
    edit(root, CLAIM, setting("commit", after))
    assert records.check(root, page=False) == [
        f"{CLAIM}/seed-{s}/receipt.json: commit is not on origin/main" for s in range(5)
    ]


@pytest.mark.parametrize(
    ("s", "k_m_t"),
    [
        (None, (5, None, None)),  # s is not set
        (0.001, (3, 0.002, 0.003)),  # t, 0.00245, rounds up
        (0.002, (3, 0.004, 0.005)),  # at 0.002 a claim needs 3 runs; m, 0.00327, rounds up
        (0.0020001, (5, 0.003, 0.004)),  # above 0.002 it needs 5
        (0.005, (5, 0.007, 0.01)),  # m, 0.00632, and t, 0.00949, round up
    ],
)
def test_k_m_and_t_follow_from_s(
    s: float | None, k_m_t: tuple[int, float | None, float | None]
) -> None:
    assert records.numbers(s) == k_m_t


@pytest.mark.usefixtures("unset")
def test_with_no_claims_the_page_says_so_and_the_rules_are_not_set(tmp_path: Path) -> None:
    assert records.render(tmp_path) == (
        f"{records.RECORDS_INTRO}\nThe rules' numbers are not set: the runs a claim needs (K, 5 "
        "for now), the margin m by which a new record must beat the standing one, and the "
        "tolerance t within which another person's rerun verifies a claim. Until they are set, "
        "no claim takes a record.\n\nNo claims yet.\n"
    )


def test_the_page_ranks_by_the_margin_and_verifies_within_the_tolerance(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(records, "K", 5)
    monkeypatch.setattr(records, "MARGIN", 0.002)
    monkeypatch.setattr(records, "TOLERANCE", 0.003)
    monkeypatch.setattr(records, "REFERENCE", CLAIM)
    root = copy(tmp_path)  # the reference claim, No table, mean 1.27378
    close = add_claim(root, "runner-a", "2026-11-10", category="open", bpb=1.272)  # by 0.0018
    best = add_claim(root, "runner-b", "2026-11-11", category="open", bpb=1.2715)  # by 0.0023
    add_claim(root, "runner-c", "2026-11-12", best, bpb=1.274)  # within 0.003 of 1.2715
    add_claim(root, "runner-d", "2026-11-13", CLAIM, bpb=1.277)  # 0.0032 over 1.27378
    studio = add_claim(root, "runner-e", "2026-11-09", hardware=STUDIO, bpb=1.29)
    assert records.check(root, page=False) == []
    page = records.render(root)
    assert "A claim's score is the mean of its runs with seeds 0 to 4 (K = 5)." in page
    assert "more than m = 0.002" in page
    assert "plus t = 0.003." in page
    ref = (
        f"[2026-11-08]({CLAIM}/) | [BhodiSea](https://github.com/BhodiSea) | Claude Code "
        "| `d92b3d3` | reference |"
    )
    assert page[page.index("## Laptops") :] == "\n".join(
        [
            "## Laptops",
            "",
            f"### `{CLASS}`",
            "",
            "Open:",
            "",
            "|  | Mean bpb | Category | Date | Person | AI system | Commit | Status |",
            "|---|---|---|---|---|---|---|---|",
            f"| record | 1.2715 | Open | [2026-11-11]({best}/) | [runner-b](https://github.com/"
            "runner-b) | Claude Code | `d92b3d3` | verified by [2026-11-12-runner-c]"
            f"({CLASS}/2026-11-12-runner-c/) |",
            f"|  | 1.2720 | Open | [2026-11-10]({close}/) | [runner-a](https://github.com/"
            "runner-a) | Claude Code | `d92b3d3` | unverified |",
            f"|  | 1.2738 | No table | {ref}",
            "",
            "No table:",
            "",
            "|  | Mean bpb | Date | Person | AI system | Commit | Status |",
            "|---|---|---|---|---|---|---|",
            f"| record | 1.2738 | {ref}",
            "",
            "## Desktops",
            "",
            f"### `{STUDIO.key()}`",
            "",
            "Open:",
            "",
            "|  | Mean bpb | Category | Date | Person | AI system | Commit | Status |",
            "|---|---|---|---|---|---|---|---|",
            f"| record | 1.2900 | No table | [2026-11-09]({studio}/) | [runner-e](https://github.com/"
            "runner-e) | Claude Code | `d92b3d3` | unverified |",
            "",
            "No table:",
            "",
            "|  | Mean bpb | Date | Person | AI system | Commit | Status |",
            "|---|---|---|---|---|---|---|",
            f"| record | 1.2900 | [2026-11-09]({studio}/) | [runner-e](https://github.com/"
            "runner-e) | Claude Code | `d92b3d3` | unverified |",
            "",
        ]
    )


def test_the_margin_is_strict_and_the_tolerance_is_not(tmp_path: Path, monkeypatch) -> None:
    # 0.25, 1.25, 1.5 and 1.75 are exact in binary, so each mean below meets its bound exactly.
    for name, value in (("K", 5), ("MARGIN", 0.25), ("TOLERANCE", 0.25)):
        monkeypatch.setattr(records, name, value)
    root = copy(tmp_path)
    edit(root, CLAIM, setting("bpb", 1.5))
    add_claim(root, "runner-a", "2026-11-10", bpb=1.25)  # beats 1.5 by exactly m: no record
    add_claim(root, "runner-b", "2026-11-12", CLAIM, bpb=1.75)  # 1.5 plus exactly t: it verifies
    assert records.check(root, page=False) == []
    page = records.render(root)
    assert page.count("| record | 1.5000 |") == 2  # in Open and in No table
    assert f"verified by [2026-11-12-runner-b]({CLASS}/2026-11-12-runner-b/)" in page


def test_a_claim_of_more_runs_than_k_scores_seeds_0_to_k_minus_1(
    tmp_path: Path, monkeypatch
) -> None:
    for name, value in (("K", 3), ("MARGIN", 0.002), ("TOLERANCE", 0.003)):
        monkeypatch.setattr(records, name, value)
    root = copy(tmp_path)  # five runs; seeds 0 to 2 score 1.2731, 1.2748 and 1.2719
    assert records.check(root, page=False) == []
    assert "| record | 1.2733 |" in records.render(root)  # not 1.2738, the mean of all five


def test_claims_of_one_day_take_the_record_in_the_order_of_their_first_runs(
    tmp_path: Path, monkeypatch
) -> None:
    for name, value in (("K", 5), ("MARGIN", 0.002), ("TOLERANCE", 0.003)):
        monkeypatch.setattr(records, name, value)
    root = copy(tmp_path)  # mean 1.27378
    add_claim(root, "runner-a", "2026-11-10", bpb=1.269)  # its runs from 04:30
    first = add_claim(root, "runner-b", "2026-11-10", bpb=1.27, utc_start="2026-11-10T00:00:00Z")
    assert records.check(root, page=False) == []
    page = records.render(root)
    # runner-b's runs start first, so it takes the record, and runner-a beats it by only 0.001.
    # In handle order, runner-a would take the record at 1.2690 instead.
    assert page.count(f"| record | 1.2700 | [2026-11-10]({first}/) |") == 1  # No table
    assert page.count(f"| record | 1.2700 | No table | [2026-11-10]({first}/) |") == 1  # Open


def test_a_claim_named_before_a_record_set_ahead_of_its_runs_is_refused(
    tmp_path: Path, monkeypatch
) -> None:
    # Named for a day before runner-b's record, runner-c's runs of a later day would come
    # ahead of that record and take it.
    for name, value in (("K", 5), ("MARGIN", 0.002), ("TOLERANCE", 0.003)):
        monkeypatch.setattr(records, name, value)
    root = copy(tmp_path)
    add_claim(root, "runner-b", "2026-11-20", bpb=1.27)
    named = add_claim(root, "runner-c", "2026-12-01", bpb=1.269)  # beats 1.2700 by only 0.001
    assert records.check(root, page=False) == []
    assert records.render(root).count("| record | 1.2700 |") == 2
    shutil.rmtree(root / named)
    early = add_claim(root, "runner-c", "2026-11-19", runs_on="2026-12-01", bpb=1.269)
    assert records.check(root, page=False) == [f"{early}: its date is not 2026-12-01, {FIRST_RUN}"]


def test_the_page_shows_an_ai_system_on_one_line(tmp_path: Path) -> None:
    root = copy(tmp_path)
    add_claim(root, "runner-a", "2026-11-08", ai_system="Claude\nCode | v2")
    assert records.check(root, page=False) == []
    assert " | Claude Code \\| v2 | " in records.render(root)


@pytest.mark.usefixtures("unset")
def test_while_the_margin_is_not_set_no_claim_is_a_record(tmp_path: Path) -> None:
    root = copy(tmp_path)
    add_claim(root, "runner-a", "2026-11-10", category="open", bpb=1.25)
    page = records.render(root)
    assert "| record |" not in page
    assert page.count("| unverified |") == 3  # two claims in Open, one in No table


def test_check_and_write_cover_both_results_pages(tmp_path: Path, monkeypatch, capsys) -> None:
    speedrun, bench = copy(tmp_path), tmp_path / "bench"
    bench.mkdir()
    monkeypatch.setattr(records, "SPEEDRUN", speedrun)
    monkeypatch.setattr(records, "BENCH", bench)
    assert records.main(["--check"]) == 1  # no pages yet
    assert capsys.readouterr().err == (
        f"results/speedrun/README.md: {STALE.format('receipts')}\n"
        f"results/bench/README.md: {STALE.format('rows')}\n"
    )
    assert records.main(["--write"]) == 0
    page = speedrun / "README.md"
    assert page.read_text(encoding="utf-8") == records.render(speedrun)
    assert (bench / "README.md").read_text(encoding="utf-8") == records.render_bench(bench)
    assert records.main(["--check"]) == 0
    written = page.read_text(encoding="utf-8")
    edit(speedrun, CLAIM, setting("practice", True), (0,))
    assert records.main(["--write"]) == 1  # refused, and the page is left as it was
    assert capsys.readouterr().err == (
        f"results/speedrun/{CLAIM}/seed-0/receipt.json: practice is not false: a practice run "
        "is never a record\n"
    )
    assert page.read_text(encoding="utf-8") == written
