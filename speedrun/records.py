"""The results pages and their checks: the speedrun's claims and the bench rows.

A speedrun claim is a folder `results/speedrun/<class>/<claim>/` with one folder per run,
`seed-<N>/`, holding the `receipt.json` and `metrics.jsonl` that `bake speedrun` wrote there.
A bench row is one `metalforge.bench-row/v1` JSON file in `results/bench/`, written by
`bake bench`. `check` and `check_row` hold claims and rows to their rules, and `render` and
`render_bench` give the two pages. Each page must equal a fresh render; with `--page-may-lag`,
the bench page may leave out rows it does not list yet.
"""

from __future__ import annotations

import argparse
import functools
import math
import re
import sys
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

from akeratos import bench, digest, speedrun
from akeratos.macops import AC, Hardware, run_command

ROOT = Path(__file__).resolve().parents[1]
SPEEDRUN = ROOT / "results" / "speedrun"
BENCH = ROOT / "results" / "bench"
PAGE = "README.md"
COMMAND = "uv run python speedrun/records.py"
# A record's commit must be on Akeratos's main.
REPO, MAIN = ROOT, "origin/main"

# The rules' numbers follow from S, the sample standard deviation of the held-out bits per byte
# of the ten reference runs: seeds 0 to 9, No table, on the M4 Max laptop. While S and the
# reference are None, the page shows the numbers as not set.
S: float | None = 0.0203  # rounded to 0.0001
# `<class>/<claim>` of the reference runs, "reference" on the page
REFERENCE: str | None = "m4-max-40c-48gb-macos26-laptop/2026-11-13-BhodiSea"
LOW_S = 0.002  # at or below it, a claim needs 3 runs, not 5


def _up(x: float) -> float:
    """`x` rounded up to 0.001."""
    return math.ceil(x * 1000) / 1000


def numbers(s: float | None) -> tuple[int, float | None, float | None]:
    """K, m and t from s; while s is not set, K is 5 and m and t are not set.

    K is 5, or 3 if s is at most `LOW_S`. m = 2 s sqrt(2 / K) and t = 3 s sqrt(2 / K), each
    rounded up to 0.001.
    """
    if s is None:
        return 5, None, None
    k = 3 if s <= LOW_S else 5
    spread = s * math.sqrt(2 / k)  # the s.d. of the difference of two means of K independent runs
    return k, _up(2 * spread), _up(3 * spread)


# K: a claim's score is the mean bits per byte of its runs with seeds 0 to K - 1. MARGIN, m: a
# new record beats the standing record's mean by more than m. TOLERANCE, t: another person's
# rerun verifies a claim if the rerun's mean is at most the claim's plus t.
K, MARGIN, TOLERANCE = numbers(S)

SHAPES = {bench.SHAPE.version: bench.SHAPE.describe()}  # each shape version a row may hold
ATTN_PATHS = ("fused", "banded")  # the paths a fresh run picks (`entry`)
FORMS = ("laptop", "desktop")
COMMIT = re.compile(r"[0-9a-f]{40}")
CLAIM = re.compile(r"(\d{4}-\d{2}-\d{2})-(.+)")  # the UTC date of the claim's first run, a handle
THERMAL_MAX = 4  # the highest thermal pressure level, sleeping
DAY_S = 86_400.0  # no step takes a day; under it, each step's time is exact in integer ns
READINGS = 2  # conditions are read at the start and at the end
NUMBER = (int, float)
OPTIONAL_STR = (str, type(None))
HARDWARE: dict[str, Any] = {
    "chip": str,
    "gpu_cores": int,
    "gpu_arch": str,
    "memory_gb": int,
    "macos": str,
    "form": str,
    "model": str,
    "class": str,
}
CONDITIONS: dict[str, Any] = {"power": list, "thermal_level": list}
ARM: dict[str, Any] = {"step_s": NUMBER, "tokens_per_s": NUMBER, "peak_bytes": int}
FIELDS: dict[str, Any] = {
    "schema": str,
    "person": str,
    "hardware": HARDWARE,
    "versions": {"metalforge": str, "commit": OPTIONAL_STR, "mlx": str, "python": str},
    "conditions": CONDITIONS,
    "shape": dict,
    "arms": {
        "off": ARM,
        "on": {**ARM, "rows_updated": NUMBER, "table_gbps_estimate": NUMBER},
    },
    "on_off_ratio": NUMBER,
    "seconds": NUMBER,
    "utc": str,
}
RECEIPT: dict[str, Any] = {
    "schema": str,
    "category": str,
    "claim": str,
    "kind": str,
    "rerun_of": OPTIONAL_STR,
    "person": str,
    "ai_system": str,
    "seed": int,
    "practice": bool,
    "commit": str,
    "metalforge": str,
    "mlx": str,
    "python": str,
    "train_py": {"sha256": str, "source": str},
    "hardware": HARDWARE,
    "conditions": CONDITIONS,
    "peak_bytes": int,
    "data": {"sha256sums": str, "store": str, "dataset": str, "tokenizer": str},
    "heldout": {"targets": int, "docs": int, "bytes": int, "seq_len": int, "doc_mask": bool},
    "rules": {"budget_s": NUMBER, "warmup_steps": int, "warmup_rule": str},
    "batch": {"rows": int, "seq_len": int},
    "steps": int,
    "tokens": int,
    "timed_s": NUMBER,
    "warmup_s": NUMBER,
    "build_s": NUMBER,
    "eval_s": NUMBER,
    "bpb": NUMBER,
    "utc_start": str,
    "utc_end": str,
}
LOG_LINE: dict[str, Any] = {
    "step": int,
    "step_s": NUMBER,
    "warmup": bool,
    "loss": NUMBER,
    "progress": NUMBER,
}
TRACK: dict[str, tuple[object, str]] = {  # what a record's receipt holds, as the track sets it
    "data": (dict(speedrun.SLICES), "data is not the track's slices"),
    "heldout": (
        {**speedrun.HELDOUT, "seq_len": speedrun.EVAL_SEQ_LEN, "doc_mask": False},
        "heldout is not the track's scored set",
    ),
    "rules": (
        {
            "budget_s": speedrun.BUDGET_S,
            "warmup_steps": speedrun.WARMUP_STEPS,
            "warmup_rule": speedrun.WARMUP_RULE,
        },
        "rules are not the track's",
    ),
    "practice": (False, "practice is not false: a practice run is never a record"),
}
RUN_FILES = ("receipt.json", "metrics.jsonl")
AGREE = ("commit", "train_py", "person", "ai_system", "category", "kind", "rerun_of")
CATEGORIES = {"open": "Open", "no-table": "No table"}
RECORDS_INTRO = (
    "# Speedrun records\n\n"
    "The claims on the 5-minute track of "
    "[the Mac speedrun](../../docs/speedrun.md): held-out bits per byte after "
    "300 seconds of step time, lower is better. Each claim is a folder here, `<class>/<claim>/`, "
    "holding the receipt and the step log of each of its runs. Claims rank within their hardware "
    "class, laptops apart from desktops, and No-table claims rank in Open as well. A claim takes "
    "a category's record only if its mean beats the standing record's by more than the margin m, "
    "and it is unverified until another person's rerun verifies it. "
    f"`{COMMAND} --write` generates this page from the receipts.\n"
)
BENCH_INTRO = (
    "# Bench rows\n\n"
    "What the n-gram table costs per training step on each Mac, for [the open problem on that "
    "cost](../../OPEN_PROBLEMS.md#the-tables-cost-on-each-mac): one row per hardware class and "
    "MLX version, each measured by `uv run bake bench` and kept as a JSON file in this folder, "
    "`<class>-mlx<version>.json`. The bench section of "
    "[the Mac speedrun page](../../docs/speedrun.md) says how to make a row and send it. "
    f"`{COMMAND} --write` generates this page from the rows.\n"
)
COLUMNS = (
    "Chip",
    "GPU cores",
    "Memory",
    "macOS",
    "Form",
    "MLX",
    "Step off (ms)",
    "Step on (ms)",
    "On/off",
    "Table GB/s (estimate)",
    "Peak on (GiB)",
    "Person",
)


class Claim(NamedTuple):
    """A claim whose runs check out: what the rerun check and the page need of it."""

    key: str  # `<class>/<claim>`
    form: str  # laptop or desktop
    category: str
    kind: str  # record or rerun
    rerun_of: str | None
    person: str
    ai_system: str
    commit: str
    train_sha: str
    mean: float  # the mean bits per byte of its runs with seeds 0 to K - 1
    start: datetime  # the earliest utc_start of its runs

    @property
    def date(self) -> str:
        """The UTC date of its first run, which its name gives (`_claim` checks it)."""
        return self.key.split("/")[1][:10]


def _cls(c: Claim) -> str:
    """The claim's hardware class, from its key."""
    return c.key.split("/")[0]


def _types(value: object, spec: object, where: str = "", top: str = "the row") -> list[str]:
    """Where `value` lacks a field of `spec`, has one more, or has a field of another type."""
    name = where or top
    if not isinstance(spec, dict):
        wanted = spec if isinstance(spec, tuple) else (spec,)
        ok = isinstance(value, wanted) and not (isinstance(value, bool) and bool not in wanted)
        return [] if ok else [f"{name} is not {' or '.join(t.__name__ for t in wanted)}"]
    if not isinstance(value, dict):
        return [f"{name} is not an object"]
    keys, fields = set(value), set(spec)
    problems = [f"{name} has no {k}" for k in sorted(fields - keys)]
    problems += [f"{name} has an unknown field {k}" for k in sorted(keys - fields)]
    for k in sorted(fields & keys):
        problems += _types(value[k], spec[k], f"{where}.{k}" if where else k, top)
    return problems


def _utc(text: str) -> bool:
    """True when `text` is an ISO 8601 time in UTC."""
    try:
        return datetime.fromisoformat(text).utcoffset() == timedelta(0)
    except ValueError:
        return False


def _hardware(row: dict[str, Any]) -> list[str]:
    """The form, and the class the hardware fields give."""
    fields = {k: v for k, v in row["hardware"].items() if k != "class"}
    if fields["form"] not in FORMS:
        return ["hardware.form is not laptop or desktop"]
    key = Hardware(**fields).key()
    return [] if row["hardware"]["class"] == key else [f"hardware.class is not {key}"]


def _files(folder: Path) -> list[Path]:
    """The files in `folder`, sorted, but for hidden ones such as .DS_Store."""
    return sorted(p for p in folder.iterdir() if not p.name.startswith("."))


def _pretty(data: bytes, value: object) -> bool:
    """Whether `data` is `value` in the pretty profile; no UTF-8 file holds a lone surrogate."""
    try:
        return data == digest.pretty_bytes(value)
    except ValueError:
        return False


# ---- the speedrun's claims ---------------------------------------------------------------------


def _handle(claim: str) -> str | None:
    """The handle in a claim named `YYYY-MM-DD-<handle>` with a real date, else None."""
    found = CLAIM.fullmatch(claim)
    if found is None:
        return None
    try:
        date.fromisoformat(found[1])
    except ValueError:
        return None
    return found[2]


@functools.cache
def _on_main(repo: Path, main: str, commit: str) -> bool:
    """Whether `commit` is an ancestor of `main` in `repo` (git exits 0)."""
    argv = ("git", "-C", str(repo), "merge-base", "--is-ancestor", commit, main)
    return run_command(argv) is not None


def _values(r: Mapping[str, Any]) -> list[str]:
    """The schema, the category and kind, who ran it and when, and a score above 0."""
    kind = "record" if r["rerun_of"] is None else "rerun"
    checks = (
        (r["schema"] == speedrun.SCHEMA, f"schema is not {speedrun.SCHEMA}"),
        (r["category"] in CATEGORIES, "category is not open or no-table"),
        (r["kind"] == kind, f"kind is not {kind}"),
        (bench.HANDLE.fullmatch(r["person"]) is not None, "person is not a GitHub handle"),
        (_handle(r["claim"]) == r["person"], "claim is not YYYY-MM-DD-<person>"),
        (r["ai_system"].strip() != "", "ai_system is empty"),
        (_utc(r["utc_start"]) and _utc(r["utc_end"]), "utc_start or utc_end is not in UTC"),
        (r["bpb"] > 0, "bpb is not above 0"),
    )
    return [problem for ok, problem in checks if not ok]


def _code(r: Mapping[str, Any]) -> list[str]:
    """The track's constants, the training file's hash and the commit."""
    problems = [problem for k, (want, problem) in TRACK.items() if r[k] != want]
    source = r["train_py"]["source"].encode("utf-8", "surrogatepass")
    if digest.digest_bytes(source) != r["train_py"]["sha256"]:
        problems.append("train_py.sha256 is not the SHA-256 of train_py.source")
    if COMMIT.fullmatch(r["commit"]) is None:
        problems.append("commit is not 40 hex digits")
    elif not _on_main(REPO, MAIN, r["commit"]):
        problems.append(f"commit is not on {MAIN}")
    return problems


def _names(r: Mapping[str, Any], folder: Path) -> list[str]:
    """The folders' names: the class, the claim and the seed."""
    want = f"{r['hardware']['class']}/{r['claim']}/seed-{r['seed']}"
    got = f"{folder.parent.parent.name}/{folder.parent.name}/{folder.name}"
    return [] if got == want else [f"its folder is not {want}"]


def _steps(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """The log's lines, each one step's, or the problem with the first line that is not."""
    lines: list[dict[str, Any]] = []
    for n, line in enumerate(data.splitlines()):
        try:
            step = digest.loads(line)
        except ValueError as e:
            return lines, [f"line {n + 1} is not strict JSON: {e}"]
        warmup = n < speedrun.WARMUP_STEPS
        if _types(step, LOG_LINE) or not isinstance(step, dict):
            return lines, [f"line {n + 1} does not hold exactly {', '.join(LOG_LINE)}"]
        if step["step"] != n or step["warmup"] != warmup or not 0 < step["step_s"] < DAY_S:
            flag = "true" if warmup else "false"
            return lines, [
                f"line {n + 1} is not step {n}, warmup {flag}, a step_s above 0 and under a day"
            ]
        lines.append(step)
    return lines, []


def _log(r: Mapping[str, Any], data: bytes) -> list[str]:
    """The receipt's numbers against its log, with the timed total replayed as the driver ran it.

    The replay takes the run's own budget, so a practice run's arithmetic checks out too; a
    record's budget is the track's (`TRACK`).
    """
    lines, problems = _steps(data)
    budget_s = r["rules"]["budget_s"]
    if problems or not (budget_s > 0 and math.isfinite(budget_s * 1e9)):
        return problems  # a budget the driver refuses fails the rules check already
    timing, budget = speedrun.Timing(), max(1, round(budget_s * 1e9))
    for n, step in enumerate(lines):
        if timing.charged_ns() >= budget:
            return [f"the timed total reached the budget before step {n}"]
        timing.add(round(step["step_s"] * 1e9))
    rows, seq_len = r["batch"]["rows"], r["batch"]["seq_len"]
    checks = (
        (r["steps"] == len(lines), "steps is not the number of lines in metrics.jsonl"),
        (r["tokens"] == r["steps"] * rows * seq_len, "tokens is not steps x rows x seq_len"),
        (timing.charged_ns() >= budget, "the timed total in metrics.jsonl is under the budget"),
        (r["timed_s"] == timing.charged_ns() / 1e9, "timed_s is not the timed total its log gives"),
        (
            r["warmup_s"] == sum(timing.steps[: speedrun.WARMUP_STEPS]) / 1e9,
            "warmup_s is not the warm-up steps' time in metrics.jsonl",
        ),
    )
    return [problem for ok, problem in checks if not ok]


def _run(folder: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """One run's receipt, when it holds every field, and the run's problems, each a sentence."""
    problems = [
        f"{p.name}: only {' and '.join(RUN_FILES)} belong here"
        for p in _files(folder)
        if p.name not in RUN_FILES
    ]
    missing = [f"{name}: missing" for name in RUN_FILES if not (folder / name).is_file()]
    if missing:
        return None, problems + missing
    data, log = (folder / "receipt.json").read_bytes(), (folder / "metrics.jsonl").read_bytes()
    try:
        r = digest.loads(data)
    except ValueError as e:
        return None, [*problems, f"receipt.json: not strict JSON: {e}"]
    found = _types(r, RECEIPT, top="the receipt")
    if found or not isinstance(r, dict):
        return None, problems + [f"receipt.json: {p}" for p in found]
    if not _pretty(data, r):
        problems.append("receipt.json: not in the pretty profile that bake speedrun writes")
    for check in (_values, _code, _hardware):
        problems += [f"receipt.json: {p}" for p in check(r)]
    problems += [f"receipt.json: {p}" for p in _names(r, folder)]
    problems += [f"metrics.jsonl: {p}" for p in _log(r, log)]
    return r, problems


def _subfolders(folder: Path, rel: str, problems: list[str], *allowed: str) -> list[Path]:
    """The folders in `folder`; any other file but those `allowed` is a problem."""
    files, what = _files(folder), " and ".join(("folders", *allowed))
    problems.extend(
        f"{rel}{p.name}: only {what} belong here"
        for p in files
        if not p.is_dir() and p.name not in allowed
    )
    return [p for p in files if p.is_dir()]


def _claim(key: str, runs: Mapping[int, Mapping[str, Any]]) -> tuple[Claim | None, list[str]]:
    """A claim from its runs by seed, or its problems: the seeds, what they share, the mean.

    The date in its name must be the UTC date of the earliest utc_start of its runs.
    """
    seeds = sorted(runs)
    problems = [
        f"{key}: its runs differ in {field}"
        for field in AGREE
        if len({repr(r[field]) for r in runs.values()}) > 1
    ]
    if seeds != list(range(len(seeds))) or len(seeds) < K:
        problems.insert(0, f"{key}: its seeds are {seeds}, not 0 to J - 1 with J at least {K}")
    starts = [datetime.fromisoformat(r["utc_start"]) for r in runs.values()]  # each in UTC
    day = min(starts).date().isoformat() if starts else None
    if day is not None and key.split("/")[1][:10] != day:
        problems.append(f"{key}: its date is not {day}, the UTC date of its first run")
    if problems:
        return None, problems
    mean = sum(runs[s]["bpb"] for s in range(K)) / K
    if not math.isfinite(mean):
        return None, [f"{key}: its mean bits per byte is not finite"]
    first = runs[0]
    claim = Claim(
        key,
        first["hardware"]["form"],
        first["category"],
        first["kind"],
        first["rerun_of"],
        first["person"],
        first["ai_system"],
        first["commit"],
        first["train_py"]["sha256"],
        mean,
        min(starts),
    )
    return claim, []


def _claims(results_dir: Path) -> tuple[dict[str, Claim], list[str]]:
    """Every claim in `results_dir` whose runs check out, by `<class>/<claim>`, and the problems."""
    claims: dict[str, Claim] = {}
    problems: list[str] = []
    for cls in _subfolders(results_dir, "", problems, PAGE):
        for folder in _subfolders(cls, f"{cls.name}/", problems):
            key, runs, bad = f"{cls.name}/{folder.name}", {}, False
            for seed in _subfolders(folder, f"{cls.name}/{folder.name}/", problems):
                r, found = _run(seed)
                problems += [f"{key}/{seed.name}/{p}" for p in found]
                bad = bad or bool(found)
                if r is not None:
                    runs[r["seed"]] = r
            claim, found = (None, []) if bad else _claim(key, runs)
            problems += found
            if claim is not None:
                claims[key] = claim
    return claims, problems


def _rerun_problem(c: Claim, claims: Mapping[str, Claim]) -> str | None:
    """Why rerun `c` does not stand, or None.

    A rerun names a record claim of its class, at the same commit and training file, by another
    person.
    """
    target = claims.get(c.rerun_of or "")
    if target is None or target.kind != "record" or _cls(target) != _cls(c):
        return f"{c.key}: rerun_of names no record claim of its class"
    if (target.commit, target.train_sha) != (c.commit, c.train_sha):
        return f"{c.key}: a rerun runs the commit and training file of the claim it reruns"
    if target.person == c.person:
        return f"{c.key}: a rerun is by another person than the claim it reruns"
    return None


def check(results_dir: Path, *, page: bool = True) -> list[str]:
    """Every problem with the claims in `results_dir` and, if they have none, with its page.

    Each problem is one line, naming its folder or file. A receipt must be strict JSON in the
    pretty profile, with every field and its type; `akeratos.speedrun`'s schema, slices, scored
    set and rules, and not a practice run; a person who is a GitHub handle, and a claim of a date
    and that handle; a training file whose SHA-256 is its source's; a commit on `origin/main`; and
    the class its hardware gives and the folders' names. Its log must give its numbers: one line
    per step, in order; tokens of the steps times the batch; and the timed total, replayed with
    the driver's own `Timing` from each step's time, under the budget before each step and at or
    over it after the last, equal to `timed_s`; no step may take a day. Its score must be above
    0. A claim's runs have seeds 0 to J - 1, J at least `K`, and agree on what ran and who ran it.
    """
    claims, problems = _claims(results_dir)
    for c in claims.values():
        found = _rerun_problem(c, claims) if c.kind == "rerun" else None
        problems += [] if found is None else [found]
    if page and not problems:
        written = results_dir / PAGE
        current = written.is_file() and written.read_text(encoding="utf-8") == render(results_dir)
        if not current:
            problems.append(
                f"{PAGE}: not the page the receipts give; write it with {COMMAND} --write"
            )
    return problems


def _record(claims: list[Claim]) -> Claim | None:
    """The standing record: by first run, a claim takes it only by beating it by more than m."""
    if MARGIN is None:
        return None
    record: Claim | None = None
    for c in sorted(claims, key=lambda c: (c.start, c.key)):
        if record is None or c.mean < record.mean - MARGIN:
            record = c
    return record


def _status(c: Claim, reruns: list[Claim]) -> str:
    """Verified by the first rerun within the tolerance, else reference or unverified."""
    if TOLERANCE is not None:
        for r in sorted(reruns, key=lambda r: (r.start, r.key)):
            if r.rerun_of == c.key and r.mean <= c.mean + TOLERANCE:
                return f"verified by [{r.key.split('/')[1]}]({r.key}/)"
    return "reference" if c.key == REFERENCE else "unverified"


def _table(claims: list[Claim], reruns: list[Claim], category: str) -> str:
    """One category's claims of one class: the record first, then the rest by mean."""
    label, mixed = CATEGORIES[category], category == "open"  # Open ranks No-table claims too
    if not claims:
        return f"{label}: no claims yet.\n"
    record = _record(claims)
    rest = sorted((c for c in claims if c != record), key=lambda c: (c.mean, c.start, c.key))
    head = ["", "Mean bpb", *(["Category"] if mixed else []), "Date", "Person", "AI system"]
    head += ["Commit", "Status"]
    lines = [f"{label}:", "", f"| {' | '.join(head)} |", f"|{'---|' * len(head)}"]
    for c in ([record] if record else []) + rest:
        cells = [
            "record" if c == record else "",
            f"{c.mean:.4f}",
            *([CATEGORIES[c.category]] if mixed else []),
            f"[{c.date}]({c.key}/)",
            f"[{c.person}](https://github.com/{c.person})",
            " ".join(c.ai_system.split()).replace("|", "\\|"),  # one line, whatever its text
            f"`{c.commit[:7]}`",
            _status(c, reruns),
        ]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def _rules() -> str:
    """The rules' numbers, or that they are not set."""
    if MARGIN is None or TOLERANCE is None:
        return (
            f"The rules' numbers are not set: the runs a claim needs (K, {K} for now), the "
            "margin m by which a new record must beat the standing one, and the tolerance t "
            "within which another person's rerun verifies a claim. Until they are set, no claim "
            "takes a record.\n"
        )
    return (
        f"A claim's score is the mean of its runs with seeds 0 to {K - 1} (K = {K}). A new record "
        f"must beat the standing record's mean by more than m = {MARGIN:.3f}, and another person's "
        f"rerun verifies a claim if the rerun's mean is at most the claim's plus t = "
        f"{TOLERANCE:.3f}.\n"
    )


def render(results_dir: Path) -> str:
    """The records page of the claims in `results_dir`, laptops first, each class by name.

    Its claims must check out (`check`). Each class's claims are ranked per category.
    """
    claims = list(_claims(results_dir)[0].values())
    records = [c for c in claims if c.kind == "record"]
    reruns = [c for c in claims if c.kind == "rerun"]
    blocks = [RECORDS_INTRO, _rules()]
    if not records:
        blocks.append("No claims yet.\n")
    for form, heading in (("laptop", "Laptops"), ("desktop", "Desktops")):
        classes = sorted({_cls(c) for c in records if c.form == form})
        blocks += [f"## {heading}\n"] if classes else []
        for cls in classes:
            mine = [c for c in records if _cls(c) == cls]
            nontable = [c for c in mine if c.category == "no-table"]
            blocks += [f"### `{cls}`\n", _table(mine, reruns, "open")]
            blocks.append(_table(nontable, reruns, "no-table"))
    return "\n".join(blocks)


# ---- the bench rows ----------------------------------------------------------------------------


def _row_identity(row: dict[str, Any]) -> list[str]:
    """The schema, the person, the commit and the time."""
    commit = row["versions"]["commit"]
    checks = (
        (row["schema"] == bench.SCHEMA, f"schema is not {bench.SCHEMA}"),
        (bench.HANDLE.fullmatch(row["person"]) is not None, "person is not a GitHub handle"),
        (commit is None or COMMIT.fullmatch(commit) is not None, "commit is not 40 hex or null"),
        (_utc(row["utc"]), "utc is not an ISO 8601 time in UTC"),
    )
    return [problem for ok, problem in checks if not ok]


def _conditions(row: dict[str, Any]) -> list[str]:
    """AC power at the start and the end, and a thermal level at each, or null."""
    power, levels = row["conditions"]["power"], row["conditions"]["thermal_level"]
    problems = [] if power == [AC, AC] else ["the power was not AC at the start and the end"]
    known = all(x is None or (type(x) is int and 0 <= x <= THERMAL_MAX) for x in levels)
    if len(levels) != READINGS or not known:
        problems.append(
            f"conditions.thermal_level is not [start, end], each 0-{THERMAL_MAX} or null"
        )
    return problems


def _shape(row: dict[str, Any]) -> list[str]:
    """A known shape version, and the attention path a fresh run picks."""
    shape = dict(row["shape"])
    path, version = shape.pop("attn_path", None), shape.get("version")
    known = SHAPES.get(version) if type(version) is int else None
    problems = [] if shape == known else ["shape is not a known version of bench.SHAPE"]
    if path not in ATTN_PATHS:
        problems.append(f"shape.attn_path is not {' or '.join(ATTN_PATHS)}")
    return problems


def _measurements(row: dict[str, Any]) -> list[str]:
    """Positive measurements, and the ratio of the two step times."""
    off, on = row["arms"]["off"], row["arms"]["on"]
    if not all(x > 0 for x in [*off.values(), *on.values(), row["seconds"]]):
        return ["a measurement is not positive"]
    ratio = on["step_s"] / off["step_s"]
    if not math.isclose(row["on_off_ratio"], ratio, rel_tol=1e-9):
        return ["on_off_ratio is not arms.on.step_s / arms.off.step_s"]
    return []


def check_row(path: Path) -> list[str]:
    """The problems with one row file, each a sentence.

    A row must be strict JSON in the pretty profile, with every field of the row and its type,
    and its file name must be `bench.file_name` of its class and MLX version. Its schema is the
    bench's, its person a GitHub handle (`bench.HANDLE`), its commit 40 hex digits or null and
    its time ISO 8601 in UTC. Its class must be the key its hardware fields give, its power AC at
    its start and its end, and its shape a known version of `bench.SHAPE` with the attention path
    a fresh run picks. Its measurements must be positive, and its ratio the on run's step time
    over the off run's.
    """
    data = path.read_bytes()
    try:
        row = digest.loads(data)
    except ValueError as e:
        return [f"not strict JSON: {e}"]
    problems = _types(row, FIELDS)
    if problems or not isinstance(row, dict):
        return problems
    if not _pretty(data, row):
        problems.append("not in the pretty profile that bake bench --out writes")
    for check in (_row_identity, _hardware, _conditions, _shape, _measurements):
        problems += check(row)
    name = bench.file_name(row["hardware"]["class"], row["versions"]["mlx"])
    if path.name != name:
        problems.append(f"the file name is not {name}")
    return problems


def check_bench(folder: Path, *, page: bool = True, may_lag: bool = False) -> list[str]:
    """Every problem with the rows in `folder` and, if they have none, with its page.

    Each problem is one line, naming its file. With `may_lag`, the page may leave rows out: it
    passes if it is the page that the rows it lists give, so a pull request may add a row alone.
    """
    problems: list[str] = []
    for path in _files(folder):
        if path.suffix == ".json":
            problems += [f"{path.name}: {p}" for p in check_row(path)]
        elif path.name != PAGE:
            problems.append(f"{path.name}: only rows (.json) and {PAGE} belong here")
    if page and not problems:
        written = folder / PAGE
        text = written.read_text(encoding="utf-8") if written.is_file() else None
        rows = _rows(folder)
        if may_lag and text is not None:
            listed = set(text.splitlines())
            rows = [row for row in rows if _line(row) in listed]
        if text != _render_rows(rows):
            problems.append(f"{PAGE}: not the page the rows give; write it with {COMMAND} --write")
    return problems


def _cells(row: Mapping[str, Any]) -> list[str]:
    """A row's cells in the table, in COLUMNS' order."""
    hw, off, on = row["hardware"], row["arms"]["off"], row["arms"]["on"]
    return [
        hw["chip"],
        str(hw["gpu_cores"]),
        f"{hw['memory_gb']} GB",
        hw["macos"],
        hw["form"],
        row["versions"]["mlx"],
        f"{off['step_s'] * 1e3:.1f}",
        f"{on['step_s'] * 1e3:.1f}",
        f"{row['on_off_ratio']:.3f}",
        f"{on['table_gbps_estimate']:.3g}",
        f"{on['peak_bytes'] / 2**30:.2f}",
        f"[{row['person']}](https://github.com/{row['person']})",
    ]


def _shape_line() -> str:
    """What every row runs: the current shape version."""
    s = bench.SHAPE
    return (
        f"Every row runs shape version {s.version}: the `{s.preset}` model, {s.rows} sequences "
        f"of {s.seq_len:,} tokens per step and a table of {s.table_params:,.0f} numbers, "
        f"{s.warmup_steps} warm-up steps and then {s.steps} timed steps with the table off and "
        f"again with it on, seed {s.seed}, and MLX's buffer cache capped at "
        f"{s.cache_limit_gib} GiB. Step times are medians over the timed steps. The table's "
        "bandwidth is an estimate: the table rows it updated per step times the bytes of each, "
        "over the step time.\n"
    )


def _line(row: Mapping[str, Any]) -> str:
    """A row's line in the table."""
    return f"| {' | '.join(_cells(row))} |"


def _rows(folder: Path) -> list[dict[str, Any]]:
    """The rows in `folder`, which must check out, in the page's order: by chip, then memory."""
    rows = [(p.name, digest.loads(p.read_bytes())) for p in _files(folder) if p.suffix == ".json"]
    rows.sort(key=lambda r: (r[1]["hardware"]["chip"], r[1]["hardware"]["memory_gb"], r[0]))
    return [row for _, row in rows]


def _render_rows(rows: list[dict[str, Any]]) -> str:
    """The page of `rows`, in their order."""
    if not rows:
        return f"{BENCH_INTRO}\n{_shape_line()}\nNo rows yet.\n"
    lines = [f"| {' | '.join(COLUMNS)} |", f"|{'---|' * len(COLUMNS)}"]
    lines += [_line(row) for row in rows]
    return f"{BENCH_INTRO}\n{_shape_line()}\n" + "\n".join(lines) + "\n"


def render_bench(folder: Path) -> str:
    """The page of the rows in `folder`, which must check out: one table, by chip, then memory."""
    return _render_rows(_rows(folder))


def main(argv: list[str]) -> int:
    """Check both results folders and their pages, or write both pages; return the exit code.

    Every problem is printed on stderr, one line each, naming its file under `results/`. Nothing
    is written while there is a problem.
    """
    parser = argparse.ArgumentParser(
        prog=COMMAND, description="Check the results pages, or write them."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="check the folders and the pages")
    mode.add_argument("--write", action="store_true", help="write the pages from the folders")
    parser.add_argument(
        "--page-may-lag",
        action="store_true",
        help="with --check, the bench page may leave out rows it does not list yet",
    )
    a = parser.parse_args(argv)
    if a.page_may_lag and not a.check:
        parser.error("--page-may-lag goes with --check")
    lag = functools.partial(check_bench, may_lag=a.page_may_lag)
    pages: tuple[tuple[str, Path, Callable[..., list[str]], Callable[[Path], str]], ...] = (
        ("results/speedrun", SPEEDRUN, check, render),
        ("results/bench", BENCH, lag, render_bench),
    )
    problems = [
        f"{label}/{p}" for label, folder, test, _ in pages for p in test(folder, page=a.check)
    ]
    if problems:
        sys.stderr.write("".join(f"{p}\n" for p in problems))
        return 1
    if a.write:
        for _, folder, _, write in pages:
            (folder / PAGE).write_text(write(folder), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
