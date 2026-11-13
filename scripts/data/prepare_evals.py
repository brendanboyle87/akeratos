"""Prepare the eval tasks as one set for `akeratos/eval/mc.py`, on a machine that can reach the Hub.

It reuses `prepare_store.py`'s Hub layer, retries, download checks and durable writes, and runs in the
environment `requirements.txt` locks. It has four modes, each run from the repository root as
`python -I -B scripts/data/prepare_evals.py MODE ...`:

  check --out ROOT: prints each task's live `main` commit, gating, licence tag, scan flags and the one file
    it will read; writes nothing.
  fetch --out ROOT --confirm-pins TASK=SHA ...: downloads each task's file and dataset card at its confirmed
    commit into ROOT/raw/, checking each file's size and hash against the listing.
  prepare --out ROOT --confirm-pins TASK=SHA ...: offline, re-hashes the fetched files, renders every row with
    `akeratos/eval/tasks.py` (lm-eval v0.4.13's zero-shot templates) and writes the set to ROOT/<set_id>/.
  verify --set ROOT/<set_id>: re-reads the set, every hash, count and item.

It never runs downloaded code, unpickles anything, sends a Hugging Face token, uses `datasets`, `transformers`
or MLX, or writes outside its own folders. ROOT must not be a data store's root, whose check refuses any other
folder in it: `~/mf-evals` is the suggested one.
"""

import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)  # under `python -I` neither is on sys.path

import prepare_store as base

from akeratos.errors import DataMismatchError
from akeratos.eval import tasks as mct

Stop, say, EXIT = base.Stop, base.say, base.EXIT
FILE_MAX = 64 << 20         # a cap, not a size: each task's file is a few MB
README_MAX = 1 << 20
make_hub = None             # None: `base.HubClient`; the tests' entry sets a fake


def key(task):
    return f"eval-{task}"


def parse_pairs(pairs, what, check=None):
    out = {}
    for p in pairs or []:
        t, sep, v = p.partition("=")
        if not sep or t not in mct.TASKS or t in out:
            raise Stop("config", "bad-option", f"{what}: {p!r} is not a task=value pair for a new task")
        if check is not None:
            check(v, f"{what} {t}")
        out[t] = v
    return out


def pins_of(args):
    """Each task's confirmed commit, from `--confirm-pins task=<40-hex>`, one per task.

    The script holds no pin: `check` prints each repository's live `main`, and `fetch` and `prepare` take each
    one back here. `survey` refuses a commit the Hub does not resolve to exactly. The pins are recorded in each
    fetch record and in the set's manifest.
    """
    pins = parse_pairs(args.confirm_pins, "--confirm-pins", base.check_revision)
    if sorted(pins) != sorted(mct.TASKS):
        raise Stop("config", "pins-not-confirmed", f"--confirm-pins needs one task=SHA for each of "
                   f"{sorted(mct.TASKS)}, as check printed them")
    return pins


def licence_ok(spec, tag, accepted):
    """The card's tag (a string, a list, or None when the card has none) is the expected one, or the
    one accepted with `--accept-licence task=<tag>`; `--accept-licence task=none` accepts a card with no tag.

    The expected tag is the task table's (`akeratos/eval/tasks.py`). `check` and `fetch` stop on a missing
    or different tag that is not accepted, and an accepted one is recorded.
    """
    tags = ["none" if t is None else str(t) for t in (tag if isinstance(tag, list) else [tag])]
    return spec.licence in tags or accepted.get(spec.name) in tags


def pick(spec, listing):
    """The one file the task reads: the first pattern that matches any file must match one.

    The patterns are `TaskSpec.files`: Parquet, or LAMBADA's JSONL. A pattern that matches two files, or a pin
    that matches none, stops the run with the listing printed.
    """
    files = [e for e in listing if e.get("type") == "file"]
    for pat in spec.files:
        hit = [e for e in files if re.fullmatch(pat, e["path"])]
        if len(hit) == 1:
            return hit[0]
        if hit:
            raise Stop("config", "layout", f"{spec.name}: {pat} matches {[e['path'] for e in hit]}")
    raise Stop("config", "layout", f"{spec.name}: no file matches {list(spec.files)}; listing: "
               f"{sorted(e['path'] for e in files)}")


def survey(hub, rev_of, accepted, attempts, strict):
    """Each task's info, listing, chosen file and card; refuses before any download.

    A layout miss stops fetch at once. Check (`strict` off) returns the misses instead, prints the
    other tasks as usual and stops as a configuration error with every miss, so one run shows each listing
    that needs a pattern."""
    plan, misses = {}, []
    for t, spec in sorted(mct.TASKS.items()):
        rev = rev_of(t)
        info = base.retry(lambda: hub.info(spec.repo, "dataset", rev), f"{spec.repo} info", attempts)
        if strict and info["sha"] != rev:
            raise Stop("config", "pin-mismatch", f"{spec.repo}: the Hub resolves {rev} to {info['sha']}")
        if info["gated"] is not False or info["private"] is not False or info["disabled"]:
            raise Stop("gated", "gated", f"{spec.repo}: gated {info['gated']!r}, private {info['private']!r}")
        listing = base.retry(lambda: hub.tree(spec.repo, "dataset", info["sha"], recursive=True),
                            f"{spec.repo} listing", attempts)
        for e in listing:
            if base.unsafe(e.get("path")):
                raise Stop("integrity", "unsafe-path", f"{spec.repo}: {e.get('path')!r} is not a safe path")
        try:
            f = pick(spec, listing)
        except Stop as e:
            if strict:
                raise
            misses.append(e.detail)
            continue
        card = next((e for e in listing if e.get("type") == "file" and e["path"] == "README.md"), None)
        for e in [f] + ([card] if card else []):
            cap = README_MAX if e["path"] == "README.md" else FILE_MAX
            if e["size"] > cap:
                raise Stop("integrity", "over-size-cap", f"{spec.repo}: {e['path']} is {e['size']} bytes > {cap}")
            if base.upstream(e)[0] is None:
                raise Stop("integrity", "no-upstream-hash", f"{spec.repo}: {e['path']} has no hash to check")
        # A Python file at a repository's root is recorded, never fetched or run.
        py = sorted(e["path"] for e in listing if e.get("type") == "file" and "/" not in e["path"]
                    and e["path"].endswith(".py"))
        plan[t] = {"spec": spec, "info": info, "file": f, "card": card, "py": py,
                   "licence_ok": licence_ok(spec, info["license"], accepted)}
    return plan, misses


def check_mode(args):
    hub = (make_hub or base.HubClient)()
    accepted = parse_pairs(args.accept_licence, "--accept-licence")
    plan, misses = survey(hub, lambda t: "main", accepted, {}, strict=False)
    out, code = [], 0
    for t, p in plan.items():
        s, i, f = p["spec"], p["info"], p["file"]
        out.append(f"{t}: {s.repo}  live main {i['sha']}  (pass it as --confirm-pins {t}={i['sha']})")
        lic = "" if p["licence_ok"] else f"  LICENCE MISMATCH: expected {s.licence!r}"
        out.append(f"  gated {i['gated']!r}, private {i['private']!r}, licence {i['license']!r}{lic}")
        for e in [f] + ([p["card"]] if p["card"] else []):
            sec = e.get("security")
            flag = "unscanned" if sec is None else ("" if sec.get("safe") is True else f"FLAGGED {sec.get('status')!r}")
            out.append(f"  fetch {e['path']}  {e['size']:,} B  {base.upstream(e)[0]}  {flag}".rstrip())
        if p["card"] is None:
            out.append("  no README.md at the root: the licence tag has no card to come from")
        if p["py"]:
            out.append(f"  root *.py (recorded, never fetched): {p['py']}")
        if not p["licence_ok"]:
            code = EXIT["gated"]
    out.extend(f"LAYOUT: {m}" for m in misses)
    base.summary(out)
    return EXIT["config"] if misses else code


def fetch_mode(args):
    pins = pins_of(args)
    accepted = parse_pairs(args.accept_licence, "--accept-licence")
    allow = set(args.allow_unsafe_file or [])
    out = os.path.abspath(args.out)
    hub = (make_hub or base.HubClient)()
    attempts = {}
    plan, _ = survey(hub, pins.__getitem__, accepted, attempts, strict=True)
    for t, p in plan.items():
        if not p["licence_ok"]:
            raise Stop("gated", "licence", f"{p['spec'].repo}: licence tag {p['info']['license']!r}, expected "
                       f"{p['spec'].licence!r} (--accept-licence {t}=<tag> is for the person running the script "
                       f"to decide)")
        base.security_check({t: [p["file"]] + ([p["card"]] if p["card"] else [])}, allow)
    raw = os.path.join(out, "raw")
    empty = os.path.join(raw, ".hf-cache-empty")
    os.makedirs(empty, exist_ok=True)
    report = []
    for t, p in plan.items():
        s, rev = p["spec"], pins[t]
        rawdir = os.path.join(raw, key(t), rev)
        os.makedirs(rawdir, exist_ok=True)
        fd = base.lock(os.path.join(rawdir, ".lock"))
        try:
            records = []
            for e in [p["file"]] + ([p["card"]] if p["card"] else []):
                path = base.retry(lambda: hub.download(s.repo, "dataset", rev, e["path"], rawdir, empty),
                                 e["path"], attempts)
                if os.listdir(empty):
                    raise Stop("integrity", "cache-written", "the library wrote into the empty cache_dir")
                digest = base.verify_download(path, rawdir, e)
                records.append({"path": e["path"], "size": e["size"], "sha256": digest, "blob_id": e.get("blob_id"),
                                "lfs_sha256": (e.get("lfs") or {}).get("sha256"), "security": e.get("security"),
                                "security_override": e["path"] in allow})
                report.append(f"  {t}: {e['path']}  {e['size']:,} B  sha256 {digest}")
            rec = {"task": t, "repo_id": s.repo, "revision": rev, "config": s.config, "split": s.split,
                   "file": p["file"]["path"], "files": records, "info": p["info"], "root_py": p["py"],
                   "licence_tag": p["info"]["license"], "licence_expected": s.licence,
                   "licence_override": accepted.get(t) if not licence_ok(s, p["info"]["license"], {}) else None,
                   "endpoint": base.ENDPOINT, "token_sent": False, "fetched_at": base.now(),
                   "fetched_by": "prep_evals"}
            base.write_file(os.path.join(rawdir, "fetch.json"), json.dumps(rec, indent=1, sort_keys=True).encode())
        finally:
            os.close(fd)
    base.summary(["fetched the five eval files at their confirmed pins"] + report)
    return 0


def read_rows(path, name):
    """The source file's rows as dicts: Parquet through pyarrow, JSONL as JSON data."""
    if name.endswith(".jsonl"):
        with open(path, "rb") as f:
            return [json.loads(line) for line in f.read().decode("utf-8", errors="strict").splitlines() if line.strip()]
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def script_commit():
    """The checkout's commit, and whether this script or the task module differs from it."""
    paths = [HERE, os.path.join(REPO, "akeratos", "eval")]
    try:
        c = base.subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        d = base.subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", *paths], capture_output=True,
                               text=True, timeout=20)
        return f"{c.stdout.strip() or None}{' (dirty)' if d.stdout.strip() else ''}"
    except (OSError, base.subprocess.SubprocessError):
        return "None"


def prepare_mode(args):
    pins = pins_of(args)
    out = os.path.abspath(args.out)
    sources, items, notes = {}, {}, []
    for t, spec in sorted(mct.TASKS.items()):
        rawdir = os.path.join(out, "raw", key(t), pins[t])
        try:
            with open(os.path.join(rawdir, "fetch.json"), "rb") as f:
                rec = json.loads(f.read())
        except OSError:
            raise Stop("refused", "not-fetched", f"{t}: no fetch record at {pins[t]}; run fetch first") from None
        if rec["revision"] != pins[t] or rec["repo_id"] != spec.repo:
            raise Stop("integrity", "fetch-record", f"{t}: its fetch record names {rec['repo_id']}@{rec['revision']}")
        files = {r["path"]: r for r in rec["files"]}
        src = files[rec["file"]]
        path = os.path.join(rawdir, src["path"])
        if base.sha256_path(path) != src["sha256"] or os.path.getsize(path) != src["size"]:
            raise Stop("integrity", "hash-mismatch", f"{t}: {src['path']} differs from its fetch record")
        try:
            items[t] = [mct.render(t, row, i) for i, row in enumerate(read_rows(path, src["path"]))]
        except Exception as e:  # a row the template cannot take: the layout is not the one expected
            raise Stop("integrity", "rows", f"{t}: {src['path']}: {type(e).__name__}: {e}") from None
        card = files.get("README.md")
        sources[t] = {"repo": spec.repo, "revision": pins[t], "config": spec.config, "split": spec.split,
                      "source": {k: src[k] for k in ("path", "size", "sha256")},
                      "card_sha256": None if card is None else card["sha256"],
                      "licence_tag": rec["licence_tag"], "licence_override": rec["licence_override"]}
        if len(items[t]) != spec.n:
            notes.append(f"  note: {t} has {len(items[t])} items, the published count is {spec.n}")
    set_id, blobs = mct.set_files(sources, items, f"prepare_evals.py at {script_commit()}")
    st = os.path.join(out, set_id)
    if os.path.isdir(st):
        if verify_set(st, quiet=True) != 0:
            raise Stop("integrity", "set-differs", f"{st} exists and does not verify; move it aside")
        base.summary([f"set {set_id} is already complete at {st}"])
        return 0
    tmp = os.path.join(out, "raw", f".set-{set_id}.tmp")  # under raw/, which the root pre-check allows
    if os.path.exists(tmp):
        for n in os.listdir(tmp):
            os.unlink(os.path.join(tmp, n))
    os.makedirs(tmp, exist_ok=True)
    for name, body in sorted(blobs.items()):
        base.write_file(os.path.join(tmp, name), body)
    os.replace(tmp, st)
    base.sync_dir(out)
    if verify_set(st, quiet=True) != 0:
        raise Stop("integrity", "verify", f"{st} does not verify after writing")
    lines = [f"set {set_id} at {st}"] + [f"  {t}: {len(items[t])} items at {pins[t]}" for t in sorted(items)]
    base.summary(lines + notes)
    return 0


def verify_set(st, quiet=False):
    try:
        mct.read_set(st)
        with open(os.path.join(st, "SHA256SUMS")) as f:
            for line in f.read().splitlines():
                digest, name = line.split("  ", 1)
                with open(os.path.join(st, name), "rb") as g:
                    if hashlib.sha256(g.read()).hexdigest() != digest:
                        raise ValueError(f"{name}: SHA-256 differs from SHA256SUMS")
        names = set(os.listdir(st))
        listed = {line.split("  ", 1)[1] for line in open(os.path.join(st, "SHA256SUMS")).read().splitlines()}
        if names != listed | {"SHA256SUMS"}:
            raise ValueError(f"files not in SHA256SUMS: {sorted(names - listed - {'SHA256SUMS'})}")
    except (OSError, ValueError, DataMismatchError) as e:
        if not quiet:
            say(f"mismatch: {e}")
        return 1
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(prog="prepare_evals.py", description="The eval sets' Mac preparation")
    ap.add_argument("mode", choices=("check", "fetch", "prepare", "verify"))
    ap.add_argument("--out")
    ap.add_argument("--set")
    ap.add_argument("--confirm-pins", nargs="+", metavar="TASK=SHA")
    ap.add_argument("--accept-licence", action="append", metavar="TASK=TAG")
    ap.add_argument("--allow-unsafe-file", action="append")
    a = ap.parse_args(argv)
    if a.mode == "verify" and not a.set:
        ap.error("verify needs --set")
    if a.mode != "verify" and not a.out:
        ap.error(f"{a.mode} needs --out")
    return a


def main(argv=None):
    args = parse(argv)
    if args.mode == "verify":
        return verify_set(os.path.abspath(args.set))
    try:
        if args.mode in ("check", "fetch"):
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        else:
            os.environ["HF_HUB_OFFLINE"] = "1"
        out = os.path.abspath(args.out)
        base.check_paths((out, os.path.join(out, "raw")))
        base.check_env(args.mode)
        base.check_locations(out)
        return {"check": check_mode, "fetch": fetch_mode, "prepare": prepare_mode}[args.mode](args)
    except Stop as s:
        say(f"stopped: {s.reason} (exit {s.code}): {s.detail}")
        return s.code


if __name__ == "__main__":
    sys.exit(main())
