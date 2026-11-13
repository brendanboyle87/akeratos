"""Prepare the data store: FineWeb-Edu tokenised with SmolLM2's tokenizer, on a machine that can reach the Hub.

It has four modes, each run from the repository root as `python -I -B scripts/data/prepare_store.py MODE ...`:

  check --sample S --files N --out ROOT --cap-tokens C: asks the Hub for the pinned commits, the live `main`
    commits, the listings, gating, licence tags and scan flags; prints them with the N files and a disk table.
  fetch ... --confirm-pins DATASET_SHA TOKENIZER_SHA: downloads the first N allowlisted Parquet files, the
    dataset card and the four tokenizer files at the pins into ROOT/raw/, checking each file's size and hash.
  prepare --sample S --out ROOT --cap-tokens C: offline, tokenises the fetched files into
    ROOT/<fingerprint[:16]>/; resumes after a stop; on a complete store it verifies and writes nothing.
  verify --store ROOT/<fingerprint[:16]>: re-hashes every file and rebuilds the manifests from the shards.

It never runs downloaded code, unpickles anything, sends a Hugging Face token, uses `datasets`, `transformers`
or `AutoTokenizer`, imports MLX, deletes anything outside its own folders, or commits or uploads data. Only
`HubClient` imports `huggingface_hub`, and only in check and fetch; `check_imports` holds prepare to the standard
library, NumPy, pyarrow, tokenizers and this folder's modules. No option or environment variable changes the
constants in the block a test may set, or skips a golden check.
"""

import argparse
import datetime
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # under `python -I` the script's own directory is not on sys.path

# ---- Pins: changing one is a code change ---------------------------------------------------------------
DATASET = "HuggingFaceFW/fineweb-edu"
DATASET_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TOKENIZER_REPO = "HuggingFaceTB/SmolLM2-135M"
TOKENIZER_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
REPOS = {DATASET: ("dataset", "dataset-fineweb-edu", "odc-by"),
         TOKENIZER_REPO: ("model", "model-smollm2-135m", "apache-2.0")}  # type, raw folder, licence tag
SAMPLES = {"sample/10BT": "sample-10BT", "sample/100BT": "sample-100BT"}
ENDPOINT = "https://huggingface.co"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md")
PARQUET_NAME = re.compile(r"sample/(10BT|100BT)/[0-9]{3}_[0-9]{5}\.parquet")
REV = re.compile(r"[0-9a-f]{40}")
# The versions requirements.txt locks.
PINNED = {"huggingface_hub": "2.2.0", "tokenizers": "0.23.3", "pyarrow": "26.0.0", "numpy": "2.5.3"}
PYARROW_FLOOR = (21, 0, 0)                                    # no option lowers it

# ---- Manifest identity and rules ----------------------------------------------------------------------
SCHEMA = "metalforge-a21-standin.data-manifest/v0"
PREP_FORMAT_VERSION = 0
CANON = "a21-1"
STAND_IN_FOR = ["#58", "#61", "#100", "#101", "#102", "#103", "#125", "#126"]
DECISION = "https://github.com/BhodiSea/MetalForge/issues/51#issuecomment-6143014262"
SPDX = "ODC-By-1.0 AND LicenseRef-CommonCrawl-ToU"
# ODC-By 1.0's example attribution notice, plus the Common Crawl line.
ATTRIBUTION = ("Contains information from FineWeb-Edu (https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), "
               "which is made available under the ODC Attribution License "
               "(https://opendatacommons.org/licenses/by/1-0/). Its use is also subject to Common Crawl's Terms of "
               "Use (https://commoncrawl.org/terms-of-use).")
HELDOUT_PERSON = b"bake-heldout-v1"
HELDOUT_LIMIT = 2 ** 64 // 1024
HELDOUT_RULE = "blake2b-8/bake-heldout-v1/lt-2^64//1024"
PREFIX_K = (64, 512, 4096)
DROPS = ("null", "non_string", "invalid_utf8", "empty", "whitespace", "too_long")
EXIT = {"integrity": 65, "unavailable": 69, "refused": 73, "disk": 74, "interrupted": 75, "gated": 77, "config": 78}

# ---- Constants a test may set; no option or environment variable changes them ------------------------
VOCAB = 49_152
SHARD_TOKENS = 1 << 27
PARQUET_MAX = 3 * 10 ** 9
TOKENIZER_MAX = 32 << 20
JSON_MAX = 64 << 10
README_MAX = 1 << 20
MAX_DOC_BYTES = 4 << 20
BIG_DOC_BYTES = 16 << 20
ROWGROUP_TEXT_MAX = 1 << 30
GOLDEN = True                  # `bpe.BPE.check`'s (2) and (3): SmolLM2's file only
BATCH_ROWS = 1024
ENCODE_BYTES = 8 << 20
EXACT_EVERY = 1000
RETRIES = 5
XET_ALLOWANCE = 5 * 10 ** 9
DISK_FIXED = 2 * 2 ** 28 + 2 ** 30
MEAN_DOC_TOKENS = 1028         # an estimate, for the disk table
CROP_ROW = 2049                # T 2,048 + 1
MERGE_PREFIXES = (32_768, 16_384, 8_192)  # `bpe.truncate`'s sizes; the tests' fixture tokenizer is far smaller

# ---- Test seams (None or the standard call in production) ----------------------------------------------------
HOOK = None                    # HOOK(event, path): the tests' spy and kill points
SLEEP = time.sleep
DISK_USAGE = shutil.disk_usage
VERSION = importlib.metadata.version
PYARROW_VERSION = None         # None: pyarrow.__version__
ST_FLAGS = lambda st: getattr(st, "st_flags", 0)  # noqa: E731
SF_DATALESS = 0x40000000
make_hub = None                # None: HubClient

np = pa = pq = pc = store_fmt = bpe = None  # imported after the environment is set
STOP_FLAG = []


class Stop(Exception):
    def __init__(self, code, reason, detail=""):
        super().__init__(f"{reason}: {detail}")
        self.code, self.reason, self.detail = EXIT.get(code, code), reason, detail


def say(msg):
    print(msg, file=sys.stderr, flush=True)


def event(kind, path):
    if HOOK is not None:
        HOOK(kind, path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canon_json(obj):
    """The hashed bodies' serialisation: sorted keys, no spaces, ASCII, no NaN, and no floats."""
    def walk(o):
        if isinstance(o, float):
            raise TypeError("a float in a hashed body")
        if isinstance(o, dict):
            for k, v in o.items():
                if not isinstance(k, str) or not k.isascii():
                    raise TypeError(f"key {k!r} is not ASCII")
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def sha256_fd(fd, chunk=1 << 24):
    h = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while b := os.read(fd, chunk):
        h.update(b)
    return h.hexdigest()


def sha256_path(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        return sha256_fd(fd)
    finally:
        os.close(fd)


def git_blob_sha1(data):
    """git's object id for a blob: SHA-1 of b"blob <len>\\0" + data (git's documented object format)."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False).hexdigest()


def parse_version(v):
    m = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


def under(path, root):
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


# ---- Durable writes ---------------------------------------------------------------------------------------
def sync_fd(fd, path, kind="sync"):
    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)  # macOS: fsync alone does not reach the medium
        except OSError:
            os.fsync(fd)  # ❓ whether APFS accepts F_FULLFSYNC on a directory descriptor
    else:
        os.fsync(fd)
    event(kind, path)


def sync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        sync_fd(fd, d, "dirsync")
    finally:
        os.close(fd)


class Out:
    """One output file, written as `<path>.tmp` with buffered writes, synced and renamed by `commit`."""

    def __init__(self, path):
        self.path, self.tmp = path, path + ".tmp"
        self.f = open(self.tmp, "wb")
        self.h, self.size = hashlib.sha256(), 0

    def write(self, b):
        event("write", self.tmp)
        self.f.write(b)
        self.h.update(b)
        self.size += len(b)

    def finish(self):
        self.f.flush()
        sync_fd(self.f.fileno(), self.tmp)
        self.f.close()

    def abort(self):
        try:
            self.f.close()
        finally:
            if os.path.exists(self.tmp):
                os.unlink(self.tmp)

    def entry(self, root):
        return {"path": os.path.relpath(self.path, root), "size": self.size, "sha256": self.h.hexdigest()}


def commit(outs):
    for o in outs:
        o.finish()
    for o in outs:
        os.replace(o.tmp, o.path)
        event("rename", o.path)
    for d in sorted({os.path.dirname(o.path) for o in outs}):
        sync_dir(d)


def write_file(path, data):
    o = Out(path)
    o.write(data)
    commit([o])
    return o


# ---- Start checks ---------------------------------------------------------------------------------------
def check_env(mode):
    ep = os.environ.get("HF_ENDPOINT")
    if ep is not None and ep != ENDPOINT:
        raise Stop("config", "foreign-endpoint", "HF_ENDPOINT is set to another host")
    if "HUGGINGFACE_CO_STAGING" in os.environ:
        raise Stop("config", "staging-endpoint", "HUGGINGFACE_CO_STAGING is set")
    if mode in ("check", "fetch"):
        for v in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if v in os.environ:
                raise Stop("config", "offline-set", f"{v} is set; the check and fetch modes must ask the Hub")


def check_revision(rev, what):
    if not isinstance(rev, str) or not REV.fullmatch(rev):
        raise Stop("config", "bad-revision", f"{what}: {rev!r} is not a 40-hex commit")


def check_paths(roots):
    """Nothing under the download or output root is on sys.path or is the current directory."""
    for p in [os.getcwd()] + [e for e in sys.path if e]:
        for r in roots:
            if os.path.exists(p) and under(p, r):
                raise Stop("refused", "path-under-root", f"{p} lies under an output root")


def check_versions(mode, allow_drift):
    names = ["tokenizers", "pyarrow", "numpy"] + (["huggingface_hub"] if mode in ("check", "fetch") else [])
    found = {}
    for n in names:
        try:
            found[n] = VERSION(n)
        except importlib.metadata.PackageNotFoundError:
            raise Stop("config", "missing-library", f"{n} is not installed") from None
    meta = parse_version(found["pyarrow"])
    if meta is None or meta < PYARROW_FLOOR:
        raise Stop("config", "security-floor", f"pyarrow {found['pyarrow']} by package metadata: needs a plain "
                   f"version at or above {'.'.join(map(str, PYARROW_FLOOR))} (no override)")
    drift = {n: found[n] for n in names if found[n] != PINNED[n]}
    if drift and not allow_drift:
        raise Stop("config", "env-drift", f"versions differ from the lock: {drift} (--allow-env-drift "
                   f"overrides and is recorded)")
    return found, drift


def check_pyarrow_runtime(found):
    rt = PYARROW_VERSION if PYARROW_VERSION is not None else pa.__version__
    if parse_version(rt) is None or parse_version(rt) < PYARROW_FLOOR or rt != found["pyarrow"]:
        raise Stop("config", "security-floor", f"pyarrow.__version__ {rt} against package metadata "
                   f"{found['pyarrow']} (no override)")


def repo_root():
    """The repository's folder, found by walking up from the script's own folder: the first folder that holds
    .git, or, when no folder above holds .git (a downloaded copy), the first that holds a pyproject.toml file.
    None when neither is found."""
    d, first = HERE, None
    while d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, ".git")):
            return d
        if first is None and os.path.isfile(os.path.join(d, "pyproject.toml")):
            first = d
        d = os.path.dirname(d)
    return first


def check_locations(out):
    """The output root: outside the repository's folder (a clone, or a copy without .git), never iCloud-synced,
    and new or holding only stores and raw/."""
    clone = repo_root()
    if clone and under(out, clone):
        raise Stop("refused", "inside-clone", "the output root lies inside the repository")
    home = os.path.expanduser("~")
    for bad in ("Library/Mobile Documents", "Library/CloudStorage"):
        if under(out, os.path.join(home, bad)):
            raise Stop("refused", "icloud-path", f"the output root is under ~/{bad}")
    for warn in ("Desktop", "Documents", "Downloads"):
        if under(out, os.path.join(home, warn)):
            say(f"warning: the output root is under ~/{warn}, which macOS may sync")
    if os.path.isdir(out):
        for e in os.scandir(out):
            if e.name == ".DS_Store" and e.is_file(follow_symlinks=False):
                continue  # Finder writes these into any folder it shows
            if not (e.name == "raw" or re.fullmatch(r"[0-9a-f]{16}", e.name)) or e.is_symlink() or not e.is_dir():
                raise Stop("refused", "root-not-empty", f"the output root holds {e.name!r}: it must be new or hold "
                           f"only stores and raw/")


def check_dataless(path):
    if ST_FLAGS(os.stat(path)) & SF_DATALESS:
        raise Stop("refused", "dataless-file", f"{os.path.basename(path)} is an iCloud placeholder (SF_DATALESS)")


ALLOWED = {"numpy", "pyarrow", "tokenizers", "__main__", "_virtualenv", "_distutils_hack"}
RUNTIME = r"_cython_[0-9_]+|cython_runtime|_sysconfigdata_[A-Za-z0-9_.-]+"


def check_imports():
    """Refuse any loaded module outside the standard library, `ALLOWED` and this folder.

    The two shims are imported by .pth files before any script runs (uv's and virtualenv's `_virtualenv`,
    setuptools' `_distutils_hack`); Cython's runtime modules come with pyarrow, and `_sysconfigdata_<platform>`
    is the interpreter's own build record, which `sys.stdlib_module_names` leaves out (Python 3.12).
    """
    bad = []
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or re.fullmatch(RUNTIME, top):
            continue
        f = getattr(sys.modules.get(top), "__file__", None) or ""
        if top == "__main__" and not (f and under(f, HERE)):
            bad.append(top)
        elif top in ALLOWED or (f and under(f, HERE)):
            continue
        else:
            bad.append(top)
    if bad:
        raise Stop("refused", "import-not-allowed", f"modules outside the allowlist: {sorted(set(bad))}")


def setup_env(args):
    """Variables that must be set before the libraries are imported."""
    if args.mode in ("check", "fetch"):
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        if args.mode == "fetch":
            os.environ["HF_XET_CACHE"] = os.path.join(os.path.abspath(args.out), "raw", ".xet-cache")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["RAYON_NUM_THREADS"] = str(args.threads or os.cpu_count() or 1)
        os.environ.pop("RAYON_RS_NUM_THREADS", None)


def import_libs(mode):
    global np, pa, pq, pc, store_fmt, bpe
    import numpy
    import pyarrow
    np, pa = numpy, pyarrow
    import layout as _store
    store_fmt = _store
    if mode == "prepare":
        import pyarrow.compute
        import pyarrow.parquet
        import bpe as _bpe
        pq, pc, bpe = pyarrow.parquet, pyarrow.compute, _bpe


# ---- The Hub layer (check and fetch only) ----------------------------------------------------------------
class Transient(Exception):
    pass


class Gated(Exception):
    pass


class Missing(Exception):
    pass


class HubClient:
    """The only code that imports huggingface_hub. Every call passes token=False and the fixed endpoint;
    results come back as plain dicts, the form the tests' fake also returns."""

    def __init__(self):
        from huggingface_hub import HfApi
        self.api = HfApi(endpoint=ENDPOINT, token=False)

    def _call(self, fn):
        from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError, HfHubHTTPError,
                                            RepositoryNotFoundError, RevisionNotFoundError)
        try:
            return fn()
        except GatedRepoError as e:  # before RepositoryNotFoundError, its base class
            raise Gated(str(e)) from None
        except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError) as e:
            raise Missing(str(e)) from None
        except (HfHubHTTPError, OSError, TimeoutError) as e:
            raise Transient(str(e)) from None

    def info(self, repo_id, repo_type, revision):
        get = self.api.dataset_info if repo_type == "dataset" else self.api.model_info
        i = self._call(lambda: get(repo_id, revision=revision, token=False))
        lic = getattr(i.card_data, "license", None) if i.card_data is not None else None
        return {"sha": i.sha, "gated": i.gated, "private": i.private, "disabled": i.disabled, "license": lic}

    def tree(self, repo_id, repo_type, revision, path=None, recursive=False):
        items = self._call(lambda: list(self.api.list_repo_tree(repo_id, path_in_repo=path, recursive=recursive,
                                                                expand=True, revision=revision, repo_type=repo_type,
                                                                token=False)))
        out = []
        for it in items:
            if not hasattr(it, "blob_id"):
                out.append({"type": "folder", "path": it.path})
                continue
            sec = it.security
            out.append({"type": "file", "path": it.path, "size": it.size, "blob_id": it.blob_id,
                        "lfs": None if it.lfs is None else {"sha256": it.lfs.sha256, "size": it.lfs.size},
                        "xet_hash": getattr(it, "xet_hash", None),
                        "security": None if sec is None else {"safe": sec.safe, "status": sec.status,
                                                              "av_scan": sec.av_scan,
                                                              "pickle_import_scan": sec.pickle_import_scan}})
        return out

    def download(self, repo_id, repo_type, revision, filename, local_dir, cache_dir):
        from huggingface_hub import hf_hub_download
        return self._call(lambda: hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision,
                                                  local_dir=local_dir, cache_dir=cache_dir, token=False,
                                                  endpoint=ENDPOINT))


def jitter(what, n):
    return int.from_bytes(hashlib.sha256(f"{what}:{n}".encode()).digest()[:2], "big") / 65536


def retry(fn, what, attempts=None):
    """At most RETRIES retries of a transient failure, min(2**n, 60) s plus deterministic jitter apart."""
    for n in range(RETRIES + 1):
        try:
            r = fn()
            if attempts is not None:
                attempts[what] = n + 1
            return r
        except Gated as e:
            raise Stop("gated", "gated", f"{what}: {e}") from None
        except Missing as e:
            raise Stop("config", "missing", f"{what}: {e}") from None
        except Transient as e:
            if n == RETRIES:
                raise Stop("interrupted", "download-failed", f"{what} after {n + 1} attempts: {e}") from None
            say(f"retrying {what} ({e})")
            SLEEP(min(2 ** n, 60) + jitter(what, n))


def unsafe(path):
    return (not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\0" in path
            or ".." in path.split("/"))


def upstream(e):
    """(hash kind, expected value), or (None, None) when the file has no upstream content hash."""
    if e.get("lfs"):
        return "lfs-sha256", e["lfs"]["sha256"]
    if e.get("xet_hash"):
        return None, None
    return "git-blob-sha1", e["blob_id"]


def cap_for(path):
    name = os.path.basename(path)
    if path.endswith(".parquet"):
        return PARQUET_MAX
    return {"tokenizer.json": TOKENIZER_MAX, "README.md": README_MAX}.get(name, JSON_MAX)


def select(hub, sample, n_files, attempts):
    """Ask the Hub for both repositories at their pins; refuse before any download."""
    check_revision(DATASET_REV, DATASET)
    check_revision(TOKENIZER_REV, TOKENIZER_REPO)
    plan = {}
    for repo, rev in ((DATASET, DATASET_REV), (TOKENIZER_REPO, TOKENIZER_REV)):
        rtype, key, licence = REPOS[repo]
        info = retry(lambda: hub.info(repo, rtype, rev), f"{repo} info", attempts)
        main = retry(lambda: hub.info(repo, rtype, "main"), f"{repo} main", attempts)
        if info["sha"] != rev:
            raise Stop("config", "pin-mismatch", f"{repo}: the Hub resolves the pin to {info['sha']}")
        if info["gated"] is not False or info["private"] is not False or info["disabled"]:
            raise Stop("gated", "gated", f"{repo}: gated {info['gated']!r}, private {info['private']!r}, "
                       f"disabled {info['disabled']!r}")
        if info["license"] != licence:
            raise Stop("gated", "licence", f"{repo}: licence tag {info['license']!r}, expected {licence!r}")
        root = retry(lambda: hub.tree(repo, rtype, rev), f"{repo} root listing", attempts)
        plan[repo] = {"info": info, "main": main["sha"], "root": root, "key": key, "type": rtype, "rev": rev}
    listing = retry(lambda: hub.tree(DATASET, "dataset", DATASET_REV, sample, recursive=True), "sample listing",
                    attempts)
    for e in listing + plan[DATASET]["root"] + plan[TOKENIZER_REPO]["root"]:
        if unsafe(e.get("path")):
            raise Stop("integrity", "unsafe-path", f"{e.get('path')!r} is not a safe path")
    for e in listing:
        if e["type"] != "file" or not PARQUET_NAME.fullmatch(e["path"]) or not e["path"].startswith(sample + "/"):
            raise Stop("integrity", "layout-changed", f"{e['path']!r} in {sample} is not an allowlisted "
                       f"Parquet file")
    files = sorted(listing, key=lambda e: e["path"])
    if len(files) < n_files:
        raise Stop("unavailable", "too-few-files", f"{sample} lists {len(files)} files, {n_files} asked for")
    droot = {e["path"]: e for e in plan[DATASET]["root"] if e["type"] == "file"}
    troot = {e["path"]: e for e in plan[TOKENIZER_REPO]["root"] if e["type"] == "file"}
    if "README.md" not in droot or any(f not in troot for f in TOKENIZER_FILES):
        raise Stop("integrity", "layout-changed",
                   "a dataset card or tokenizer file is missing at the pin")
    py = sorted(p for p in droot if p.endswith(".py"))
    want = {DATASET: files[:n_files] + [droot["README.md"]], TOKENIZER_REPO: [troot[f] for f in TOKENIZER_FILES]}
    for repo, entries in want.items():
        for e in entries:
            if e["size"] > cap_for(e["path"]):
                raise Stop("integrity", "over-size-cap",
                           f"{e['path']}: {e['size']} bytes > {cap_for(e['path'])}")
            kind, _ = upstream(e)
            if kind is None or (e["path"].endswith(".parquet") and not e.get("lfs")):
                raise Stop("integrity", "no-upstream-hash", f"{e['path']}: no lfs.sha256 to check against")
    return plan, files, want, py


def security_check(want, allow):
    flagged = []
    for repo, entries in want.items():
        for e in entries:
            sec = e.get("security")
            if sec is not None and sec.get("safe") is not True:
                if e["path"] not in allow:
                    raise Stop("integrity", "flagged-unsafe", f"{e['path']}: the Hub's scan says {sec.get('status')!r} "
                               f"(--allow-unsafe-file overrides that file and is recorded)")
                flagged.append({"path": e["path"], "status": sec.get("status")})
    return flagged


def disk_table(rows):
    """Required bytes per volume against free bytes; rows are (label, path, bytes)."""
    vols = {}
    for label, path, need in rows:
        p = path
        while not os.path.exists(p):
            p = os.path.dirname(p)
        dev = os.stat(p).st_dev
        v = vols.setdefault(dev, {"labels": [], "need": 0, "free": DISK_USAGE(p).free})
        v["labels"].append(label)
        v["need"] += need
    short = []
    for v in vols.values():
        say(f"disk: {', '.join(v['labels'])}: need {v['need'] / 1e9:.2f} GB, free {v['free'] / 1e9:.2f} GB")
        if v["need"] > v["free"]:
            short.append(v)
    return [{"labels": v["labels"], "need": v["need"], "free": v["free"]} for v in vols.values()], short


def store_estimate(cap):
    """An estimate of the store's bytes, from the planned tokens: the cap plus one shard."""
    planned = cap + SHARD_TOKENS
    docs = planned // MEAN_DOC_TOKENS
    return int(1.05 * (planned * 2 * (1 + 1 / 1024) + docs * 8)) + DISK_FIXED


def dir_bytes(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def hf_home():
    return os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def summary(lines):
    print("\n".join(lines), flush=True)


def check_mode(args, found, drift):
    hub = (make_hub or HubClient)()
    attempts = {}
    plan, files, want, py = select(hub, args.sample, args.files, attempts)
    out = []
    for repo, p in plan.items():
        moved = "" if p["main"] == p["rev"] else \
            "  (main has moved: report it in the thread; changing a pin is a code change)"
        out.append(f"{repo}: pin {p['rev']}  live main {p['main']}{moved}")
        i = p["info"]
        out.append(f"  gated {i['gated']!r}, private {i['private']!r}, licence {i['license']!r}")
    out.append(f"{args.sample}: {len(files)} files, {sum(e['size'] for e in files):,} bytes listed")
    if py:
        out.append(f"dataset root *.py (recorded, never fetched): {py}")
    for repo, entries in want.items():
        for e in entries:
            sec = e.get("security")
            flag = "unscanned" if sec is None else ("" if sec.get("safe") is True else f"FLAGGED {sec.get('status')!r}")
            kind = "lfs" if e.get("lfs") else ("NO LFS (git blob)" if upstream(e)[0] else "NO UPSTREAM HASH")
            out.append(f"  fetch {e['path']}  {e['size']:,} B  {kind}  {flag}".rstrip())
    out.append("HF_* variables set: " + (", ".join(sorted(k for k in os.environ if k.startswith("HF_"))) or "none"))
    rows = [("raw files", os.path.join(args.out, "raw"), sum(e["size"] for es in want.values() for e in es)),
            ("store", args.out, store_estimate(args.cap_tokens)),
            ("Xet cache allowance (download root)", os.path.join(args.out, "raw"), XET_ALLOWANCE),
            ("Xet cache allowance (HF_HOME)", hf_home(), XET_ALLOWANCE)]
    _, short = disk_table(rows)
    out.append(f"disk: {'SHORT on ' + str(len(short)) + ' volume(s)' if short else 'enough on every volume'}")
    out.append(f"versions: {found}{' (drift: ' + str(drift) + ')' if drift else ''}")
    summary(out)
    return EXIT["refused"] if short else 0


def verify_download(path, rawdir, e):
    """Check one downloaded file: no link, inside its directory, exact size, matching hash."""
    expect = os.path.join(rawdir, e["path"])
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as err:
        if err.errno == errno.ELOOP:
            raise Stop("integrity", "linked-file", f"{e['path']} is a symlink") from None
        raise
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise Stop("integrity", "linked-file", f"{e['path']}: not a regular file with one link")
        if os.path.realpath(path) != os.path.realpath(expect) or not under(path, rawdir):
            raise Stop("integrity", "linked-file", f"{e['path']} resolves outside its download directory")
        if st.st_size != e["size"]:
            raise Stop("integrity", "size-mismatch", f"{e['path']}: {st.st_size} bytes, listing {e['size']}")
        digest = sha256_fd(fd)
        kind, want = upstream(e)
        if kind == "lfs-sha256":
            ok = digest == want and e["lfs"]["size"] == st.st_size
        else:
            os.lseek(fd, 0, os.SEEK_SET)
            data = b""
            while b := os.read(fd, 1 << 24):
                data += b
            ok = git_blob_sha1(data) == want
        if not ok:
            raise Stop("integrity", "hash-mismatch", f"{e['path']}: content does not match the listing's {kind}")
    finally:
        os.close(fd)
    os.chmod(path, 0o444)
    return digest


def lock(path, shared=False):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise Stop("refused", "locked", f"{path} is held by another run") from None
    return fd


def fetch_mode(args, found, drift):
    if list(args.confirm_pins or []) != [DATASET_REV, TOKENIZER_REV]:
        for r in args.confirm_pins or []:
            check_revision(r, "--confirm-pins")
        raise Stop("config", "pins-not-confirmed", "--confirm-pins must equal the script's two pins, as the check "
                   "mode printed them")
    out = os.path.abspath(args.out)
    hub = (make_hub or HubClient)()
    attempts = {}
    home = hf_home()
    before = {"hf_home": dir_bytes(home), "hf_home_xet": dir_bytes(os.path.join(home, "xet")),
              "raw_xet": dir_bytes(os.path.join(out, "raw", ".xet-cache"))}
    plan, files, want, py = select(hub, args.sample, args.files, attempts)
    flagged = security_check(want, set(args.allow_unsafe_file or []))
    raw = os.path.join(out, "raw")
    missing = 0
    for repo, entries in want.items():
        rawdir = os.path.join(raw, plan[repo]["key"], plan[repo]["rev"])
        for e in entries:
            p = os.path.join(rawdir, e["path"])
            if not (os.path.exists(p) and os.path.getsize(p) == e["size"]):
                missing += e["size"]
    table, short = disk_table([("raw files", raw, missing), ("store", out, store_estimate(args.cap_tokens)),
                               ("Xet cache allowance (download root)", raw, XET_ALLOWANCE),
                               ("Xet cache allowance (HF_HOME)", home, XET_ALLOWANCE)])
    if short:
        raise Stop("refused", "disk", "too little free space")
    empty_cache = os.path.join(raw, ".hf-cache-empty")
    os.makedirs(empty_cache, exist_ok=True)
    os.makedirs(os.path.join(raw, ".xet-cache"), exist_ok=True)
    report = []
    for repo, entries in want.items():
        p = plan[repo]
        rawdir = os.path.join(raw, p["key"], p["rev"])
        os.makedirs(rawdir, exist_ok=True)
        fd = lock(os.path.join(rawdir, ".lock"))
        try:
            records = []
            for e in entries:
                path = retry(lambda: hub.download(repo, p["type"], p["rev"], e["path"], rawdir, empty_cache),
                             e["path"], attempts)
                if os.listdir(empty_cache):
                    raise Stop("integrity", "cache-written", "the library wrote into the empty cache_dir")
                digest = verify_download(path, rawdir, e)
                kind = "dataset-file" if repo == DATASET else "tokenizer-file"
                records.append({"kind": kind, "host": "huggingface.co", "repo_id": repo, "revision": p["rev"],
                                "path": e["path"], "sha256": digest, "size": e["size"], "blob_id": e["blob_id"],
                                "lfs_sha256": (e.get("lfs") or {}).get("sha256"), "xet_hash": e.get("xet_hash"),
                                "license_tag": p["info"]["license"], "license_source": "card_data.license at the pin",
                                "gated": p["info"]["gated"], "fetched_at": now(), "attempts": attempts.get(e["path"]),
                                "fetched_by": "a21-stand-in", "run_id": None, "run_attempt": None, "source_url": None,
                                "security": e.get("security"),
                                "security_override": e["path"] in set(args.allow_unsafe_file or [])})
                report.append(f"  {e['path']}  {e['size']:,} B  sha256 {digest}")
            sums = "".join(f"{r['sha256']}  {r['path']}\n" for r in sorted(records, key=lambda r: r["path"]))
            write_file(os.path.join(rawdir, "SHA256SUMS"), sums.encode())
            write_file(os.path.join(rawdir, "listing-root.json"), canon_json(strip_security(p["root"])))
            if repo == DATASET:
                name = "listing-" + args.sample.replace("/", "-") + ".json"
                write_file(os.path.join(rawdir, name), canon_json(strip_security(files)))
            fetch_rec = {"repo_id": repo, "repo_type": p["type"], "revision": p["rev"], "pins_confirmed": True,
                         "main_commit_at_fetch": p["main"], "info": p["info"], "files": records,
                         "sample": args.sample if repo == DATASET else None,
                         "py_scan_root": py if repo == DATASET else None,
                         "endpoint": ENDPOINT, "token_sent": False,
                         "token_configured": "HF_TOKEN" in os.environ or os.path.exists(os.path.join(home, "token")),
                         "hf_home_set": "HF_HOME" in os.environ,
                         "hf_env": sorted(k for k in os.environ if k.startswith("HF_")),
                         "flagged_overridden": flagged, "versions": found, "env_drift": drift, "disk": table}
            write_file(os.path.join(rawdir, "fetch.json"), json.dumps(fetch_rec, indent=1, sort_keys=True).encode())
        finally:
            os.close(fd)
    after = {"hf_home": dir_bytes(home), "hf_home_xet": dir_bytes(os.path.join(home, "xet")),
             "raw_xet": dir_bytes(os.path.join(raw, ".xet-cache"))}
    with open(os.path.join(raw, "cache-sizes.json"), "w") as f:
        json.dump({"before": before, "after": after}, f, indent=1)
    summary([f"fetched {args.sample} at {DATASET_REV} and {TOKENIZER_REPO} at {TOKENIZER_REV}"] + report +
            [f"cache bytes before {before}, after {after}"])
    return 0


def strip_security(entries):
    """Listings are saved without `security`, which a re-scan can change."""
    return [{k: v for k, v in e.items() if k != "security"} for e in sorted(entries, key=lambda e: e["path"])]


# ---- Parsing untrusted bytes ------------------------------------------------------------------------------
class Input:
    """A Parquet file opened once without following links, hashed, then handed to pyarrow."""

    def __init__(self, path, rec):
        self.path, self.rec = path, rec
        check_dataless(path)
        self.fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        st = os.fstat(self.fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size != rec["size"]:
            raise Stop("integrity", "size-mismatch",
                       f"{rec['path']}: {st.st_size} bytes, recorded {rec['size']}")
        self.mark = (st.st_size, st.st_ino, st.st_mtime_ns)
        if sha256_fd(self.fd) != rec["sha256"]:
            raise Stop("integrity", "hash-mismatch", f"{rec['path']} differs from its fetch record")
        event("hashed", path)
        os.lseek(self.fd, 0, os.SEEK_SET)
        self.f = os.fdopen(self.fd, "rb")
        self.pf = pq.ParquetFile(self.f, arrow_extensions_enabled=False)

    def unchanged(self):
        st = os.fstat(self.fd)
        if (st.st_size, st.st_ino, st.st_mtime_ns) != self.mark:
            raise Stop("integrity", "input-changed", f"{self.rec['path']} changed while it was read")

    def close(self):
        self.f.close()


def walk_fields(fields, prefix=""):
    for f in fields:
        yield prefix + f.name, f
        t = f.type
        if pa.types.is_struct(t):
            yield from walk_fields([t.field(i) for i in range(t.num_fields)], prefix + f.name + ".")
        elif pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
            yield from walk_fields([t.value_field], prefix + f.name + ".")
        elif pa.types.is_map(t):
            yield from walk_fields([t.key_field, t.item_field], prefix + f.name + ".")


def schema_check(inp):
    """Refuse extension types and extension metadata anywhere; text and id must be strings."""
    s = inp.pf.schema_arrow
    for name, f in walk_fields(list(s)):
        if isinstance(f.type, pa.ExtensionType) or (f.metadata and b"ARROW:extension:name" in f.metadata):
            raise Stop("integrity", "parquet-extension-rejected", f"{inp.rec['path']}: field {name} has "
                       f"an extension type or extension metadata")
    strings = (pa.string(), pa.large_string())
    for col in ("text", "id"):
        if col not in s.names or s.field(col).type not in strings:
            raise Stop("integrity", "schema", f"{inp.rec['path']}: column {col!r} missing or not a string")
    cols = ["text", "id"]
    extra = {}
    if "token_count" in s.names and s.field("token_count").type == pa.int64():
        cols.append("token_count")
        extra["token_count"] = True
    if "dump" in s.names and s.field("dump").type in strings:
        cols.append("dump")
        extra["dump"] = True
    return cols, extra


def prescan(inp):
    """Before any document of the file is written: per-row-group size caps from the footer, then the share
    of documents over the per-document limit."""
    md_ = inp.pf.metadata
    sizes, oversized, big = [], 0, 0
    text_i = inp.pf.schema_arrow.get_field_index("text")
    for i in range(md_.num_row_groups):
        rg = md_.row_group(i)
        col = rg.column(text_i)
        if col.total_uncompressed_size > ROWGROUP_TEXT_MAX:
            raise Stop("integrity", "row-group-too-large", f"{inp.rec['path']} row group {i}: "
                       f"{col.total_uncompressed_size} bytes of text > {ROWGROUP_TEXT_MAX}")
        sizes.append({"row_group": i, "num_rows": rg.num_rows, "total_byte_size": rg.total_byte_size,
                      "text_uncompressed": col.total_uncompressed_size, "text_compressed": col.total_compressed_size})
        for b in inp.pf.iter_batches(batch_size=BATCH_ROWS, row_groups=[i], columns=["text"]):
            lens = pc.binary_length(b.column(0).cast(pa.binary()))
            oversized += int(pc.sum(pc.greater(lens, MAX_DOC_BYTES)).as_py() or 0)
            big += int(pc.sum(pc.greater(lens, BIG_DOC_BYTES)).as_py() or 0)
    if oversized > md_.num_rows // 10_000:
        raise Stop("integrity", "oversized-rows", f"{inp.rec['path']}: {oversized} documents over {MAX_DOC_BYTES} "
                   f"bytes, more than ⌊{md_.num_rows} ÷ 10,000⌋")
    return sizes, oversized, big


def canonical(raw):
    """The stored bytes, decoded strictly; returns (text, None) or (None, drop reason)."""
    if raw is None:
        return None, "null"
    if not isinstance(raw, bytes):
        return None, "non_string"
    if len(raw) > MAX_DOC_BYTES:
        return None, "too_long"
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, "invalid_utf8"
    if text == "":
        return None, "empty"
    if text.isspace():
        return None, "whitespace"
    return text, None


def heldout_digest(raw):
    return hashlib.blake2b(raw, digest_size=8, person=HELDOUT_PERSON).digest()


def is_heldout(digest):
    return int.from_bytes(digest, "big") < HELDOUT_LIMIT


# ---- The store --------------------------------------------------------------------------------------------
SPLIT_DIRS = ("train", "heldout", "local/train", "local/heldout", "sources", "tokenizer")
ROOT_FILES = ("fingerprint.json", "manifest.json", "provenance.json", "SHA256SUMS", "token_bytes.u16")
SOURCE_FILES = ("dataset-listing-sample.json", "dataset-listing-root.json", "tokenizer-listing-root.json",
                "dataset-README.md", "tokenizer-README.md")
FORMS = {"": [re.escape(n) for n in ROOT_FILES],
         "train": [r"[0-9]{6}\.(bin|idx|json)", r"manifest\.json"],
         "heldout": [r"[0-9]{6}\.(bin|idx|bytes|json)", r"part-[0-9]{6}\.(bin|idx|bytes|digests)", r"manifest\.json"],
         "local/train": [r"[0-9]{6}\.ids"],
         "local/heldout": [r"[0-9]{6}\.ids", r"part-[0-9]{6}\.ids"],
         "sources": [re.escape(n) for n in SOURCE_FILES],
         "tokenizer": [re.escape(n) for n in TOKENIZER_FILES[:3]] + [r"tokenizer\.prefix-[0-9]+\.json"]}


def is_tmp(folder, name):
    return name.endswith(".tmp") and any(re.fullmatch(f, name[:-4]) for f in FORMS[folder])


def u16(ids):
    return np.asarray(ids).astype(store_fmt.U16)


class Unit:
    """One row group's running record."""

    def __init__(self, path, rg, num_rows):
        self.rec = {"file": path, "row_group": rg, "num_rows": num_rows, "rows_read": 0, "complete": False,
                    "docs_accepted": 0, "drops": {d: 0 for d in DROPS}, "over_16mib": 0, "utf8_bytes": 0,
                    "tokens": 0, "tokens_heldout": 0}
        self.canon, self.content = hashlib.sha256(), hashlib.sha256()

    def fold(self, raw, ids):
        self.canon.update(struct.pack("<Q", len(raw)) + raw)
        self.content.update(struct.pack("<Q", len(ids)) + u16(ids).tobytes())
        self.rec["docs_accepted"] += 1
        self.rec["utf8_bytes"] += len(raw)

    def record(self, rows_read, complete):
        return dict(self.rec, rows_read=rows_read, complete=complete, canon_sha256=self.canon.hexdigest(),
                    content_sha256=self.content.hexdigest())


class Shard:
    """Train shard i and its held-out part, written as *.tmp until `commit`."""

    def __init__(self, st, i):
        self.st, self.i = st, i
        n = f"{i:06d}"
        self.bin = Out(os.path.join(st, "train", n + ".bin"))
        self.ids = Out(os.path.join(st, "local", "train", n + ".ids"))
        self.hbin = Out(os.path.join(st, "heldout", f"part-{n}.bin"))
        self.hbytes = Out(os.path.join(st, "heldout", f"part-{n}.bytes"))
        self.hdig = Out(os.path.join(st, "heldout", f"part-{n}.digests"))
        self.hids = Out(os.path.join(st, "local", "heldout", f"part-{n}.ids"))
        self.offsets, self.hoffsets = [0], [0]
        self.tokens = self.htokens = self.utf8 = self.hutf8 = 0
        self.max_id, self.hmax = None, None
        self.units = []

    def add(self, doc, doc_id, nbytes, held, digest=None):
        b = doc.tobytes()
        line = (json.dumps([doc_id, nbytes], ensure_ascii=True) + "\n").encode()
        m = int(doc.max())
        if held:
            self.hbin.write(b)
            self.hbytes.write(struct.pack("<Q", nbytes))
            self.hdig.write(digest)
            self.hids.write(line)
            self.htokens += len(doc)
            self.hoffsets.append(self.htokens)
            self.hutf8 += nbytes
            self.hmax = m if self.hmax is None else max(self.hmax, m)
        else:
            self.bin.write(b)
            self.ids.write(line)
            self.tokens += len(doc)
            self.offsets.append(self.tokens)
            self.utf8 += nbytes
            self.max_id = m if self.max_id is None else max(self.max_id, m)

    def outs(self):
        return [self.bin, self.idx, self.ids, self.hbin, self.hidx, self.hbytes, self.hdig, self.hids]

    def commit(self, end_cursor, totals, stats, stop=None):
        n = f"{self.i:06d}"
        self.idx = Out(os.path.join(self.st, "train", n + ".idx"))
        self.idx.write(np.asarray(self.offsets, store_fmt.U64).tobytes())
        self.hidx = Out(os.path.join(self.st, "heldout", f"part-{n}.idx"))
        self.hidx.write(np.asarray(self.hoffsets, store_fmt.U64).tobytes())
        commit(self.outs())
        rec = {"path": f"train/{n}.bin", "split": "train", "index": self.i, "dtype": "<u2", "tokens": self.tokens,
               "documents": len(self.offsets) - 1, "utf8_bytes": self.utf8, "max_id": self.max_id,
               "sha256_bin": self.bin.h.hexdigest(), "sha256_idx": self.idx.h.hexdigest(), "end_cursor": end_cursor,
               "files": [o.entry(self.st) for o in self.outs()],
               "part": {"documents": len(self.hoffsets) - 1, "tokens": self.htokens, "utf8_bytes": self.hutf8,
                        "max_id": self.hmax},
               "units": self.units, "totals": dict(totals), "stats": json.loads(json.dumps(stats)), "stop": stop}
        write_file(os.path.join(self.st, "train", n + ".json"), canon_json(rec))
        event("record", os.path.join(self.st, "train", n + ".json"))
        return rec

    def abort(self):
        for o in (self.bin, self.ids, self.hbin, self.hbytes, self.hdig, self.hids, getattr(self, "idx", None),
                  getattr(self, "hidx", None)):
            if o is not None:
                o.abort()


def records(st, split, pattern=r"[0-9]{6}\.json"):
    d = os.path.join(st, split)
    out = []
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if re.fullmatch(pattern, name):
            with open(os.path.join(d, name), "rb") as f:
                try:
                    out.append(json.loads(f.read()))
                except ValueError:
                    raise Stop("integrity", "torn-record", f"{split}/{name} does not parse") from None
    for k, r in enumerate(out):
        if r.get("index") != k:
            raise Stop("integrity", "torn-record", f"{split}: records are not contiguous at {k}")
    return out


def check_record_files(st, rec, rehash):
    for e in rec["files"]:
        p = os.path.join(st, e["path"])
        if not os.path.isfile(p) or os.path.islink(p) or os.path.getsize(p) != e["size"]:
            raise Stop("integrity", "torn-record", f"{e['path']}: missing or not {e['size']} bytes")
        if rehash and sha256_path(p) != e["sha256"]:
            raise Stop("integrity", "torn-record", f"{e['path']}: SHA-256 differs from its record")


def resume_scan(st):
    """Check the records (deleting nothing if one is torn), then delete the script's own temporary files
    and any shard or part file without a record, within the store's own folders only."""
    for folder in [""] + list(SPLIT_DIRS):
        d = os.path.join(st, folder)
        if os.path.isdir(d):
            for e in os.scandir(d):
                if e.is_symlink():
                    raise Stop("integrity", "symlink-in-store", f"{folder}/{e.name} is a symlink")
    train = records(st, "train")
    final = records(st, "heldout")
    for k, r in enumerate(train):
        check_record_files(st, r, rehash=k == len(train) - 1)
    for k, r in enumerate(final):
        check_record_files(st, r, rehash=k == len(final) - 1)
    have_t, have_h = len(train), len(final)
    for folder in [""] + list(SPLIT_DIRS):
        d = os.path.join(st, folder)
        if not os.path.isdir(d):
            continue
        for e in os.scandir(d):
            name = e.name
            if not e.is_file(follow_symlinks=False):
                continue
            kill = is_tmp(folder, name)
            m = re.fullmatch(r"(part-)?([0-9]{6})\.(bin|idx|bytes|digests|ids)", name)
            if m and folder in ("train", "local/train") and not m.group(1):
                kill = kill or int(m.group(2)) >= have_t
            elif m and m.group(1) and folder in ("heldout", "local/heldout"):
                kill = kill or int(m.group(2)) >= have_t
            elif m and folder in ("heldout", "local/heldout"):
                kill = kill or int(m.group(2)) >= have_h
            if kill:
                os.unlink(os.path.join(d, name))
                event("delete", os.path.join(d, name))
    return train, final


class Prep:
    def __init__(self, args, st, files, raw_dir, tok, golden, versions, drift, fp, overrides):
        self.args, self.st, self.files, self.raw_dir, self.tok = args, st, files, raw_dir, tok
        self.golden, self.versions, self.drift, self.fp, self.overrides = golden, versions, drift, fp, overrides
        self.tb = tok.token_bytes()
        self.V = tok.vocab
        self.totals = {"tokens_train_actual": 0, "documents_train": 0, "utf8_bytes_train": 0, "tokens_heldout": 0,
                       "documents_heldout": 0, "utf8_bytes_heldout": 0, "gpt2_token_count_sum": 0,
                       "gpt2_missing": False}
        self.stats = {"len_hist": {}, "crop_excess": 0, "tokens_all": 0, "dump_counts": {}, "gpt2_docs": 0,
                      "gpt2_bad": 0, "dump_missing": False}
        self.rowgroups, self.t0, self.read_bytes = [], time.time(), 0
        self.shard = None
        self.inp = None

    def check_doc(self, ids, raw, unit_index):
        """The id range check, and the check that the tokens' byte lengths sum to the document's bytes.

        Every EXACT_EVERY-th document's tokens are also joined and compared with its bytes.
        """
        if ids.size and (int(ids.min()) < store_fmt.FIRST_ID or int(ids.max()) > self.V - 1):
            raise Stop("integrity", "id-range", f"ids in [{int(ids.min())}, {int(ids.max())}] outside "
                       f"[{store_fmt.FIRST_ID}, {self.V - 1}]")
        if int(self.tb[ids].sum()) != len(raw):
            raise Stop("integrity", "byte-check", f"token bytes sum to {int(self.tb[ids].sum())}, the document has "
                       f"{len(raw)}")
        if unit_index % EXACT_EVERY == 0 and b"".join(self.tok.raw(int(i)) for i in ids) != raw:
            raise Stop("integrity", "byte-check", "the tokens' raw bytes do not rebuild the document")

    def encode(self, texts):
        out, batch, size = [], [], 0
        for t in texts:
            n = len(t.encode())
            if batch and size + n > ENCODE_BYTES:
                out += self.tok.encode(batch)
                batch, size = [], 0
            batch.append(t)
            size += n
        if batch:
            out += self.tok.encode(batch)
        return out

    def stat_doc(self, ntok, token_count, dump, extra):
        s = self.stats
        k = str(max(ntok - 1, 1).bit_length())
        s["len_hist"][k] = s["len_hist"].get(k, 0) + 1
        s["crop_excess"] += max(0, ntok - CROP_ROW)
        s["tokens_all"] += ntok
        if extra.get("dump"):
            key = (dump or "").encode("ascii", "backslashreplace").decode()[:64]
            s["dump_counts"][key] = s["dump_counts"].get(key, 0) + 1
        if extra.get("token_count"):
            if token_count is None or token_count < 0:
                s["gpt2_bad"] += 1
            else:
                self.totals["gpt2_token_count_sum"] += int(token_count)
                s["gpt2_docs"] += 1

    def run(self, cursor, next_index):
        """Read from `cursor` = (file index, row group, row) until the cap; returns the stop record."""
        cap = self.args.cap_tokens
        fi0, rg0, row0 = cursor
        self.shard = Shard(self.st, next_index)
        for fi in range(fi0, len(self.files)):
            rec = self.files[fi]
            self.inp = Input(os.path.join(self.raw_dir, rec["path"]), rec)
            cols, extra = schema_check(self.inp)
            if not extra.get("token_count"):
                self.totals["gpt2_missing"] = True
            if not extra.get("dump"):
                self.stats["dump_missing"] = True
            sizes, _, big = prescan(self.inp)
            self.rowgroups += [dict(s, file=rec["path"]) for s in sizes]
            n_rg = self.inp.pf.metadata.num_row_groups
            for rg in range(rg0 if fi == fi0 else 0, n_rg):
                replay = row0 if (fi == fi0 and rg == rg0) else 0
                unit = Unit(rec["path"], rg, self.inp.pf.metadata.row_group(rg).num_rows)
                row = 0
                for batch in self.inp.pf.iter_batches(batch_size=BATCH_ROWS, row_groups=[rg], columns=cols):
                    raws = batch.column("text").cast(pa.binary()).to_pylist()
                    ids_col = batch.column("id").to_pylist()
                    tcs = batch.column("token_count").to_pylist() if extra.get("token_count") else [None] * len(raws)
                    dumps = batch.column("dump").to_pylist() if extra.get("dump") else [None] * len(raws)
                    canon = [canonical(raw) for raw in raws]
                    kept = [j for j, (_, drop) in enumerate(canon) if drop is None]
                    encoded = dict(zip(kept, self.encode([canon[j][0] for j in kept])))
                    for j, raw in enumerate(raws):  # in row order, so a stop record counts only rows before it
                        if STOP_FLAG:
                            raise Stop("interrupted", "interrupted",
                                       "a signal arrived; committed shards are kept")
                        r, drop = row + j, canon[j][1]
                        if drop:
                            unit.rec["drops"][drop] += 1
                            unit.rec["over_16mib"] += drop == "too_long" and len(raw) > BIG_DOC_BYTES
                            continue
                        ids, doc_id, tc, dump = encoded[j], ids_col[j], tcs[j], dumps[j]
                        self.check_doc(ids, raw, unit.rec["docs_accepted"])
                        digest = heldout_digest(raw)
                        held = is_heldout(digest)
                        doc = np.concatenate([np.zeros(1, store_fmt.U16), u16(ids)])
                        if r < replay:  # committed already: restore the unit's running digests only
                            unit.fold(raw, ids)
                            unit.rec["tokens_heldout" if held else "tokens"] += len(doc)
                            continue
                        if not held and self.shard.tokens and self.shard.tokens + len(doc) > SHARD_TOKENS:
                            done = self.totals["tokens_train_actual"] + self.shard.tokens >= cap
                            stop = None
                            end = {"file": rec["path"], "row_group": rg, "row": r}
                            self.totals["tokens_train_actual"] += self.shard.tokens
                            if done:
                                stop = {"cursor": end, "unit": unit.record(r, False)}
                            self.inp.unchanged()
                            self.shard.commit(end, self.totals, self.stats, stop)
                            say(f"shard {self.shard.i:06d}: {self.totals['tokens_train_actual']:,} train tokens")
                            if done:
                                self.inp.close()
                                return stop
                            self.shard = Shard(self.st, self.shard.i + 1)
                        self.shard.add(doc, doc_id, len(raw), held, digest if held else None)
                        unit.fold(raw, ids)
                        self.stat_doc(len(doc), tc, dump, extra)
                        if held:
                            unit.rec["tokens_heldout"] += len(doc)
                            self.totals["tokens_heldout"] += len(doc)
                            self.totals["documents_heldout"] += 1
                            self.totals["utf8_bytes_heldout"] += len(raw)
                        else:
                            unit.rec["tokens"] += len(doc)
                            self.totals["documents_train"] += 1
                            self.totals["utf8_bytes_train"] += len(raw)
                    row += len(raws)
                    self.read_bytes += batch.nbytes
                self.shard.units.append(unit.record(unit.rec["num_rows"], True))
            self.inp.unchanged()
            self.inp.close()
            self.inp = None
            if fi == fi0:
                el = time.time() - self.t0
                rate = self.totals["tokens_train_actual"] + self.shard.tokens
                if el > 0 and rate:
                    left = max(cap - rate, 0) / (rate / el)
                    say(f"projected finish in about {left / 60:.0f} min (ESTIMATE from the first file's throughput)")
        raise Stop("unavailable", "input-exhausted", "the fetched files end before the cap; fetch more files (a larger "
                   "--files) and run prepare again to resume")


def heldout_finals(st, train):
    """Concatenate the parts in order into the final held-out shards, skipping those with a record."""
    final = records(st, "heldout")
    docs = []  # (part index, position in the part, slice start, end) over the concatenated parts
    sizes = []
    for r in train:
        n = f"{r['index']:06d}"
        idx = np.fromfile(os.path.join(st, "heldout", f"part-{n}.idx"), store_fmt.U64).astype(np.int64)
        for j, (a, b) in enumerate(zip(idx[:-1], idx[1:])):
            docs.append((r["index"], j, int(a), int(b)))
            sizes.append(int(b - a))
    cuts = store_fmt.close_points(sizes, SHARD_TOKENS) if sizes else [0]
    cache = {}

    def part(i):
        if i not in cache:
            n = f"{i:06d}"
            cache.clear()
            cache[i] = (np.fromfile(os.path.join(st, "heldout", f"part-{n}.bin"), store_fmt.U16),
                        np.fromfile(os.path.join(st, "heldout", f"part-{n}.bytes"), store_fmt.U64),
                        open(os.path.join(st, "local", "heldout", f"part-{n}.ids"), "rb").read().splitlines(True))
        return cache[i]
    out = list(final)
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        if k < len(final):
            continue
        n = f"{k:06d}"
        o_bin, o_bytes = Out(os.path.join(st, "heldout", n + ".bin")), Out(os.path.join(st, "heldout", n + ".bytes"))
        o_ids = Out(os.path.join(st, "local", "heldout", n + ".ids"))
        offs, tok, utf8, mx_ = [0], 0, 0, None
        for (p, j, s, e) in docs[a:b]:
            pb, pbytes, pids = part(p)
            d = pb[s:e]
            o_bin.write(d.tobytes())
            o_bytes.write(pbytes[j].tobytes())
            o_ids.write(pids[j])
            tok += e - s
            offs.append(tok)
            utf8 += int(pbytes[j])
            mx_ = int(d.max()) if mx_ is None else max(mx_, int(d.max()))
        o_idx = Out(os.path.join(st, "heldout", n + ".idx"))
        o_idx.write(np.asarray(offs, store_fmt.U64).tobytes())
        outs = [o_bin, o_idx, o_bytes, o_ids]
        commit(outs)
        rec = {"path": f"heldout/{n}.bin", "split": "heldout", "index": k, "dtype": "<u2", "tokens": tok,
               "documents": len(offs) - 1, "utf8_bytes": utf8, "max_id": mx_, "sha256_bin": o_bin.h.hexdigest(),
               "sha256_idx": o_idx.h.hexdigest(), "end_cursor": None, "files": [o.entry(st) for o in outs]}
        write_file(os.path.join(st, "heldout", n + ".json"), canon_json(rec))
        event("record", os.path.join(st, "heldout", n + ".json"))
        out.append(rec)
    return out


def store_files(st):
    """Every file in the store, relative, sorted; SHA256SUMS lists all but provenance.json, .lock, itself and
    Finder's .DS_Store files, which Finder rewrites whenever it shows a folder."""
    out = []
    for dp, dns, fs in os.walk(st):
        dns.sort()
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), st)
            if rel not in ("provenance.json", ".lock", "SHA256SUMS") and f != ".DS_Store":
                out.append(rel)
    return sorted(out)


def build_manifests(st):
    """The split manifests and the root manifest body, from the records and the files.
    The prepare mode writes what this returns; the verify mode compares it with what was written."""
    with open(os.path.join(st, "fingerprint.json"), "rb") as f:
        fpj = json.loads(f.read())
    train, final = records(st, "train"), records(st, "heldout")
    last = train[-1]
    V = fpj["vocab"]

    def split_manifest(recs, split):
        names = [os.path.basename(r["path"]) for r in recs]
        m = {"dtype": "uint16", "eos": 0, "vocab": V, "shards": names, "tokens": [r["tokens"] for r in recs],
             "format": store_fmt.FORMAT, "split": split, "sep_id": 0, "sep_position": "start", "reserved_ids": [1, 16],
             "idx": [n[:-4] + ".idx" for n in names], "documents": [r["documents"] for r in recs],
             "utf8_bytes": [r["utf8_bytes"] for r in recs], "max_id": [r["max_id"] for r in recs],
             "sha256_bin": [r["sha256_bin"] for r in recs], "sha256_idx": [r["sha256_idx"] for r in recs],
             "end_cursor": [r["end_cursor"] for r in recs]}
        if split == "heldout":
            docs, toks = sum(m["documents"]), sum(m["tokens"])
            m |= {"bytes": [n[:-4] + ".bytes" for n in names], "documents_total": docs, "tokens_total": toks,
                  "target_tokens_total": toks - docs, "utf8_bytes_total": sum(m["utf8_bytes"])}
        return m
    mt, mh = split_manifest(train, "train"), split_manifest(final, "heldout")
    stream, digests, firsts = hashlib.sha256(), [], {k: None for k in PREFIX_K}
    pref = hashlib.sha256()
    count = 0
    for r in train:
        n = f"{r['index']:06d}"
        dg = open(os.path.join(st, "heldout", f"part-{n}.digests"), "rb").read()
        digests += [dg[i:i + 8] for i in range(0, len(dg), 8)]
        b = np.fromfile(os.path.join(st, "heldout", f"part-{n}.bin"), store_fmt.U16)
        idx = np.fromfile(os.path.join(st, "heldout", f"part-{n}.idx"), store_fmt.U64).astype(np.int64)
        stream.update(b.tobytes())
        for a, e in zip(idx[:-1], idx[1:]):
            pref.update(b[a:e].tobytes())
            count += 1
            if count in firsts:
                firsts[count] = pref.hexdigest()
    with open(os.path.join(st, "sources", "dataset-listing-sample.json"), "rb") as f:
        listing = json.loads(f.read())
    listed = {e["path"]: e for e in listing}
    units = [u for r in train for u in r["units"]] + ([last["stop"]["unit"]] if last.get("stop") else [])
    read = {}
    for u in units:
        read.setdefault(u["file"], {"rows": 0, "row_groups": set()})
        read[u["file"]]["rows"] += u["rows_read"]
        read[u["file"]]["row_groups"].add(u["row_group"])
    # Every Parquet file has an lfs.sha256 (select refuses one without), and fetch and prepare both check the
    # file against it, so the listing in the store gives each file's SHA-256: the store does not depend on
    # how many files were fetched before a resume.
    files = [{"path": p, "size": listed[p]["size"], "sha256": listed[p]["lfs"]["sha256"],
              "lfs_sha256": listed[p]["lfs"]["sha256"], "blob_id": listed[p]["blob_id"],
              "xet_hash": listed[p]["xet_hash"], "row_groups": len(read[p]["row_groups"]), "read": read[p]["rows"]}
             for p in sorted(read) if read[p]["rows"] > 0]
    tot = last["totals"]
    src = fpj["inputs"]["source"]
    sample = src["sample"]
    tokfiles = fpj["tokenizer_files"]
    body = {
        "schema": SCHEMA, "prep_format_version": PREP_FORMAT_VERSION, "stand_in_for": STAND_IN_FOR,
        "research_only": True, "decision": DECISION,
        "source": {"repo_id": DATASET, "publisher": "HuggingFaceFW", "sample": sample, "config": SAMPLES[sample],
                   "revision": src["revision"], "access": "public", "data_files": sample + "/*.parquet",
                   "source_key": "fineweb-edu-" + sample.replace("/", "-").lower(), "text_column": "text",
                   "filters": [],
                   "listing_bytes_total": sum(e["size"] for e in listing),
                   "size": {"bytes": sum(e["size"] for e in listing), "documents": None, "files": len(listing)},
                   "py_scan": {"scope": ["dataset root", sample], "root_py_files": fpj["py_scan_root"]},
                   "files": files, "stop_cursor": last["stop"]["cursor"] if last.get("stop") else None},
        "licence": {"hf_license_tag": fpj["licence_tags"][DATASET], "spdx_expected": SPDX,
                    "attribution_notice": ATTRIBUTION, "card_sha256": fpj["card_sha256"][DATASET],
                    "license_evidence_url":
                        f"https://huggingface.co/datasets/{DATASET}/blob/{src['revision']}/README.md",
                    "redistributable": False},
        "tokenizer": {"repo_id": TOKENIZER_REPO, "revision": fpj["tokenizer_revision"], "files": tokfiles,
                      "hf_license_tag": fpj["licence_tags"][TOKENIZER_REPO],
                      "vocab_size": V, "golden": fpj["golden"], "encode_special_tokens": True,
                      "add_special_tokens": False, "merge_prefix": fpj["inputs"]["params"]["merge_prefix"]},
        "rules": {"canon": CANON, "max_doc_bytes": fpj["max_doc_bytes"], "abort_per_10000": 1,
                  "heldout_rule": HELDOUT_RULE, "sep_id": 0, "sep_position": "start",
                  "shard_tokens": fpj["inputs"]["params"]["shard_tokens"], "dtype": "<u2",
                  "cap_tokens": fpj["inputs"]["params"]["cap_tokens"]},
        "units": units,
        "totals": {k: tot[k] for k in ("tokens_train_actual", "documents_train", "utf8_bytes_train", "tokens_heldout",
                                       "documents_heldout", "utf8_bytes_heldout")}
        | {"gpt2_token_count_sum": None if tot["gpt2_missing"] else tot["gpt2_token_count_sum"]},
        "outputs": {"token_bytes_sha256": sha256_path(os.path.join(st, "token_bytes.u16")),
                    "train_manifest_sha256": hashlib.sha256(canon_json(mt)).hexdigest(),
                    "heldout_manifest_sha256": hashlib.sha256(canon_json(mh)).hexdigest(),
                    "heldout_stream_sha256": stream.hexdigest(),
                    "heldout_digests_sha256": hashlib.sha256(b"".join(sorted(digests))).hexdigest(),
                    "heldout_prefix_sha256": {str(k): v for k, v in firsts.items()},
                    "fingerprint": fpj["fingerprint"]}}
    return mt, mh, body


def finish(st, train, prov):
    """The end of a run, each step durable: final held-out shards, split manifests, root manifest,
    provenance, SHA256SUMS last."""
    final = heldout_finals(st, train)
    if not final:
        say("warning: the held-out split has no document; the trainer's loader cannot read it")
    mt, mh, body = build_manifests(st)
    write_file(os.path.join(st, "train", "manifest.json"), canon_json(mt))
    write_file(os.path.join(st, "heldout", "manifest.json"), canon_json(mh))
    raw_body = canon_json(body)
    write_file(os.path.join(st, "manifest.json"), raw_body)
    event("manifest", os.path.join(st, "manifest.json"))
    prov["manifest_sha256"] = hashlib.sha256(raw_body).hexdigest()
    prov["end"] = now()
    write_file(os.path.join(st, "provenance.json"), json.dumps(prov, indent=1, sort_keys=True).encode())
    sums = "".join(f"{sha256_path(os.path.join(st, p))}  {p}\n" for p in store_files(st))
    write_file(os.path.join(st, "SHA256SUMS"), sums.encode())
    event("sums", os.path.join(st, "SHA256SUMS"))
    return body, prov["manifest_sha256"]


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in diff(u, v, f"{path}[{i}]")]
    return [] if a == b else [path]


def verify_store(st, quiet=False):
    """Re-hash every file and rebuild the manifests from the shards and records; returns the exit code."""
    pr = (lambda m: None) if quiet else print
    if not os.path.exists(os.path.join(st, "manifest.json")):
        pr("incomplete: no manifest.json")
        return 3
    try:
        lines = open(os.path.join(st, "SHA256SUMS")).read().splitlines()
    except OSError:
        pr("incomplete: no SHA256SUMS")
        return 3
    listed = dict(reversed(ln.split("  ", 1)) for ln in lines if ln)
    if sorted(listed) != store_files(st):
        pr("incomplete: SHA256SUMS lists a different set of files from the store's")
        return 3
    bad = []
    for p, want in sorted(listed.items()):
        if sha256_path(os.path.join(st, p)) != want:
            bad.append(p)
            pr(f"mismatch: {p}: bytes differ from SHA256SUMS")
    if bad:
        return 1
    errs = []
    for split in ("train", "heldout"):
        for r in records(st, split):
            b = np.fromfile(os.path.join(st, r["path"]), store_fmt.U16)
            idx = np.fromfile(os.path.join(st, r["path"][:-4] + ".idx"), store_fmt.U64)
            got = {"tokens": int(b.size), "documents": int(idx.size - 1), "max_id": int(b.max()) if b.size else None}
            try:
                store_fmt.check_idx(idx, b.size, r["path"])
                zeros = np.flatnonzero(b == 0)
                if not np.array_equal(zeros, idx[:-1].astype(np.int64)):
                    raise ValueError("id 0 is not exactly at the document starts")
            except ValueError as e:
                errs.append(f"{r['path']}: {e}")
            for k, v in got.items():
                if r[k] != v:
                    errs.append(f"{r['path']}: {k} {v}, recorded {r[k]}")
    mt, mh, body = build_manifests(st)
    for name, want in (("train/manifest.json", mt), ("heldout/manifest.json", mh), ("manifest.json", body)):
        have = json.loads(open(os.path.join(st, name), "rb").read())
        errs += [f"{name}: field {f} differs from what the shards and records give" for f in diff(want, have)]
    for e in errs:
        pr(f"mismatch: {e}")
    return 1 if errs else 0


def load_raw(out, sample):
    recs = {}
    for repo in (DATASET, TOKENIZER_REPO):
        rtype, key, _ = REPOS[repo]
        rev = DATASET_REV if repo == DATASET else TOKENIZER_REV
        d = os.path.join(out, "raw", key, rev)
        try:
            with open(os.path.join(d, "fetch.json"), "rb") as f:
                recs[repo] = (d, json.loads(f.read()))
        except OSError:
            raise Stop("unavailable", "not-fetched",
                       f"no fetch record for {repo} at {rev}; run the fetch mode") from None
    d, fr = recs[DATASET]
    if fr["sample"] != sample:
        raise Stop("config", "sample-mismatch", f"the fetch was for {fr['sample']}, not {sample}")
    for repo, (d, fr) in recs.items():
        for r in fr["files"]:
            p = os.path.join(d, r["path"])
            if os.path.islink(p) or not os.path.isfile(p) or os.path.getsize(p) != r["size"]:
                raise Stop("integrity", "size-mismatch", f"{r['path']}: missing or not its recorded size")
            check_dataless(p)
    return recs


def read_small(path, rec, cap):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        data = os.read(fd, cap + 1)
    finally:
        os.close(fd)
    if len(data) > cap or len(data) != rec["size"] or hashlib.sha256(data).hexdigest() != rec["sha256"]:
        raise Stop("integrity", "hash-mismatch", f"{rec['path']} differs from its fetch record")
    return data


def library_record(found, drift):
    rec = {"python": platform.python_version(), **found}
    try:
        rec["hf-xet"] = VERSION("hf-xet")
    except importlib.metadata.PackageNotFoundError:
        rec["hf-xet"] = None
    try:
        rec["mlx"] = VERSION("mlx")  # package metadata only; MLX is never imported
    except importlib.metadata.PackageNotFoundError:
        rec["mlx"] = None
    rec["drift"] = drift
    return rec


def script_commit():
    try:
        c = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        d = subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", HERE], capture_output=True,
                           text=True, timeout=20)
        return {"commit": c.stdout.strip() or None, "dirty": bool(d.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def prepare_mode(args, found, drift):
    out = os.path.abspath(args.out)
    recs = load_raw(out, args.sample)
    ddir, dfr = recs[DATASET]
    tdir, tfr = recs[TOKENIZER_REPO]
    locks = [lock(os.path.join(d, ".lock"), shared=True) for d, _ in recs.values()]
    tfiles = {r["path"]: r for r in tfr["files"]}
    tok_bytes = read_small(os.path.join(tdir, "tokenizer.json"), tfiles["tokenizer.json"], TOKENIZER_MAX)
    side = {f: read_small(os.path.join(tdir, f), tfiles[f], JSON_MAX if f.endswith(".json") else README_MAX)
            for f in TOKENIZER_FILES[1:]}
    for f in ("tokenizer_config.json", "special_tokens_map.json"):
        json.loads(side[f])  # parsed as data, never executed
    try:
        tok = bpe.BPE(tok_bytes, sha256=tfiles["tokenizer.json"]["sha256"])
        check = tok.check(golden=GOLDEN)
        if tok.vocab != VOCAB:
            raise Stop("integrity", "vocab", f"the tokenizer has {tok.vocab} ids, expected {VOCAB}")
        cut = None
        if args.merge_prefix:
            cut_bytes, info = bpe.truncate(tok_bytes, args.merge_prefix)
            tok = bpe.BPE(cut_bytes)
            tok.check(golden=False)
            cut = (cut_bytes, info)
    except bpe.TokenizerError as e:
        raise Stop("integrity", "tokenizer", f"{e}") from None
    dfiles = sorted((r for r in dfr["files"] if r["path"].endswith(".parquet")), key=lambda r: r["path"])
    listing_name = "listing-" + args.sample.replace("/", "-") + ".json"
    with open(os.path.join(ddir, listing_name), "rb") as f:
        listing_raw = f.read()
    listing = json.loads(listing_raw)
    by_path = {e["path"]: e for e in listing}
    for r in dfiles:
        e = by_path.get(r["path"])
        if e is None or not e.get("lfs") or e["lfs"]["sha256"] != r["sha256"] or e["size"] != r["size"]:
            raise Stop("integrity", "hash-mismatch",
                       f"{r['path']}: its fetch record and the saved listing disagree")
    fp_inputs = {"prep_format_version": PREP_FORMAT_VERSION,
                 "source": {"repo_id": DATASET, "revision": DATASET_REV, "sample": args.sample,
                            "listing": sorted(({"path": e["path"], "upstream_hash": upstream(e)[1],
                                                "hash_kind": upstream(e)[0]} for e in listing),
                                              key=lambda e: e["path"])},
                 "canon": CANON, "tokenizer_sha256": hashlib.sha256(tok_bytes).hexdigest(),
                 "params": {"shard_tokens": SHARD_TOKENS, "dtype": "<u2", "heldout_rule": HELDOUT_RULE,
                            "cap_tokens": args.cap_tokens, "merge_prefix": args.merge_prefix}}
    fingerprint = hashlib.sha256(canon_json(fp_inputs)).hexdigest()
    st = os.path.join(out, fingerprint[:16])
    new = not os.path.exists(os.path.join(st, "fingerprint.json"))
    if new:
        _, short = disk_table([("store", out, store_estimate(args.cap_tokens))])
        if short:
            raise Stop("refused", "disk", "too little free space for the store")
    for d in SPLIT_DIRS:
        os.makedirs(os.path.join(st, d), exist_ok=True)
    locks.append(lock(os.path.join(st, ".lock")))
    libs = library_record(found, drift)
    tok_name = "tokenizer.json" if cut is None else f"tokenizer.prefix-{args.merge_prefix}.json"
    if new:
        dsrc = {r["path"]: r for r in dfr["files"]}
        card = read_small(os.path.join(ddir, "README.md"), dsrc["README.md"], README_MAX)
        tokfiles = [{k: tfiles[f][k] for k in ("path", "size", "sha256", "blob_id", "lfs_sha256", "xet_hash")}
                    for f in TOKENIZER_FILES[:3]]
        if cut is not None:
            tokfiles.append({"path": tok_name, "size": len(cut[0]), "sha256": hashlib.sha256(cut[0]).hexdigest(),
                             "blob_id": None, "lfs_sha256": None, "xet_hash": None})
        fpj = {"fingerprint": fingerprint, "inputs": fp_inputs, "vocab": tok.vocab, "max_doc_bytes": MAX_DOC_BYTES,
               "py_scan_root": dfr.get("py_scan_root") or [], "licence_tags": {DATASET: dfr["info"]["license"],
                                                                                TOKENIZER_REPO: tfr["info"]["license"]},
               "card_sha256": {DATASET: hashlib.sha256(card).hexdigest(),
                               TOKENIZER_REPO: hashlib.sha256(side["README.md"]).hexdigest()},
               "tokenizer_revision": TOKENIZER_REV, "tokenizer_files": tokfiles, "tokenizer_load": tok_name,
               "golden": {"checked": bool(GOLDEN), "cases": [[t, ids] for t, ids in bpe.GOLDEN]},
               "merge_prefix": None if cut is None else {"size": args.merge_prefix,
                                                        "merges_kept": cut[1]["merges_kept"],
                                                        "identity": cut[1]["identity"]}}
        write_file(os.path.join(st, "fingerprint.json"), canon_json(fpj))
        write_file(os.path.join(st, "token_bytes.u16"), tok.token_bytes().astype(store_fmt.U16).tobytes())
        write_file(os.path.join(st, "sources", "dataset-listing-sample.json"), listing_raw)
        write_file(os.path.join(st, "sources", "dataset-listing-root.json"),
                   open(os.path.join(ddir, "listing-root.json"), "rb").read())
        write_file(os.path.join(st, "sources", "tokenizer-listing-root.json"),
                   open(os.path.join(tdir, "listing-root.json"), "rb").read())
        write_file(os.path.join(st, "sources", "dataset-README.md"), card)
        write_file(os.path.join(st, "sources", "tokenizer-README.md"), side["README.md"])
        write_file(os.path.join(st, "tokenizer", "tokenizer.json"), tok_bytes)
        for f in TOKENIZER_FILES[1:3]:
            write_file(os.path.join(st, "tokenizer", f), side[f])
        if cut is not None:
            write_file(os.path.join(st, "tokenizer", tok_name), cut[0])
        prov = {"libraries": libs}
        write_file(os.path.join(st, "provenance.json"), json.dumps(prov, indent=1, sort_keys=True).encode())
    else:
        with open(os.path.join(st, "fingerprint.json"), "rb") as f:
            if json.loads(f.read())["fingerprint"] != fingerprint:
                raise Stop("integrity", "fingerprint", "the store's fingerprint.json does not match its name")
        with open(os.path.join(st, "provenance.json"), "rb") as f:
            prov = json.loads(f.read())
        first = prov["libraries"]
        moved = {k: (first.get(k), libs.get(k)) for k in ("tokenizers", "pyarrow", "numpy", "python")
                 if first.get(k) != libs.get(k)}
        if moved and not args.allow_env_drift:
            raise Stop("config", "env-drift", f"library versions differ from the run that created the store: {moved} "
                       f"(--allow-env-drift overrides and is recorded)")
        if moved:
            prov.setdefault("overrides", []).append({"env_drift_on_resume": moved})
        if os.path.exists(os.path.join(st, "manifest.json")) and os.path.exists(os.path.join(st, "SHA256SUMS")):
            code = verify_store(st)
            summary([f"store {fingerprint[:16]} is complete; verify {'matches' if code == 0 else 'MISMATCH'}"])
            return 0 if code == 0 else EXIT["integrity"]
    train, final = resume_scan(st)
    prov.update({"script": script_commit(), "libraries": libs, "start": now(),
                 "threads": os.environ["RAYON_NUM_THREADS"],
                 "machine": {"platform": platform.platform(), "machine": platform.machine(),
                             "mac_ver": platform.mac_ver()[0] or None,
                             "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")},
                 "fetch": {"dataset": dfr, "tokenizer": tfr}, "tokenizer_check": check,
                 "overrides": prov.get("overrides", []) + ([{"env_drift": drift}] if drift else [])})
    check_imports()
    p = Prep(args, st, dfiles, ddir, tok, check, found, drift, fingerprint, prov.get("overrides"))
    signal.signal(signal.SIGINT, lambda *a: STOP_FLAG.append(1))
    signal.signal(signal.SIGTERM, lambda *a: STOP_FLAG.append(1))
    try:
        if not (train and train[-1].get("stop")):
            if train:
                last = train[-1]
                p.totals, p.stats = dict(last["totals"]), last["stats"]
                c = last["end_cursor"]
                fi = next(i for i, r in enumerate(dfiles) if r["path"] == c["file"])
                cursor = (fi, c["row_group"], c["row"])
            else:
                cursor = (0, 0, 0)
            p.run(cursor, len(train))
            train = records(st, "train")
        body, digest = finish(st, train, prov | {"statistics": statistics(train[-1]), "row_groups_largest":
                                                  sorted(p.rowgroups, key=lambda r: -r["text_uncompressed"])[:5],
                                                  "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                  "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
                                                  "seconds": round(time.time() - p.t0, 1)})
    except OSError as e:
        if e.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
            if p.shard is not None:
                p.shard.abort()
            need = max(args.cap_tokens - p.totals["tokens_train_actual"], 0) * 2 + SHARD_TOKENS * 2
            raise Stop("disk", "disk-full", f"no space left; about {need / 1e9:.2f} GB more is needed (ESTIMATE: "
                       "2 bytes per token still to write up to the cap, plus one shard)") from None
        raise
    except Stop:
        if p.shard is not None:
            p.shard.abort()
        raise
    finally:
        for fd in locks:
            os.close(fd)
    t = body["totals"]
    summary([f"sample {args.sample} at {DATASET_REV}; tokenizer {TOKENIZER_REPO} at {TOKENIZER_REV}",
             *[f"  {f['path']}  {f['size']:,} B  sha256 {f['sha256']}  lfs {f['lfs_sha256']}"
               for f in body["source"]["files"]],
             f"train: {t['tokens_train_actual']:,} tokens, {t['documents_train']:,} documents; held-out: "
             f"{t['tokens_heldout']:,} tokens, {t['documents_heldout']:,} documents",
             f"drops: {sum_drops(body['units'])}", f"manifest sha256 {digest}", f"store {fingerprint[:16]}",
             f"disk: store {dir_bytes(st):,} B, {DISK_USAGE(st).free:,} B free on its volume"])
    return 0


def sum_drops(units):
    out = {d: 0 for d in DROPS}
    for u in units:
        for d, n in u["drops"].items():
            out[d] += n
    return out


def statistics(last):
    """What the sizing estimates assume, measured (provenance only, never hashed)."""
    t, s = last["totals"], last["stats"]
    toks = s["tokens_all"]
    content = toks - t["documents_train"] - t["documents_heldout"]
    utf8 = t["utf8_bytes_train"] + t["utf8_bytes_heldout"]
    return {"bytes_per_token": utf8 / content if content else None,
            "smollm2_per_gpt2_token": (content / t["gpt2_token_count_sum"]) if t["gpt2_token_count_sum"] else None,
            "tokens_per_document": toks / max(t["documents_train"] + t["documents_heldout"], 1),
            "crop_floor_t2048": s["crop_excess"] / toks if toks else None,
            "length_histogram_pow2": s["len_hist"], "dump_counts": s["dump_counts"], "gpt2_bad": s["gpt2_bad"]}


def parse(argv):
    ap = argparse.ArgumentParser(prog="prepare_store.py", description="The stand-in data store's preparation")
    ap.add_argument("mode", choices=("check", "fetch", "prepare", "verify"))
    ap.add_argument("--sample", choices=sorted(SAMPLES))
    ap.add_argument("--files", type=int, default=9)
    ap.add_argument("--out")
    ap.add_argument("--cap-tokens", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--confirm-pins", nargs=2, metavar=("DATASET_SHA", "TOKENIZER_SHA"))
    ap.add_argument("--allow-env-drift", action="store_true")
    ap.add_argument("--allow-unsafe-file", action="append")
    ap.add_argument("--merge-prefix", type=int, choices=MERGE_PREFIXES)
    ap.add_argument("--store")
    a = ap.parse_args(argv)
    if a.mode == "verify":
        if not a.store:
            ap.error("verify needs --store")
    else:
        for need in ("sample", "out", "cap_tokens"):
            if getattr(a, need) is None:
                ap.error(f"{a.mode} needs --{need.replace('_', '-')}")
        if a.cap_tokens <= 0 or a.files <= 0:
            ap.error("--cap-tokens and --files must be positive")
    return a


def main(argv=None):
    args = parse(argv)
    if args.mode == "verify":
        import_libs("verify")
        return verify_store(os.path.abspath(args.store))
    try:
        setup_env(args)
        out = os.path.abspath(args.out)
        check_paths((out, os.path.join(out, "raw")))
        check_env(args.mode)
        check_revision(DATASET_REV, DATASET)
        check_revision(TOKENIZER_REV, TOKENIZER_REPO)
        found, drift = check_versions(args.mode, args.allow_env_drift)
        import_libs(args.mode)
        check_pyarrow_runtime(found)
        check_locations(out)
        return {"check": check_mode, "fetch": fetch_mode, "prepare": prepare_mode}[args.mode](args, found, drift)
    except Stop as s:
        say(f"stopped: {s.reason} (exit {s.code}): {s.detail}")
        return s.code


if __name__ == "__main__":
    sys.exit(main())
