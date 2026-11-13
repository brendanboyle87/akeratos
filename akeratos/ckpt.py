"""Atomic checkpoints, and resume from the newest complete one.

A checkpoint is `out/ckpt-<step>/`, written to `out/.tmp-ckpt-<step>/` and renamed into place,
so a run killed mid-write leaves no checkpoint that resume picks up. A checkpoint counts only if
`state.json` parses, has this format and step, and names files that all exist with their
recorded sizes; resume skips any other and falls back to the next older one.
`akeratos/train/saver.py` resumes from it and refuses configs that differ from the saved ones.

No global RNG is consumed in training, so the step counter and the source's cursor are the whole
random state, and a resumed run reads exactly the rows it would have read. The schedule is
rebuilt from the configs. Nothing here depends on the model's size: the files are whatever the
run holds, and the interval and `keep` come from the caller.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx

from akeratos._compat.mlx_seam import load_safetensors, save_safetensors

FORMAT = 1
STATE = "state.json"
PREFIX = "ckpt-"
TMP = ".tmp-"
SEP = "::"
KEEP = 2

log = logging.getLogger(__name__)
Tree = Mapping[str, Any]
Writer = Callable[[Path], object]  # what it returns is ignored (`Table.write` lists its files)


def flatten(tree: Tree, prefix: str = "") -> dict[str, mx.array]:
    """A nested dict of arrays as one flat dict, keys joined by `::`."""
    out: dict[str, mx.array] = {}
    for k, v in tree.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(cast(Tree, v), key + SEP))
        else:
            out[key] = cast(mx.array, v)
    return out


def restore(tree: dict[str, Any], flat: Mapping[str, mx.array]) -> None:
    """Write `flat` back into `tree` in place; the key sets must match exactly.

    The compiled update holds the optimiser's state dictionaries, so replacing them would leave
    it reading the old ones.
    """
    if set(flatten(tree)) != set(flat):
        diff = sorted(set(flatten(tree)) ^ set(flat))[:8]
        raise KeyError(f"checkpoint and live state differ in {diff}")
    for key, v in flat.items():
        *path, leaf = key.split(SEP)
        node = tree
        for p in path:
            node = node[p]
        node[leaf] = v


def _sync(path: Path) -> None:
    """Flush `path` to the drive.

    On macOS a plain `fsync` does not flush the drive's cache, so `F_FULLFSYNC` is used where
    `fcntl` has it (Apple's fsync(2) man page), else `fsync`.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        full = getattr(fcntl, "F_FULLFSYNC", None)
        if full is None:
            os.fsync(fd)
        else:
            try:
                fcntl.fcntl(fd, full)
            except OSError:
                os.fsync(fd)
    finally:
        os.close(fd)


def _step_of(path: Path) -> int | None:
    tail = path.name[len(PREFIX) :]
    return int(tail) if path.name.startswith(PREFIX) and tail.isdigit() else None


def save(
    out: str | Path,
    step: int,
    *,
    arrays: Mapping[str, Mapping[str, mx.array]],
    state: Mapping[str, Any],
    dirs: Mapping[str, Writer] | None = None,
    keep: int = KEEP,
    pin: bool = False,
) -> Path:
    """Write checkpoint `step` atomically, then prune to the newest `keep`.

    Each of `arrays` is saved as `<stem>.safetensors`: `dense` holds the FP32 master weights,
    `optim-<group>` each group's AdamW (its step, its rate, m and v per weight) and `ema` the EMA
    when on. Each of `dirs` is a subdirectory its writer fills, such as the table's shards.
    `state.json` is written last: the format, the step, every other file's size, and the caller's
    JSON `state` (the skip count, the configs, the source's cursor and the log's loss window).
    Every file and directory is synced before the rename, and the parent after it. `keep` counts
    complete unpinned checkpoints, this one included unless `pin` keeps it outside the rotation.
    """
    out = Path(out)
    name = f"{PREFIX}{step:08d}"
    tmp, final = out / f"{TMP}{name}", out / name
    out.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out.glob(f"{TMP}{PREFIX}*")):
        shutil.rmtree(stale, ignore_errors=True)
    tmp.mkdir()
    for stem, flat in arrays.items():
        save_safetensors(tmp / f"{stem}.safetensors", dict(flat))
    for sub, write in (dirs or {}).items():
        (tmp / sub).mkdir()
        write(tmp / sub)
    files: dict[str, int] = {}
    for p in sorted(tmp.rglob("*")):
        _sync(p)
        if p.is_file():
            files[p.relative_to(tmp).as_posix()] = p.stat().st_size
    doc: dict[str, Any] = {"format": FORMAT, "step": step, "files": files, "state": dict(state)}
    if pin:
        doc["pinned"] = True
    (tmp / STATE).write_text(json.dumps(doc))
    _sync(tmp / STATE)
    _sync(tmp)
    if final.exists():
        if complete(final):
            raise FileExistsError(f"{final} already holds a complete checkpoint")
        shutil.rmtree(final)  # torn: a run that fell back past it has reached its step again
    tmp.replace(final)
    _sync(out)
    prune(out, keep)
    return final


def _read(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads((path / STATE).read_text())
    except (OSError, ValueError):
        return None
    return cast(dict[str, Any], doc) if isinstance(doc, dict) else None


def complete(path: str | Path) -> bool:
    """True when `state.json` parses and every file it names has its recorded size."""
    path = Path(path)
    doc = _read(path)
    if doc is None or doc.get("format") != FORMAT or doc.get("step") != _step_of(path):
        return False
    files = doc.get("files")
    if not isinstance(files, dict):
        return False
    for rel, size in cast(dict[str, int], files).items():
        p = path / rel
        if not p.is_file() or p.stat().st_size != size:
            return False
    return True


def _dirs(out: Path) -> list[Path]:
    if not out.is_dir():
        return []
    found = [(s, p) for p in sorted(out.iterdir()) if p.is_dir() and (s := _step_of(p)) is not None]
    return [p for _, p in sorted(found)]


def checkpoints(out: str | Path) -> list[Path]:
    """The complete checkpoints in `out`, oldest first."""
    return [p for p in _dirs(Path(out)) if complete(p)]


def latest(out: str | Path) -> Path | None:
    """The newest complete checkpoint, skipping (and logging) any newer incomplete one."""
    for p in reversed(_dirs(Path(out))):
        if complete(p):
            return p
        log.warning("skipping incomplete checkpoint %s", p)
    return None


def pinned(path: str | Path) -> bool:
    """True when checkpoint `path` is complete and has `pinned: true` in its `state.json`."""
    doc = _read(Path(path))
    return doc is not None and doc.get("pinned") is True and complete(path)


def pin(path: str | Path) -> None:
    """Pin the complete checkpoint `path` in place, rewriting its `state.json` atomically.

    `state.json` is not among the files it lists, so its new size leaves it complete. The run
    pins the last stable checkpoint before the decay this way (`akeratos/train/saver.py`).
    """
    path = Path(path)
    doc = _read(path)
    if doc is None or not complete(path):
        raise ValueError(f"incomplete checkpoint {path}")
    tmp = path / f"{TMP}{STATE}"
    tmp.write_text(json.dumps(doc | {"pinned": True}))
    _sync(tmp)
    tmp.replace(path / STATE)
    _sync(path)


def prune(out: str | Path, keep: int = KEEP) -> None:
    """Remove all but the newest `keep` unpinned complete checkpoints, and every incomplete one.

    Pinned checkpoints are always kept and do not count towards `keep`.
    """
    done = checkpoints(out)
    rotating = [p for p in done if not pinned(p)]
    kept = (set(done) - set(rotating)) | (set(rotating[-keep:]) if keep > 0 else set(rotating))
    for p in _dirs(Path(out)):
        if p not in kept:
            shutil.rmtree(p)


@dataclass(frozen=True)
class Checkpoint:
    """A complete checkpoint, read lazily."""

    path: Path
    step: int
    state: dict[str, Any]

    def arrays(self, stem: str) -> dict[str, mx.array]:
        """The flat array dict saved as `<stem>.safetensors`."""
        return load_safetensors(self.path / f"{stem}.safetensors")[0]


def load(path: str | Path) -> Checkpoint:
    """Open checkpoint `path`, which must be complete."""
    path = Path(path)
    doc = _read(path)
    if doc is None or not complete(path):
        raise ValueError(f"incomplete checkpoint {path}")
    return Checkpoint(path, int(doc["step"]), cast(dict[str, Any], doc["state"]))


def disk_note(path: Path) -> str | None:
    """A warning when the disk has less free space than checkpoint `path` takes."""
    doc = _read(path) or {}
    size = sum(cast(dict[str, int], doc.get("files", {})).values())
    free = shutil.disk_usage(path.parent).free
    if free >= size:
        return None
    return f"{free / 1e9:.1f} GB free, and a checkpoint takes {size / 1e9:.1f} GB: the next save may fail"
