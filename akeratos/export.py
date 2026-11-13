"""The release folder a checkpoint ships in, and its loader.

`export` writes a new folder from a complete checkpoint (`akeratos/ckpt.py`):

| File | What it holds |
|---|---|
| `WEIGHTS` | the FP32 raw weights, and the EMA where the run kept one |
| `TABLE` | every file of the checkpoint's table: its shards (rows, v and stamps), `table.json` and the read counts |
| `TOKENIZER`, `LICENSE`, `NOTICE` | the files given |
| `RUN`, `LOG` | the training logs: the run's manifest and its log, one line per step, from the run directory |
| `CONFIG` | Akeratos's own schema, `SCHEMA`: the step, the weight files and the run's configuration from the checkpoint's `ckpt.STATE` |
| `SUMS` | the SHA-256 of every other file, in the format `shasum -a 256 -c` reads; written last |

No path on the machine that trained the run ships, and the optimiser state stays behind. `load`
reads an export back, and `checkpoint` opens a checkpoint or an export for the readouts and
post-training.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tokenizers import Tokenizer

from akeratos import __version__, ckpt
from akeratos._compat.mlx_seam import load_safetensors
from akeratos.configs import ModelConfig, from_keys
from akeratos.data.chat import IM_END, IM_START, headers
from akeratos.data.packer import RESERVED, SEP_ID
from akeratos.errors import AkeratosError, ConfigError, DataMismatchError
from akeratos.model import Params
from akeratos.table.config import TableConfig
from akeratos.table.table import Table

SCHEMA = "metalforge.export/v1"
ANY_MAJOR = re.compile(r"metalforge\.export/v([1-9][0-9]*)")  # a later major too
CONFIG, SUMS, TOKENIZER, TABLE = "config.json", "SHA256SUMS", "tokenizer.json", "table"
RUN, LOG = "logs/run.json", "logs/metrics.jsonl"  # the training logs, from the run directory
WEIGHTS = {"raw": "dense.safetensors", "ema": "ema.safetensors"}
CHUNK = 1 << 24  # bytes per read


def _copy(src: Path, dst: Path) -> str:
    """Copy `src` to `dst` byte for byte; returns the SHA-256 of the bytes copied."""
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as r, dst.open("wb") as w:
        while chunk := r.read(CHUNK):
            h.update(chunk)
            w.write(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _shipped(src: Path) -> list[str]:
    """The checkpoint files that ship, as its `ckpt.STATE` lists them: the weights and the table."""
    listed: dict[str, int] = json.loads((src / ckpt.STATE).read_text())["files"]
    return sorted(n for n in listed if n in WEIGHTS.values() or n.startswith(f"{TABLE}/"))


def _config(c: ckpt.Checkpoint, weights: dict[str, str]) -> dict[str, Any]:
    """`CONFIG`: the schema, the writer, the step, the weight files and the run's config.

    It lets the Hugging Face Hub count downloads; it is not the configuration file that mlx-lm
    and transformers read.
    """
    run = dict(c.state["config"])
    # A post-trained run's `post` record keeps only the folder name of its base checkpoint.
    if run.get("post") is not None:
        run["post"] = run["post"] | {"base": Path(run["post"]["base"]).name}
    head = {"schema": SCHEMA, "schema_minor": 0, "written_by": f"bake {__version__}"}
    return head | {"step": c.step, "weights": weights, "config": run}


def _cut(doc: object) -> object:
    """`doc` with each absolute path in it cut to its last part."""
    if isinstance(doc, dict):
        return {k: _cut(v) for k, v in cast(dict[str, object], doc).items()}
    if isinstance(doc, list):
        return [_cut(v) for v in cast(list[object], doc)]
    return Path(doc).name if isinstance(doc, str) and doc.startswith("/") else doc


def _manifest(path: Path) -> object:
    """The run's `run.json` at `path`, with each absolute path in it cut to its last part."""
    try:
        return _cut(json.loads(path.read_text()))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{path}: {e}") from e


def _tokenizer(path: Path) -> Tokenizer:
    # `tokenizers` raises a plain Exception for a missing file and for one with no tokenizer.
    try:
        return Tokenizer.from_file(str(path))
    except Exception as e:
        raise ConfigError(f"{path}: {e}") from e


def _check(out: Path, given: dict[str, Path], vocab: int) -> None:
    """Refuse an existing `out`, a missing file and a tokenizer that does not fit the model."""
    if out.exists():
        raise ConfigError(f"{out} exists; an export goes to a new folder")
    missing = [str(p) for p in given.values() if not p.is_file()]
    if missing:
        raise ConfigError(f"no such file: {', '.join(missing)}")
    if _tokenizer(given[TOKENIZER]).get_vocab_size() != vocab:
        raise ConfigError(f"{given[TOKENIZER]}: not the model's {vocab}-token vocabulary")


def export(src: Path, out: Path, *, tokenizer: Path, licence: Path, notice: Path) -> dict[str, str]:
    """Write the export of the complete checkpoint `src` to the new folder `out`.

    The run directory, `src`'s parent, gives the training logs: `metrics.jsonl` byte for byte, and
    `run.json` with each absolute path cut to its last part, in the JSON form `Run.manifest`
    writes, so a run that names none ships its `run.json` byte for byte. Returns the SHA-256 of
    each file in `out` but `SUMS`, by its path in `out`.
    """
    try:
        c = ckpt.load(src)
        cfg = from_keys(c.state["config"]["model"])
    except (KeyError, ValueError) as e:
        raise ConfigError(f"{src}: {e}") from e
    manifest = _manifest(src.parent / "run.json")
    given = {TOKENIZER: tokenizer, "LICENSE": licence, "NOTICE": notice}
    given[LOG] = src.parent / "metrics.jsonl"
    _check(out, given, cfg.vocab_size)
    out.mkdir(parents=True)
    sums = {n: _copy(src / n, out / n) for n in _shipped(src)}
    sums |= {n: _copy(p, out / n) for n, p in given.items()}
    weights = {k: v for k, v in WEIGHTS.items() if v in sums}
    (out / RUN).write_text(json.dumps(manifest, indent=1))
    (out / CONFIG).write_text(json.dumps(_config(c, weights), indent=1, sort_keys=True) + "\n")
    sums |= {n: _sha256(out / n) for n in (RUN, CONFIG)}
    (out / SUMS).write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    # Read the folder back, so the files that ship hold the bytes the scorer read.
    verify(out)
    return sums


def verify(path: Path) -> set[str]:
    """Check the SHA-256 of every file `SUMS` lists in the export `path`; returns those files."""
    try:
        lines = (path / SUMS).read_text().splitlines()
    except OSError as e:
        raise DataMismatchError(f"{path}: no {SUMS} ({e})") from e
    if not lines or any("  " not in line for line in lines):
        raise DataMismatchError(f"{path / SUMS}: not one '<sha256>  <file>' line per file")
    listed = [line.split("  ", 1) for line in lines]
    for want, name in listed:
        if not (path / name).is_file() or _sha256(path / name) != want:
            raise DataMismatchError(f"{path / name}: missing, or not the SHA-256 in {SUMS}")
    return {name for _, name in listed}


def _no_write_back() -> int:
    """The table store's memory counter, read only around a write-back, which a loader never makes."""
    raise AkeratosError("an exported table is read-only")


def _table(path: Path, saved: dict[str, Any]) -> Table:
    """The export's table, read-only, checked against the run's table configuration."""
    keys = dict(saved)
    keys["orders"] = tuple(keys["orders"])  # a JSON list in config.json
    return Table.load(path / TABLE, TableConfig(**keys).validate(), _no_write_back)


@dataclass(frozen=True)
class Export:
    """A loaded export: the model configuration, one set of weights, the table and the tokenizer.

    `chat` is true for a post-trained model, which answers in the chat format
    (`akeratos/data/chat.py`).
    """

    cfg: ModelConfig
    params: Params
    table: Table | None
    tokenizer: Tokenizer
    chat: bool

    def encode(self, text: str) -> list[int]:
        """The tokenizer's ids for `text`, with no special tokens added."""
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def context(self, text: str) -> list[int]:
        """The prompt `text` as the decoder's context, id 0 first.

        A base model reads the text as a document's start. A post-trained one (`chat`) reads the
        user's turn, then the assistant's header, so it answers.
        """
        body = self.encode(text)
        if any(t <= RESERVED[1] for t in body):  # no training text holds a reserved id
            raise ConfigError("the prompt holds a reserved token (ids 0-16), such as <|im_end|>")
        if not self.chat:
            return [SEP_ID, *body]
        head = headers(self.encode)
        return [SEP_ID, IM_START, *head["user"], *body, IM_END, IM_START, *head["assistant"]]


def _read(path: Path) -> dict[str, Any]:
    """The `CONFIG` of the export at `path`, if this release reads its schema's major."""
    raw = json.loads((path / CONFIG).read_text())
    doc: dict[str, Any] = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    found = ANY_MAJOR.fullmatch(str(doc.get("schema")))
    if found is None:
        raise ConfigError(f"{path / CONFIG}: not a {SCHEMA} export")
    if found[1] != "1":
        by = str(doc.get("written_by"))
        raise ConfigError(
            f"SCHEMA_TOO_NEW kind=export found=v{found[1]} supported=v1..v1 "
            f"written_by={by} FIX: upgrade bake to >= {by.removeprefix('bake ')}"
        )
    return doc


def load(path: Path, weights: str = "raw") -> Export:
    """Read the export at `path` with its `weights` (`raw` or `ema`), its table and tokenizer."""
    try:
        doc = _read(path)
        if weights not in doc["weights"]:
            raise ConfigError(f"{path} has no {weights!r} weights")
        run = doc["config"]
        cfg = from_keys(run["model"])
        table = _table(path, run["table"]) if cfg.table else None
        params = load_safetensors(path / doc["weights"][weights])[0]
        tok = _tokenizer(path / TOKENIZER)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as e:
        raise ConfigError(f"{path}: {e}") from e
    return Export(cfg, params, table, tok, run.get("post") is not None)


def checkpoint(path: Path) -> ckpt.Checkpoint:
    """The checkpoint at `path`, a run directory's newest complete one, or the export at `path`.

    An export is checked against `SUMS` and read as the checkpoint it was made from: its step,
    its weights and table under the checkpoint's names, and the run's configuration as its state,
    which is all a readout or post-training reads of a checkpoint. So the released files are
    scored by the commands that scored the checkpoint. A reader takes the EMA wherever its
    `WEIGHTS` file is, so the weight files must be the ones `CONFIG` names and `SUMS` lists.
    """
    found = path if ckpt.complete(path) else ckpt.latest(path)
    if found is not None:
        return ckpt.load(found)
    try:
        doc = _read(path)
        step, run, named = int(doc["step"]), doc["config"], doc["weights"]
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"{path}: no complete checkpoint or export ({e})") from e
    listed = verify(path)
    held = {k: n for k, n in WEIGHTS.items() if (path / n).is_file()}
    if named != held or not listed >= set(held.values()):
        raise DataMismatchError(
            f"{path}: its weight files are not the ones {CONFIG} names and {SUMS} lists"
        )
    return ckpt.Checkpoint(path, step, {"config": run})
