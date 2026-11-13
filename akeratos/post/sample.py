"""Greedy samples of a chat checkpoint and their format compliance.

For each of the first documents of an SFT split, the prompt is every turn but the last answer,
then that turn's id 1 and the assistant's header (`akeratos/post/data.py`). The decoder
(`akeratos/infer.py`) runs greedily until id 2 or the token cap. The samples are ids: the
tokenizer is not in `akeratos/`, so they are decoded where the data was prepared. A model with
the table term reads the checkpoint's `table/`, as `eval.heldout` does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from akeratos.configs import from_keys
from akeratos.data.chat import IM_START
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout import make_parent_dirs
from akeratos.infer import greedy
from akeratos.post.data import documents, prompt_of
from akeratos.post.run import SFTRun, base_weights, load_table
from akeratos.train.loop import DTYPES

EXIT_CONFIG, EXIT_DATA = 78, 22


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m akeratos.post.sample",
        description="Greedy samples and their format compliance.",
    )
    p.add_argument("--ckpt", type=Path, required=True, help="a checkpoint, run directory or export")
    p.add_argument("--data", type=Path, required=True, help="a chat directory of prompts")
    p.add_argument("--weights", choices=("raw", "ema"), default="raw", help="which weights")
    p.add_argument("--n", type=int, default=64, help="samples: the first N conversations")
    p.add_argument("--max-new", type=int, default=256, help="new tokens at most per sample")
    p.add_argument("--compute-dtype", choices=tuple(DTYPES), default="float32")
    p.add_argument("--out", type=Path, default=None, help="one JSON line per sample")
    return p


def _samples(a: argparse.Namespace) -> dict[str, Any]:
    """Sample, write one JSON line per sample to `--out`, and return the summary record.

    A sample complies with the format when the decoder stopped at id 2 and the answer is not
    empty and holds no id 1; the format compliance is the share of samples that comply.
    """
    make_parent_dirs(a.out)
    c = SFTRun.base_checkpoint(a.ckpt)
    try:
        cfg = from_keys(c.state["config"]["model"])
    except (KeyError, ValueError) as e:
        raise ConfigError(f"{c.path}: the run's model configuration: {e}") from e
    params = base_weights(c, a.weights)
    table = load_table(c, cfg)
    docs = documents(a.data, cfg.vocab_size)
    lines: list[dict[str, Any]] = []
    for i, doc in enumerate(docs[: a.n]):
        start, _ = prompt_of(doc)
        s = greedy(
            params, cfg, start.tolist(), a.max_new, table=table, dtype=DTYPES[a.compute_dtype]
        )
        ok = s.stopped and bool(s.ids) and IM_START not in s.ids
        lines.append({"i": i, "ids": list(s.ids), "stopped": s.stopped, "complies": ok})
    if not lines:
        raise DataMismatchError(f"{a.data}: no conversation to sample")
    if a.out is not None:
        a.out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    n = len(lines)
    return {
        "ckpt": str(c.path),
        "step": c.step,
        "data": str(a.data),
        "weights": a.weights,
        "n": n,
        "max_new": a.max_new,
        "stopped": sum(x["stopped"] for x in lines) / n,
        "format_compliance": sum(x["complies"] for x in lines) / n,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Sample and print the summary record; returns the exit code."""
    a = _parser().parse_args(argv)
    try:
        rec = _samples(a)
    except ConfigError as e:
        sys.stderr.write(f"configuration error: {e}\n")
        return EXIT_CONFIG
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA
    sys.stdout.write(json.dumps(rec, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
