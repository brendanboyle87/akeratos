"""The flags of the held-out readout, for `python -m akeratos.eval.heldout` and `bake evaluate`.

This module imports no MLX, so `bake --help` can list the flags without loading `mlx.core`. The
readout itself is `akeratos/eval/heldout.py`.
"""

import argparse
from pathlib import Path

COMPUTE_DTYPES = ("bfloat16", "float32")
DESCRIPTION = "Held-out bits per byte over a checkpoint."


def add_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", type=Path, required=True, help="a checkpoint, run directory or export")
    p.add_argument("--data", type=Path, required=True, help="the data store's heldout/ split")
    p.add_argument("--seq-len", type=int, default=2048, help="tokens per row (T)")
    p.add_argument("--rows", type=int, default=4, help="rows per forward pass (B)")
    p.add_argument(
        "--tokens", type=float, default=None, help="score whole documents up to N targets"
    )
    p.add_argument(
        "--compute-dtype", choices=COMPUTE_DTYPES, default=None, help="default: the run's"
    )
    p.add_argument("--table", choices=("on", "off"), default="on", help="off: no table term")
    p.add_argument(
        "--mask", choices=("own", "base"), default="own", help="base: also mask ids 1-16"
    )
    p.add_argument("--out", type=Path, default=None, help="also write the record here")
    p.add_argument(
        "--records", type=Path, default=None, help="also write per-document and per-target records"
    )
