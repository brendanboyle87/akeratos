"""The pinned environment for bitwise CPU runs.

MLX's Linux CPU wheel uses bundled reference BLAS with no thread knob, and the
repro runs load no torch, so nothing on the MLX side needs pinning. The pins
below still go into every child a `repro` test starts, so a library that does
read them cannot vary a run. Each such child gets `DETERMINISM_ENV` on top of
the child environment of `Children.env` in `tests/support/proc.py`, which
already sets the thread variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and macOS Accelerate's `VECLIB_MAXIMUM_THREADS` to 1.
"""

from __future__ import annotations

DETERMINISM_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_OFFLINE": "1",
    # entry turns TF32 off itself before MLX loads; the pin keeps a child that
    # imports MLX some other way on the same FP32 path.
    "MLX_ENABLE_TF32": "0",
}
