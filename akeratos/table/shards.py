"""The table's pieces (shards), the one object that owns them, and the check against copies.

A shard is a contiguous range of rows with these fields: `rows` (BF16 for F2, INT8 with an FP32
per-row `scale` for F1, FP32 for the F32 oracle), the FP32 second moment `v`, the int32 `stamp`,
and under `table.bias` per_row the int32 `count`. MLX copies a whole shard when its write-back
runs while anything else still references it: a slice view, an unevaluated `take`, a NumPy view
or an unevaluated statistic. So `ShardStore` is the sole owner: it never returns a shard, hands
out only evaluated gathers, and checks every write-back for a copy.

F2 rows round stochastically (add 16 random low bits, truncate), F1 rows are INT8 with an FP32
per-row absmax scale and stochastic rounding, and F32 is the test oracle. The rounding key is
`derive(seed, "table.round", step, shard)`, so a resumed run draws the same bits.
"""

import os
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx

from akeratos._compat.mlx_seam import evaluate, load_safetensors, save_safetensors
from akeratos.errors import AkeratosError, ConfigError
from akeratos.rng import derive

SCHEMA = "metalforge.table-shard/v1"
Shard = dict[str, mx.array]
MemoryCounter = Callable[[], int]


class ShardCopiedError(AkeratosError):
    """A write-back allocated a second copy of a shard: something else still references it."""


def round_key(seed: int, step: int, shard: int) -> mx.array:
    """The MLX key of one shard's rounding draws at one step."""
    return mx.random.key(derive(seed, "table.round", step, shard))


def sr_bf16(x: mx.array, key: mx.array) -> mx.array:
    """Unbiased stochastic rounding of FP32 to BF16: add 16 random low bits, truncate."""
    bits = x.view(mx.uint32)
    r = mx.random.randint(0, 1 << 16, x.shape, dtype=mx.uint32, key=key)
    high = mx.array(0xFFFF0000, dtype=mx.uint32)
    return ((bits + r) & high).view(mx.float32).astype(mx.bfloat16)


def sr_int8(x: mx.array, key: mx.array) -> tuple[mx.array, mx.array]:
    """INT8 rows in [-127, 127] and their FP32 absmax scales; zero rows stay zero."""
    scale = mx.maximum(mx.max(mx.abs(x), axis=1), 1e-30) / 127.0
    u = mx.random.uniform(shape=x.shape, key=key)
    return mx.clip(mx.floor(x / scale[:, None] + u), -127, 127).astype(mx.int8), scale


def encode(x: mx.array, fmt: str, stochastic: bool, key: mx.array) -> Shard:
    """FP32 rows [n, w] in the stored format's fields."""
    if fmt == "F32":
        return {"rows": x}
    if fmt == "F1":
        q, scale = sr_int8(x, key)
        return {"rows": q, "scale": scale}
    return {"rows": sr_bf16(x, key) if stochastic else x.astype(mx.bfloat16)}


def decode(fields: Shard, fmt: str) -> mx.array:
    """Gathered stored rows back to FP32 [n, w]."""
    if fmt == "F1":
        return fields["rows"].astype(mx.float32) * fields["scale"][:, None]
    return fields["rows"].astype(mx.float32)


def row_fields(fmt: str) -> tuple[str, ...]:
    """The fields that hold a format's row values."""
    return ("rows", "scale") if fmt == "F1" else ("rows",)


def zeros(n: int, width: int, fmt: str, per_row: bool) -> Shard:
    """A shard of n zero rows with zero moments, stamps and counts."""
    rows: Shard
    if fmt == "F1":
        rows = {"rows": mx.zeros((n, width), mx.int8), "scale": mx.zeros((n,), mx.float32)}
    else:
        rows = {"rows": mx.zeros((n, width), mx.bfloat16 if fmt == "F2" else mx.float32)}
    rows |= {"v": mx.zeros((n,), mx.float32), "stamp": mx.zeros((n,), mx.int32)}
    return rows | ({"count": mx.zeros((n,), mx.int32)} if per_row else {})


def shard_file(s: int) -> str:
    """The checkpoint file name of shard s."""
    return f"shard_{s:04d}.safetensors"


def fsync(path: Path) -> None:
    """Flush a file or directory to the device with the portable fsync.

    `akeratos/ckpt.py` uses F_FULLFSYNC where `fcntl` has it.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ShardStore:
    """The sole owner of a table's shards.

    The store takes `shards` over, and the caller keeps no reference to them. `active_memory` is
    the active-memory counter in bytes, read around each write-back; it is injected because only
    `akeratos/_compat/mlx_seam.py` reads MLX's memory counters.
    """

    def __init__(self, shards: list[Shard], active_memory: MemoryCounter) -> None:
        self._shards = shards
        for sh in self._shards:
            evaluate(list(sh.values()))
        self._nbytes = [sum(a.nbytes for a in sh.values()) for sh in self._shards]
        self._active = active_memory

    def __len__(self) -> int:
        return len(self._shards)

    def nbytes(self, s: int | None = None) -> int:
        """Resident bytes of shard s, or of every shard."""
        return sum(self._nbytes) if s is None else self._nbytes[s]

    def rows(self, s: int) -> int:
        return self._shards[s]["v"].shape[0]

    def fields(self, s: int) -> tuple[str, ...]:
        return tuple(self._shards[s])

    def take(
        self, s: int, names: tuple[str, ...], rows: mx.array, *, evaluated: bool = True
    ) -> Shard:
        """The fields `names` of shard s at local rows, evaluated, so no pending gather holds it.

        With `evaluated=False` the caller must evaluate every returned field, or arrays computed
        from all of them, before the next `write_back` on any shard; the copy check stops the run
        otherwise.
        """
        sh = self._shards[s]
        out = {name: mx.take(sh[name], rows, axis=0) for name in names}
        if evaluated:
            evaluate(list(out.values()))
        return out

    def write_back(self, s: int, rows: mx.array, values: Shard) -> None:
        """Scatter evaluated `values` into shard s at local rows, then check for a copy.

        Active memory that grows by more than half the shard's bytes means a copy, and
        `ShardCopiedError` stops the run before the next shard, so a stray reference costs at most
        one shard, never the table.
        """
        sh = self._shards[s]
        before = self._active()
        for name, value in values.items():
            sh[name][rows] = value
        evaluate(list(sh.values()))
        grew = self._active() - before
        if grew > self._nbytes[s] // 2:
            msg = (
                f"shard {s}: write-back allocated {grew / 1e9:.2f} GB of a "
                f"{self._nbytes[s] / 1e9:.2f} GB shard; a view, an unevaluated op or a NumPy "
                "view still references it"
            )
            raise ShardCopiedError(msg)

    def save(self, path: Path, row_starts: list[int]) -> list[str]:
        """Write and flush one safetensors file per shard into `path`, with no second copy.

        Each file holds the shard's fields in sorted key order, since MLX's bytes follow the
        insertion order, and its metadata; shard s starts at global row `row_starts[s]`.
        """
        names: list[str] = []
        for s, sh in enumerate(self._shards):
            names.append(shard_file(s))
            meta = {
                "format": "mlx",
                "schema": SCHEMA,
                "schema_minor": "0",
                "shard": str(s),
                "row_start": str(row_starts[s]),
                "row_count": str(self.rows(s)),
            }
            save_safetensors(path / names[-1], {k: sh[k] for k in sorted(sh)}, meta)
            fsync(path / names[-1])
        return names

    @classmethod
    def load(cls, path: Path, row_starts: list[int], active_memory: MemoryCounter) -> "ShardStore":
        """The shards `save` wrote to `path`, each checked against its metadata."""
        shards: list[Shard] = []
        for s, start in enumerate(row_starts):
            file = path / shard_file(s)
            try:
                arrays, meta = load_safetensors(file)
            except (OSError, ValueError, RuntimeError) as e:
                raise ConfigError(f"{file}: {e}") from e
            want = {"format": "mlx", "schema": SCHEMA, "schema_minor": "0"}
            want |= {"shard": str(s), "row_start": str(start)}
            got = {k: meta.get(k) for k in want}
            if got != want or meta.get("row_count") != str(arrays["v"].shape[0]):
                raise ConfigError(f"{file}: metadata {meta}, expected {want}")
            shards.append(arrays)
            evaluate(list(arrays.values()))
        return cls(shards, active_memory)
