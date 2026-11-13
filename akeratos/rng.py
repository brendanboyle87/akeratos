"""Seed derivation for every random stream.

`derive` hashes a domain tag, the seed, the stream name and any indices with SHA-256 and keeps
the first 8 bytes as a little-endian uint64. The run's three seeds are `run_seeds`, and
`seed_hex` is the form a seed is written in. The tag holds the project's first name, MetalForge,
and never changes: every seed of the released runs derives from it.
"""

import hashlib

_DOMAIN = b"metalforge.rng/v1\0"
_U64_END = 2**64


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "little")


def derive(seed: int, stream: str, *indices: int) -> int:
    """A uint64 from the first 8 bytes, little-endian, of sha256 over the seed, stream and indices.

    Raises OverflowError for a seed or index outside 0..2**64 - 1, and UnicodeEncodeError for a
    stream name that is not ASCII.
    """
    message = _DOMAIN + _u64(seed) + stream.encode("ascii") + b"\0"
    message += b"".join(_u64(i) for i in indices)
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


def run_seeds(
    seed: int, init_seed: int | None = None, data_seed: int | None = None
) -> dict[str, int]:
    """The run's master seed and the two sub-seeds, each derived from it unless given.

    `init_seed` and `data_seed` default to `derive(seed, "init")` and `derive(seed, "data")`, so a
    sweep over consecutive master seeds never reuses one run's data seed as another's init seed.
    """
    seeds = {
        "seed": seed,
        "init_seed": derive(seed, "init") if init_seed is None else init_seed,
        "data_seed": derive(seed, "data") if data_seed is None else data_seed,
    }
    for name, value in seeds.items():
        if not 0 <= value < _U64_END:
            raise OverflowError(f"{name} {value} is outside 0..2**64 - 1")
    return seeds


def seed_hex(value: int) -> str:
    """A seed as it is written to disk."""
    return f"0x{value:016x}"
