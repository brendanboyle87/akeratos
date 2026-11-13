"""The seed derivation in `akeratos/rng.py`, which the packer's data order uses.

`derive` follows its formula by hand, gives different keys for each stream and index, refuses
values out of range and non-ASCII streams, and matches pinned golden values.
"""

import hashlib

import pytest

from akeratos.rng import derive


def by_hand(seed: int, stream: str, *indices: int) -> int:
    message = b"metalforge.rng/v1\0" + seed.to_bytes(8, "little") + stream.encode() + b"\0"
    for i in indices:
        message += i.to_bytes(8, "little")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little")


@pytest.mark.parametrize(
    ("seed", "stream", "indices"),
    [(0, "data", ()), (7, "data", (0,)), (2**64 - 1, "table.round", (3, 1)), (1, "init", ())],
)
def test_derive_follows_the_spec_formula(seed: int, stream: str, indices: tuple[int, ...]) -> None:
    value = derive(seed, stream, *indices)
    assert value == by_hand(seed, stream, *indices)
    assert 0 <= value < 2**64


def test_streams_and_indices_give_different_keys() -> None:
    keys = {derive(1, "data"), derive(1, "init"), derive(1, "data", 0), derive(1, "data", 1)}
    assert len(keys) == 4


@pytest.mark.parametrize(("seed", "index"), [(-1, 0), (2**64, 0), (0, -1)])
def test_out_of_range_values_are_refused(seed: int, index: int) -> None:
    with pytest.raises(OverflowError):
        derive(seed, "data", index)


def test_a_non_ascii_stream_is_refused() -> None:
    with pytest.raises(UnicodeEncodeError):
        derive(0, "dàta")


def test_derive_matches_pinned_values() -> None:
    """Golden values, so a change to the formula or its encoding cannot pass silently."""
    assert derive(0, "data", 0) == 12761468709178515585
    assert derive(2**64 - 1, "table.round", 3, 1) == 16025407123371746949
