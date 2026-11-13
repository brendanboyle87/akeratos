"""Canonical JSON bytes, SHA-256 digests and the JSON form the repository commits.

`canonical` gives the RFC 8785 bytes of a value, from the pinned `rfc8785` package.
`digest_bytes` and `file_digest` give digests in the form `sha256:<64 lowercase hex>`.
`pretty` is the one form for JSON files the repository commits, and `loads` and `load` are a
strict reader that refuses duplicate keys and numbers that are not finite.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, NoReturn, cast

import rfc8785

ALGORITHM = "sha256"


class DuplicateKeyError(ValueError):
    """A JSON object names the same key twice."""


class NonFiniteNumberError(ValueError):
    """A JSON text uses NaN, Infinity or -Infinity, or a number that overflows to infinity."""


class NonStringKeyError(ValueError):
    """A value to write as committed JSON has a key that is not a `str`."""


def digest_bytes(data: bytes) -> str:
    return f"{ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def file_digest(path: str | Path) -> str:
    """The digest of a file's bytes as checked out, with no newline conversion."""
    return digest_bytes(Path(path).read_bytes())


def canonical(value: object) -> bytes:
    """The RFC 8785 bytes of a value.

    `rfc8785.CanonicalizationError`, a ValueError, rejects NaN, an infinity, an integer beyond
    2**53-1, a non-`str` key and a value with no JSON form.
    """
    return rfc8785.dumps(cast("Any", value))


def _check_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, val in cast("dict[object, object]", value).items():
            if not isinstance(key, str):
                raise NonStringKeyError(f"non-str key {key!r}")
            _check_keys(val)
    elif isinstance(value, (list, tuple)):
        for val in cast("list[object] | tuple[object, ...]", value):
            _check_keys(val)


def pretty(value: object) -> str:
    """The form for committed JSON: two-space indent, sorted keys, UTF-8 text, final newline."""
    _check_keys(value)
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def pretty_bytes(value: object) -> bytes:
    return pretty(value).encode("utf-8")


def write_pretty(path: str | Path, value: object) -> None:
    """Write a committed JSON file in the `pretty` form, replacing the target atomically."""
    target = Path(path)
    data = pretty_bytes(value)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".tmp-{target.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, val in pairs:
        if key in out:
            raise DuplicateKeyError(f"duplicate key {key!r}")
        out[key] = val
    return out


def _constant(name: str) -> NoReturn:
    raise NonFiniteNumberError(f"non-finite number {name}")


def _float(text: str) -> float:
    value = float(text)
    if math.isinf(value):
        raise NonFiniteNumberError(f"number {text} overflows to infinity")
    return value


def loads(text: str | bytes) -> object:
    """Parse JSON strictly: duplicate keys, NaN, Infinity, overflow and a BOM raise ValueError."""
    if isinstance(text, bytes):
        if text.startswith(b"\xef\xbb\xbf"):
            raise UnicodeDecodeError("utf-8", text, 0, 3, "byte-order mark")
        text = text.decode("utf-8")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)


def load(path: str | Path) -> object:
    """Parse a JSON file strictly, as `loads` does."""
    return loads(Path(path).read_bytes())
