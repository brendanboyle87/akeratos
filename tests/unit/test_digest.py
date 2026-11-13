"""Tests of `akeratos/digest.py`.

Canonical bytes (RFC 8785) and their digests, the pretty JSON form the repository commits
and its atomic write, and the strict JSON reader.
"""

import pytest
import rfc8785

from akeratos import digest


@pytest.mark.parametrize(
    ("value", "expect"),
    [
        (1.5e300, b"1.5e+300"),
        (-2.5e-8, b"-2.5e-8"),
        (123456.789, b"123456.789"),
        (1e21, b"1e+21"),
        (1e20, b"100000000000000000000"),
        (-0.0, b"0"),
        (2**53 - 1, b"9007199254740991"),
        ((1, "a"), b'[1,"a"]'),
        ({"b": [None, True], "a": "\u00e9\u2028"}, b'{"a":"\xc3\xa9\xe2\x80\xa8","b":[null,true]}'),
    ],
)
def test_canonical_forms(value: object, expect: bytes) -> None:
    assert digest.canonical(value) == expect


@pytest.mark.parametrize("value", [{"a": object()}, {1: "a"}, 2**53, float("nan"), float("inf")])
def test_canonical_refuses_a_value_with_no_canonical_form(value: object) -> None:
    with pytest.raises(rfc8785.CanonicalizationError):
        digest.canonical(value)


def test_digest_bytes() -> None:
    assert digest.digest_bytes(b"abc") == (
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_file_digest_reads_bytes_without_newline_conversion(tmp_path) -> None:
    path = tmp_path / "f.txt"
    path.write_bytes(b"a\r\nb\n")
    assert digest.file_digest(path) == digest.digest_bytes(b"a\r\nb\n")
    assert digest.file_digest(str(path)) != digest.digest_bytes(b"a\nb\n")


def test_pretty_form() -> None:
    value = {"b": "\u00e9", "a": [1, 2.5, None, {}]}
    text = '{\n  "a": [\n    1,\n    2.5,\n    null,\n    {}\n  ],\n  "b": "\u00e9"\n}\n'
    assert digest.pretty(value) == text
    assert digest.pretty_bytes(value) == text.encode("utf-8")


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({1: "a"}, digest.NonStringKeyError),
        ({"a": [{2: "b"}]}, digest.NonStringKeyError),
        (float("nan"), ValueError),
    ],
)
def test_pretty_refuses(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        digest.pretty(value)


def test_write_pretty_replaces_atomically(tmp_path) -> None:
    out = tmp_path / "state.json"
    out.write_bytes(b"old")
    digest.write_pretty(out, {"a": [1, {"b": 2}]})
    assert out.read_bytes() == b'{\n  "a": [\n    1,\n    {\n      "b": 2\n    }\n  ]\n}\n'
    with pytest.raises(digest.NonStringKeyError):
        digest.write_pretty(out, {"a": [{1: "a"}]})
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        ('{"x": [1.5, "\u00e9"]}', {"x": [1.5, "\u00e9"]}),
        (b'{"x": [1.5, "\xc3\xa9"]}', {"x": [1.5, "\u00e9"]}),
    ],
)
def test_loads_reads_str_and_utf8_bytes(text: str | bytes, expect: object) -> None:
    assert digest.loads(text) == expect


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ('{"a": 1, "a": 2}', digest.DuplicateKeyError),
        ('{"o": {"a": 1, "a": 1}}', digest.DuplicateKeyError),
        ("NaN", digest.NonFiniteNumberError),
        ("[Infinity]", digest.NonFiniteNumberError),
        ("-Infinity", digest.NonFiniteNumberError),
        ("[1e400]", digest.NonFiniteNumberError),
        (b"\xef\xbb\xbf{}", UnicodeDecodeError),
        (b'"\xff"', UnicodeDecodeError),
    ],
)
def test_loads_refuses(text: str | bytes, error: type[Exception]) -> None:
    with pytest.raises(error):
        digest.loads(text)


def test_load_reads_a_file_strictly(tmp_path) -> None:
    path = tmp_path / "f.json"
    path.write_bytes(b'{"a": [1, "\xc3\xa9"]}')
    assert digest.load(path) == {"a": [1, "\u00e9"]}
    path.write_bytes(b'{"a": 1, "a": 2}')
    with pytest.raises(digest.DuplicateKeyError):
        digest.load(str(path))
