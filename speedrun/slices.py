"""Cut the 5-minute speedrun track's slices from Akeratos's data store.

The store is the one `scripts/data/prepare_store.py` builds (docs/reproduce.md). `cut` writes
the slices into one new folder and opens them with Akeratos's readers, `Split` and `HeldOut`
over the scored set. The script then prints each file's size and two pins for
`akeratos/speedrun.py`: `SLICES` (the SHA-256 of `SHA256SUMS`, the store, the dataset and the
tokenizer) and `HELDOUT` (the scored set's targets, documents and bytes). The same store gives
the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

from akeratos import digest
from akeratos.configs import VOCAB
from akeratos.data.store import Split
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout import HeldOut
from akeratos.speedrun import EVAL_SEQ_LEN, EVAL_TARGETS

# The SHA-256 of the store's manifest.json (docs/reproduce.md). No option changes it.
STORE_MANIFEST_SHA256 = "503beba3d802bfc2302d20ed31ceba167e8040e7573e1fad02373ae73fd2ba70"
TRAIN_SHARDS = 1  # the store's first training shard, about 134M tokens
SCHEMA = "metalforge.speedrun-slices/v1"
SUMS, CARD = "SHA256SUMS", "README.md"
# A split manifest's lists with one entry per shard, as `scripts/data/prepare_store.py` writes them.
PER_SHARD = frozenset(
    {
        "shards",
        "tokens",
        "idx",
        "documents",
        "utf8_bytes",
        "max_id",
        "sha256_bin",
        "sha256_idx",
        "end_cursor",
    }
)
EXIT_DATA, EXIT_CONFIG = 22, 78

CARD_TEXT = """\
---
license: odc-by
---

# Akeratos 5-minute speedrun slices

The fixed data of the 5-minute Mac speedrun track of
[Akeratos](https://github.com/cogvera-labs/akeratos): a training slice and a held-out slice
of [FineWeb-Edu](https://huggingface.co/datasets/{dataset}/tree/{dataset_rev}) (`{sample}`), as
token ids of [SmolLM2](https://huggingface.co/{tokenizer}/tree/{tokenizer_rev})'s tokenizer.
`bake speedrun` trains a model on the first for 300 seconds of step time and scores its held-out
bits per byte on the second. The track's rules are in
[docs/speedrun.md](https://github.com/cogvera-labs/akeratos/blob/main/docs/speedrun.md).

## Files

- `train/`: {train_tokens:,} tokens in {train_docs:,} documents, the first training shard of
  Akeratos's data store `{store}`, as a `.bin` and an `.idx` file.
- `heldout/`: the store's whole held-out split, {held_tokens:,} tokens in {held_docs:,}
  documents, as `.bin`, `.idx` and `.bytes` files.
- `train/manifest.json` and `heldout/manifest.json`: each split's shards, with their SHA-256s.
- `token_bytes.u16`: each token id's length in bytes of text, as little-endian uint16.
- `manifest.json`: the store and its manifest's SHA-256, the dataset and tokenizer revisions,
  the licence and the attribution notice.
- `SHA256SUMS`: the SHA-256 of every file but itself and this card.

A `.bin` file holds little-endian uint16 token ids, each document stored as id 0 and then its
ids. An `.idx` file holds each document's start offset in the `.bin`, then the `.bin`'s length
in ids, as little-endian uint64. A `.bytes` file holds each document's length in bytes of UTF-8
text, as little-endian uint64. The tokenizer's files are not included: they are in SmolLM2's repository
at the revision linked above.

## Check the files

    shasum -a 256 -c SHA256SUMS

Akeratos pins the SHA-256 of `SHA256SUMS` in `akeratos/speedrun.py`, and `bake speedrun`
checks every file it lists before each run.

## How they were made

Akeratos's `speedrun/slices.py` copied the shards, `heldout/manifest.json` and
`token_bytes.u16` byte for byte from the data store that
[`scripts/data/prepare_store.py`](https://github.com/cogvera-labs/akeratos/blob/main/scripts/data/prepare_store.py)
builds, store `{store}`, whose `manifest.json` has the SHA-256 `{manifest_sha256}`, and wrote
the other files. Anyone who builds that store can cut the same bytes:
[Reproduce the runs](https://github.com/cogvera-labs/akeratos/blob/main/docs/reproduce.md#2-build-the-training-data)
shows how.

## Licence

These files are released under the
[Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/),
as FineWeb-Edu is. {attribution}
"""


class Cut(NamedTuple):
    """What a cut wrote: each file's size in bytes, and the pins for `akeratos/speedrun.py`."""

    sizes: dict[str, int]
    slices: dict[str, str]
    heldout: dict[str, int]


class _Store(NamedTuple):
    """The store's three manifests, parsed, and every file the slices copy, checked."""

    root: dict[str, Any]
    train: dict[str, Any]
    held: dict[str, Any]
    files: dict[str, bytes]  # by path in the slices


def _read(path: Path, sha256: str) -> bytes:
    """`path`'s bytes, refused unless their SHA-256 is `sha256` (hex)."""
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise DataMismatchError(f"{path}: SHA-256 {got}, expected {sha256}")
    return data


def _shards(store: Path, split: str, m: dict[str, Any], n: int | None) -> dict[str, bytes]:
    """The first `n` shards of a split (all when None), each `.bin` and `.idx` checked."""
    files: dict[str, bytes] = {}
    for i, name in enumerate(m["shards"][:n]):
        idx = m["idx"][i]
        files[f"{split}/{name}"] = _read(store / split / name, m["sha256_bin"][i])
        files[f"{split}/{idx}"] = _read(store / split / idx, m["sha256_idx"][i])
    return files


def _sidecars(store: Path, held: dict[str, Any]) -> dict[str, bytes]:
    """The held-out `.bytes` files, each checked against the store's `SHA256SUMS`."""
    lines = (store / SUMS).read_text(encoding="utf-8").splitlines()
    sums = {name: sha for sha, name in (line.split("  ", 1) for line in lines)}
    files: dict[str, bytes] = {}
    for name in held["bytes"]:
        rel = f"heldout/{name}"
        if rel not in sums:
            raise DataMismatchError(f"{store / SUMS} does not list {rel}")
        files[rel] = _read(store / rel, sums[rel])
    return files


def _checked(store: Path, pin: str) -> _Store:
    """Read the store's manifests and the files the slices copy, each checked as it is read."""
    root = json.loads(_read(store / "manifest.json", pin))
    outputs = root["outputs"]
    train = json.loads(_read(store / "train" / "manifest.json", outputs["train_manifest_sha256"]))
    raw = _read(store / "heldout" / "manifest.json", outputs["heldout_manifest_sha256"])
    held = json.loads(raw)
    files = {"heldout/manifest.json": raw}
    files["token_bytes.u16"] = _read(store / "token_bytes.u16", outputs["token_bytes_sha256"])
    files |= _shards(store, "train", train, TRAIN_SHARDS)
    files |= _shards(store, "heldout", held, None)
    files |= _sidecars(store, held)
    return _Store(root, train, held, files)


def _root_manifest(root: dict[str, Any], pin: str) -> dict[str, Any]:
    """The slices' `manifest.json`: their source, licence and attribution notice."""
    src, tok, lic = root["source"], root["tokenizer"], root["licence"]
    return {
        "schema": SCHEMA,
        "source": {"store": root["outputs"]["fingerprint"][:16], "manifest_sha256": pin},
        "dataset": f"{src['repo_id']}@{src['revision']}",
        "tokenizer": f"{tok['repo_id']}@{tok['revision']}",
        "spdx": lic["spdx_expected"],
        "attribution": lic["attribution_notice"],
        "outputs": {"token_bytes_sha256": root["outputs"]["token_bytes_sha256"]},
    }


def _card(s: _Store, meta: dict[str, Any]) -> str:
    """The dataset card, `README.md`."""
    src, tok = s.root["source"], s.root["tokenizer"]
    return CARD_TEXT.format(
        dataset=src["repo_id"],
        dataset_rev=src["revision"],
        sample=src["sample"],
        tokenizer=tok["repo_id"],
        tokenizer_rev=tok["revision"],
        store=meta["source"]["store"],
        manifest_sha256=meta["source"]["manifest_sha256"],
        train_tokens=sum(s.train["tokens"][:TRAIN_SHARDS]),
        train_docs=sum(s.train["documents"][:TRAIN_SHARDS]),
        held_tokens=s.held["tokens_total"],
        held_docs=s.held["documents_total"],
        attribution=meta["attribution"],
    )


def cut(store: Path, out: Path, pin: str) -> Cut:
    """Cut the slices from `store` into the new folder `out`, which must not exist.

    The folder holds:

    - `train/`: the store's first training shard, its `.bin` and `.idx`, and a one-shard
      `manifest.json` that keeps the store's SHA-256s for them;
    - `heldout/`: the store's whole held-out split, its `manifest.json` and each shard's `.bin`,
      `.idx` and `.bytes`;
    - `token_bytes.u16`, each token id's length in bytes of text;
    - `manifest.json`: the store and its manifest's SHA-256, the dataset and tokenizer revisions,
      the licence, the attribution notice and `token_bytes.u16`'s SHA-256;
    - `SHA256SUMS`, in `shasum -a 256` format, for every file but itself and `README.md`;
    - `README.md`, the card of the Hugging Face dataset.

    Nothing is written until every file to copy is checked. The store's `manifest.json` must have
    the SHA-256 `pin` (hex); the split manifests and `token_bytes.u16` must have the SHA-256s it
    records, and each shard's `.bin` and `.idx` those its split manifest records. A held-out
    `.bytes` file, which no manifest hashes, must match the store's `SHA256SUMS`.
    """
    if out.exists():
        raise ConfigError(f"{out} exists; cut into a new folder")
    s = _checked(store, pin)
    meta = _root_manifest(s.root, pin)
    one = {k: v[:TRAIN_SHARDS] if k in PER_SHARD else v for k, v in s.train.items()}
    files = s.files | {"train/manifest.json": digest.canonical(one)}
    files["manifest.json"] = digest.canonical(meta)
    for rel, data in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    Split(out / "train", vocab=VOCAB)
    scored = HeldOut(out / "heldout", EVAL_SEQ_LEN, EVAL_TARGETS, doc_mask=False, vocab=VOCAB)
    sums = "".join(f"{hashlib.sha256(files[r]).hexdigest()}  {r}\n" for r in sorted(files))
    (out / SUMS).write_text(sums, encoding="utf-8")
    (out / CARD).write_text(_card(s, meta), encoding="utf-8")
    slices = {"sha256sums": digest.digest_bytes(sums.encode()), "store": meta["source"]["store"]}
    slices |= {"dataset": meta["dataset"], "tokenizer": meta["tokenizer"]}
    sizes = {rel: (out / rel).stat().st_size for rel in sorted([*files, SUMS, CARD])}
    counts = {"targets": scored.targets, "docs": scored.docs, "bytes": scored.bytes}
    return Cut(sizes, slices, counts)


def main(argv: list[str] | None = None) -> int:
    """Cut the slices and print each file's size and the pins; returns the exit code."""
    p = argparse.ArgumentParser(
        prog="speedrun/slices.py",
        description="Cut the 5-minute speedrun track's slices from Akeratos's data store.",
        allow_abbrev=False,
    )
    p.add_argument(
        "--store", type=Path, required=True, help="the store scripts/data/prepare_store.py built"
    )
    p.add_argument("--out", type=Path, required=True, help="a new folder for the slices")
    a = p.parse_args(argv)
    try:
        c = cut(a.store.expanduser(), a.out.expanduser(), STORE_MANIFEST_SHA256)
    except DataMismatchError as e:
        sys.stderr.write(f"slices: {e}\n")
        return EXIT_DATA
    except (ConfigError, OSError) as e:
        sys.stderr.write(f"slices: {e}\n")
        return EXIT_CONFIG
    lines = [f"{rel} {size}" for rel, size in c.sizes.items()]
    lines.append(f"{len(c.sizes)} files, {sum(c.sizes.values())} bytes")
    lines.append(f"SLICES = {json.dumps(c.slices, indent=4)}")
    lines.append(f"HELDOUT = {json.dumps(c.heldout)}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
