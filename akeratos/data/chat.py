r"""The chat format of the post-training data: ChatML on ids 1 and 2, with an answer-only mask.

Ids 1-16 are reserved; `<|im_start|>` is 1 and `<|im_end|>` is 2. This module is the one
definition of the template: the preparation script writes the SFT store with it, and
post-training and the decoder read the same constants. A document is one conversation,
`[0] + turns`, and a turn is

    [1] + encode(role + "\n") + encode(content) + [2]

The header and the content are encoded separately, so no token spans the boundary. No newline
follows id 2, so every turn ends at id 2 and id 0 appears only at the document's start: id 0
opens a segment in the packer, the attention mask and the hasher, so an id 0 inside a
conversation would cut its earlier turns out of attention. Roles are `user` and `assistant`
only, alternating, starting with `user` and ending with `assistant`; there is no system prompt.
This module imports NumPy, the standard library and `akeratos.data` only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from akeratos.data.packer import RESERVED, SEP_ID
from akeratos.data.store import Split
from akeratos.errors import DataMismatchError

IM_START = 1  # `<|im_start|>`
IM_END = 2  # `<|im_end|>`: every turn ends here, and decoding stops here
ROLES = ("user", "assistant")
FORMAT = "chatml-ids-1-2/v1"
MASK_DTYPE = "uint8-target"
MAX_DOC_TOKENS = 2048  # Billet's trained context, id 0 included, so a document never needs a crop

Encode = Callable[[str], Sequence[int]]
Message = Mapping[str, str]


class ChatDropError(ValueError):
    """A conversation the format cannot carry; `reason` names the drop counter."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class Rendered:
    """One conversation as a document: ids and mask of equal length, and whether turns were cut."""

    ids: NDArray[np.uint16]
    mask: NDArray[np.uint8]
    cut: bool


def check_messages(messages: Sequence[Message]) -> None:
    """Raise `ChatDropError("shape")` unless roles alternate from `user` and end on `assistant`."""
    if len(messages) < len(ROLES) or len(messages) % len(ROLES):
        raise ChatDropError(
            "shape", f"{len(messages)} turns; a conversation is user/assistant pairs"
        )
    for i, m in enumerate(messages):
        if set(m) != {"role", "content"}:
            raise ChatDropError("shape", f"turn {i} has keys {sorted(m)}")
        if m["role"] != ROLES[i % 2]:
            raise ChatDropError("shape", f"turn {i} is {m['role']!r}, expected {ROLES[i % 2]!r}")
        content = cast("object", m["content"])  # data from JSON, whatever the annotation says
        if not isinstance(content, str) or not content.strip():
            raise ChatDropError("empty", f"turn {i} has no text")


def headers(encode: Encode) -> dict[str, tuple[int, ...]]:
    r"""Each role's header tokens, `encode(role + "\n")`."""
    return {role: tuple(int(t) for t in encode(role + "\n")) for role in ROLES}


def _content(encode: Encode, text: str, turn: int) -> list[int]:
    ids = [int(t) for t in encode(text)]
    if not ids:
        raise ChatDropError("empty", f"turn {turn} encodes to no tokens")
    low, high = SEP_ID, RESERVED[1]
    if any(low <= t <= high for t in ids):
        raise ChatDropError("reserved_in_text", f"turn {turn} encodes to an id in {low}-{high}")
    return ids


def render(
    messages: Sequence[Message], encode: Encode, max_tokens: int = MAX_DOC_TOKENS
) -> Rendered:
    """One conversation as `[0] + turns` with its answer-only mask.

    The mask has one byte per token: 1 means the token is a loss target (predicted from the token
    before it). It is 1 on an assistant turn's content and its closing id 2, and 0 on id 0, on user
    turns and on an assistant turn's id 1 and header. With rows of T + 1 tokens, the loss mask for
    `targets = row[1:]` is `mask_row[1:]`. A conversation longer than `max_tokens` is cut to its
    longest prefix that ends with an assistant turn and fits. A turn whose text encodes to an id
    in 0-16 is dropped as `reserved_in_text`, and one whose first exchange does not fit as
    `too_long`.
    """
    check_messages(messages)
    head = headers(encode)
    ids, mask = [SEP_ID], [0]
    kept = 0
    for i in range(0, len(messages), 2):
        pair_ids: list[int] = []
        pair_mask: list[int] = []
        for m in messages[i : i + 2]:
            body = _content(encode, m["content"], i)
            framed = [IM_START, *head[m["role"]], *body, IM_END]
            answer = m["role"] == "assistant"
            pair_ids += framed
            pair_mask += [0] * (1 + len(head[m["role"]])) + [int(answer)] * (len(body) + 1)
        if len(ids) + len(pair_ids) > max_tokens:
            break
        ids += pair_ids
        mask += pair_mask
        kept += 2
    if kept == 0:
        raise ChatDropError("too_long", f"the first exchange needs more than {max_tokens} tokens")
    return Rendered(
        np.asarray(ids, dtype=np.uint16),
        np.asarray(mask, dtype=np.uint8),
        kept < len(messages),
    )


def _bad(why: str) -> DataMismatchError:
    return DataMismatchError(f"chat document breaks {FORMAT}: {why}")


def _turn_bounds(ids: NDArray[np.int64]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Each turn's opener and end positions, checked to tile the document after its id 0."""
    starts = np.flatnonzero(ids == IM_START)
    ends = np.flatnonzero(ids == IM_END)
    reserved = (ids >= IM_START) & (ids <= RESERVED[1])
    if reserved.sum() != len(starts) + len(ends):
        raise _bad("a reserved id other than 1 and 2")
    if len(starts) != len(ends) or len(starts) < len(ROLES) or len(starts) % len(ROLES):
        raise _bad(f"{len(starts)} turn openers and {len(ends)} turn ends")
    if starts[0] != 1 or ends[-1] != len(ids) - 1 or (starts[1:] != ends[:-1] + 1).any():
        raise _bad("turns must tile the document, each [1] ... [2] with nothing between them")
    return starts, ends


def check_document(
    ids: NDArray[np.integer],
    mask: NDArray[np.integer],
    head: Mapping[str, Sequence[int]],
    max_tokens: int = MAX_DOC_TOKENS,
) -> int:
    """Check one stored conversation against the format; return its number of turns."""
    ids = np.asarray(ids, dtype=np.int64)
    mask = np.asarray(mask, dtype=np.int64)
    if len(ids) != len(mask):
        raise _bad(f"{len(ids)} ids but {len(mask)} mask bytes")
    if not 1 < len(ids) <= max_tokens:
        raise _bad(f"length {len(ids)} outside 2-{max_tokens}")
    if ids[0] != SEP_ID or (ids[1:] == SEP_ID).any():
        raise _bad("id 0 must open the document and appear nowhere else")
    if mask[0] != 0 or not np.isin(mask, (0, 1)).all():
        raise _bad("mask bytes must be 0 or 1, and 0 on id 0")
    starts, ends = _turn_bounds(ids)
    for turn, (s, e) in enumerate(zip(starts, ends, strict=True)):
        role = ROLES[turn % len(ROLES)]
        h = tuple(int(t) for t in head[role])
        if tuple(int(t) for t in ids[s + 1 : s + 1 + len(h)]) != h or e <= s + len(h) + 1:
            raise _bad(f"turn {turn} does not open with the {role!r} header and some content")
        want = np.zeros(e - s + 1, dtype=np.int64)
        if role == "assistant":
            want[1 + len(h) :] = 1
        if (mask[s : e + 1] != want).any():
            raise _bad(f"turn {turn}'s mask is not the {role} rule")
    return len(starts)


class MaskedSplit(Split):
    """An SFT split: `Split` plus one uint8 mask map per shard, and the manifest's chat block.

    Beside each shard sits `NNNNNN.mask`, uint8, one byte per token of the `.bin`, and the
    manifest's `chat` block names the format and the mask type.
    """

    def __init__(self, path: str | Path, vocab: int | None = None) -> None:
        super().__init__(path, vocab)
        m = json.loads((self.path / "manifest.json").read_text())
        chat: dict[str, Any] = m.get("chat") or {}
        if (chat.get("format"), chat.get("mask")) != (FORMAT, MASK_DTYPE):
            raise DataMismatchError(f"{self.path}: not a {FORMAT} split with {MASK_DTYPE} masks")
        self.masks: list[NDArray[np.uint8]] = []
        for name, shard in zip(m["shards"], self.shards, strict=True):
            p = self.path / f"{name[:-4]}.mask"
            if p.stat().st_size != len(shard):
                msg = f"{p}: {p.stat().st_size} bytes for {len(shard)} tokens"
                raise DataMismatchError(msg)
            self.masks.append(np.memmap(p, dtype=np.uint8, mode="r", shape=(len(shard),)))


def read_mask(split: MaskedSplit, index: int, count: int) -> NDArray[np.uint8]:
    """The mask bytes of the first `count` tokens of document `index`, beside `split.read`."""
    shard, start = int(split.doc_shard[index]), int(split.doc_start[index])
    n = min(count, int(split.lengths[index]))
    return np.asarray(split.masks[shard][start : start + n], dtype=np.uint8)


def turn_count(ids: NDArray[np.integer]) -> int:
    """The number of turns in a rendered document (its id 1s)."""
    return int((np.asarray(ids) == IM_START).sum())


def last_answer(
    ids: NDArray[np.integer], mask: NDArray[np.integer]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """(prompt, answer): a document before its last answer, and that answer without its id 2.

    `mask` is the document's mask, one byte per token. The prompt is every turn but the last
    answer, then that turn's id 1 and the assistant's header, which is where a decoder starts.
    """
    ids = np.asarray(ids, dtype=np.int32)
    live = np.flatnonzero(np.asarray(mask))
    openers = np.flatnonzero(ids == IM_START)
    after = live[live > openers[-1]] if len(openers) else live[:0]
    if not len(after):
        raise _bad("a document with no answer after its last id 1")
    first = int(after[0])
    return ids[:first], ids[first:-1]


def prompts(path: str | Path, vocab: int) -> list[tuple[NDArray[np.int32], NDArray[np.int32]]]:
    """`last_answer` of every document of an SFT split, in order: the readouts' questions."""
    try:
        split = MaskedSplit(Path(path).expanduser(), vocab)
    except OSError as e:  # a missing manifest, shard or mask file
        raise DataMismatchError(f"{path}: not an SFT split: {e}") from e
    out: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []
    for i, n in enumerate(split.lengths.tolist()):
        ids = split.read(i, n)
        if ids[0] != SEP_ID or (ids[1:] == SEP_ID).any():
            raise DataMismatchError(f"{split.path}: document {i} breaks `[0] + turns`")
        out.append(last_answer(ids, read_mask(split, i, n)))
    return out
