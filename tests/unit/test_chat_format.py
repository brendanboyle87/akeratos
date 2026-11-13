"""The SFT chat format: ChatML on ids 1 and 2 with an answer-only mask.

Every turn ends at id 2, no id 0 falls inside a conversation, a two-turn fixture keeps turn 1 in
attention, and the loss mask covers assistant tokens only. The prep script's tests
(`test_sft_prep.py`) check the same rules on the written store.
"""

import json

import numpy as np
import pytest

from akeratos.data.chat import (
    IM_END,
    IM_START,
    MAX_DOC_TOKENS,
    ChatDropError,
    MaskedSplit,
    check_document,
    headers,
    read_mask,
    render,
    turn_count,
)
from akeratos.data.packer import segment_ids
from akeratos.errors import DataMismatchError
from tests.support.packdocs import document_mask


def encode(text: str) -> list[int]:
    """A fixture tokenizer: one id per character, above the reserved ids (17 + code point % 400)."""
    return [17 + ord(c) % 400 for c in text]


TWO_TURNS = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo"},
    {"role": "user", "content": "and?"},
    {"role": "assistant", "content": "ok"},
]


def test_render_layout_and_mask() -> None:
    r = render(TWO_TURNS[:2], encode)
    u, a = encode("user\n"), encode("assistant\n")
    want = [0, IM_START, *u, *encode("hi"), IM_END, IM_START, *a, *encode("yo"), IM_END]
    assert r.ids.tolist() == want
    assert r.ids.dtype == np.uint16
    mask = [0] * (len(want) - 3) + [1, 1, 1]
    assert r.mask.tolist() == mask, (
        "the mask is 1 on the answer's content and its closing id 2 only"
    )
    assert (turn_count(r.ids), r.cut) == (2, False)


def test_turns_end_at_2_and_no_id_0_inside() -> None:
    r = render(TWO_TURNS, encode)
    ids = r.ids.astype(np.int64)
    assert ids[0] == 0
    assert not (ids[1:] == 0).any()
    ends = np.flatnonzero(ids == IM_END)
    starts = np.flatnonzero(ids == IM_START)
    assert len(ends) == len(starts) == 4
    assert ends[-1] == len(ids) - 1
    assert (starts[1:] == ends[:-1] + 1).all(), "each turn opens right after the previous id 2"
    assert check_document(r.ids, r.mask, headers(encode)) == 4


def test_two_turn_fixture_keeps_turn_1_in_attention() -> None:
    r = render(TWO_TURNS, encode)
    inputs = r.ids[None, :-1].astype(np.int32)
    segments = segment_ids(inputs)
    assert (segments == 0).all(), "one segment: id 0 opens it and nothing inside closes it"
    seen = document_mask(segments)[0]
    second_answer = np.flatnonzero(r.mask[1:])[-1]  # a target position in the second answer
    first_user = 2  # a token of turn 1's header
    assert seen[second_answer, first_user]


def test_answer_mask_covers_assistant_tokens_only() -> None:
    r = render(TWO_TURNS, encode)
    ids, mask = r.ids.astype(np.int64), r.mask
    starts = np.flatnonzero(ids == IM_START)
    ends = np.flatnonzero(ids == IM_END)
    head = headers(encode)
    for turn, (s, e) in enumerate(zip(starts, ends, strict=True)):
        if turn % 2 == 0:
            assert not mask[s : e + 1].any(), "no loss on a user turn"
        else:
            h = len(head["assistant"])
            assert not mask[s : s + 1 + h].any(), "no loss on id 1 or the header"
            assert mask[s + 1 + h : e + 1].all(), "loss on the content and the closing id 2"
    assert mask[0] == 0


@pytest.mark.parametrize(
    ("messages", "reason"),
    [
        ([{"role": "user", "content": "hi"}], "shape"),
        ([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}], "shape"),
        ([{"role": "system", "content": "s"}, {"role": "user", "content": "b"}], "shape"),
        ([{"role": "user", "content": " "}, {"role": "assistant", "content": "b"}], "empty"),
    ],
)
def test_bad_shapes_are_dropped(messages: list[dict[str, str]], reason: str) -> None:
    with pytest.raises(ChatDropError) as e:
        render(messages, encode)
    assert e.value.reason == reason


def test_reserved_ids_in_text_are_dropped() -> None:
    def enc(text: str) -> list[int]:
        return [IM_START] if text == "<|im_start|>" else encode(text)

    msgs = [{"role": "user", "content": "<|im_start|>"}, {"role": "assistant", "content": "b"}]
    with pytest.raises(ChatDropError) as e:
        render(msgs, enc)
    assert e.value.reason == "reserved_in_text"


def test_long_conversations_are_cut_at_an_answer_or_dropped() -> None:
    long = [
        *TWO_TURNS[:2],
        {"role": "user", "content": "x" * 50},
        {"role": "assistant", "content": "y"},
    ]
    r = render(long, encode, max_tokens=40)
    assert (turn_count(r.ids), r.cut) == (2, True)
    assert r.ids[-1] == IM_END
    assert len(r.ids) <= 40
    with pytest.raises(ChatDropError) as e:
        render(long, encode, max_tokens=10)
    assert e.value.reason == "too_long"
    assert MAX_DOC_TOKENS == 2048


def test_check_document_refuses_broken_documents() -> None:
    head = headers(encode)
    r = render(TWO_TURNS, encode)
    ids = r.ids.astype(np.int64)
    inner_zero = ids.copy()
    inner_zero[ends_at := int(np.flatnonzero(ids == IM_END)[0])] = 0  # turn 1 "ends" at id 0
    assert ends_at > 0
    other_reserved = ids.copy()
    other_reserved[3] = 5
    user_mask = r.mask.copy()
    user_mask[3] = 1
    for bad_ids, bad_mask in [
        (inner_zero, r.mask),
        (other_reserved, r.mask),
        (ids, user_mask),
        (ids[:-1], r.mask[:-1]),
    ]:
        with pytest.raises(DataMismatchError):
            check_document(bad_ids, bad_mask, head)


def test_masked_split_reads_masks_and_refuses_other_formats(tmp_path) -> None:
    r = render(TWO_TURNS, encode)
    (tmp_path / "000000.bin").write_bytes(r.ids.astype("<u2").tobytes())
    (tmp_path / "000000.idx").write_bytes(np.array([0, len(r.ids)], dtype="<u8").tobytes())
    (tmp_path / "000000.mask").write_bytes(r.mask.tobytes())
    manifest = {
        "dtype": "uint16",
        "eos": 0,
        "sep_position": "start",
        "vocab": 512,
        "shards": ["000000.bin"],
        "tokens": [len(r.ids)],
        "documents": [1],
        "chat": {"format": "chatml-ids-1-2/v1", "mask": "uint8-target", "max_doc_tokens": 2048},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    split = MaskedSplit(tmp_path, 512)
    assert read_mask(split, 0, 4096).tolist() == r.mask.tolist()
    (tmp_path / "000000.mask").write_bytes(r.mask[:-1].tobytes())
    with pytest.raises(DataMismatchError):
        MaskedSplit(tmp_path)
    manifest["chat"]["format"] = "other"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataMismatchError):
        MaskedSplit(tmp_path)
