"""Stage 1's documents, packed rows and prompts, read from the SFT store.

The chat template is `akeratos.data.chat`'s: a document is one conversation, `[0] + turns`,
each turn `[1] + header + content + [2]`, so id 0 opens the conversation's segment and appears
nowhere else in it, and every turn ends at id 2, never at id 0. Its mask is 1 on each assistant
turn's content and closing id 2, and 0 elsewhere, so user and template tokens carry no loss.

A `Doc`'s mask is aligned to the targets: row position i is scored on token i + 1 when that
token's mask byte is 1. This module imports no MLX.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from akeratos.configs import ModelConfig
from akeratos.data.chat import IM_END, IM_START, MaskedSplit, last_answer, read_mask
from akeratos.data.packer import SEP_ID, ExhaustedError, segment_ids
from akeratos.errors import DataMismatchError
from akeratos.rng import derive, seed_hex
from akeratos.train.source import StepRows

STAGE1_IDS = (IM_START, IM_END)  # the reserved ids stage 1 unmasks at Billet


@dataclass(frozen=True)
class Doc:
    """One conversation: ids [n] int32 opening with id 0, and its target-aligned mask [n - 1]."""

    ids: NDArray[np.int32]
    mask: NDArray[np.float32]


def stage_config(cfg: ModelConfig, stage: int | None) -> ModelConfig:
    """The model configuration under `post.stage`: ids 1 and 2 unmasked when a stage is set.

    Every other id of `head.masked_ids` stays masked.
    """
    if stage is None:
        return cfg
    return replace(cfg, masked_ids=tuple(i for i in cfg.masked_ids if i not in STAGE1_IDS))


def documents(path: str | Path, vocab: int) -> list[Doc]:
    """Every document of an SFT split, with its target-aligned mask.

    The split holds one uint8 mask byte per token. `chat.MaskedSplit` refuses a split that is not
    in the chat format or whose vocabulary is not `vocab`, with `DataMismatchError`.
    """
    try:
        split = MaskedSplit(Path(path).expanduser(), vocab)
    except OSError as e:  # a missing manifest, shard or mask file
        raise DataMismatchError(f"{path}: not an SFT split: {e}") from e
    out: list[Doc] = []
    for i, n in enumerate(split.lengths.tolist()):
        ids = split.read(i, n)
        if ids[0] != SEP_ID or (ids[1:] == SEP_ID).any():
            raise DataMismatchError(f"{split.path}: document {i} breaks `[0] + turns`")
        mask = read_mask(split, i, n)
        out.append(Doc(ids.astype(np.int32), mask[1:].astype(np.float32)))
    return out


def epochs_of(path: str | Path) -> int:
    """The split's store's `epochs` (2 when the training tokens are under 20M), else 1."""
    p = Path(path).expanduser()
    for m in (p / "manifest.json", p.parent / "manifest.json"):
        if m.is_file():
            epochs = json.loads(m.read_text()).get("epochs")
            if isinstance(epochs, int) and not isinstance(epochs, bool) and epochs >= 1:
                return epochs
    return 1


def prompt_of(doc: Doc) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """(prompt, answer): the document before its last answer, and that answer without its id 2.

    The prompt, for the decoder, is every turn but the last answer, then that turn's id 1 and the
    assistant's header.
    """
    if not doc.mask.any():
        raise DataMismatchError("a document with no answer token")
    return last_answer(doc.ids, np.concatenate([[0.0], doc.mask]))


class ChatSource:
    """Whole documents packed in order into rows of `rows` x `seq_len`, mask included.

    Several short conversations share a row, each its own segment; the rest of a row is id 0
    with mask 0. A document is never cut, since that would cut an answer and its id 2: one longer
    than a row is counted in `dropped` and skipped. Epoch e reads the documents in the order of a
    Philox permutation keyed by `derive(data_seed, "chat", e)`. The cursor (epoch, position) is
    the whole state, so a resumed run reads the same rows. A run that reads past its last epoch
    stops with `ExhaustedError`.
    """

    def __init__(
        self,
        docs: Sequence[Doc],
        rows: int,
        seq_len: int,
        *,
        data_seed: int,
        epochs: int = 1,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.rows, self.seq_len = rows, seq_len
        self.data_seed, self.epochs = data_seed, epochs
        self.docs = [d for d in docs if len(d.ids) <= seq_len + 1]
        self.dropped = len(docs) - len(self.docs)
        if not self.docs:
            raise DataMismatchError(f"no document fits a row of {seq_len + 1} tokens")
        saved = dict(state or {})
        if saved and saved.get("seed") != seed_hex(data_seed):
            raise DataMismatchError("the saved chat cursor is another data seed's")
        self.epoch, self.pos = int(saved.get("epoch", 0)), int(saved.get("pos", 0))
        self.order: NDArray[np.int64] = self._permutation(self.epoch)

    def _permutation(self, epoch: int) -> NDArray[np.int64]:
        philox = np.random.Philox(key=derive(self.data_seed, "chat", epoch))
        return np.random.Generator(philox).permutation(len(self.docs))

    def rows_available(self) -> int:
        """Rows every epoch together fills from a fresh start, counted on lengths alone.

        A row is emitted only once the next document does not fit it, so the last part-filled
        row is not counted: the source runs out while building it.
        """
        n, rows, o = self.seq_len + 1, 0, 0
        for e in range(self.epochs):
            for i in self._permutation(e):
                k = len(self.docs[int(i)].ids)
                if o and o + k > n:
                    rows, o = rows + 1, 0
                o += k
        return rows

    def _take(self) -> Doc:
        if self.pos == len(self.order):
            if self.epoch + 1 >= self.epochs:
                raise ExhaustedError(f"the chat data ran out after {self.epochs} epoch(s)")
            self.epoch, self.pos = self.epoch + 1, 0
            self.order = self._permutation(self.epoch)
        return self.docs[int(self.order[self.pos])]

    def _row(self) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        n = self.seq_len + 1
        ids = np.zeros(n, np.int32)
        mask = np.zeros(n - 1, np.float32)
        o = 0
        while True:
            d = self._take()
            if o and o + len(d.ids) > n:
                return ids, mask
            ids[o : o + len(d.ids)] = d.ids
            mask[o : o + len(d.mask)] = d.mask
            o += len(d.ids)
            self.pos += 1

    def next(self) -> StepRows:
        built = [self._row() for _ in range(self.rows)]
        full = np.stack([r[0] for r in built])
        inputs = full[:, :-1]
        mask = np.stack([r[1] for r in built])
        return StepRows(inputs, full[:, 1:], segment_ids(inputs), mask)

    def state(self) -> dict[str, Any]:
        """The cursor a checkpoint records."""
        return {
            "kind": "chat",
            "epoch": self.epoch,
            "pos": self.pos,
            "seed": seed_hex(self.data_seed),
        }


def pack_rows(docs: Sequence[Doc], seq_len: int) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Every document that fits, in order: ids [R, seq_len + 1] and masks [R, seq_len].

    This is the held-out form: no shuffle, and the last part-filled row is kept.
    """
    n = seq_len + 1
    rows: list[list[Doc]] = [[]]
    for d in docs:
        if len(d.ids) > n:
            continue
        if sum(len(x.ids) for x in rows[-1]) + len(d.ids) > n:
            rows.append([])
        rows[-1].append(d)
    rows = [r for r in rows if r]
    ids = np.zeros((len(rows), n), np.int32)
    mask = np.zeros((len(rows), n - 1), np.float32)
    for i, row in enumerate(rows):
        o = 0
        for d in row:
            ids[i, o : o + len(d.ids)] = d.ids
            mask[i, o : o + len(d.mask)] = d.mask
            o += len(d.ids)
    return ids, mask
