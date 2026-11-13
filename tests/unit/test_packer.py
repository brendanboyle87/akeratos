"""The best-fit packer in `akeratos/data/packer.py`.

Rows start at documents and segments change only at separators; the row filling equals
nanochat's best fit; the crop share stays within the planning figure; a saved state resumes only
the same seed, buffer, documents and batch shape; one epoch stops and two wrap; `epoch_rows`
counts an epoch's rows from the lengths, where the first rows crop far less than the rest; the
document mask equals per-document causal attention; and the packer imports no MLX.
"""

import math
import subprocess
import sys

import numpy as np
import pytest
from numpy.typing import NDArray

from akeratos.data.packer import (
    ExhaustedError,
    Packer,
    PackerState,
    epoch_rows,
    segment_ids,
)
from akeratos.errors import DataMismatchError
from akeratos.rng import derive
from tests.support.packdocs import TokenDocs, V, document_mask, store, zipf_docs


@pytest.fixture(scope="module")
def docs() -> TokenDocs:
    return store(zipf_docs(0, 3000))


def crop_share(state: PackerState) -> float:
    """The share of the tokens read so far that cropping discarded."""
    return 1 - state["used"] / state["read"] if state["read"] else 0.0


def nanochat_rows(docs: TokenDocs, order: NDArray[np.int64], cap: int, size: int, n_rows: int):
    """nanochat's best-fit row filling, transcribed with a single-document refill."""
    buf: list[int] = []
    it = iter(order)
    lengths = docs.lengths
    rows = []
    for _ in range(n_rows):
        pos, row = 0, []
        while pos < cap:
            while len(buf) < size:
                buf.append(int(next(it)))
            remaining = cap - pos
            best_idx, best_len = -1, 0
            for i, d in enumerate(buf):
                if best_len < lengths[d] <= remaining:
                    best_idx, best_len = i, lengths[d]
            if best_idx >= 0:
                d = buf.pop(best_idx)
                row.append(docs.read(d, int(lengths[d])))
                pos += int(lengths[d])
            else:
                j = min(range(len(buf)), key=lambda i: lengths[buf[i]])
                row.append(docs.read(buf.pop(j), remaining))
                pos += remaining
        rows.append(np.concatenate(row))
    return np.stack(rows)


def test_rows_start_at_documents_and_segments_change_only_at_separators(docs: TokenDocs) -> None:
    packer = Packer(docs, 4, 64, vocab=V, data_seed=1, buffer=50)
    for _ in range(40):
        batch = next(packer)
        for field in (batch.inputs, batch.targets, batch.segment_ids):
            assert field.shape == (4, 64)
            assert field.dtype == np.int32
        assert (batch.inputs[:, 0] == 0).all()
        assert np.array_equal(batch.inputs[:, 1:], batch.targets[:, :-1])
        steps = np.diff(batch.segment_ids, axis=1)
        assert set(np.unique(steps).tolist()) <= {0, 1}
        assert np.array_equal(steps == 1, batch.inputs[:, 1:] == 0)
        assert (batch.segment_ids[:, 0] == 0).all()


def test_best_fit_equals_nanochats_row_filling(docs: TokenDocs) -> None:
    packer = Packer(docs, 4, 64, vocab=V, data_seed=5, buffer=50)
    got = np.concatenate(
        [
            np.concatenate([b.inputs, b.targets[:, -1:]], axis=1)
            for b in (next(packer) for _ in range(40))
        ]
    )
    philox = np.random.Philox(key=derive(5, "data", 0))
    order = np.random.Generator(philox).permutation(len(docs.lengths))
    assert np.array_equal(got, nanochat_rows(docs, order, 65, 50, 160))
    assert 0 < crop_share(packer.state()) < 1


@pytest.mark.parametrize(("seed", "buffer"), [(8, 40), (9, 41)])
def test_a_state_from_another_seed_or_buffer_is_refused(
    docs: TokenDocs, seed: int, buffer: int
) -> None:
    saved = Packer(docs, 2, 64, vocab=V, data_seed=9, buffer=40).state()
    with pytest.raises(ValueError, match="other documents"):
        Packer(docs, 2, 64, vocab=V, data_seed=seed, buffer=buffer, state=saved)


def test_a_state_from_other_documents_is_refused(docs: TokenDocs) -> None:
    saved = Packer(docs, 2, 64, vocab=V).state()
    with pytest.raises(ValueError, match="other documents"):
        Packer(store(zipf_docs(1, 3000)), 2, 64, vocab=V, state=saved)


class Synthetic:
    """Documents of given lengths whose tokens are made on read: `[0, 17, 17, ...]`."""

    def __init__(self, lengths: NDArray[np.int64]) -> None:
        self.lengths = lengths

    def read(self, index: int, count: int) -> NDArray[np.int64]:
        out = np.full(min(count, int(self.lengths[index])), 17, dtype=np.int64)
        out[0] = 0
        return out


SPEC_CROP = 0.346  # nanochat's share of tokens cropped at T 2,048
MEAN = 1028  # estimated GPT-2 tokens per FineWeb-Edu document


@pytest.mark.parametrize("sigma", [0.8, 1.0, 1.2])
def test_the_crop_share_at_t_2048_is_within_the_planning_figure(sigma: float) -> None:
    """The crop share on synthetic lognormal lengths of mean 1,028 ❓.

    20,000 rows at T 2,048 with a 1,000-document buffer. The share stays at or below 34.6%, and
    above the floor no packer can beat: the tails of documents longer than a row.
    """
    rng = np.random.default_rng(0)
    draw = rng.lognormal(math.log(MEAN) - sigma * sigma / 2, sigma, 80_000)
    lengths = 1 + np.maximum(1, np.round(draw)).astype(np.int64)
    packer = Packer(Synthetic(lengths), 100, 2048, vocab=V, buffer=1000)
    for _ in range(200):
        next(packer)
    share = crop_share(packer.state())
    floor = np.maximum(lengths - 2049, 0).sum() / lengths.sum()
    assert floor < share <= SPEC_CROP


def take(packer: Packer, n: int) -> int:
    """Read up to n batches; the number read before `ExhaustedError`."""
    for i in range(n):
        try:
            next(packer)
        except ExhaustedError:
            return i
    return n


def test_one_epoch_stops_and_two_wrap_with_a_warning() -> None:
    small = store(zipf_docs(1, 300, long_every=0))
    count = take(Packer(small, 2, 64, vocab=V, buffer=20), 10_000)
    assert 0 < count < 10_000
    once = Packer(small, 2, 64, vocab=V, buffer=20)
    take(once, count)
    with pytest.raises(ExhaustedError, match="1 time"):
        next(once)
    twice = Packer(small, 2, 64, vocab=V, buffer=20, epochs=2)
    take(twice, count)
    with pytest.warns(UserWarning, match="data epoch 1"):
        assert take(twice, 3) == 3
    assert twice.state()["epoch"] == 1


@pytest.mark.parametrize(("seed", "buffer", "seq_len"), [(0, 50, 64), (3, 8, 64), (7, 1000, 48)])
def test_epoch_rows_counts_the_rows_one_epoch_gives(
    docs: TokenDocs, seed: int, buffer: int, seq_len: int
) -> None:
    packer = Packer(docs, 1, seq_len, vocab=V, data_seed=seed, buffer=buffer)
    rows = epoch_rows(docs.lengths, seq_len, data_seed=seed, buffer=buffer)
    assert rows == take(packer, 10**9) > 0


def test_a_buffer_the_documents_cannot_fill_gives_no_row(docs: TokenDocs) -> None:
    buffer = len(docs.lengths) + 1
    assert epoch_rows(docs.lengths, 64, buffer=buffer) == 0
    assert take(Packer(docs, 1, 64, vocab=V, buffer=buffer), 1) == 0


def test_the_first_rows_crop_far_less_than_the_epoch() -> None:
    """Why the count is of a whole epoch. Lognormal lengths of mean 1,028 at T 2,048.

    The buffer holds back the documents longer than a row and crops them later, so a crop
    measured on the first thousand rows promises at least a quarter more rows than the epoch
    gives.
    """
    rng = np.random.default_rng(0)
    draw = rng.lognormal(math.log(MEAN) - 0.5, 1.0, 40_000)
    lengths = 1 + np.maximum(1, np.round(draw)).astype(np.int64)
    packer = Packer(Synthetic(lengths), 1, 2048, vocab=V, buffer=1000)
    assert take(packer, 1000) == 1000
    early = crop_share(packer.state())
    rows = epoch_rows(lengths, 2048, buffer=1000)
    assert rows == 1000 + take(packer, 10**9)
    whole = crop_share(packer.state())
    assert early < 0.1
    assert whole > 0.25
    assert rows < 0.8 * lengths.sum() * (1 - early) / 2049


def with_token(where: int, value: int) -> TokenDocs:
    """200 documents whose first, `[0] + [40] * 10`, has `value` at position `where`."""
    docs = zipf_docs(2, 200, long_every=0)
    docs[0] = np.full(10, 40, dtype=np.int64)
    base = store(docs)
    tokens = np.asarray(base.read(0, 11))
    tokens[where] = value
    flat = np.concatenate([tokens, *(base.read(i, int(base.lengths[i])) for i in range(1, 200))])
    return TokenDocs(flat, np.concatenate([[0], np.cumsum(base.lengths)]))


@pytest.mark.parametrize(("where", "value"), [(1, 0), (1, 5), (2, V), (3, -1)])
def test_a_document_that_breaks_the_format_stops_the_packer(where: int, value: int) -> None:
    packer = Packer(with_token(where, value), 4, 64, vocab=V, buffer=20)
    with pytest.raises(DataMismatchError, match="breaks the format"):
        take(packer, 100)


def test_an_allowed_reserved_id_after_the_separator_packs() -> None:
    """An allowed reserved id, the safety data's tag, may follow the separator."""
    docs = with_token(1, 5)
    packer = Packer(docs, 4, 64, vocab=V, buffer=20, allow_reserved=frozenset({5}))
    seen = []
    for _ in range(100):
        try:
            seen.append(next(packer).inputs)
        except ExhaustedError:
            break
    rows = np.concatenate(seen)
    starts = np.flatnonzero(rows.ravel() == 5)
    assert len(starts) == 1
    assert rows.ravel()[starts[0] - 1] == 0
    other = Packer(docs, 4, 64, vocab=V, buffer=20, allow_reserved=frozenset({6}))
    with pytest.raises(DataMismatchError, match="breaks the format"):
        take(other, 100)


@pytest.mark.parametrize("ids", [{0}, {17}, {1, 16, 17}])
def test_allow_reserved_names_only_ids_1_to_16(docs: TokenDocs, ids: set[int]) -> None:
    with pytest.raises(ValueError, match="allow_reserved"):
        Packer(docs, 2, 64, vocab=V, allow_reserved=frozenset(ids))


def test_a_state_for_another_batch_shape_is_refused(docs: TokenDocs) -> None:
    saved = Packer(docs, 2, 64, vocab=V).state()
    with pytest.raises(ValueError, match="batch shape"):
        Packer(docs, 2, 32, vocab=V, state=saved)
    with pytest.raises(ValueError, match="batch shape"):
        Packer(docs, 3, 64, vocab=V, state=saved)


@pytest.mark.parametrize("offsets", [[], [0], [1, 5, 10], [0, 5, 9], [0, 5, 5, 10]])
def test_bad_offsets_are_refused(offsets: list[int]) -> None:
    with pytest.raises(ValueError, match="offsets"):
        TokenDocs(np.zeros(10, dtype=np.uint16), np.asarray(offsets, dtype=np.int64))


def test_the_document_mask_equals_per_document_causal_attention() -> None:
    ids = np.array([[0, 30, 31, 0, 32, 0, 33, 34], [0, 30, 31, 32, 33, 34, 35, 36]])
    segments = segment_ids(ids)
    assert segments.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2], [0] * 8]
    mask = document_mask(segments)
    for b in range(2):
        for i in range(8):
            for j in range(8):
                want = j <= i and segments[b, i] == segments[b, j]
                assert mask[b, i, j] == want
    assert np.array_equal(mask[1], np.tril(np.ones((8, 8), dtype=bool)))


def test_the_packer_imports_no_mlx() -> None:
    code = "import sys, akeratos.data.packer; print('mlx' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=30
    )
    assert out.stdout.strip() == "False"
