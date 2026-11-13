"""The loss's host-side inputs: the count of valid targets and its bounds.

The host count must equal the device sum of the weights. The refusal of negative and
out-of-range targets (DataMismatchError) is `test_out_of_range_targets_are_refused`, in
tests/unit/test_model_head.py.
"""

import mlx.core as mx
import numpy as np
import pytest

from akeratos.errors import DataMismatchError
from akeratos.train.loop import N_VALID_MAX, n_valid, prepare_targets
from tests.unit import model_ref as ref


def test_n_valid_matches_weights() -> None:
    """The host count equals the device sum of the weights over the micro-batches."""
    cfg = ref.tiny()
    ids, tg = ref.batch(1, b=4)
    tg[0, 3] = 9  # a masked-id target: weight 0
    lmask = np.ones(ids.shape, np.float32)
    lmask[1, :10] = 0
    parts = [prepare_targets(tg[r : r + 2], lmask[r : r + 2], cfg) for r in (0, 2)]
    count, n = n_valid(w for _, w in parts)
    device = sum(float(mx.array(w).sum()) for _, w in parts)
    assert count == device == ids.size - 11
    assert n.dtype == mx.float32
    assert n.shape == ()
    assert float(n) == count


@pytest.mark.parametrize("size", [0, N_VALID_MAX + 1])
def test_n_valid_refuses_an_empty_or_inexact_count(size: int) -> None:
    """No valid target is a data error; above 2**24 FP32 no longer holds the count exactly."""
    with pytest.raises(DataMismatchError, match="N_valid"):
        n_valid([np.ones(size, np.float32)] if size else [np.zeros(8, np.float32)])
