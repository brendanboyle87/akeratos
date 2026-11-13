"""A resumed packer reads the same batches.

A packer resumed from a state saved as JSON reads the batches an unbroken packer reads. The
documents are the packer suite's (`tests/support/packdocs.py`).
"""

import json

import numpy as np
import pytest

from akeratos.data.packer import (
    Packer,
)
from tests.support.packdocs import TokenDocs, V, store, zipf_docs


@pytest.fixture(scope="module")
def docs() -> TokenDocs:
    return store(zipf_docs(0, 3000))


def test_a_resumed_packer_reads_the_same_batches(docs: TokenDocs) -> None:
    """Resumed from a state saved as JSON, it reads the rest and ends in the same state."""
    first = Packer(docs, 2, 64, vocab=V, data_seed=9, buffer=40)
    batches, saved = [], None
    for i in range(7):
        if i == 3:
            saved = json.loads(json.dumps(first.state()))
        batches.append(next(first))
    assert saved is not None
    assert saved["buffer"]
    resumed = Packer(docs, 2, 64, vocab=V, data_seed=9, buffer=40, state=saved)
    for want in batches[3:]:
        got = next(resumed)
        assert np.array_equal(got.inputs, want.inputs)
        assert np.array_equal(got.targets, want.targets)
    assert resumed.state() == first.state()
