"""A one-shard data store split on disk, for the training entry point's tests.

The repro suite's child-process kills (`tests/repro/test_resume.py`) and its in-process resumes
(`tests/repro/test_ckpt_resume.py`) read the same split, so both write it with this helper.
"""

import json

import numpy as np


def write_split(path, n=200):
    """A one-shard split in the layout `akeratos/data/store.py` reads."""
    rng = np.random.default_rng(0)
    docs = [rng.integers(17, 512, size=int(rng.integers(5, 50))) for _ in range(n)]
    path.mkdir(parents=True)
    body = np.concatenate([np.concatenate([[0], d]) for d in docs]).astype("<u2")
    idx = np.concatenate([[0], np.cumsum([len(d) + 1 for d in docs])]).astype("<u8")
    body.tofile(path / "000000.bin")
    idx.tofile(path / "000000.idx")
    manifest = {
        "dtype": "uint16",
        "eos": 0,
        "vocab": 512,
        "shards": ["000000.bin"],
        "tokens": [len(body)],
        "documents": [n],
        "sep_position": "start",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path
