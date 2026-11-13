"""The named configurations in `akeratos/configs.py`.

Billet's and its twin's configurations equal the `config.json` that ships with each released
run, the starter is Billet's configuration at the 10M model's size, and each run's launch rows
fill one optimiser update.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from akeratos.configs import (
    BILLET,
    BILLET_TWIN,
    LAUNCH,
    MODEL_10M,
    STARTER,
    STARTER_LAUNCH,
    from_keys,
)
from akeratos.train.recipe import steps_for

RELEASED = Path(__file__).resolve().parents[1] / "fixtures" / "released"


@pytest.mark.parametrize(("named", "arm"), [(BILLET, "billet"), (BILLET_TWIN, "billet-twin")])
def test_a_named_config_equals_its_released_config_json(named: dict[str, Any], arm: str) -> None:
    saved = json.loads((RELEASED / arm / "config.json").read_text())["config"]
    assert named == saved
    assert named["model"]["attn_path"] == "banded"


def test_the_launch_rows_fill_one_update() -> None:
    tokens = LAUNCH["rows"] * LAUNCH["seq_len"] * BILLET["accum"]
    assert tokens == BILLET["optim"]["tokens_per_update"]


def test_the_starter_launch_rows_fill_one_update() -> None:
    tokens = STARTER_LAUNCH["rows"] * STARTER_LAUNCH["seq_len"] * STARTER["accum"]
    assert tokens == STARTER["optim"]["tokens_per_update"] == 8192
    assert steps_for(STARTER["optim"]["tokens"], tokens) == 9_156


def test_the_starter_is_billet_at_the_10m_size() -> None:
    assert list(STARTER) == [k for k in BILLET if k not in ("init_seed", "data_seed")]
    shape = ("d_model", "n_layers", "n_heads", "head_dim", "ffn")
    assert STARTER["model"] == BILLET["model"] | {k: MODEL_10M[k] for k in shape}
    assert from_keys(STARTER["model"]) == from_keys(MODEL_10M | {"attn_path": "banded"})
    changed = {"lr_peak": 0.003, "tokens": 7.5e7, "tokens_per_update": 8192}
    assert STARTER["optim"] == BILLET["optim"] | changed
    assert STARTER["recipe"] == BILLET["recipe"]
    assert STARTER["table"] == BILLET["table"] | {"params": 1e9, "width": 256}
    same = ("seed", "compute_dtype", "mtp")
    assert [STARTER[k] for k in same] == [BILLET[k] for k in same]
    assert STARTER["accum"] == 2


# The fifteen model keys the old presets table gave its 150M row.
MODEL_150M_KEYS: dict[str, Any] = {
    "d_model": 768,
    "n_layers": 18,
    "n_heads": 6,
    "head_dim": 128,
    "n_kv_heads": 2,
    "ffn": 3456,
    "ffn_kind": "relu2",
    "vocab_size": 49152,
    "window": 512,
    "global_every": 5,
    "rope_theta_local": 10000.0,
    "rope_theta_global": 1000000.0,
    "rms_norm_eps": 1e-05,
    "value_embed": "odd",
    "ve_gate_width": 12,
}


def test_the_10m_model_builds() -> None:
    cfg = from_keys(MODEL_10M)
    assert (cfg.d_model, cfg.n_layers, cfg.ffn) == (256, 8, 1152)


def test_billet_is_the_150m_model_plus_its_switches() -> None:
    switches = {k: v for k, v in BILLET["model"].items() if k not in MODEL_150M_KEYS}
    assert len(switches) == len(BILLET["model"]) - len(MODEL_150M_KEYS)
    assert from_keys(BILLET["model"]) == from_keys(MODEL_150M_KEYS | switches)
