"""The released runs' saved files still load.

`tests/fixtures/released/` holds the `config.json` that ships with Billet and with its twin, and
the `state.json` of Billet's final checkpoint. Each test reads them through the code a user's
command runs: the export loader, the checkpoint loader and the configuration classes.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from akeratos import ckpt, export
from akeratos.configs import from_keys
from akeratos.table.config import TableConfig
from akeratos.train.recipe import OptimConfig, RecipeConfig

RELEASED = Path(__file__).resolve().parents[1] / "fixtures" / "released"
STEP = 24_415
ARMS = {"billet": True, "billet-twin": False}  # the arm, and whether it has the table


def plain(value: object) -> Any:
    """`value` as JSON holds it: a tuple becomes a list."""
    return json.loads(json.dumps(value))


def saved(arm: str) -> dict[str, Any]:
    return json.loads((RELEASED / arm / export.CONFIG).read_text())


def an_export(arm: str, folder: Path) -> Path:
    """An export folder with the arm's real `config.json` and empty stand-ins for its weights."""
    folder.mkdir()
    shutil.copy(RELEASED / arm / export.CONFIG, folder / export.CONFIG)
    names = [export.CONFIG, *export.WEIGHTS.values()]
    for name in names[1:]:
        (folder / name).write_bytes(b"")
    sums = {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names}
    (folder / export.SUMS).write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    return folder


@pytest.mark.parametrize(("arm", "table"), ARMS.items())
def test_a_released_config_json_loads(arm: str, table: bool, tmp_path: Path) -> None:
    c = export.checkpoint(an_export(arm, tmp_path / arm))
    run = c.state["config"]
    assert c.step == STEP
    cfg = from_keys(run["model"])
    assert plain(asdict(cfg)) == run["model"]
    assert cfg.attn_path == "banded"
    assert cfg.table is table
    assert asdict(OptimConfig(**run["optim"]).validate()) == run["optim"]
    assert asdict(RecipeConfig(**run["recipe"]).validate()) == run["recipe"]
    if table:
        keys = run["table"] | {"orders": tuple(run["table"]["orders"])}
        assert plain(asdict(TableConfig(**keys).validate())) == run["table"]
    else:
        assert run["table"] is None


def test_the_twin_differs_from_billet_only_in_the_table() -> None:
    billet, twin = saved("billet"), saved("billet-twin")
    model = billet["config"]["model"] | {"table": False}
    assert twin == billet | {"config": billet["config"] | {"model": model, "table": None}}


def test_the_final_state_json_loads_and_holds_the_released_config(tmp_path: Path) -> None:
    doc = json.loads((RELEASED / "billet" / ckpt.STATE).read_text())
    folder = tmp_path / f"{ckpt.PREFIX}{STEP:08d}"
    folder.mkdir()
    # The weight files are 4 GB and are not in the repository, so the copy lists none.
    (folder / ckpt.STATE).write_text(json.dumps(doc | {"files": {}}))
    c = ckpt.load(folder)
    assert c.step == STEP
    assert c.state["config"] == saved("billet")["config"]
    assert set(c.state) == {"config", "log", "skipped", "source", "table_event"}
    assert c.state["source"]["kind"] == "packed"
