"""`akeratos/cli.py`: `bake`'s help, `bake train`, `bake evaluate`, and `run_trainer`, the guarded
trainer's entry.

`bake train`'s flags are checked against the named configurations without building a model:
the trainer parses them and resolves them, and the result must be the released record, or for
the starter Billet's configuration at the 10M model's size.

`run_trainer` installs macops' MLX cache cap as `entry.main`'s setup hook.
The fakes check the peeling and the hand-over in process; one test runs the guard's own child
command for a tiny CPU-sized run, without caffeinate, through the shared `children` helper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from akeratos import cli, macops
from akeratos.configs import BILLET_TWIN, LAUNCH, MODEL_10M, STARTER_LAUNCH, from_keys
from akeratos.eval import heldout
from akeratos.table.config import TableConfig
from akeratos.train import entry
from akeratos.train.loop import Run
from akeratos.train.recipe import OptimConfig, RecipeConfig, steps_for
from akeratos.train.saver import Checkpointer
from tests.unit.test_heldout import TINY as TINY_MODEL
from tests.unit.test_heldout import T, docs_for, save, write_store

ROOT = Path(__file__).resolve().parents[2]
# `bake --help` at 100 columns. `run_trainer` is not a subcommand and must not change it.
BAKE_HELP = (
    "usage: bake [-h] [--version] COMMAND ...\n\n"
    "Akeratos: an open lab for training language models from scratch on Apple silicon Macs with "
    "MLX.\n\n"
    "positional arguments:\n"
    "  COMMAND\n"
    "    train     train a named configuration: billet, billet-twin or starter\n"
    "    evaluate  bits per byte on the held-out split\n"
    "    generate  greedy text from an export\n"
    "    export    write a checkpoint's release folder\n"
    "    speedrun  train speedrun/train.py's model for 300 s and score it\n"
    "    bench     what the n-gram table costs per training step on this Mac\n\n"
    "options:\n"
    "  -h, --help  show this help message and exit\n"
    "  --version   show program's version number and exit\n"
)
COMMANDS = ["train", "evaluate", "generate", "export", "speedrun", "bench"]
TINY = [
    "table=false",
    "d_model=64",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    "vocab_size=512",
    "window=16",
    "param.kind=sp",
]


def trainer_args(out: Path) -> list[str]:
    args = ["--data", "synthetic", "--tokens", str(4 * 2 * 32), "--rows", "2", "--seq-len", "32"]
    args += ["--out", str(out), "--compute-dtype", "float32", "--print-every", "0"]
    args += [f"--model={k}" for k in TINY]
    return [*args, "--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]


def test_bake_help_is_unchanged(monkeypatch, capsys) -> None:
    monkeypatch.setenv("COLUMNS", "100")
    assert cli.main([]) == 0
    assert capsys.readouterr().out == BAKE_HELP
    with pytest.raises(SystemExit) as done:
        cli.main(["--help"])
    assert done.value.code == 0
    assert capsys.readouterr().out == BAKE_HELP


def test_run_trainer_peels_the_cache_cap_and_passes_the_rest(monkeypatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    seen: list[tuple[list[str], object, object]] = []
    caps: list[int | None] = []

    def fake_main(argv, run_cls, saver_cls, clock=None, *, setup=None):
        assert clock is None
        assert setup is not None
        seen.append((list(argv), run_cls, saver_cls))
        setup()
        return entry.EXIT_RESUMABLE

    monkeypatch.setattr(entry, "main", fake_main)
    monkeypatch.setattr(macops, "apply_cache_limit", caps.append)
    rest = ["--data", "synthetic", "--out", "runs/a", "--cache", "kept"]  # no prefix matching
    for argv in (
        ["--cache-limit-gb", "3.81", *rest],
        [*rest[:2], "--cache-limit-gb=0.5", *rest[2:]],
        rest,
    ):
        assert cli.run_trainer(argv) == entry.EXIT_RESUMABLE  # the trainer's code, unchanged
    assert [s[0] for s in seen] == [rest] * 3
    assert all(s[1] is Run and s[2] is Checkpointer for s in seen)
    assert caps == [int(3.81 * 2**30), 2**29, None]


def test_the_guards_child_command_runs_the_trainer_with_the_cap(tmp_path, children) -> None:
    argv = macops.child_argv(trainer_args(tmp_path / "out"), 0.5)
    argv = argv[argv.index(sys.executable) :]  # caffeinate is the guard's, not this test's
    proc = children.run(argv, cwd=ROOT, timeout=120)
    assert proc.returncode == entry.EXIT_OK, proc.stderr
    assert "macops: MLX cache limit 0.50 GB (536,870,912 bytes)" in proc.stderr
    assert (tmp_path / "out" / "metrics.jsonl").is_file()


def commands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """The parser's subcommands by name, in the order it adds them."""
    groups = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return dict(cast("dict[str, argparse.ArgumentParser]", groups[0].choices))


def flags_of(parser: argparse.ArgumentParser) -> list[tuple[Any, ...]]:
    """Each flag's spelling, destination, default, choices, whether it is required, and help."""
    return [
        (a.option_strings, a.dest, a.default, a.choices, a.required, a.help)
        for a in parser._actions
    ]


def record(c: entry.Resolved) -> dict[str, Any]:
    """What `Run.config` records for a resolved configuration, as `config.json` holds it."""
    table = None if c.table is None else SimpleNamespace(cfg=c.table)
    run = SimpleNamespace(
        cfg=c.cfg,
        o=c.o,
        r=c.r,
        seeds=c.seeds,
        compute_dtype=c.compute_dtype,
        mtp=c.mtp,
        accum=c.accum,
        table=table,
    )
    return json.loads(json.dumps(Run.config(cast(Run, run))))


@pytest.mark.parametrize("name", ["billet", "billet-twin"])
def test_train_flags_resolve_to_the_released_configuration(name: str) -> None:
    run = cli.RUNS[name]
    a = entry._parser().parse_args(cli.train_flags(run, "DIR", "DIR"))
    c = entry.resolve(a)
    got = record(c)
    assert list(got) == list(run)
    for key, want in run.items():
        if isinstance(want, dict):
            for field, value in cast("dict[str, Any]", want).items():
                assert got[key][field] == value, f"{key}.{field}"
        assert got[key] == want, key
    assert c.names_path
    assert c.cfg.attn_path == "banded"
    assert (a.init_seed, a.data_seed) == (None, None)  # both derived from --seed
    assert (got["init_seed"], got["data_seed"]) == (run["init_seed"], run["data_seed"])
    launch = (a.rows, a.seq_len, a.ckpt_every, a.keep, a.pin_decay)
    assert launch == tuple(
        LAUNCH[k] for k in ("rows", "seq_len", "ckpt_every", "keep", "pin_decay")
    )
    assert (a.data, a.out, a.steps, a.minutes) == ("DIR", Path("DIR"), None, None)


def test_bake_train_runs_the_trainer_with_the_cache_cap_or_under_the_guard(monkeypatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(cli, "run_trainer", lambda argv: calls.append(("trainer", argv)) or 3)
    monkeypatch.setattr(macops, "main", lambda argv: calls.append(("guard", argv)) or 75)
    args = ["billet-twin", "--data", "d", "--out", "o", "--steps", "20", "--print-every", "5"]
    assert cli.main(["train", *args]) == 3
    assert cli.main(["train", *args, "--guard"]) == 75
    flags = cli.train_flags(BILLET_TWIN, "d", "o", steps=20, print_every=5)
    assert flags[-4:] == ["--steps", "20", "--print-every", "5"]
    cap = ["--cache-limit-gb", "3.81"]
    assert calls == [("trainer", [*cap, *flags]), ("guard", [*cap, "--", *flags])]


def test_the_starter_resolves_to_billets_configuration_at_the_10m_size() -> None:
    a = entry._parser().parse_args(
        cli.train_flags(cli.RUNS["starter"], "DIR", "DIR", launch=cli.LAUNCHES["starter"])
    )
    c = entry.resolve(a)
    assert c.cfg == from_keys(MODEL_10M | {"attn_path": "banded", "table": True})
    assert c.names_path
    assert c.o == OptimConfig(tokens=7.5e7, tokens_per_update=8192)
    assert c.r == RecipeConfig()
    assert c.table == TableConfig(params=1e9, width=256, seed=0)
    assert (a.rows, a.seq_len, c.accum, c.compute_dtype, c.mtp) == (2, 2048, 2, "bfloat16", False)
    assert (a.ckpt_every, a.keep, a.pin_decay) == (2000, 2, False)
    assert (a.init_seed, a.data_seed) == (None, None)  # both derived from --seed
    assert steps_for(c.o.tokens, c.o.tokens_per_update) == 9_156


@pytest.mark.parametrize(("name", "cap"), [("starter", "2.0"), ("billet", "3.81")])
def test_bake_train_caps_the_cache_at_the_runs_own_limit(monkeypatch, name: str, cap: str) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "run_trainer", lambda argv: calls.append(argv) or 0)
    assert cli.main(["train", name, "--data", "d", "--out", "o"]) == 0
    flags = cli.train_flags(cli.RUNS[name], "d", "o", launch=cli.LAUNCHES[name])
    assert calls == [["--cache-limit-gb", cap, *flags]]


def test_billet_and_its_twin_keep_the_released_launch() -> None:
    assert cli.LAUNCHES["billet"] is LAUNCH
    assert cli.LAUNCHES["billet-twin"] is LAUNCH
    assert cli.LAUNCHES["starter"] is STARTER_LAUNCH
    for name in ("billet", "billet-twin"):
        run = cli.RUNS[name]
        assert cli.train_flags(run, "d", "o", launch=LAUNCH) == cli.train_flags(run, "d", "o")


@pytest.mark.parametrize("name", ["billet", "billet-twin", "starter"])
def test_seed_changes_the_trainers_seed_and_nothing_else(monkeypatch, name: str) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "run_trainer", lambda argv: calls.append(argv) or 0)
    args = ["train", name, "--data", "d", "--out", "o"]
    assert cli.main(args) == 0
    assert cli.main([*args, "--seed", "7"]) == 0
    before, after = calls
    at = before.index("--seed") + 1
    assert before[at] == str(int(cli.RUNS[name]["seed"], 16))
    assert after[at] == "7"
    assert before[:at] + before[at + 1 :] == after[:at] + after[at + 1 :]
    assert entry._parser().parse_args(after[2:]).seed == 7


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "billet-mini", "--data", "d", "--out", "o"],
        ["train", "starter", "--data", "d", "--out", "o", "--seed", "seven"],
        ["train", "starter", "--data", "d", "--out", "o", "--seed", "1.5"],
    ],
)
def test_bake_train_refuses_an_unknown_name_and_a_seed_that_is_not_an_integer(
    capsys, argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as done:
        cli.build_parser().parse_args(argv)
    assert done.value.code == 2
    assert "invalid" in capsys.readouterr().err


def test_bake_evaluate_takes_the_readouts_flags() -> None:
    assert flags_of(commands(cli.build_parser())["evaluate"]) == flags_of(heldout._parser())


def test_bake_evaluate_prints_what_the_readout_prints(tmp_path: Path, capsys) -> None:
    split = write_store(tmp_path / "store", docs_for(6, 20))
    save(tmp_path / "run", TINY_MODEL, ema=True)
    data = ["--data", str(split), "--seq-len", str(T)]
    scored = ["--ckpt", str(tmp_path / "run"), *data]
    refused = ["--ckpt", str(tmp_path / "none"), *data]
    printed = {}
    for args, code in ((scored, 0), (refused, heldout.EXIT_CONFIG)):
        assert heldout.main(args) == code
        printed[code] = capsys.readouterr()
        assert cli.main(["evaluate", *args]) == code
        assert capsys.readouterr() == printed[code]
    assert json.loads(printed[0].out)["raw"]["bpb"] > 0
    assert printed[heldout.EXIT_CONFIG].err.startswith("configuration error: ")


def test_bake_has_these_commands_in_this_order_and_names_no_internal_module(
    monkeypatch, capsys
) -> None:
    assert list(commands(cli.build_parser())) == COMMANDS
    monkeypatch.setenv("COLUMNS", "100")
    for argv in (["--help"], ["train", "--help"]):
        with pytest.raises(SystemExit) as done:
            cli.main(argv)
        assert done.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("usage: bake")
        assert "introspect" not in out
        assert "entry" not in out
