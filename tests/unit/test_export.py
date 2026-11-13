"""The release folder (`akeratos/export.py`), `bake export` and `bake generate`.

The checkpoints are two tiny runs that keep an EMA, three steps each on MLX's CPU device
(`akeratos/train/entry.py`): one with the table, as Billet, and one without it, as Billet's
twin. A word-level tokenizer with the model's 512 ids stands in for SmolLM2's.
The held-out split and the multiple-choice set are `test_heldout.py`'s and `test_mc.py`'s.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from akeratos import __version__, ckpt, cli, export
from akeratos.configs import ModelConfig
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval import heldout, mc
from akeratos.infer import greedy
from akeratos.model import init_params
from akeratos.post.run import SFTRun, load_table
from akeratos.train import entry
from akeratos.train.loop import Run
from akeratos.train.saver import Checkpointer
from tests.unit.test_heldout import docs_for
from tests.unit.test_heldout import write_store as heldout_store
from tests.unit.test_mc import items_for, write_set
from tests.unit.test_mc import write_store as mc_store

mx.set_default_device(mx.cpu)

V, B, T = 512, 2, 32
MODEL = ["d_model=64", "n_layers=2", "n_heads=2", "head_dim=32", "n_kv_heads=1", "ffn=128"]
MODEL += [f"vocab_size={V}", "window=16", "param.kind=sp", "table=true"]
TABLE = [f"params={4096 * 64}", f"max_shard_elements={1024 * 64}"]  # 4 shards
WORDS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    *[f"<|reserved_{i}|>" for i in range(3, 17)],
]
WORDS += ["user", "assistant", "[UNK]"]
WORDS += [f"w{i}" for i in range(len(WORDS), V)]
USER, ASSISTANT = WORDS.index("user"), WORDS.index("assistant")
# Each file an export copies from the files given: its name there, and the keyword that names it.
GIVEN = {"tokenizer.json": "tokenizer", "LICENSE": "licence", "NOTICE": "notice"}
LOGS = ("run.json", "metrics.jsonl")  # the run directory's, shipped under logs/


@pytest.fixture(autouse=True)
def tf32_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


def train(out: Path, model: list[str], table: list[str]) -> Path:
    """The newest checkpoint of a three-step run at `out` that keeps an EMA."""
    old = os.environ.get("MLX_ENABLE_TF32")
    os.environ["MLX_ENABLE_TF32"] = "0"
    args = ["--data", "synthetic", "--tokens", str(3 * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--out", str(out), "--compute-dtype", "float32"]
    args += [f"--model={k}" for k in model] + [f"--table={k}" for k in table]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    try:
        assert entry.main([*args, "--print-every", "0"], Run, Checkpointer) == 0
    finally:
        if old is None:
            os.environ.pop("MLX_ENABLE_TF32", None)
        else:
            os.environ["MLX_ENABLE_TF32"] = old
    found = ckpt.latest(out)
    assert found is not None
    return found


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The newest checkpoint of a three-step table-on run that keeps an EMA, as Billet's."""
    return train(tmp_path_factory.mktemp("run") / "run", MODEL, TABLE)


@pytest.fixture(scope="module")
def twin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same run with the table off, as Billet's twin, which keeps an EMA too."""
    model = [k for k in MODEL if k != "table=true"] + ["table=false"]
    return train(tmp_path_factory.mktemp("twin") / "run", model, [])


def tokenizer(path: Path, words: list[str]) -> Path:
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, "[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    tok.save(str(path))
    return path


@pytest.fixture(scope="module")
def given(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The tokenizer, LICENSE and NOTICE files an export copies."""
    d = tmp_path_factory.mktemp("given")
    (d / "LICENSE").write_text("Apache License, Version 2.0\n")
    (d / "NOTICE").write_text("A fixture NOTICE\n")
    tok = tokenizer(d / "tokenizer.json", WORDS)
    return {"tokenizer": tok, "licence": d / "LICENSE", "notice": d / "NOTICE"}


def files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def same(a: dict[str, mx.array], b: dict[str, mx.array]) -> bool:
    return set(a) == set(b) and all(mx.array_equal(a[k], b[k]).item() for k in a)


def sft_like(run: Path, at: Path) -> Path:
    """A copy of `run` and its logs that say stage 1 of post-training made it, as `post.sft` does."""
    dst = at / run.name
    shutil.copytree(run, dst)
    doc = json.loads((dst / ckpt.STATE).read_text())
    config = doc["state"]["config"]
    config["model"]["masked_ids"] = [i for i in config["model"]["masked_ids"] if i not in (1, 2)]
    config["post"] = {"stage": 1, "base": f"/Users/someone/mf-runs/base/{run.name}", "base_step": 3}
    (dst / ckpt.STATE).write_text(json.dumps(doc))
    manifest = json.loads((run.parent / "run.json").read_text()) | {"config": config}
    (at / "run.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(run.parent / "metrics.jsonl", at)
    return dst


# ---- the folder and its loader ---------------------------------------------------------------


def test_an_export_loads_back_the_checkpoints_own_tensors(
    run: Path, given: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "export"
    export.export(run, out, **given)
    c = ckpt.load(run)
    got = files(out)
    want = {n: b for n, b in files(run).items() if n.startswith("table/")}
    want |= {n: (run / n).read_bytes() for n in ("dense.safetensors", "ema.safetensors")}
    want |= {n: given[k].read_bytes() for n, k in GIVEN.items()}
    want |= {f"logs/{n}": (run.parent / n).read_bytes() for n in LOGS}  # run.json names no path
    assert {n: got[n] for n in want} == want  # every shipped file is byte for byte the source
    assert set(got) - set(want) == {"config.json", "SHA256SUMS"}
    assert len([n for n in got if n.startswith("table/shard_")]) == 4
    assert "table/reads.npy" in got  # the table is copied whole, read counts included
    raw, ema = export.load(out), export.load(out, "ema")
    assert same(raw.params, c.arrays("dense"))
    assert same(ema.params, c.arrays("ema"))
    cfg = raw.cfg
    base = load_table(c, cfg)
    assert raw.table is not None
    assert base is not None
    assert raw.table.meta() == base.meta()
    ids = np.arange(17, 17 + 2 * 24, dtype=np.int64).reshape(2, 24)
    a, b = raw.table.gather(raw.table.keys(ids)), base.gather(base.keys(ids))
    assert mx.array_equal(a.W, b.W).item()
    doc = json.loads((out / "config.json").read_text())
    assert doc["schema"] == export.SCHEMA == "metalforge.export/v1"
    assert (doc["schema_minor"], doc["step"]) == (0, c.step)
    assert doc["weights"] == {"raw": "dense.safetensors", "ema": "ema.safetensors"}
    assert doc["config"] == c.state["config"]
    assert not raw.chat


def test_sha256sums_covers_every_file_and_shasum_reads_it(
    run: Path, given: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "export"
    sums = export.export(run, out, **given)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    listed = {name: h for h, name in (line.split("  ", 1) for line in lines)}
    on_disk = {n: hashlib.sha256(b).hexdigest() for n, b in files(out).items() if n != "SHA256SUMS"}
    assert listed == on_disk == sums
    assert {f"logs/{n}" for n in LOGS} <= set(listed)  # the training logs are hashed too
    assert [line.split("  ", 1)[1] for line in lines] == sorted(listed)
    tool = ["shasum", "-a", "256", "-c"] if shutil.which("shasum") else ["sha256sum", "-c"]
    check = subprocess.run([*tool, "SHA256SUMS"], cwd=out, capture_output=True, check=False)
    assert check.returncode == 0, check.stdout
    export.verify(out)
    shard = out / "table" / "shard_0001.safetensors"
    kept = shard.read_bytes()
    shard.write_bytes(kept[:-1] + bytes([kept[-1] ^ 1]))
    with pytest.raises(DataMismatchError, match="shard_0001"):
        export.verify(out)
    flipped = subprocess.run([*tool, "SHA256SUMS"], cwd=out, capture_output=True, check=False)
    assert flipped.returncode != 0
    shard.write_bytes(kept)
    export.verify(out)
    text = (out / "SHA256SUMS").read_text()
    for bad in ("", text + "\n"):  # empty, and a blank line
        (out / "SHA256SUMS").write_text(bad)
        with pytest.raises(DataMismatchError, match="line per file"):
            export.verify(out)
    (out / "SHA256SUMS").write_text(text)
    (out / "NOTICE").unlink()
    with pytest.raises(DataMismatchError, match="NOTICE"):
        export.verify(out)
    gone = subprocess.run([*tool, "SHA256SUMS"], cwd=out, capture_output=True, check=False)
    assert gone.returncode != 0


def test_a_post_trained_export_names_its_base_and_prompts_in_the_chat_format(
    run: Path, given: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sft = sft_like(run, tmp_path / "sft")
    out = tmp_path / "export"
    export.export(sft, out, **given)
    post = json.loads((out / "config.json").read_text())["config"]["post"]
    assert post == {"stage": 1, "base": run.name, "base_step": 3}  # no local path ships
    logged = (out / "logs" / "run.json").read_text()
    assert json.loads(logged)["config"]["post"] == post  # nor in the training logs
    assert "/Users/someone" not in logged
    e = export.load(out)
    assert e.chat
    assert 2 not in e.cfg.masked_ids
    chat = [0, 1, USER, 20, 21, 2, 1, ASSISTANT]
    assert e.context("w20 w21") == chat
    plain = replace(e, chat=False)
    assert plain.context("w20 w21") == [0, 20, 21]
    assert plain.context("") == [0]
    for text in ("w20 <|im_end|>", "<|endoftext|> w20", "<|reserved_9|>", "<|reserved_16|>"):
        with pytest.raises(ConfigError, match="reserved"):
            e.context(text)
    c = ckpt.load(sft)
    want = greedy(c.arrays("dense"), e.cfg, chat, 4, table=load_table(c, e.cfg))
    argv = ["generate", str(out), "--prompt", "w20 w21", "--max-tokens", "4", "--json"]
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["ids"] == list(want.ids)  # the chat context


def test_a_table_off_export_without_an_ema(
    given: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = ModelConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        n_kv_heads=1,
        ffn=128,
        vocab_size=V,
        window=16,
        table=False,
        param_kind="sp",
    )
    p = init_params(cfg, 3)
    state = {"config": {"model": asdict(cfg), "table": None, "compute_dtype": "float32"}}
    src = ckpt.save(tmp_path / "twin", 5, arrays={"dense": p}, state=state)
    (src.parent / "run.json").write_text(json.dumps(state, indent=1))
    (src.parent / "metrics.jsonl").write_text("")
    out = tmp_path / "export"
    export.export(src, out, **given)
    shipped = ["SHA256SUMS", "config.json", "dense.safetensors", *GIVEN]
    assert sorted(files(out)) == sorted([*shipped, *(f"logs/{n}" for n in LOGS)])
    e = export.load(out)
    assert e.table is None
    assert same(e.params, p)
    with pytest.raises(ConfigError, match="'ema'"):
        export.load(out, "ema")
    assert cli.main(["generate", str(out), "--prompt", "w20", "--weights", "ema"]) == 78
    assert "no 'ema' weights" in capsys.readouterr().err
    assert export.checkpoint(out).step == 5
    shutil.copy(out / "dense.safetensors", out / "ema.safetensors")  # as an earlier download left
    with pytest.raises(DataMismatchError, match="weight files"):  # a scorer would take it as EMA
        export.checkpoint(out)


def test_the_export_refuses_what_it_cannot_ship(
    run: Path, given: dict[str, Path], tmp_path: Path
) -> None:
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(ConfigError, match="exists"):
        export.export(run, out, **given)
    gone = given | {"notice": tmp_path / "NOTICE"}
    with pytest.raises(ConfigError, match="no such file"):
        export.export(run, tmp_path / "a", **gone)
    small = given | {"tokenizer": tokenizer(tmp_path / "small.json", WORDS[:100])}
    with pytest.raises(ConfigError, match="512-token"):
        export.export(run, tmp_path / "b", **small)
    torn = tmp_path / "torn" / run.name
    shutil.copytree(run, torn)
    (torn / "ema.safetensors").unlink()
    with pytest.raises(ConfigError, match="incomplete"):
        export.export(torn, tmp_path / "c", **given)
    wrong = given | {"tokenizer": given["licence"]}  # a file that holds no tokenizer
    with pytest.raises(ConfigError, match="LICENSE"):
        export.export(run, tmp_path / "d", **wrong)
    lone = tmp_path / "lone" / run.name  # a checkpoint away from its run's logs
    shutil.copytree(run, lone)
    with pytest.raises(ConfigError, match=r"run\.json"):
        export.export(lone, tmp_path / "e", **given)
    (lone.parent / "run.json").write_text("{")
    with pytest.raises(ConfigError, match=r"run\.json"):
        export.export(lone, tmp_path / "e", **given)
    shutil.copy(run.parent / "run.json", lone.parent)
    with pytest.raises(ConfigError, match=r"no such file: .*metrics\.jsonl"):
        export.export(lone, tmp_path / "e", **given)
    for path in (tmp_path / "a", tmp_path / "b", tmp_path / "c", tmp_path / "d", tmp_path / "e"):
        assert not path.exists()
    with pytest.raises(ConfigError, match=r"config\.json"):
        export.load(tmp_path / "none")
    good = tmp_path / "good"
    export.export(run, good, **given)
    doc = json.loads((good / "config.json").read_text())
    (good / "config.json").write_text(json.dumps(doc | {"schema": "metalforge.export/v2"}))
    line = "SCHEMA_TOO_NEW kind=export found=v2 supported=v1..v1 "
    line += f"written_by=bake {__version__} FIX: upgrade bake to >= {__version__}"
    with pytest.raises(ConfigError, match=re.escape(line)):  # a schema newer than this bake reads
        export.load(good)
    (good / "config.json").write_text(json.dumps(doc | {"schema": "metalforge.checkpoint/v1"}))
    with pytest.raises(ConfigError, match=re.escape(f"not a {export.SCHEMA} export")):
        export.load(good)
    (good / "config.json").write_text(json.dumps(doc | {"schema": "metalforge.flagship-export/v1"}))
    with pytest.raises(ConfigError, match=re.escape("not a metalforge.export/v1 export")):
        export.load(good)  # the format's old name


# ---- the readouts and post-training read an export -------------------------------------------


def record(
    main: Callable[[list[str]], int], argv: list[str], capsys: pytest.CaptureFixture[str]
) -> dict[str, Any]:
    """The record `main` prints for `argv`, less the folder it read, which is `--ckpt`."""
    assert main(argv) == 0
    rec: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert rec.pop("ckpt") == argv[argv.index("--ckpt") + 1]
    return rec


@pytest.mark.parametrize(
    ("arm", "table", "cut"),
    [("run", "on", "table/shard_0001.safetensors"), ("twin", "off", "dense.safetensors")],
    ids=["run", "twin"],
)
def test_the_scorers_score_an_export_as_they_score_its_checkpoint(
    arm: str,
    table: str,
    cut: str,
    request: pytest.FixtureRequest,
    given: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Held-out bits per byte, its records and the multiple-choice scores equal the checkpoint's.

    In Billet's layout (`run`, the table on) and its twin's (the table off), both weights are
    scored, each with and without the table term where there is one. An export is refused before
    anything is scored when its bytes are not its `SHA256SUMS`, or its weight files are not the
    ones `config.json` names and `SHA256SUMS` lists.
    """
    src: Path = request.getfixturevalue(arm)
    out = tmp_path / "export"
    export.export(src, out, **given)
    split = heldout_store(tmp_path / "store", docs_for(15, 20, long=1))
    held = ["--data", str(split), "--seq-len", str(T)]
    want = record(heldout.main, ["--ckpt", str(src), *held], capsys)
    assert want["table"] == table
    assert {"raw", "ema"} <= set(want)
    assert record(heldout.main, ["--ckpt", str(out), *held], capsys) == want
    sha: list[dict[str, str]] = []
    for path in (src, out):
        argv = ["--ckpt", str(path), *held, "--records", str(tmp_path / f"records-{path.name}")]
        rec = record(heldout.main, argv, capsys)
        sha.append(rec.pop("records")["sha256"])
        assert rec == want
    assert sha[0] == sha[1]  # documents.npy and targets.npy, byte for byte
    evals = write_set(tmp_path / "evals", items_for(3))
    sets = ["--set", str(evals), "--store", str(mc_store(tmp_path / "mc")), "--seq-len", str(T)]
    want = record(mc.main, ["--ckpt", str(src), *sets], capsys)
    assert want["table"] == table
    assert {"raw", "ema"} <= set(want)
    assert record(mc.main, ["--ckpt", str(out), *sets], capsys) == want
    bad = tmp_path / "bad"  # the EMA's line gone from SHA256SUMS, and another file in its place
    shutil.copytree(out, bad)
    lines = (bad / "SHA256SUMS").read_text().splitlines(keepends=True)
    (bad / "SHA256SUMS").write_text("".join(x for x in lines if "  ema.safetensors" not in x))
    shutil.copy(bad / "dense.safetensors", bad / "ema.safetensors")
    assert heldout.main(["--ckpt", str(bad), *held]) == heldout.EXIT_DATA
    assert mc.main(["--ckpt", str(bad), *sets]) == mc.EXIT_DATA
    assert capsys.readouterr().err.count("weight files") == 2
    (bad / "ema.safetensors").unlink()  # gone, while config.json still names it
    with pytest.raises(DataMismatchError, match="weight files"):
        export.checkpoint(bad)
    (out / cut).write_bytes((out / cut).read_bytes()[:-1])  # an interrupted download
    assert heldout.main(["--ckpt", str(out), *held]) == heldout.EXIT_DATA
    assert mc.main(["--ckpt", str(out), *sets]) == mc.EXIT_DATA
    assert Path(cut).name in capsys.readouterr().err


def test_post_training_opens_an_export_as_its_checkpoint(
    run: Path, given: dict[str, Path], tmp_path: Path
) -> None:
    """`post.sft --base` and `post.sample --ckpt` read what the checkpoint gives them.

    That is the step, the run's configuration, both weights and the table with its read counts.
    """
    out = tmp_path / "export"
    export.export(run, out, **given)
    c, e = ckpt.load(run), SFTRun.base_checkpoint(out)
    assert (e.path, e.step, e.state) == (out, c.step, {"config": c.state["config"]})
    assert same(e.arrays("dense"), c.arrays("dense"))
    assert same(e.arrays("ema"), c.arrays("ema"))
    cfg = export.load(out).cfg
    table, base = SFTRun.table_of(e, cfg), SFTRun.table_of(c, cfg)
    assert table is not None
    assert base is not None
    assert table.meta() == base.meta()
    assert table.reads is not None
    assert base.reads is not None
    assert np.array_equal(table.reads, base.reads)  # the counts a stage carries forward


# ---- the commands ----------------------------------------------------------------------------


def export_argv(run: Path, out: Path, given: dict[str, Path]) -> list[str]:
    argv = ["export", str(run), "--out", str(out), "--tokenizer", str(given["tokenizer"])]
    return [*argv, "--licence", str(given["licence"]), "--notice", str(given["notice"])]


def test_bake_export_prints_the_sums_and_exits_78_and_22(
    run: Path,
    given: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = tmp_path / "export"
    assert cli.main(export_argv(run, out, given)) == 0
    rec = json.loads(capsys.readouterr().out)
    lines = (out / "SHA256SUMS").read_text().splitlines()
    assert rec == {"export": str(out), "sha256": {n: h for h, n in (x.split("  ") for x in lines)}}
    assert cli.main(export_argv(run, out, given)) == cli.EXIT_CONFIG == 78
    assert "exists" in capsys.readouterr().err
    real = export._copy

    def wrong(src: Path, dst: Path) -> str:
        real(src, dst)
        return "0" * 64

    monkeypatch.setattr(export, "_copy", wrong)
    assert cli.main(export_argv(run, tmp_path / "bad", given)) == cli.EXIT_DATA == 22
    assert "data error" in capsys.readouterr().err


def test_bake_generate_decodes_the_export_as_the_checkpoint_decodes(
    run: Path, given: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "export"
    export.export(run, out, **given)
    c = ckpt.load(run)
    e = export.load(out)
    want = greedy(c.arrays("dense"), e.cfg, [0, 20, 21, 22], 4, table=load_table(c, e.cfg))
    argv = ["generate", str(out), "--prompt", "w20 w21 w22", "--max-tokens", "4"]
    assert cli.main([*argv, "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    text = Tokenizer.from_file(str(given["tokenizer"])).decode(list(want.ids))
    assert rec == {"ids": list(want.ids), "stopped": want.stopped, "text": text}
    assert rec["ids"]  # the tiny model decodes past the prompt, so the match is not empty
    assert rec["stopped"] is False  # a base model never emits id 2: ids 1-16 stay masked
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("w20 w21 w22")
    assert cli.main(["generate", str(out), "--prompt-file", str(prompt), "--max-tokens", "4"]) == 0
    assert capsys.readouterr().out == text + "\n"
    ema = greedy(c.arrays("ema"), e.cfg, [0, 20, 21, 22], 4, table=load_table(c, e.cfg))
    assert cli.main([*argv, "--weights", "ema", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ids"] == list(ema.ids)
    assert cli.main([*argv[:1], str(tmp_path / "none"), *argv[2:]]) == 78
    assert cli.main([*argv[:3], "w20 <|im_end|>"]) == 78
    assert "reserved" in capsys.readouterr().err


def test_bake_generate_refuses_a_broken_export_in_one_line(
    run: Path, given: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good"
    export.export(run, good, **given)
    dense = (good / "dense.safetensors").read_bytes()
    broken = [  # what an interrupted download or a stray edit leaves
        ("dense.safetensors", None),
        ("dense.safetensors", dense[: len(dense) // 2]),
        ("tokenizer.json", None),
        ("config.json", b"[1, 2]"),
    ]
    for i, (name, data) in enumerate(broken):
        bad = tmp_path / f"bad{i}"
        shutil.copytree(good, bad)
        if data is None:
            (bad / name).unlink()
        else:
            (bad / name).write_bytes(data)
        assert cli.main(["generate", str(bad), "--prompt", "w20"]) == 78, name
        err = capsys.readouterr().err
        assert err.startswith("configuration error: "), err
        assert err.count("\n") == 1, err
    latin = tmp_path / "latin.txt"
    latin.write_bytes("café".encode("latin-1"))  # not UTF-8
    for prompt in (latin, tmp_path / "none.txt"):
        assert cli.main(["generate", str(good), "--prompt-file", str(prompt)]) == 78
        assert "--prompt-file" in capsys.readouterr().err


def test_the_commands_refuse_an_abbreviated_flag(tmp_path: Path) -> None:
    none = str(tmp_path / "none")
    for argv in (  # `allow_abbrev=False` on every parser
        ["generate", none, "--prompt", "w20", "--max", "2"],
        ["export", none, "--out", none, "--tok", none, "--licence", none, "--notice", none],
    ):
        with pytest.raises(SystemExit) as stop:
            cli.main(argv)
        assert stop.value.code == 2  # argparse's usage error
