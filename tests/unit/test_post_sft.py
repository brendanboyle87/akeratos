"""Stage 1's supervised fine-tuning on a frozen table, and the greedy decoder.

The stage keeps the table, puts loss on answer tokens only, unmasks ids 1 and 2 only in
post-training, lets turn 2 attend to turn 1, and resumes to the state of an unbroken run. The
data is the SFT store format (`akeratos.data.chat`), written here with a fixture tokenizer.
Everything runs on MLX's CPU device on a tiny model whose base checkpoint
`akeratos/train/entry.py` writes, table on.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from akeratos import ckpt
from akeratos.configs import ModelConfig
from akeratos.data import chat
from akeratos.data.packer import ExhaustedError, segment_ids
from akeratos.errors import DataMismatchError
from akeratos.infer import greedy
from akeratos.model import (
    EMBED,
    forward,
    head_logits,
    hidden,
    init_params,
    masked_id_mask,
    ve_name,
)
from akeratos.post import sample, sft
from akeratos.post.data import (
    ChatSource,
    Doc,
    documents,
    epochs_of,
    pack_rows,
    prompt_of,
    stage_config,
)
from akeratos.post.run import SFTRun, load_table, marker_rows
from akeratos.train import entry
from akeratos.train.loop import Run, make_loss_fn, prepare
from akeratos.train.recipe import OptimConfig
from akeratos.train.saver import Checkpointer
from akeratos.train.source import StepRows
from tests.fakes.clock import FakeClock

mx.set_default_device(mx.cpu)

V, B, T = 512, 2, 32  # the base run's shape
S = 64  # the stage's rows: room for two short conversations
TINY = ModelConfig(
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
MODEL_KEYS = [
    "d_model=64",
    "n_layers=2",
    "n_heads=2",
    "head_dim=32",
    "n_kv_heads=1",
    "ffn=128",
    f"vocab_size={V}",
    "window=16",
    "param.kind=sp",
]
TABLE_KEYS = [f"params={4096 * 64}", f"max_shard_elements={1024 * 64}"]
STEP_TOKENS = B * S * 2  # --rows B, --seq-len S, --accum 2


def encode(text: str) -> list[int]:
    """A fixture tokenizer: one id per character, above the reserved ids."""
    return [17 + ord(c) % 400 for c in text]


@pytest.fixture(autouse=True)
def tf32_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")


def messages(rng: np.random.Generator, pairs: int = 1) -> list[dict[str, str]]:
    def word() -> str:
        return "".join(chr(97 + int(c)) for c in rng.integers(0, 26, int(rng.integers(1, 5))))

    return [{"role": r, "content": word()} for _ in range(pairs) for r in chat.ROLES]


def doc_of(r: chat.Rendered) -> Doc:
    return Doc(r.ids.astype(np.int32), r.mask[1:].astype(np.float32))


def rendered(n: int = 40, seed: int = 0) -> list[chat.Rendered]:
    rng = np.random.default_rng(seed)
    return [chat.render(messages(rng, int(rng.integers(1, 3))), encode) for _ in range(n)]


def write_split(path: Path, docs: Sequence[chat.Rendered], epochs: int | None = None) -> Path:
    path.mkdir(parents=True)
    lengths = [len(r.ids) for r in docs]
    (path / "000000.bin").write_bytes(np.concatenate([r.ids for r in docs]).astype("<u2").tobytes())
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype("<u8")
    (path / "000000.idx").write_bytes(starts.tobytes())
    (path / "000000.mask").write_bytes(np.concatenate([r.mask for r in docs]).tobytes())
    manifest = {
        "dtype": "uint16",
        "eos": 0,
        "sep_position": "start",
        "vocab": V,
        "shards": ["000000.bin"],
        "tokens": [sum(lengths)],
        "documents": [len(docs)],
        "chat": {"format": chat.FORMAT, "mask": "uint8-target", "max_doc_tokens": 2048},
        "allow_reserved": [1, 2],
    }
    if epochs is not None:
        manifest["epochs"] = epochs
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def store(path: Path, n: int = 60, seed: int = 0, epochs: int | None = None) -> Path:
    return write_split(path, rendered(n, seed), epochs)


def base_run(out: Path, steps: int = 3, *extra: str, recipe: Sequence[str] = ()) -> Path:
    keys = [*MODEL_KEYS, "table=true", *extra]
    args = ["--data", "synthetic", "--tokens", str(steps * B * T), "--rows", str(B)]
    args += ["--seq-len", str(T), "--out", str(out), "--compute-dtype", "float32"]
    args += [f"--model={k}" for k in keys] + [f"--table={k}" for k in TABLE_KEYS]
    args += ["--recipe=lr.warmup=1", "--recipe=param.kind=sp", "--recipe=opt.decay_ind=0.0"]
    args += ["--recipe=opt.clip_norm=0.5"]  # a switch stage 1 does not set, so it carries over
    args += [f"--recipe={k}" for k in recipe]
    assert entry.main([*args, "--print-every", "0"], Run, Checkpointer) == 0
    return out


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    old = os.environ.get("MLX_ENABLE_TF32")
    os.environ["MLX_ENABLE_TF32"] = "0"
    try:
        yield base_run(tmp_path_factory.mktemp("base") / "run")
    finally:
        if old is None:
            os.environ.pop("MLX_ENABLE_TF32", None)
        else:
            os.environ["MLX_ENABLE_TF32"] = old


def sft_args(base: Path, data: Path, out: Path, tokens: float, *extra: str) -> list[str]:
    args = ["--base", str(base), "--data", str(data), "--out", str(out), "--tokens", str(tokens)]
    args += ["--rows", str(B), "--seq-len", str(S), "--accum", "2", "--print-every", "0"]
    return [*args, *extra]


def run_sft(argv: list[str]) -> int:
    return sft.main(argv, SFTRun, Checkpointer, clock=FakeClock())


def newest(path: Path) -> ckpt.Checkpoint:
    found = ckpt.latest(path)
    assert found is not None
    return ckpt.load(found)


def digests(path: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
        if p.is_file()
    }


# ---- the configuration, the masks and attention ----------------------------------------------


def test_stage_config_unmasks_ids_1_and_2_only() -> None:
    """Ids 1 and 2 have zero probability unless post.stage is set."""
    assert stage_config(TINY, None) is TINY
    staged = stage_config(TINY, 1)
    assert staged.masked_ids == tuple(range(3, 17))
    p = init_params(TINY, 0)
    ids = mx.array(np.arange(30, 30 + T, dtype=np.int32)[None])
    seg = mx.zeros(ids.shape, mx.int32)
    base = np.array(forward(p, TINY, ids, seg, mx.float32))
    post = np.array(forward(p, staged, ids, seg, mx.float32))
    assert np.isneginf(base[..., 1:17]).all()
    assert np.isfinite(post[..., 1:3]).all()
    assert np.isneginf(post[..., 3:17]).all()
    assert np.isfinite(post[..., 17:]).all()
    assert np.isfinite(post[..., 0]).all()


def test_user_and_template_tokens_carry_no_loss() -> None:
    """The loss is the mean over answer targets; nothing else moves it."""
    cfg = stage_config(TINY, 1)
    p = init_params(cfg, 0)
    ids, mask = pack_rows([doc_of(r) for r in rendered(6)], S)
    inputs = ids[:, :-1]
    batch = StepRows(inputs, ids[:, 1:], segment_ids(inputs), mask)
    args = prepare(batch, cfg)
    assert np.array_equal(np.array(args[3]), mask)  # id 2 is a live target under post.stage
    targets = ids[:, 1:]
    assert (targets[mask > 0] == chat.IM_END).sum() == (targets == chat.IM_END).sum() // 2
    assert np.isin(targets[mask > 0], [chat.IM_END, *range(17, V)]).all()
    loss = float(make_loss_fn(cfg, mx.float32)(p, *args))
    z = head_logits(p, cfg, hidden(p, cfg, args[0], args[1], mx.float32), masked_id_mask(cfg))
    y = np.array(args[2])
    zz = np.array(z)
    nll = np.array(mx.logsumexp(z, axis=-1)) - np.take_along_axis(zz, y[..., None], -1)[..., 0]
    assert loss == pytest.approx(float(nll[mask > 0].mean()), rel=1e-5)

    # Every non-answer target can change and the loss does not.
    other = ids.copy()
    other[:, 1:][mask == 0] = 0
    moved = StepRows(inputs, other[:, 1:], segment_ids(inputs), mask)
    assert float(make_loss_fn(cfg, mx.float32)(p, *prepare(moved, cfg))) == loss

    # Under the pretraining configuration id 2 is a masked target, so it carries no weight.
    w0 = np.array(prepare(batch, TINY)[3])
    assert (w0[targets == chat.IM_END] == 0).all()


def test_turn_two_attends_to_turn_one() -> None:
    """Turns end at id 2, so turn 2's logits depend on turn 1."""
    cfg = stage_config(replace(TINY, window=None), 1)
    p = init_params(cfg, 1)

    def logits(first: str, split: bool = False) -> np.ndarray:
        turns = [first, "yo", "and", "ok"]
        msgs = [{"role": chat.ROLES[i % 2], "content": t} for i, t in enumerate(turns)]
        ids = chat.render(msgs, encode).ids.astype(np.int32)
        if split:  # what id 0 at turn 1's end would do: open a new segment
            ids[np.flatnonzero(ids == chat.IM_START)[2]] = 0
        x = ids[None, :]
        out = forward(p, cfg, mx.array(x), mx.array(segment_ids(x)), mx.float32)
        return np.array(out)[0, -3:]

    assert not np.allclose(logits("hi"), logits("no"))
    # Split by id 0, the second exchange no longer sees the first.
    np.testing.assert_array_equal(logits("hi", True), logits("no", True))


def test_marker_rows_start_at_the_mean_row() -> None:
    p = init_params(TINY, 0)
    q = marker_rows(p)
    assert ve_name(1) in p
    for name in (EMBED, ve_name(1)):
        E, F = np.array(p[name]), np.array(q[name])
        mean = E.mean(axis=0)
        np.testing.assert_allclose(F[1], mean, rtol=1e-6)
        np.testing.assert_allclose(F[2], mean, rtol=1e-6)
        np.testing.assert_array_equal(np.delete(F, [1, 2], 0), np.delete(E, [1, 2], 0))
    others = [n for n in p if n not in (EMBED, ve_name(1))]
    assert all(np.array_equal(np.array(p[n]), np.array(q[n])) for n in others)


# ---- the documents and the chat source --------------------------------------------------------


def test_documents_read_the_store_with_target_aligned_masks(tmp_path: Path) -> None:
    rs = rendered(5)
    docs = documents(write_split(tmp_path / "train", rs), V)
    for r, d in zip(rs, docs, strict=True):
        assert d.ids.tolist() == r.ids.tolist()
        assert d.mask.tolist() == r.mask[1:].tolist()
    with pytest.raises(DataMismatchError):
        documents(tmp_path / "train", V + 1)
    manifest = json.loads((tmp_path / "train" / "manifest.json").read_text())
    manifest["chat"]["format"] = "other"
    (tmp_path / "train" / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataMismatchError):
        documents(tmp_path / "train", V)


def test_epochs_come_from_the_store_manifest(tmp_path: Path) -> None:
    assert epochs_of(write_split(tmp_path / "a" / "train", rendered(2), epochs=2)) == 2
    assert epochs_of(write_split(tmp_path / "b" / "train", rendered(2))) == 1
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "manifest.json").write_text('{"epochs": 2}')
    assert epochs_of(write_split(tmp_path / "c" / "train", rendered(2))) == 2


def test_the_prompt_stops_after_the_last_assistant_header() -> None:
    msgs = [
        {"role": chat.ROLES[i % 2], "content": t} for i, t in enumerate(["hi", "yo", "x", "ok"])
    ]
    d = doc_of(chat.render(msgs, encode))
    head, answer = prompt_of(d)
    assert answer.tolist() == encode("ok")
    assert head.tolist() == d.ids[: len(d.ids) - len(answer) - 1].tolist()
    assert head[-len(encode("assistant\n")) - 1 :].tolist() == [
        chat.IM_START,
        *encode("assistant\n"),
    ]


def test_rows_hold_whole_conversations_each_in_its_segment() -> None:
    src = ChatSource([doc_of(r) for r in rendered()], B, S, data_seed=5)
    x = src.next()
    assert x.lmask is not None
    assert x.ids.shape == x.targets.shape == x.lmask.shape == (B, S)
    for r in range(B):
        opens = np.flatnonzero(x.ids[r] == 0)
        assert opens[0] == 0
        assert (np.diff(x.segment_ids[r]) >= 0).all()
        # Each segment's answer targets end at an id 2.
        ends = x.targets[r][x.lmask[r] > 0]
        assert ends[-1] == chat.IM_END
    assert set(np.unique(x.lmask)) <= {0.0, 1.0}


def test_the_cursor_resumes_the_same_rows_and_the_data_runs_out() -> None:
    docs = [doc_of(r) for r in rendered()]
    a = ChatSource(docs, B, S, data_seed=5, epochs=2)
    a.next()
    b = ChatSource(docs, B, S, data_seed=5, epochs=2, state=a.state())
    for _ in range(3):
        xa, xb = a.next(), b.next()
        assert np.array_equal(xa.ids, xb.ids)
        assert xa.lmask is not None
        assert xb.lmask is not None
        assert np.array_equal(xa.lmask, xb.lmask)
    with pytest.raises(DataMismatchError, match="data seed"):
        ChatSource(docs, B, S, data_seed=6, state=a.state())
    c = ChatSource(docs, 1, S, data_seed=5, epochs=2)
    for _ in range(c.rows_available()):
        c.next()
    with pytest.raises(ExhaustedError):
        c.next()


def test_a_conversation_longer_than_a_row_is_dropped_not_cut() -> None:
    long = doc_of(
        chat.render(
            [{"role": "user", "content": "x" * S}, *messages(np.random.default_rng(0))[1:]], encode
        )
    )
    src = ChatSource([long, *(doc_of(r) for r in rendered(5))], B, S, data_seed=0)
    assert src.dropped == 1
    with pytest.raises(DataMismatchError):
        ChatSource([long], B, S, data_seed=0)


# ---- the stage ---------------------------------------------------------------------------------


def test_stage_one_keeps_the_table_and_trains_the_dense_weights(base: Path, tmp_path: Path) -> None:
    """The table's shard checksums are identical before and after the stage."""
    c = newest(base)
    before = digests(c.path / "table")
    data = store(tmp_path / "train")
    assert run_sft(sft_args(base, data, tmp_path / "sft", 4 * STEP_TOKENS, "--rate", "1")) == 0
    assert digests(c.path / "table") == before
    done = newest(tmp_path / "sft")
    assert digests(done.path / "table") == before
    # The base's read counts load with the table and are saved unchanged: SFT counts nothing.
    base_reads = np.load(c.path / "table" / "reads.npy")
    assert base_reads.sum() > 0
    assert np.array_equal(np.load(done.path / "table" / "reads.npy"), base_reads)
    assert done.state["table_event"] == c.state["table_event"]
    a, b = c.arrays("dense"), done.arrays("dense")
    assert any(not np.array_equal(np.array(a[n]), np.array(b[n])) for n in a)
    post = done.state["config"]["post"]
    assert (post["stage"], post["unmasked"], post["base_step"]) == (1, [1, 2], c.step)
    assert done.state["config"]["model"]["masked_ids"] == list(range(3, 17))
    assert done.state["config"]["optim"]["lr_peak"] == c.state["config"]["optim"]["lr_peak"]
    assert done.state["config"]["optim"]["clip"] == c.state["config"]["optim"]["clip"] == 0.5
    assert done.state["config"]["recipe"]["schedule"] == "d2z"
    assert done.state["config"]["recipe"]["ema"] is False
    lines = [json.loads(x) for x in (tmp_path / "sft" / "metrics.jsonl").read_text().splitlines()]
    assert all(x["table"]["frozen"] for x in lines)
    assert all(x["table"]["event"] == c.state["table_event"] for x in lines)
    assert lines[-1]["s"] < lines[1]["s"]  # linear decay toward 0


def test_a_stopped_and_resumed_stage_matches_an_unbroken_one(base: Path, tmp_path: Path) -> None:
    """Stop after a step, rerun the same command, and the end state is the same."""
    data = store(tmp_path / "train")
    tokens = 4 * STEP_TOKENS
    whole = sft_args(base, data, tmp_path / "whole", tokens, "--rate", "1", "--ckpt-every", "1")
    assert run_sft(whole) == 0
    part = sft_args(base, data, tmp_path / "part", tokens, "--rate", "1", "--ckpt-every", "1")
    assert run_sft([*part, "--steps", "2"]) == sft.EXIT_STOPPED
    assert run_sft(part) == 0
    a = newest(tmp_path / "whole")
    b = newest(tmp_path / "part")
    assert a.step == b.step == 4
    for stem in ("dense", "optim-hidden"):
        x, y = a.arrays(stem), b.arrays(stem)
        assert all(np.array_equal(np.array(x[n]), np.array(y[n])) for n in x)
    assert a.state["source"] == b.state["source"]
    # Another setting is refused on resume.
    assert run_sft([*part[:-2], "--rate", "0.5"]) == sft.EXIT_CONFIG


def test_the_sweep_picks_the_lowest_held_out_loss_and_ties_go_low(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, held = store(tmp_path / "train"), store(tmp_path / "heldout", n=8, seed=1)
    argv = sft_args(base, data, tmp_path / "sweep", 2 * STEP_TOKENS, "--sweep", "0.2,0.05,0.1")
    assert run_sft([*argv, "--heldout", str(held)]) == 0
    rec = json.loads((tmp_path / "sweep" / "sweep.json").read_text())
    assert set(rec["loss"]) == {"0.05", "0.1", "0.2"}
    assert rec["pick"] == min((v, float(k)) for k, v in rec["loss"].items())[1]
    for f in ("0.05", "0.1", "0.2"):
        man = json.loads((tmp_path / "sweep" / f"rate-{f}" / "run.json").read_text())
        assert man["config"]["post"]["rate_factor"] == float(f)

    monkeypatch.setattr(SFTRun, "assistant_loss", lambda self, rows, b: 1.0)
    out = tmp_path / "tie"
    tie = sft_args(base, data, out, 2 * STEP_TOKENS, "--sweep", "0.2,0.1", "--heldout", str(held))
    assert run_sft(tie) == 0
    assert json.loads((out / "sweep.json").read_text())["pick"] == 0.1


def test_the_stage_refuses_what_it_cannot_run(
    base: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    small = store(tmp_path / "small" / "train", n=6)
    assert run_sft(sft_args(base, small, tmp_path / "a", 1e6, "--rate", "1")) == sft.EXIT_CONFIG
    assert "the budget needs" in capsys.readouterr().err
    data = store(tmp_path / "train")
    no_held = sft_args(base, data, tmp_path / "b", STEP_TOKENS, "--sweep", "0.1")
    assert run_sft(no_held) == sft.EXIT_CONFIG
    none = sft_args(tmp_path / "none", data, tmp_path / "c", STEP_TOKENS, "--rate", "1")
    assert run_sft(none) == sft.EXIT_CONFIG
    (tmp_path / "plain").mkdir()
    plain = sft_args(base, tmp_path / "plain", tmp_path / "d", 2 * STEP_TOKENS, "--rate", "1")
    assert run_sft(plain) == sft.EXIT_DATA
    one = sft_args(base, data, tmp_path / "e", STEP_TOKENS, "--rate", "1")
    assert run_sft(one) == sft.EXIT_CONFIG
    ema = sft_args(base, data, tmp_path / "f", 2 * STEP_TOKENS, "--rate", "1", "--weights", "ema")
    assert run_sft(ema) == 0  # the base kept an EMA
    causal = base_run(tmp_path / "causal", 3, "mask.packing=causal")
    packed = sft_args(causal, data, tmp_path / "g", 2 * STEP_TOKENS, "--rate", "1")
    assert run_sft(packed) == sft.EXIT_CONFIG
    assert "mask.packing is causal" in capsys.readouterr().err


def test_a_coupled_base_still_gets_no_weight_decay(tmp_path: Path) -> None:
    keys = ("opt.weight_decay_mode=coupled", "opt.decay_tokens=1e12")
    coupled = base_run(tmp_path / "coupled", 3, recipe=keys)
    data = store(tmp_path / "train")
    assert run_sft(sft_args(coupled, data, tmp_path / "sft", 2 * STEP_TOKENS, "--rate", "1")) == 0
    assert newest(coupled).state["config"]["optim"]["weight_decay_mode"] == "coupled"
    optim = newest(tmp_path / "sft").state["config"]["optim"]
    assert optim["weight_decay_mode"] == "independent"
    assert OptimConfig(**optim).hidden_decay() == 0.0


def test_the_store_epochs_bound_the_budget(
    base: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    once = store(tmp_path / "a" / "train", n=12)
    twice = store(tmp_path / "b" / "train", n=12, epochs=2)
    have = ChatSource([doc_of(r) for r in rendered(12)], B, S, data_seed=0).rows_available()
    tokens = (have // (2 * B) + 1) * STEP_TOKENS  # one step more than one epoch feeds
    assert run_sft(sft_args(base, once, tmp_path / "x", tokens, "--rate", "1")) == sft.EXIT_CONFIG
    assert "1 epoch(s)" in capsys.readouterr().err
    assert run_sft(sft_args(base, twice, tmp_path / "y", tokens, "--rate", "1")) == 0
    assert json.loads((tmp_path / "y" / "run.json").read_text())["config"]["post"]["epochs"] == 2


def test_the_entry_point_module_loads_no_mlx() -> None:
    code = "import sys, akeratos.post.sft; print('mlx.core' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---- the decoder and the samples --------------------------------------------------------------


def test_greedy_follows_the_arg_max_and_stops_at_its_stop_id() -> None:
    cfg = stage_config(TINY, 1)
    p = init_params(cfg, 3)
    start = [0, 1, 20, 21, 40, 41, 2, 1, 22, 21]
    s = greedy(p, cfg, start, 6)
    assert s.ids  # this seed decodes past the prompt
    oracle = replace(cfg, attn_path="masked")  # the full-mask path takes any length
    ids = list(start)
    for tok in s.ids:
        x = np.asarray([ids], np.int32)
        z = forward(p, oracle, mx.array(x), mx.zeros(x.shape, mx.int32), mx.float32)
        assert int(np.argmax(np.array(z)[0, -1])) == tok
        ids.append(tok)
    assert not set(s.ids) & set(range(3, 17))
    stop = greedy(p, cfg, start, 6, stop=s.ids[0])
    assert stop.ids == ()
    assert stop.stopped
    for bad in ([1, 2], [0, 0], [0, V]):
        with pytest.raises(ValueError, match="prompt"):
            greedy(p, cfg, bad, 2)


def test_greedy_reads_the_frozen_table(base: Path) -> None:
    c = newest(base)
    cfg = stage_config(replace(TINY, table=True), 1)
    table = load_table(c, cfg)
    assert table is not None
    assert table.event == c.state["table_event"]
    s = greedy(c.arrays("dense"), cfg, [0, 1, 20, 21, 40, 2, 1, 22, 21], 4, table=table)
    assert len(s.ids) <= 4
    with pytest.raises(ValueError, match="together"):
        greedy(c.arrays("dense"), cfg, [0, 40], 2)


def test_samples_report_format_compliance(
    base: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = store(tmp_path / "train")
    assert run_sft(sft_args(base, data, tmp_path / "sft", 4 * STEP_TOKENS, "--rate", "1")) == 0
    out = tmp_path / "new" / "samples.jsonl"  # its folder does not exist yet
    argv = ["--ckpt", str(tmp_path / "sft"), "--data", str(data), "--n", "3", "--max-new", "4"]
    assert sample.main([*argv, "--out", str(out)]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["n"] == 3
    assert 0 <= rec["format_compliance"] <= rec["stopped"] <= 1
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 3
    assert all(len(x["ids"]) <= 4 for x in lines)
    missing = ["--ckpt", str(tmp_path / "none"), "--data", str(data)]
    assert sample.main(missing) == sample.EXIT_CONFIG
