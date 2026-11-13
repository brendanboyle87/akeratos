"""Entry point for the `bake` and `akeratos` commands, and for the trainer the Mac guard starts.

`bake train NAME` trains Billet, its twin or the starter: it turns the configuration `RUNS[NAME]`
holds, and how that run is launched (`LAUNCHES[NAME]`), into the trainer's flags (`train_flags`),
then runs the trainer in this process or under the Mac guard. `--seed N` replaces the
configuration's seed and the init and data seeds derived from it. `bake evaluate` prints
held-out bits per byte for a checkpoint (`akeratos/eval/heldout.py`). `bake generate` decodes
greedily from an export (`akeratos/infer.py`). `bake export` writes a checkpoint's release
folder (`akeratos/export.py`). `bake bench` measures what the table costs per training step
on this Mac (`akeratos/bench.py`); its handler reads the hardware class, passes macops'
conditions reader in and caps MLX's buffer cache, all through macops. `bake speedrun` trains the
model in `speedrun/train.py` for the 5-minute track's 300 s of step time, scores it and writes
its receipt (`akeratos/speedrun.py`). Each handler imports MLX only when it runs, so
`bake --help` and `bake --version` never load `mlx.core`.
"""

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from akeratos import __version__, configs, digest, macops
from akeratos.clock import SystemClock
from akeratos.configs import SWITCHES
from akeratos.env import tf32_off
from akeratos.errors import ConfigError, DataMismatchError
from akeratos.eval.heldout_flags import DESCRIPTION as HELDOUT_DESCRIPTION
from akeratos.eval.heldout_flags import add_flags as add_heldout_flags
from akeratos.train import entry
from akeratos.train.recipe import KEYS as RECIPE_KEYS

# `run_trainer` is called by name from the guard's child command (`macops.RUN_TRAINER`).

# The runs `bake train` starts, by name: the two released runs and the starter.
RUNS: dict[str, dict[str, Any]] = {
    "billet": configs.BILLET,
    "billet-twin": configs.BILLET_TWIN,
    "starter": configs.STARTER,
}
# How each run is launched, by name.
LAUNCHES: dict[str, dict[str, Any]] = {
    "billet": configs.LAUNCH,
    "billet-twin": configs.LAUNCH,
    "starter": configs.STARTER_LAUNCH,
}
# The trainer takes these from --tokens and from --rows x --seq-len x --accum, not --recipe.
_FROM_FLAGS = {"tokens.budget", "batch.tokens_per_update"}

# FATAL_CONFIG, a refusal, and DATA_MISMATCH, data that does not match its SHA-256s: an export
# that does not read back, or slices that differ from their pins.
EXIT_CONFIG, EXIT_DATA = 78, 22


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the `bake` command."""
    parser = argparse.ArgumentParser(
        prog="bake",
        description="Akeratos: an open lab for training language models from scratch on Apple silicon Macs with MLX.",
    )
    parser.add_argument("--version", action="version", version=f"akeratos {__version__}")
    groups = parser.add_subparsers(dest="group", metavar="COMMAND")
    train = groups.add_parser(
        "train",
        help="train a named configuration: billet, billet-twin or starter",
        description="Train Billet or its twin from its released configuration, or the starter: "
        "Billet's configuration at the 10M model's size, a first run of about 40 minutes.",
        allow_abbrev=False,
    )
    evaluate = groups.add_parser(
        "evaluate",
        help="bits per byte on the held-out split",
        description=HELDOUT_DESCRIPTION,
        allow_abbrev=False,
    )
    # No abbreviated flags: `--max` is not taken for `--max-tokens`.
    generate = groups.add_parser("generate", help="greedy text from an export", allow_abbrev=False)
    export = groups.add_parser(
        "export", help="write a checkpoint's release folder", allow_abbrev=False
    )
    speedrun = groups.add_parser(
        "speedrun",
        help="train speedrun/train.py's model for 300 s and score it",
        allow_abbrev=False,
    )
    bench = groups.add_parser(
        "bench",
        help="what the n-gram table costs per training step on this Mac",
        allow_abbrev=False,
    )
    _train_args(train)
    add_heldout_flags(evaluate)
    _export_args(export)
    _generate_args(generate)
    _bench_args(bench)
    _speedrun_args(speedrun)
    return parser


def _train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "name", choices=tuple(RUNS), metavar="NAME", help="billet, billet-twin or starter"
    )
    p.add_argument("--data", required=True, metavar="DIR", help="the data store's train/ split")
    p.add_argument("--out", required=True, metavar="DIR", help="the run directory")
    p.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="the master seed, from which the trainer derives the init and data seeds "
        "(default: the configuration's own)",
    )
    p.add_argument("--steps", type=int, metavar="N", help="stop after N steps")
    p.add_argument("--minutes", type=float, metavar="M", help="stop after M minutes")
    p.add_argument("--print-every", type=int, metavar="N", help="a progress line every N steps")
    p.add_argument(
        "--guard",
        action="store_true",
        help="run under the Mac guard, which stops the run cleanly on battery, heat, memory "
        "pressure or low disk",
    )


def _json(value: object) -> str:
    """`value` as the trainer's `key=value` switches read it: JSON, or a bare string."""
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return value  # the trainer reads text that is not JSON as this string
    return json.dumps(value, separators=(",", ":"))


def train_flags(
    run: dict[str, Any],
    data: str,
    out: str,
    *,
    launch: dict[str, Any] | None = None,
    seed: int | None = None,
    steps: int | None = None,
    minutes: float | None = None,
    print_every: int | None = None,
) -> list[str]:
    """The trainer's flags that start `run`, one of `RUNS`, launched as `launch` says.

    `launch` is one of `LAUNCHES`; by default it is `configs.LAUNCH`, how Billet and its twin
    were launched. `seed` replaces the run's master seed; the trainer derives the init seed and
    the data seed from it. Every model field goes by its switch key where it has one and by its
    field name otherwise, `attn.path` among them. Every optimiser and recipe value goes by its
    recipe key; a value with no key is one the trainer cannot set and keeps at its default. The
    table's values go by `--table`, apart from its width and seed, which the trainer takes from
    the model's width and from `--seed`.
    """
    launch = configs.LAUNCH if launch is None else launch
    master = int(run["seed"], 16) if seed is None else seed
    flags = ["--data", data, "--out", out, "--seed", str(master)]
    flags += ["--tokens", repr(run["optim"]["tokens"]), "--accum", str(run["accum"])]
    flags += ["--rows", str(launch["rows"]), "--seq-len", str(launch["seq_len"])]
    flags += ["--compute-dtype", run["compute_dtype"], *(["--mtp"] if run["mtp"] else [])]
    model_keys = {field: key for key, field in SWITCHES.items()}
    for field, value in run["model"].items():
        flags += ["--model", f"{model_keys.get(field, field)}={_json(value)}"]
    recipe_keys = {where: key for key, where in RECIPE_KEYS.items() if key not in _FROM_FLAGS}
    for part in ("optim", "recipe"):
        for field, value in run[part].items():
            if (key := recipe_keys.get((part, field))) is not None:
                flags += ["--recipe", f"{key}={_json(value)}"]
    table_keys = {field: key for key, field in entry.TABLE_KEYS.items()}
    table: dict[str, Any] = run["table"] or {}  # None with the table off
    for field, value in table.items():
        if field not in ("width", "seed"):
            flags += ["--table", f"{table_keys.get(field, field)}={_json(value)}"]
    flags += ["--ckpt-every", str(launch["ckpt_every"]), "--keep", str(launch["keep"])]
    flags += ["--pin-decay"] if launch["pin_decay"] else []
    for flag, value in (("--steps", steps), ("--minutes", minutes), ("--print-every", print_every)):
        flags += [] if value is None else [flag, str(value)]
    return flags


def _train(a: argparse.Namespace) -> int:
    """`bake train`: run the trainer in this process, or under the Mac guard with `--guard`.

    The run's launch values give the trainer's flags and the cap on MLX's cache.
    """
    launch = LAUNCHES[a.name]
    flags = train_flags(
        RUNS[a.name],
        a.data,
        a.out,
        launch=launch,
        seed=a.seed,
        steps=a.steps,
        minutes=a.minutes,
        print_every=a.print_every,
    )
    cap = ["--cache-limit-gb", str(launch["cache_limit_gb"])]
    return macops.main([*cap, "--", *flags]) if a.guard else run_trainer([*cap, *flags])


def _evaluate(a: argparse.Namespace) -> int:
    """`bake evaluate`: the held-out score, with its own record, messages and exit codes."""
    heldout: Any = importlib.import_module("akeratos.eval.heldout")
    return heldout.report(a)


def _export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("ckpt", type=Path, metavar="CKPT", help="a complete checkpoint directory")
    p.add_argument("--out", type=Path, required=True, metavar="DIR", help="the new export folder")
    p.add_argument("--tokenizer", type=Path, required=True, metavar="FILE", help="tokenizer.json")
    p.add_argument("--licence", type=Path, required=True, metavar="FILE", help="the LICENSE")
    p.add_argument("--notice", type=Path, required=True, metavar="FILE", help="the NOTICE")


def _generate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("export", type=Path, metavar="EXPORT", help="a folder `bake export` wrote")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", dest="text", metavar="TEXT", help="the prompt")
    source.add_argument("--prompt-file", type=Path, metavar="PATH", help="a UTF-8 prompt file")
    p.add_argument(
        "--max-tokens", type=int, default=256, metavar="N", help="new tokens at most (default: 256)"
    )
    p.add_argument("--weights", choices=("raw", "ema"), default="raw", help="raw (default) or ema")
    p.add_argument("--json", action="store_true", help="print one JSON object")


def _bench_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="print the row alone, as JSON")
    p.add_argument("--out", type=Path, metavar="DIR", help="also write the row into DIR")
    p.add_argument("--person", metavar="HANDLE", help="the GitHub handle the row credits")


def _speedrun_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar="DIR",
        help="the folder that holds the track's slices",
    )
    p.add_argument(
        "--claim",
        required=True,
        metavar="YYYY-MM-DD-HANDLE",
        help="your claim's name: the UTC date of its first run, then your handle",
    )
    p.add_argument("--person", required=True, metavar="HANDLE", help="your GitHub handle")
    p.add_argument("--ai-system", required=True, help="the AI system that helped, or none")
    p.add_argument(
        "--train",
        type=Path,
        default="speedrun/train.py",
        help="the training file (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sets the data order and the seed build() receives (default: 0)",
    )
    p.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="default: results/speedrun; practice: runs/local/speedrun",
    )
    p.add_argument("--category", choices=("open", "no-table"), default="open", help="default: open")
    p.add_argument(
        "--rerun-of", metavar="CLASS/CLAIM", help="another person's claim that this run reruns"
    )
    p.add_argument(
        "--seconds",
        type=float,
        default=300.0,
        help="seconds of step time (default: 300); another value makes a practice run",
    )


def _export(a: argparse.Namespace) -> str:
    """`bake export`: write the folder, then print each file's SHA-256 as JSON."""
    export: Any = importlib.import_module("akeratos.export")
    sums = export.export(a.ckpt, a.out, tokenizer=a.tokenizer, licence=a.licence, notice=a.notice)
    return json.dumps({"export": str(a.out), "sha256": sums}, indent=1, sort_keys=True)


def _generate(a: argparse.Namespace) -> str:
    """`bake generate`: greedy text from an export, or one JSON object with `--json`."""
    try:  # the prompt first: loading an export reads gigabytes
        text = a.text if a.prompt_file is None else a.prompt_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as err:
        raise ConfigError(f"--prompt-file: {err}") from err
    export: Any = importlib.import_module("akeratos.export")
    infer: Any = importlib.import_module("akeratos.infer")
    e = export.load(a.export, a.weights)
    s = infer.greedy(e.params, e.cfg, e.context(text), a.max_tokens, table=e.table)
    out = e.tokenizer.decode(list(s.ids))
    rec = {"text": out, "ids": list(s.ids), "stopped": s.stopped}
    return json.dumps(rec, sort_keys=True) if a.json else out


def _bench(a: argparse.Namespace) -> str:
    """`bake bench`: the table's cost per step on this Mac, as a short table or the row as JSON.

    Before the run, which takes minutes, it refuses a person that is not a GitHub handle, a row
    file that exists and an out folder that cannot be made.
    """
    if a.out is not None and a.person is None:
        raise ConfigError("--out needs --person HANDLE, the GitHub handle the row credits")
    bench: Any = importlib.import_module("akeratos.bench")
    if a.person is not None and not bench.HANDLE.fullmatch(a.person):
        raise ConfigError(f"--person {a.person} is not a GitHub handle")
    hardware = macops.read_hardware()
    path = _row_file(a.out, bench.file_name(hardware.key(), metadata.version("mlx")))
    macops.apply_cache_limit(bench.SHAPE.cache_limit_gib * 2**30)  # before the first allocation
    row = bench.run(hardware, macops.read_conditions, SystemClock(), a.person, bench.SHAPE)
    if path is not None:
        digest.write_pretty(path, row)
        sys.stderr.write(f"bench: wrote {path}\n")
    return digest.pretty(row).rstrip("\n") if a.json else bench.summary(row)


def _row_file(out: Path | None, name: str) -> Path | None:
    """The row's file in `out`, or None without one."""
    if out is None:
        return None
    path = out / name
    if path.exists():
        raise ConfigError(f"{path} exists: a class holds one row per MLX version")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ConfigError(f"--out: {err}") from err
    return path


def _speedrun(a: argparse.Namespace) -> str:
    """`bake speedrun`: train, score and write the receipt; the summary line."""
    speedrun: Any = importlib.import_module("akeratos.speedrun")
    opts = speedrun.Options(
        data=a.data,
        train=a.train,
        seed=a.seed,
        out=a.out,
        category=a.category,
        claim=a.claim,
        person=a.person,
        ai_system=a.ai_system,
        rerun_of=a.rerun_of,
        hardware=macops.read_hardware(),
        conditions=macops.read_conditions,
        clock=SystemClock(),
        budget_s=a.seconds,
    )
    return speedrun.run(opts)


def _run(command: Callable[[argparse.Namespace], str], a: argparse.Namespace) -> int:
    """Run `command` with TF32 off and print its text; a refusal prints one line on stderr."""
    try:
        tf32_off()  # before the command imports MLX, which reads MLX_ENABLE_TF32 once
        text = command(a)
    except ConfigError as e:
        sys.stderr.write(f"configuration error: {e}\n")
        return EXIT_CONFIG
    except DataMismatchError as e:
        sys.stderr.write(f"data error: {e}\n")
        return EXIT_DATA
    sys.stdout.write(text + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the command, or print the help text; return the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.group == "train":
        return _train(args)
    if args.group == "evaluate":
        return _evaluate(args)
    if args.group == "export":
        return _run(_export, args)
    if args.group == "generate":
        return _run(_generate, args)
    if args.group == "bench":
        return _run(_bench, args)
    if args.group == "speedrun":
        return _run(_speedrun, args)
    parser.print_help()
    return 0


def run_trainer(argv: list[str] | None = None) -> int:
    """Run `akeratos.train.entry` with macops' MLX cache cap; return the trainer's exit code.

    The trainer never imports macops, so the CLI installs the cap as `entry.main`'s `setup` hook.
    The Mac guard launches the trainer through this function. It peels the guard's
    `--cache-limit-gb` (GB of 2^30 bytes) off `argv` and passes the rest to `entry.main`
    unchanged, so without it MLX keeps its default cache. It is not a `bake` subcommand.
    """
    peel = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    peel.add_argument("--cache-limit-gb", type=float)
    a, rest = peel.parse_known_args(sys.argv[1:] if argv is None else argv)
    limit = None if a.cache_limit_gb is None else int(a.cache_limit_gb * 2**30)
    # The loop imports MLX, which reads MLX_ENABLE_TF32 once, so TF32 goes off first.
    tf32_off()
    loop = importlib.import_module("akeratos.train.loop")
    saver = importlib.import_module("akeratos.train.saver")
    return entry.main(
        rest, loop.Run, saver.Checkpointer, setup=lambda: macops.apply_cache_limit(limit)
    )
