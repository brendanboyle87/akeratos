#!/usr/bin/env python3
"""M1's readouts for MetalForge Billet and its twin (#235).

M1 is the analysis plan posted on #235 on 2026-11-01 (`M1` below). This file computes its
readouts by its rules. A dated amendment on #235 records this file's SHA-256 before any
Billet − twin figure is computed; after that the file changes only by a dated amendment. Figures
M1 does not name are labelled exploratory: each zero-shot item's margin, and some memorisation
figures (Readouts below).

    .venv/bin/python -I docs/research/billet/analysis.py --out report.json
        --heldout B/heldout.json T/heldout.json --split <store>/heldout
        [--pinned B/heldout-19532.json T/heldout-19532.json]
        [--reads reads.npy] [--overlap overlap.npy]
        [--mc B/mc.json T/mc.json --samples B/mc.items.jsonl T/mc.items.jsonl --set <set>]
        [--metrics B/metrics.jsonl T/metrics.jsonl --probe <probe>/summary.json]
        [--s6 B/s6.json T/s6.json --thirds thirds.npy]

(one line; B is Billet, T its twin). It needs NumPy and the standard library only. Never run it
on the twin's files before the twin has ended: M1 computes every Billet − twin readout only
after then, and every readout here needs the twin's final checkpoint or its last step. A
readout whose inputs are not given is reported as not run. A failed check refuses the run
(exit 2) before anything is written.

Inputs, each checked before use:
- heldout.json: `python -m metalforge.eval.heldout --compute-dtype float32 --records DIR --out
  heldout.json` (#481). Its dtype must be float32, it has no `mask`, and DIR's documents.npy and
  targets.npy have the SHA-256s it records. Billet's has the table on, the twin's off. Both score
  the same documents, targets, bytes and windows; each set's per-target nats sum to its
  documents' nats, and those give the record's bpb.
- --split: the held-out split of M1's store, 9545f2d205efd574 by the store's fingerprint.json
  (as mc.py reads it), read without scores for each document's targets and each target's bytes
  (`token_bytes.u16` at its id). The records must match it exactly.
- reads.npy: integers [targets], in the scorer's reading order: the summed training read count
  of the bigram and trigram rows read at the input position that predicts each target, from
  Billet's final `table/reads.npy` (S3's helper writes it).
- overlap.npy: [documents], fields `distinct` and `found`: each held-out document's distinct
  13-grams and how many of them occur in the store's training split (M8's scan writes it).
- mc.json, its `--samples` file and --set: `python -m metalforge.eval.mc --decontam DIR
  --compute-dtype float32 --samples FILE --out mc.json` over the prepared set --set, whose
  <task>.jsonl give each item's strings and gold index. Both records are float32 at seq_len
  2,048, have no --limit, share their tokenizer_sha256, and drop the same items; the arms score
  the same kept items, and each task's accuracy from the samples must equal its record's.
- metrics.jsonl: each run's training log (step, loss, batch hash), through its last step.
- summary.json: the timing probe's, written by its onoff.py (`--analyse`), whose SHA-256 the
  probe-rule comment on #235 records. Its design must be the rule's: a 5-minute warm-up, then 16
  six-minute blocks in the order ABBABAABBAABABBA, each block's first 3 steps dropped.
- --s6, --thirds: heldout.json records of the memorisation split, and integers [documents] of
  that split, 0, 1 or 2 for a document read in the first, middle or last third of training.
  Give them only if the dated amendment recorded the SHA-256 of S6's split writer before any Δ
  was computed: M1 runs memorisation only then, and otherwise reports it as not run.

Readouts, by M1's sections:
- Primary: Δ = bpb(Billet) − bpb(twin), final checkpoints, raw weights, where bpb = Σ nats ÷
  (ln 2 · Σ bytes) over the 7,314 documents. δ = max(0.01 nats per target in bpb, 4σ) with
  σ = 0.00108; Δ ≤ −δ: the table helps; Δ ≥ +δ: it hurts; otherwise no measurable difference.
  Equivalence is claimed if, and only if, |Δ| < 0.0018.
- EMA: Δ with EMA weights, in the same bands. Pinned: Δ at step 19,532, raw weights.
- Splits of the primary Δ, each in bpb over its own targets and bytes: (1) deciles of targets
  by reads.npy; (2) each document's first window; (3) documents with no 13-gram found in
  training, documents with any, and the latter by quartile of overlap share (found ÷ distinct).
  "None" includes documents too short to hold a 13-gram; the bins' summary counts them. Every
  bin is fixed from these score-free inputs before any score is read. An empty bin has no bpb.
- Zero-shot, raw weights, per task: McNemar's exact two-sided test on the headline accuracy
  (acc_norm for HellaSwag, acc for the others; LAMBADA's acc is every target greedy),
  Holm-adjusted over the five tasks at a family-wise 5%; Δ accuracy and Δ gold-answer bpb.
- Training loss: Billet − twin per step, over steps logged by both arms with finite losses and
  the same batch hash; its mean over five 1,000-step windows ending at 10, 25, 50, 80 and 100%
  of training. Steps left out are counted by reason.
- Memorisation (S6): Δ on the training documents, beside the held-out Δ, and each arm's
  training − held-out gap by third. Also given, and labelled exploratory: Δ by third, each arm's
  gap overall, and the difference of the arms' gaps, by third and overall.
- Per Mac-hour (M9): r and its 95% interval by the probe's rule (#235 comment 6543579907;
  onoff.py's crossover and t975, copied unchanged), recomputed from summary.json's pairs and
  checked against its result. k(s) = s′/s, where s′ is the first step at which the twin's smoothed
  training loss is at or below Billet's at step s, both smoothed by a centred 1,001-step mean of
  the finite losses, for s and s′ from 1,500 to 19,031; an s with no such s′ is left out and
  counted by reason; k is the median of k(s). The table leads in the hold if k is above r's
  interval, trails if below it, and otherwise neither.

Intervals: every Billet − twin figure except k gets a 95% percentile interval (NumPy's linear
rule at 2.5 and 97.5%) from 10,000 paired bootstrap resamples, labelled evaluation noise. They
resample held-out documents, zero-shot items (per task), training documents (within each third,
with the held-out documents resampled independently for the gaps) or moving blocks of 50
consecutive steps (20 per 1,000-step window, their starts drawn from all 951). Resample b draws
its indices with one call of `Generator.integers`; each family has its own stream from SEED,
20261101. NumPy may change those streams between versions (NEP 19), so the report records the
Python and NumPy versions.

Choices M1 leaves to this code:
- A window ending at p% of N steps holds steps ⌊pN⌋ − 1,000 to ⌊pN⌋ − 1: the 1,000 steps done
  when ⌊pN⌋ of the N steps are done. At N = 24,415 they start at 1,441, 5,103, 11,207, 18,532
  and 23,415, so the 80% window is the hold's last 1,000 steps and the 100% window training's.
- Deciles and quartiles: edge j of k bins is the ⌈jn/k⌉-th smallest value; a value equal to an
  edge goes to the bin below it, so tied values share a bin.
- A gold answer's bytes are the UTF-8 bytes of its continuation as scored, its leading space
  included (with any whitespace `encode_pair` moves from the context); a task's gold bpb is its
  items' total bits over their total bytes.
- A zero-shot item's margin (exploratory): its gold score minus the best other choice's, in the
  headline's units (Σ log p, or per character for acc_norm); LAMBADA has none.
- Δ accuracy is McNemar's test's own effect: (b − c) ÷ n, where b items of n are right for
  Billet only and c for the twin only. So it is reported with the test, not as exploratory.
- A step's loss is used whatever its skip flag, when it is finite in both arms.
- Equivalence is read for the primary Δ only. M1 reads EMA's Δ with "the same bands", and
  equivalence is a claim beside its three bands, not one of them.
- k(s): if the twin's smoothed loss is already at or below Billet's at s when s′ = 1,500, the
  range cannot show the step at which the twin got there, so s has no s′, as when the twin is
  not there by 19,031. Both kinds of s are left out and counted apart. The maintainer confirmed
  this rule on 2026-11-01.
- k(s): a smoothed loss is undefined where its 1,001-step window holds no finite loss. An s at
  which Billet's is undefined is left out and counted; an s′ at which the twin's is undefined is
  one at which the twin is not at or below Billet's.
- r's interval has onoff.py's degrees of freedom, the pairs used − 2: 6 with all 8 pairs, as the
  probe rule's text says, and 5 if a pair is void. The rule reads the probe "with the rule its
  code already applies, and no other", so the code's count is used.
- A resample that draws none of a split bin's bytes gives the bin no bpb. The bin's interval is
  the percentile interval of the other resamples, and the report counts the ones left out
  ("resamples left out, no bytes"): about 10,000 · e^−m of them for a bin whose bytes lie in m
  documents. An empty bin has no bpb and no interval.
- A split bin whose nats and bytes lie in one document has no interval and no count of
  resamples left out, and its interval's label says why ("one document"): every resample that
  draws the document gives the bin the same bpb. The maintainer decided this on 2026-11-01.
- Both mc records must have seq_len 2,048, as the held-out records must; a record with any
  other value, or none, is refused. The maintainer decided this on 2026-11-01.

Sources: M1 and the probe rule (#235, comments 6553969708 and 6543579907); #481 (the records);
metalforge/eval/heldout.py, mc.py and tasks.py; metalforge/train/log.py; the probe's onoff.py
(SHA-256 d55caf76…, as the probe rule records); docs/research/arch/wp-f/analysis.py (Holm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

M1 = "https://github.com/cogvera-labs/MetalForge/issues/235#issuecomment-6553969708"
PROBE_RULE = "https://github.com/cogvera-labs/MetalForge/issues/235#issuecomment-6543579907"

LN2 = math.log(2.0)
SIGMA = 0.00108  # bpb: M1's σ
NATS_BAR = 0.01  # nats per target: the bar's other term
EQUIVALENCE = 0.0018  # bpb
ALPHA = 0.05  # Holm's family-wise level
RESAMPLES = 10_000
SEED = 20_261_101
STREAMS = {"documents": 1, "pinned": 2, "items": 3, "steps": 4, "training documents": 5}
BLOCK, WINDOW, ENDS = 50, 1_000, (10, 25, 50, 80, 100)
HOLD, HALF = (1_000, 19_531), 500  # the hold's first and last step; the smoothing's half-width
SEQ_LEN = 2_048
FLOAT32 = "mlx.core.float32"  # what heldout.py and mc.py record for float32
TASKS = {
    "piqa": "acc",
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "lambada_openai": "acc",
    "winogrande": "acc",
}
DELIM = " "  # the delimiter that opens a multiple-choice continuation (tasks.py)
STORE = "9545f2d205efd574"  # M1's store: how its fingerprint.json's fingerprint starts
# M1's held-out set (in STORE) and checkpoints; "steps" is N, the run's steps.
EXPECT = {"docs": 7_314, "targets": 7_636_111, "bytes": 34_765_120, "steps": 24_415, "pinned": 19_532}
# The probe rule's design (PROBE.md; onoff.py's defaults), as summary.json records it.
DESIGN = {"order": "ABBABAABBAABABBA", "minutes": 6, "warmup_minutes": 5, "drop": 3}
# The probe rule's t quantiles (onoff.py's T975 and t975, copied unchanged).
T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228)
T975 += (2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086)
NOT_RUN = "not run: inputs not given"


class Refused(Exception):
    """An input fails a check, so nothing is computed from it."""


# ---------------------------------------------------------------- inputs


def sha256(path: Path) -> str:
    """The file's SHA-256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {e}") from None


def _lines(path: Path) -> list[Any]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {e}") from None


def _missing(path: Path) -> str:
    raise Refused(f"{path}: missing")


def _npy(path: Path) -> np.ndarray:
    try:
        return np.load(path)  # allow_pickle stays False
    except (OSError, ValueError) as e:
        raise Refused(f"{path}: {e}") from None


@dataclass(frozen=True)
class Layout:
    """The held-out split as the scorer reads it: counts and positions, no score."""

    doc_targets: np.ndarray  # [documents]
    doc_bytes: np.ndarray  # [documents]
    target_bytes: np.ndarray  # [targets], in reading order
    document: np.ndarray  # [targets]
    window: np.ndarray  # [targets]


def read_split(split: Path) -> Layout:
    """The held-out split of M1's store, read as metalforge.eval.heldout reads it.

    Documents come in the split's order, each [0] + ids, and a document's targets are its ids,
    so a target's bytes are token_bytes.u16 at its id. Window k of a document holds its targets
    kT to kT + T − 1 (T = 2,048).
    """
    fp = _json(split.parent / "fingerprint.json")
    if not isinstance(fp, dict) or not str(fp.get("fingerprint")).startswith(STORE):
        raise Refused(f"{split.parent}: not store {STORE} by its fingerprint.json")
    m, tb_path, root = _json(split / "manifest.json"), split.parent / "token_bytes.u16", split.parent / "manifest.json"
    try:
        tb = np.fromfile(tb_path, dtype="<u2").astype(np.int64)
        want = _json(root).get("outputs", {}).get("token_bytes_sha256") if root.exists() else None
        shards = [
            (np.fromfile(split / name, dtype="<u2").astype(np.int64), np.fromfile(split / f"{name[:-4]}.idx", dtype="<u8").astype(np.int64), int(n))
            for name, n in zip(m["shards"], m["tokens"], strict=True)
        ]
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise Refused(f"{split}: {e!r}") from None
    if want is not None and want != sha256(tb_path):
        raise Refused(f"{tb_path}: SHA-256 differs from the store manifest's")
    if m.get("dtype") != "uint16" or m.get("eos") != 0 or tb[:17].any():
        raise Refused(f"{split}: not an id-0-first uint16 split with no bytes on ids 0-16")
    counts, tbytes = [], []
    for ids, idx, n in shards:
        if len(ids) != n or len(idx) < 2 or idx[0] != 0 or idx[-1] != n or (np.diff(idx) <= 0).any():
            raise Refused(f"{split}: a shard's offsets do not match its tokens")
        if (ids[idx[:-1]] != 0).any() or (ids == 0).sum() != len(idx) - 1 or ids.max() >= len(tb):
            raise Refused(f"{split}: a document does not open with id 0, or an id is out of range")
        body = np.ones(n, bool)
        body[idx[:-1]] = False
        counts.append(np.diff(idx) - 1)
        tbytes.append(tb[ids[body]])
    doc_targets, target_bytes = np.concatenate(counts), np.concatenate(tbytes)
    docs = len(doc_targets)
    document = np.repeat(np.arange(docs), doc_targets)
    first = np.repeat(np.cumsum(doc_targets) - doc_targets, doc_targets)
    window = (np.arange(len(target_bytes)) - first) // SEQ_LEN
    doc_bytes = np.bincount(document, weights=target_bytes, minlength=docs).astype(np.int64)
    return Layout(doc_targets, doc_bytes, target_bytes, document, window)


@dataclass(frozen=True)
class Scored:
    """One heldout.json record with its documents.npy and targets.npy."""

    path: Path
    rec: dict[str, Any]
    documents: np.ndarray
    targets: np.ndarray


def read_scored(path: Path, table: str, sets: tuple[str, ...] = ("raw", "ema")) -> Scored:
    """A held-out record and its records files (#481), checked.

    Raises:
        Refused: not float32, the wrong table, a --mask record, a hash that differs, or nats
            that do not sum to the documents' nats and the record's bpb.
    """
    rec = _json(path)
    if rec.get("dtype") != FLOAT32:
        raise Refused(f"{path}: dtype {rec.get('dtype')!r}, not {FLOAT32}")
    if rec.get("table") != table or "mask" in rec:
        raise Refused(f"{path}: table {rec.get('table')!r} (want {table!r}), or a --mask record")
    try:
        d, want = Path(rec["records"]["dir"]), dict(rec["records"]["sha256"])
    except (KeyError, TypeError, ValueError):
        raise Refused(f"{path}: no records block (score with --records)") from None
    # heldout.py records --records as given: a relative DIR is the scorer's working directory's
    # (tried first) or, for a moved run folder, the record's own.
    d = next((c for c in (d, path.parent / d) if (c / "documents.npy").is_file()), d)
    for f in ("documents.npy", "targets.npy"):
        if not (d / f).is_file() or sha256(d / f) != want.get(f):
            raise Refused(f"{d / f}: missing, or its SHA-256 differs from {path}'s")
    docs, tg = _npy(d / "documents.npy"), _npy(d / "targets.npy")
    n = len(docs)
    counts = (n, len(tg), int(docs["bytes"].sum()))
    if (rec.get("docs"), rec.get("targets"), rec.get("bytes")) != counts or len(tg) != int(docs["targets"].sum()):
        raise Refused(f"{path}: its counts differ from its records'")
    for s in sets:
        if s not in (docs.dtype.names or ()) or s not in (tg.dtype.names or ()) or "bpb" not in rec.get(s, {}):
            raise Refused(f"{path}: no {s!r} set")
        per_doc = np.bincount(tg["document"], weights=tg[s].astype(np.float64), minlength=n)
        if not np.allclose(per_doc, docs[s], rtol=0.0, atol=1e-9):
            raise Refused(f"{path}: the {s} targets' nats do not sum to their documents'")
        bpb = float(docs[s].sum() / (LN2 * docs["bytes"].sum()))
        if abs(bpb - rec[s]["bpb"]) > 1e-10:
            raise Refused(f"{path}: the {s} documents give bpb {bpb}, the record {rec[s]['bpb']}")
    return Scored(path, rec, docs, tg)


def check_pair(b: Scored, t: Scored, step: int, layout: Layout | None = None) -> None:
    """Billet's and the twin's records score the same documents, targets and windows at `step`."""
    for s in (b, t):
        if s.rec.get("step") != step or s.rec.get("seq_len") != SEQ_LEN:
            raise Refused(f"{s.path}: step {s.rec.get('step')}, T {s.rec.get('seq_len')}; want {step}, {SEQ_LEN}")
    same = all(np.array_equal(b.documents[f], t.documents[f]) for f in ("targets", "bytes"))
    same &= all(np.array_equal(b.targets[f], t.targets[f]) for f in ("document", "window"))
    if not same:
        raise Refused(f"{b.path} and {t.path} score different documents or targets")
    if layout is not None and not (
        np.array_equal(b.documents["targets"], layout.doc_targets)
        and np.array_equal(b.documents["bytes"], layout.doc_bytes)
        and np.array_equal(b.targets["document"], layout.document)
        and np.array_equal(b.targets["window"], layout.window)
    ):
        raise Refused(f"{b.path}: its documents, bytes or windows differ from the split's")


# ---------------------------------------------------------------- bins (no score)


def lower_bins(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Bins 0 to k − 1 of x and their k − 1 edges.

    Edge j is the ⌈jn/k⌉-th smallest value (the lower empirical j/k quantile); a value equal to
    an edge goes to the bin below it.
    """
    n = len(x)
    edges = np.sort(x)[[-(-j * n // k) - 1 for j in range(1, k)]]
    return np.searchsorted(edges, x, side="left"), edges


@dataclass(frozen=True)
class Bins:
    """The splits' bins, fixed from score-free inputs before any score is read."""

    first: np.ndarray  # [targets] bool: in its document's first window
    reads: np.ndarray | None  # [targets] decile 0-9
    reads_edges: np.ndarray | None
    overlap: dict[str, np.ndarray] | None  # group: [documents] bool
    share_edges: np.ndarray | None
    no_13grams: int | None = None  # documents with no 13-gram to find (distinct 0): all in "none"

    def summary(self) -> dict[str, Any]:
        """What each bin holds, and one SHA-256 over every bin array."""
        h = hashlib.sha256(self.first.tobytes())
        out: dict[str, Any] = {"first window targets": int(self.first.sum())}
        if self.reads is not None and self.reads_edges is not None:
            h.update(self.reads.tobytes())
            out["read-count deciles"] = {
                "targets": np.bincount(self.reads, minlength=10).tolist(),
                "edges": self.reads_edges.tolist(),
            }
        if self.overlap is not None and self.share_edges is not None:
            for v in self.overlap.values():
                h.update(v.tobytes())
            out["13-gram overlap"] = {
                "documents": {g: int(v.sum()) for g, v in self.overlap.items()},
                "documents in none with no 13-gram to find": self.no_13grams,
                "share edges": self.share_edges.tolist(),
            }
        out["sha256"] = h.hexdigest()
        return out


def fix_bins(layout: Layout, reads: Path | None, overlap: Path | None) -> Bins:
    """Every split's bins, from the split's layout, reads.npy and overlap.npy only."""
    first = layout.window == 0
    deciles = decile_edges = groups = share_edges = no_13grams = None
    if reads is not None:
        x = _npy(reads)
        if x.ndim != 1 or len(x) != len(layout.window) or x.dtype.kind not in "iu" or (x < 0).any():
            raise Refused(f"{reads}: not one non-negative integer per held-out target")
        deciles, decile_edges = lower_bins(x, 10)
    if overlap is not None:
        o = _npy(overlap)
        if len(o) != len(layout.doc_targets) or not {"distinct", "found"} <= set(o.dtype.names or ()):
            raise Refused(f"{overlap}: not one (distinct, found) row per held-out document")
        distinct, found = o["distinct"].astype(np.int64), o["found"].astype(np.int64)
        if (found < 0).any() or (found > distinct).any():
            raise Refused(f"{overlap}: found must lie in 0..distinct")
        hit, no_13grams = found > 0, int((distinct == 0).sum())
        empty = (np.zeros(0, np.int64), np.zeros(0))
        quartile, share_edges = lower_bins(found[hit] / distinct[hit], 4) if hit.any() else empty
        groups = {"none": ~hit, "any": hit}
        for j in range(4):
            groups[f"any, quartile {j + 1}"] = np.zeros(len(hit), bool)
            groups[f"any, quartile {j + 1}"][np.flatnonzero(hit)[quartile == j]] = True
    for a in (first, deciles, *(groups or {}).values()):
        if a is not None:
            a.flags.writeable = False
    return Bins(first, deciles, decile_edges, groups, share_edges, no_13grams)


# ---------------------------------------------------------------- statistics


def rng(stream: str, index: int = 0) -> np.random.Generator:
    """The fixed random stream of one family of resamples (and its index-th member)."""
    return np.random.default_rng([SEED, STREAMS[stream], index])


def resample_sums(x: np.ndarray, gen: np.random.Generator) -> np.ndarray:
    """Column sums of x [n, m] over each of RESAMPLES draws of n rows with replacement."""
    n = len(x)
    out = np.empty((RESAMPLES, x.shape[1]))
    for a in range(0, RESAMPLES, 256):
        rows = range(a, min(a + 256, RESAMPLES))
        counts = np.stack([np.bincount(gen.integers(0, n, n), minlength=n) for _ in rows])
        out[a : a + len(rows)] = counts @ x
    return out


def interval(reps: np.ndarray) -> list[float] | None:
    """The 95% percentile interval of the replicates; None if there are none or any is undefined."""
    if not len(reps) or not np.isfinite(reps).all():
        return None
    return [float(v) for v in np.percentile(reps, [2.5, 97.5], method="linear")]


def noise(of: str) -> str:
    """An interval's label."""
    return f"evaluation noise only: 95% percentile interval of {RESAMPLES:,} paired bootstrap resamples of {of}"


def mcnemar(b: int, c: int) -> float:
    """McNemar's exact two-sided p: 2·P(X ≤ min(b, c)), X ~ Binomial(b + c, 1/2), at most 1."""
    n = b + c
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)  # exact integers, one rounding


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjusted p-values (as docs/research/arch/wp-f/analysis.py's holm)."""
    out, run = {}, 0.0
    order = sorted(p, key=lambda k: (p[k], k))
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * p[k]))
        out[k] = run
    return out


def block_means(d: np.ndarray, kept: np.ndarray, gen: np.random.Generator) -> np.ndarray:
    """Moving-block resamples of the mean of d over the kept steps.

    Each draws len(d) / BLOCK blocks of BLOCK consecutive steps, their starts uniform over all
    len(d) − BLOCK + 1; a step left out adds nothing to its block.
    """
    n, k = len(d), len(d) // BLOCK
    if n % BLOCK:
        raise ValueError(f"{n} steps are not whole blocks of {BLOCK}")
    view = np.lib.stride_tricks.sliding_window_view
    sums = view(np.where(kept, d, 0.0), BLOCK).sum(axis=1)
    used = view(kept.astype(np.int64), BLOCK).sum(axis=1)
    out = np.empty(RESAMPLES)
    for r in range(RESAMPLES):
        s = gen.integers(0, n - BLOCK + 1, k)
        out[r] = sums[s].sum() / used[s].sum() if used[s].sum() else np.nan
    return out


def t975(df: int) -> float:
    """The probe rule's 97.5% t quantile (onoff.py's t975)."""
    if df <= len(T975):
        return T975[df - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def crossover(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The probe rule: the 2×2 crossover estimate of the on/off ratio and its 95% t-interval.

    onoff.py's crossover, its body copied unchanged: the mean of log r over the pairs that ran
    the table first, and over those that ran it second, averaged; a t-interval on pairs − 2
    degrees of freedom.
    """
    first = [math.log(p["ratio"]) for p in pairs if p["table_first"]]
    second = [math.log(p["ratio"]) for p in pairs if not p["table_first"]]
    res: dict[str, Any] = {"pairs_used": len(pairs), "table_first": len(first),
                           "table_second": len(second)}
    if not first or not second:
        res["note"] = "needs at least one pair of each order"
        return res
    est = (statistics.fmean(first) + statistics.fmean(second)) / 2
    res["ratio"] = math.exp(est)
    res["order_effect"] = math.exp((statistics.fmean(second) - statistics.fmean(first)) / 2)
    df = len(pairs) - 2
    if df < 1:
        res["note"] = "an interval needs at least 3 pairs"
        return res
    ss = sum((d - statistics.fmean(first)) ** 2 for d in first)
    ss += sum((d - statistics.fmean(second)) ** 2 for d in second)
    se = math.sqrt(ss / df / 4 * (1 / len(first) + 1 / len(second)))
    t = t975(df)
    res.update(ci95=[math.exp(est - t * se), math.exp(est + t * se)], df=df, t=t, se_log=se)
    return res


def bar(targets: int, nbytes: int) -> dict[str, float]:
    """M1's δ = max(0.01 nats per target in bpb, 4σ), with both terms."""
    nats_term = NATS_BAR * targets / (LN2 * nbytes)
    terms = {"sigma": SIGMA, "4 sigma": 4 * SIGMA, "0.01 nats per target": nats_term}
    return terms | {"delta": max(nats_term, 4 * SIGMA)}


def band(delta: float, edge: float) -> str:
    """M1's outcome band for a raw or EMA Δ, at δ = `edge`."""
    if delta <= -edge:
        return "the table helps"
    if delta >= edge:
        return "the table hurts"
    return "no measurable difference"


# ---------------------------------------------------------------- held-out readouts


class Ratios:
    """Per-document columns: Billet's nats, the twin's nats and the bytes of each named figure."""

    def __init__(self) -> None:
        self.cols: list[np.ndarray] = []
        self.at: dict[str, int] = {}

    def add(self, name: str, nb: np.ndarray, nt: np.ndarray, nbytes: np.ndarray) -> None:
        """A figure's per-document nats for Billet and the twin, and its bytes."""
        self.at[name] = len(self.cols)
        self.cols += [np.asarray(c, dtype=np.float64) for c in (nb, nt, nbytes)]

    def figures(self, gen: np.random.Generator) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
        """Each figure's bpb and Δ with its interval, and each figure's per-arm bpb replicates.

        A figure with no bytes (an empty bin) has no bpb: its figures are None. A resample that
        draws none of a figure's bytes gives it no bpb either: its interval is over the other
        resamples, and it counts the ones left out. A figure whose nats and bytes lie in one
        document has no interval, and says so: every resample that draws it gives the same bpb.
        """
        x = np.column_stack(self.cols)
        point, reps = x.sum(axis=0), resample_sums(x, gen)
        out, arm = {}, {}
        for name, i in self.at.items():
            b, t, y = point[i : i + 3]
            has = reps[:, i + 2] > 0  # the resamples that draw some of its bytes
            one = np.count_nonzero(x[:, i : i + 3].any(axis=1)) == 1  # one document holds it
            with np.errstate(divide="ignore", invalid="ignore"):
                rb, rt = reps[:, i] / (LN2 * reps[:, i + 2]), reps[:, i + 1] / (LN2 * reps[:, i + 2])
            out[name] = {
                "billet_bpb": float(b / (LN2 * y)) if y else None,
                "twin_bpb": float(t / (LN2 * y)) if y else None,
                "delta": float((b - t) / (LN2 * y)) if y else None,
                "ci95": None if one else interval((rb - rt)[has]),
                "resamples left out, no bytes": None if one else int((~has).sum()),
                "interval": "none: one document, and every resample that draws it gives the same bpb" if one else noise("held-out documents"),
                "bytes": int(y),
            }
            arm[name] = np.column_stack([rb, rt])
        return out, arm


def heldout(b: Scored, t: Scored, bins: Bins, target_bytes: np.ndarray, edge: float) -> tuple[dict[str, Any], np.ndarray]:
    """The primary, EMA and split readouts, and the primary's per-arm bpb replicates."""
    db, dt, docs = b.documents, t.documents, len(b.documents)
    doc = b.targets["document"].astype(np.int64)
    per_target = [s.targets["raw"].astype(np.float64) for s in (b, t)] + [target_bytes.astype(np.float64)]
    r = Ratios()
    r.add("primary", db["raw"], dt["raw"], db["bytes"])
    r.add("ema", db["ema"], dt["ema"], db["bytes"])
    if bins.reads is not None:
        g = doc * 10 + bins.reads
        sb, st, sy = (np.bincount(g, weights=w, minlength=docs * 10).reshape(docs, 10) for w in per_target)
        for j in range(10):
            r.add(f"read-count decile {j + 1}", sb[:, j], st[:, j], sy[:, j])
    r.add("first windows", *(np.bincount(doc, weights=w * bins.first, minlength=docs) for w in per_target))
    for name, m in (bins.overlap or {}).items():
        r.add(f"13-gram overlap: {name}", db["raw"] * m, dt["raw"] * m, db["bytes"] * m)
    figs, arm = r.figures(rng("documents"))
    p, e = figs.pop("primary"), figs.pop("ema")
    p |= {"band": band(p["delta"], edge), "status": "read against δ by M1's bands"}
    p["equivalence"] = {"edge": EQUIVALENCE, "claimed": abs(p["delta"]) < EQUIVALENCE}
    e |= {"band": band(e["delta"], edge), "status": "read against δ by M1's bands"}
    for v in figs.values():
        v["status"] = "described, not tested" if v["bytes"] else "empty bin: no bytes, so no bpb"
    if bins.reads is None:
        figs["read-count deciles"] = NOT_RUN
    if bins.overlap is None:
        figs["13-gram overlap"] = NOT_RUN
    return {"primary": p, "ema": e, "splits": figs}, arm["primary"]


def pinned(b: Scored, t: Scored) -> dict[str, Any]:
    """Δ at both arms' step-19,532 checkpoints, raw weights: described, not tested."""
    r = Ratios()
    r.add("pinned", b.documents["raw"], t.documents["raw"], b.documents["bytes"])
    figs, _ = r.figures(rng("pinned"))
    return figs["pinned"] | {"status": "described, not tested"}


def memorisation(b: Scored, t: Scored, thirds: np.ndarray, primary: dict[str, Any], held: np.ndarray) -> dict[str, Any]:
    """S6: Δ on training documents by third and overall, and each arm's training − held-out gap.

    M1 names the overall Δ and each arm's gap by third; the other figures are labelled
    exploratory. Training documents are resampled within their third, and the held-out
    replicates are the primary's, drawn independently of them.
    """
    if thirds.ndim != 1 or len(thirds) != len(b.documents) or thirds.dtype.kind not in "iu" or set(np.unique(thirds).tolist()) != {0, 1, 2}:
        raise Refused("thirds: not 0, 1 or 2 for each memorisation document, with every third used")
    groups = {"first third": thirds == 0, "middle third": thirds == 1, "last third": thirds == 2}
    cols = [np.asarray(c, dtype=np.float64) for c in (b.documents["raw"], t.documents["raw"], b.documents["bytes"])]
    point, reps = {}, {}
    for j, (name, m) in enumerate(groups.items()):
        x = np.column_stack([c[m] for c in cols])
        point[name], reps[name] = x.sum(axis=0), resample_sums(x, rng("training documents", j))
    point["all"], reps["all"] = sum(point[g] for g in groups), sum(reps[g] for g in groups)
    hb, ht = primary["billet_bpb"], primary["twin_bpb"]
    out: dict[str, Any] = {
        "held-out delta": primary["delta"],
        "status": "described, not tested",
        "interval": noise("training documents within each third, and of held-out documents"),
        "exploratory": "each third's delta, the gaps of 'all' and every gap delta, with their intervals: M1 does not name them",
    }
    for name in (*groups, "all"):
        (sb, st, sy), r = point[name], reps[name]
        rb, rt = r[:, 0] / (LN2 * r[:, 2]), r[:, 1] / (LN2 * r[:, 2])
        train_b, train_t = float(sb / (LN2 * sy)), float(st / (LN2 * sy))
        out[name] = {
            "documents": int(groups[name].sum()) if name in groups else len(thirds),
            "billet_bpb": train_b,
            "twin_bpb": train_t,
            "delta": train_b - train_t,
            "ci95": interval(rb - rt),
            "gap billet": train_b - hb,
            "gap billet ci95": interval(rb - held[:, 0]),
            "gap twin": train_t - ht,
            "gap twin ci95": interval(rt - held[:, 1]),
            "gap delta": (train_b - hb) - (train_t - ht),
            "gap delta ci95": interval((rb - held[:, 0]) - (rt - held[:, 1])),
        }
    return out


# ---------------------------------------------------------------- zero-shot


def scored_continuation(context: str, continuation: str) -> str:
    """The continuation as tasks.encode_pair scores it: the context's trailing whitespace joins it."""
    n = len(context) - len(context.rstrip())
    return context[len(context) - n :] + continuation if n else continuation


def read_set(root: Path) -> tuple[Any, dict[str, dict[str, dict[str, Any]]]]:
    """The set's id and each task's items by doc_id, each file checked against the manifest."""
    m = _json(root / "manifest.json")
    items: dict[str, dict[str, dict[str, Any]]] = {}
    for task in TASKS:
        path, row = root / f"{task}.jsonl", m.get("tasks", {}).get(task) or {}
        if not path.is_file() or sha256(path) != row.get("sha256"):
            raise Refused(f"{path}: missing, or its SHA-256 differs from the manifest's")
        items[task] = {it["doc_id"]: it for it in _lines(path)}
        if len(items[task]) != row.get("n"):
            raise Refused(f"{path}: {len(items[task])} distinct items, the manifest's n {row.get('n')}")
    return m.get("set_id"), items


def read_samples(path: Path, term: str) -> dict[tuple[str, str], dict[str, Any]]:
    """The raw-weights rows of an mc --samples file for `term` (table, off), by (task, doc_id)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _lines(path):
        if row.get("weights") == "raw" and row.get("term") == term:
            key = (row["task"], row["doc_id"])
            if key in out:
                raise Refused(f"{path}: {key} appears twice")
            out[key] = row
    return out


def check_mc(path: Path, rec: dict[str, Any], table: str, set_id: Any) -> None:
    """An mc record M1 can use: float32 at seq_len 2,048, decontaminated, every item, at the final step."""
    if rec.get("dtype") != FLOAT32 or rec.get("table") != table or rec.get("limit") is not None:
        raise Refused(f"{path}: not float32, with the table {table} and no --limit")
    if rec.get("seq_len") != SEQ_LEN:
        raise Refused(f"{path}: seq_len {rec.get('seq_len')}; want {SEQ_LEN}")
    decontam = rec.get("decontam") or {}
    if not all(t in decontam for t in TASKS) or rec.get("set_id") != set_id or rec.get("step") != EXPECT["steps"]:
        raise Refused(f"{path}: not decontaminated for every task, another set, or not step {EXPECT['steps']}")
    for task, headline in TASKS.items():
        if rec.get("raw", {}).get(task, {}).get("headline") != headline:
            raise Refused(f"{path}: {task}'s headline is not {headline}")


def item_scores(task: str, item: dict[str, Any], row: dict[str, Any]) -> tuple[bool, float, int, float]:
    """(correct by the headline, gold bits, gold bytes, margin) of one scored item."""
    reqs, gold = item["requests"], item["gold"]
    ll = np.asarray(row["ll"], dtype=np.float64)
    if len(ll) != len(reqs) or len(row["greedy"]) != len(reqs) or not np.isfinite(ll).all():
        raise Refused(f"{task} item {item['doc_id']}: its samples do not fit its requests")
    q = reqs[gold]
    nbytes = len(scored_continuation(q["context"], q["continuation"]).encode())
    bits = float(-ll[gold] / LN2)
    if task == "lambada_openai":
        return bool(row["greedy"][0]), bits, nbytes, math.nan
    score = ll
    if TASKS[task] == "acc_norm":
        score = ll / np.array([float(len(r["continuation"][len(DELIM) :])) for r in reqs])
    margin = float(score[gold] - np.delete(score, gold).max())
    return int(np.argmax(score)) == gold, bits, nbytes, margin


def zero_shot(mc: list[Path], samples: list[Path], set_root: Path) -> dict[str, Any]:
    """Per task: McNemar on the headline accuracy (Holm over five), Δ gold bpb, Δ margin."""
    set_id, items = read_set(set_root)
    recs = [_json(p) for p in mc]
    for p, rec, table in zip(mc, recs, ("on", "off"), strict=True):
        check_mc(p, rec, table, set_id)
    drop = [{k: v for k, v in r["decontam"].items() if k != "dir"} for r in recs]
    if drop[0] != drop[1]:
        raise Refused("the two mc records drop different items")
    if recs[0].get("tokenizer_sha256") is None or recs[0].get("tokenizer_sha256") != recs[1].get("tokenizer_sha256"):
        raise Refused("the two mc records differ in tokenizer_sha256, or lack it")
    rows = [read_samples(samples[0], "table"), read_samples(samples[1], "off")]
    tasks: dict[str, Any] = {}
    for i, task in enumerate(TASKS):
        ids = [d for d in items[task] if (task, d) in rows[0]]
        if any({d for k, d in rw if k == task} != set(ids) for rw in rows) or len(ids) != drop[0][task].get("kept"):
            raise Refused(f"{task}: the arms scored different items, or not the kept ones")
        arm = []
        for rec, rw, p in zip(recs, rows, samples, strict=True):
            a = np.array([item_scores(task, items[task][d], rw[(task, d)]) for d in ids])
            if float(a[:, 0].mean()) != rec["raw"][task][TASKS[task]]:
                raise Refused(f"{p}: {task}'s accuracy differs from its record's")
            arm.append(a)
        (cb, bb, yb, mb), (ct, bt, _, mt) = arm[0].T, arm[1].T
        cols = [cb - ct, bb - bt, yb] + ([mb - mt] if task != "lambada_openai" else [])
        reps = resample_sums(np.column_stack(cols), rng("items", i))
        n, only_b, only_t = len(ids), int(((cb == 1) & (ct == 0)).sum()), int(((cb == 0) & (ct == 1)).sum())
        tasks[task] = {
            "items": n,
            "headline": TASKS[task],
            "billet_acc": float(cb.mean()),
            "twin_acc": float(ct.mean()),
            "delta_acc": float((cb - ct).mean()),
            "delta_acc ci95": interval(reps[:, 0] / n),
            "right for Billet only": only_b,
            "right for the twin only": only_t,
            "p": mcnemar(only_b, only_t),
            "gold bpb billet": float(bb.sum() / yb.sum()),
            "gold bpb twin": float(bt.sum() / yb.sum()),
            "gold bpb delta": float((bb - bt).sum() / yb.sum()),
            "gold bpb delta ci95": interval(reps[:, 1] / reps[:, 2]),
            "interval": noise("zero-shot items"),
        }
        if task != "lambada_openai":
            tasks[task]["margin delta (exploratory)"] = float((mb - mt).mean())
            tasks[task]["margin delta ci95 (exploratory)"] = interval(reps[:, 3] / n)
    adjusted = holm({t: tasks[t]["p"] for t in TASKS})
    for t in TASKS:
        tasks[t] |= {"p holm": adjusted[t], "difference claimed": adjusted[t] <= ALPHA}
    claims = "only the Holm-adjusted McNemar test claims a difference; the other figures are described"
    return {"tasks": tasks, "family": f"Holm over the five tasks at {ALPHA:.0%} family-wise", "claims": claims}


# ---------------------------------------------------------------- training loss and M9


@dataclass(frozen=True)
class Log:
    """A run's training log by step: loss (NaN where absent) and batch hash (None where absent)."""

    loss: np.ndarray
    present: np.ndarray
    batch: np.ndarray


def read_log(path: Path) -> Log:
    """metrics.jsonl over the run's N steps; refuses a run that has not logged its last step."""
    n = EXPECT["steps"]
    loss, present, batch = np.full(n, np.nan), np.zeros(n, bool), np.full(n, None, dtype=object)
    for r in _lines(path):
        try:
            s, value, h = int(r["step"]), float(r["loss"]), r["batch"]
        except (KeyError, TypeError, ValueError):
            raise Refused(f"{path}: a line that is not a step record") from None
        if not isinstance(h, str):
            raise Refused(f"{path}: step {s} has no batch hash")
        if not 0 <= s < n or present[s]:
            raise Refused(f"{path}: step {s} is outside 0..{n - 1} or repeated")
        present[s], loss[s], batch[s] = True, value, h
    if not present[n - 1]:
        raise Refused(f"{path}: no step {n - 1}, the last; the run has not ended")
    return Log(loss, present, batch)


def training_loss(lb: Log, lt: Log) -> dict[str, Any]:
    """Billet − twin per step on the same batch, over 1,000-step windows ending at 10-100%."""
    n = len(lb.loss)
    missing = ~(lb.present & lt.present)
    nonfinite = ~missing & ~(np.isfinite(lb.loss) & np.isfinite(lt.loss))
    differs = ~missing & ~nonfinite & (lb.batch != lt.batch)
    kept = ~(missing | nonfinite | differs)
    d = np.where(kept, lb.loss - lt.loss, 0.0)

    def counts(w: slice) -> dict[str, int]:
        return {
            "kept": int(kept[w].sum()),
            "left out, batch hashes differ": int(differs[w].sum()),
            "left out, missing in an arm": int(missing[w].sum()),
            "left out, loss not finite": int(nonfinite[w].sum()),
        }

    windows = {}
    for i, pct in enumerate(ENDS):
        end = n * pct // 100
        w, k = slice(end - WINDOW, end), kept[end - WINDOW : end]
        windows[f"{pct}%"] = {
            "steps": [end - WINDOW, end - 1],
            "delta": float(d[w][k].mean()) if k.any() else None,
            "ci95": interval(block_means(d[w], k, rng("steps", i))) if k.any() else None,
            **counts(w),
        }
    return {
        "units": "nats per token",
        "status": "described, not tested",
        "interval": noise(f"{BLOCK}-step moving blocks of steps"),
        "all steps": counts(slice(0, n)),
        "windows": windows,
    }


def smoothed(loss: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The mean of the finite losses over steps s − 500 to s + 500, at each s; NaN if none."""
    ok = np.isfinite(loss)
    cv = np.concatenate([[0.0], np.cumsum(np.where(ok, loss, 0.0))])
    cn = np.concatenate([[0], np.cumsum(ok)])
    num, cnt = cv[s + HALF + 1] - cv[s - HALF], cn[s + HALF + 1] - cn[s - HALF]
    return np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)


def multiplier(lb: Log, lt: Log) -> dict[str, Any]:
    """M9's k: the median over s of k(s) = s′/s, with s and s′ from 1,500 to 19,031.

    An s has no s′ if the twin's smoothed loss is already at or below Billet's at s when s′ =
    1,500 (the range cannot show when it got there), if it is not there by 19,031, or if
    Billet's smoothed loss at s is undefined. The twin is not there at an s′ where its own
    smoothed loss is undefined.
    """
    s = np.arange(HOLD[0] + HALF, HOLD[1] - HALF + 1)
    mb, mt = smoothed(lb.loss, s), smoothed(lt.loss, s)
    low = np.minimum.accumulate(np.where(np.isfinite(mt), mt, np.inf))  # the twin's lowest so far
    j = np.searchsorted(-low, -mb, side="left")  # the first s′ with the twin at or below Billet
    undefined = ~np.isfinite(mb)
    before, after = ~undefined & (j == 0), ~undefined & (j == len(s))
    used = ~(undefined | before | after)
    ks = s[j[used]] / s[used]
    return {
        "k": float(np.median(ks)) if used.any() else None,
        "s from": int(s[0]),
        "s to": int(s[-1]),
        "s used": int(used.sum()),
        "s left out (no s')": int((~used).sum()),
        "left out, the twin at or below already at s' = 1,500": int(before.sum()),
        "left out, the twin not at or below by s' = 19,031": int(after.sum()),
        "left out, Billet's smoothed loss undefined": int(undefined.sum()),
        "status": "no interval (M1)",
    }


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-12)
    return a == b


def per_mac_hour(lb: Log, lt: Log, summary_path: Path) -> dict[str, Any]:
    """r by the probe's rule, checked against summary.json and the rule's design; k; and the reading."""
    summary = _json(summary_path)
    try:
        if any(summary["design"].get(key) != value for key, value in DESIGN.items()):
            raise Refused(f"{summary_path}: its design (order, block minutes, warm-up, steps dropped) is not the rule's")
        used = [p for p in summary["pairs"] if not p["void"]]
        if not all(_close(p["ratio"], p["on_mean_s"] / p["off_mean_s"]) for p in used):
            raise Refused(f"{summary_path}: a pair's ratio is not its on ÷ off mean")
        r = crossover(used)
        given = summary["result"]
    except (AttributeError, KeyError, TypeError, ZeroDivisionError, ValueError) as e:
        raise Refused(f"{summary_path}: {e!r}") from None
    if any(not _close(r.get(key), given.get(key)) for key in ("pairs_used", "ratio", "ci95", "df")):
        raise Refused(f"{summary_path}: its result differs from its pairs under the rule")
    r |= {"rule": PROBE_RULE, "interval": "the probe rule's 95% t-interval over its block pairs"}
    k = multiplier(lb, lt)
    out = {"r": r, "probe checks": summary.get("checks"), "k": k}
    if "ci95" not in r or k["k"] is None:
        out["reading"] = "not run: r has no interval, or no s has an s′"
    elif k["k"] > r["ci95"][1]:
        out["reading"] = "the table leads per Mac-hour in the hold"
    elif k["k"] < r["ci95"][0]:
        out["reading"] = "the table trails per Mac-hour in the hold"
    else:
        out["reading"] = "neither leads per Mac-hour in the hold"
    return out


# ---------------------------------------------------------------- main


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="analysis.py", description="M1's readouts for Billet and its twin (#235).")
    two: dict[str, Any] = {"nargs": 2, "type": Path, "metavar": ("BILLET", "TWIN")}
    p.add_argument("--out", type=Path, required=True, help="the report (JSON)")
    p.add_argument("--heldout", required=True, help="final heldout.json records", **two)
    p.add_argument("--split", type=Path, required=True, help="the store's held-out split")
    p.add_argument("--pinned", help="step-19,532 heldout.json records", **two)
    p.add_argument("--reads", type=Path, help="per-target read counts (S3)")
    p.add_argument("--overlap", type=Path, help="per-document 13-gram counts (M8)")
    p.add_argument("--mc", help="mc.json records", **two)
    p.add_argument("--samples", help="their --samples files", **two)
    p.add_argument("--set", type=Path, help="the prepared set they scored")
    p.add_argument("--metrics", help="metrics.jsonl training logs", **two)
    p.add_argument("--probe", type=Path, help="the timing probe's summary.json")
    p.add_argument("--s6", help="memorisation heldout.json records", **two)
    p.add_argument("--thirds", type=Path, help="each memorisation document's third (0-2)")
    return p


def _pair(paths: list[Path], step: int, layout: Layout | None, sets: tuple[str, ...]) -> tuple[Scored, Scored]:
    b, t = read_scored(paths[0], "on", sets), read_scored(paths[1], "off", sets)
    check_pair(b, t, step, layout)
    return b, t


def run(a: argparse.Namespace) -> dict[str, Any]:
    """Every readout M1 names whose inputs are given; the bins are fixed before any score is read."""
    if (a.mc is None) != (a.samples is None) or (a.mc is None) != (a.set is None):
        raise Refused("--mc, --samples and --set go together")
    if (a.s6 is None) != (a.thirds is None) or (a.probe is not None and a.metrics is None):
        raise Refused("--s6 needs --thirds, and --probe needs --metrics")
    layout = read_split(a.split)
    bins = fix_bins(layout, a.reads, a.overlap)
    files = [a.split / "manifest.json", a.split.parent / "fingerprint.json", a.split.parent / "token_bytes.u16"]
    files += [a.set / "manifest.json"] if a.set else []
    for v in vars(a).values():
        files += v if isinstance(v, list) else [v] if isinstance(v, Path) and v not in (a.out, a.split, a.set) else []
    report: dict[str, Any] = {
        "what": "M1's readouts for MetalForge Billet and its twin",
        "m1": M1,
        "analysis sha256": sha256(Path(__file__)),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "inputs": {str(p): sha256(p) if p.is_file() else _missing(p) for p in files},
        "bins": bins.summary(),
    }
    b, t = _pair(a.heldout, EXPECT["steps"], layout, ("raw", "ema"))
    if (len(b.documents), len(b.targets), int(b.documents["bytes"].sum())) != (EXPECT["docs"], EXPECT["targets"], EXPECT["bytes"]):
        raise Refused(f"{b.path}: not the {EXPECT['docs']:,} held-out documents M1 names")
    report["bar"] = bar(EXPECT["targets"], EXPECT["bytes"])
    held, held_reps = heldout(b, t, bins, layout.target_bytes, report["bar"]["delta"])
    report |= held
    report["pinned"] = pinned(*_pair(a.pinned, EXPECT["pinned"], layout, ("raw",))) if a.pinned else NOT_RUN
    report["zero_shot"] = zero_shot(a.mc, a.samples, a.set) if a.mc else NOT_RUN
    report["training_loss"] = report["per_mac_hour"] = NOT_RUN
    if a.metrics:
        lb, lt = read_log(a.metrics[0]), read_log(a.metrics[1])
        report["training_loss"] = training_loss(lb, lt)
        if a.probe:
            report["per_mac_hour"] = per_mac_hour(lb, lt, a.probe)
    report["memorisation"] = NOT_RUN
    if a.s6:
        sb, st = _pair(a.s6, EXPECT["steps"], None, ("raw",))
        report["memorisation"] = memorisation(sb, st, _npy(a.thirds), held["primary"], held_reps)
    return report


def main(argv: list[str] | None = None) -> int:
    """Write the report; returns 0, or 2 when an input is refused."""
    a = _parser().parse_args(argv)
    try:
        report = run(a)
    except Refused as e:
        sys.stderr.write(f"refused: {e}\n")
        return 2
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
