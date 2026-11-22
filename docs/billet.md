# Billet and its twin

Cogvera Labs trained one model with a large n-gram table and one without, on the same tokens, and fixed how to compare them before the second run finished.

## What was trained

- **Billet.** A decoder-only transformer of 162M parameters (18 layers, width 768) with a table of 7,812,500 rows of 768 numbers, 6.0 billion in all. Each position reads the row of its hashed bigram and the row of its hashed trigram. Their sum, divided by √2, is added to the residual stream before every block. [The n-gram memory](learn/n-gram-memory.md) explains it.
- **Its twin.** The same model and settings with the table off.
- **Data.** FineWeb-Edu's `sample/10BT`, tokenised with SmolLM2's tokenizer. Each run read 3.2 billion tokens in the same order. Held out and never trained on: 7,314 documents, 7.6 million tokens.
- **Runs.** 24,415 steps of 131,072 tokens on one MacBook Pro (M4 Max, 48 GB), with MLX 0.32.3 and the banded attention path. Billet's steps took 107.6 hours in all and its memory peaked at 24.1 GiB; its twin's took 111.1 hours and peaked at 12.4 GiB. The twin's steps may have been the slower ones because the Mac was swapping more during its run: a median of 18 GB of swap in use, against 0.6 GB during Billet's. The timing probe below prices the table with both runs under the same conditions.
- **Once each.** One seed per model.

The code as trained ships beside each model's weights, as an unedited snapshot. This repository holds the same code, tidied. A 20-step replay of each run from this repository matches the released training log: every batch hash, and the loss within 0.00005.

## The rule, fixed first

[The analysis plan](analysis-plan.md) was posted on 1 November 2026 at 05:34 UTC, eight days before the twin's last step. It has not been edited. It says:

- Δ is Billet's held-out bits per byte minus its twin's: final checkpoints, raw weights, all 7,314 documents. Below zero, Billet is better.
- δ, the bar, is the larger of 0.01 nats per token and 4σ, where σ is 0.00108 bits per byte.
- Δ at or below −δ: the table helps. Δ at or above +δ: the table hurts. Otherwise: no measurable difference.
- Equivalence is claimed only if Δ is within 0.0018 of zero.

One amendment, on 2 November, recorded the SHA-256 of the analysis code before any Δ was computed. That code is [`analysis.py`](https://github.com/cogvera-labs/akeratos/blob/main/scripts/analysis/analysis.py), unchanged since.

## Primary readout

Bits per byte is the held-out text's total loss, in bits, divided by its bytes. Lower is better, even between models with different tokenizers.

| | Held-out bits per byte |
|---|---|
| Billet | 0.8866 |
| Twin | 0.9030 |
| Δ | −0.0164 (95% interval −0.0172 to −0.0157) |
| δ | 0.00432 |

**The table helps.** Δ is −0.0164 bits per byte, below −δ.

## Secondary readouts

The plan tests one of these, the zero-shot tasks. The rest it describes and does not test.

| Readout | Δ | 95% interval | The plan's status |
|---|---|---|---|
| EMA weights, final checkpoints | −0.0165 | −0.0173 to −0.0158 | read against δ: the table helps |
| Raw weights at step 19,532 | −0.0173 | −0.0181 to −0.0165 | described, not tested |
| Each document's first window | −0.0176 | −0.0184 to −0.0168 | described, not tested |
| Targets by how often their table rows were read, in tenths | −0.0310 in the least-read tenth, −0.0117 in the most-read; smallest in the eighth, −0.0095 | every tenth's interval is below zero | described, not tested |
| Documents with no 13-gram found in the training data (2,456) | −0.0079 | −0.0084 to −0.0075 | described, not tested |
| Documents with one or more (4,858) | −0.0186; by quarter of overlap, lowest to highest: −0.0096, −0.0122, −0.0189, −0.0385 | −0.0196 to −0.0177 | described, not tested |
| Zero-shot accuracy: HellaSwag (9,864 items) | +0.85 points | +0.29 to +1.41 | tested: Holm-adjusted p = 0.016, a difference |
| PIQA, ARC-Easy, LAMBADA, Winogrande | −0.66, −0.90, 0.00, +0.24 points | each spans zero | tested: p = 1 for each, no difference claimed |
| Training loss in nats per token, mean over 1,000 steps ending at 10, 25, 50, 80 and 100% of training | −0.0475, −0.0406, −0.0538, −0.0586, −0.0584 | the widest, at 25%: −0.0425 to −0.0384 | described, not tested |
| Per Mac-hour: k against r | k = 1.39; r = 1.052 | r: 1.050 to 1.054 | no interval for k. k is above r's interval: the table leads per Mac-hour in the hold |
| Memorisation: 7,314 training documents | −0.0456, against −0.0164 held out | −0.0466 to −0.0447 | described, not tested |

Accuracy is in percentage points, and above zero Billet is better. Everywhere else Δ is in bits per byte unless the row says otherwise, and below zero Billet is better.

r is how much longer a step takes with the table on: 5.2%. k is how many times more steps the twin needed to reach Billet's training loss: the median over the 9,007 steps of the hold where it did. At the other 8,525 the twin had not reached it by the hold's end, and they are left out.

On text both models trained on, Billet's lead is 0.029 bits per byte larger than on held-out text (interval 0.028 to 0.030). The plan does not name this figure, so it is exploratory.

The plan's learning-rate sweeps, 50M tokens at each of four rates, scored lowest at 3e-2 for both models. The pair stayed at 1e-2, as the plan fixed, and its trigger for a second twin did not fire. The sweeps' logs are not in this repository.

## Both curves

![Training loss of Billet and its twin, step by step](https://github.com/cogvera-labs/akeratos/raw/main/results/billet/training-loss.svg)

Both bend at step 19,532, where the learning rate starts to fall.

## What one seed cannot say

- Each model was trained once. The interval above resamples held-out documents. It says how well these two checkpoints are told apart, not what another seed would do.
- σ is borrowed. The plan takes it from another paper's three-seed runs at 135M parameters. Cogvera Labs has not measured seed noise at this size.
- One Mac, one dataset, one model size, one table size. Nothing here says what happens at another.
- The comparison is at equal tokens, not equal time. The table makes each step slower, and only the per-Mac-hour row prices that.

## What the plan left open

The amendment of 2 November names three things not yet fixed when it was posted: the code that writes each target's table read count, the code that picks the memorisation documents, and the commit of the script that scores a peer model. It says a dated amendment would record each before any Δ was computed, and that a readout without one is reported as not run. No second amendment was posted. The first two were fixed in one commit on 4 November, before the twin finished and a week before any Δ was computed; each output records that commit and its code's SHA-256, and the report lists the outputs. So the read-count and memorisation rows above are run, but nothing on the thread fixed their code in advance: read them as described, with less weight than the rest. The peer model's seed spread was not run. The thread's last entry is the closing note of 10 November, at the end of [the plan's page](analysis-plan.md).

## Check it

- [`results/billet/`](https://github.com/cogvera-labs/akeratos/blob/main/results/billet/README.md) holds the report these numbers are read from, and the command that writes it.
- [Reproduce](reproduce.md) rebuilds the data, reruns both models and scores them.
- If your number differs from this page's, open an issue. That is a finding.
