# Akeratos

I train small language models from scratch on one Mac with MLX, to test one idea: a very large learned n-gram table, held in the Mac's unified memory beside a small transformer.

## The result

Akeratos Billet is a 162M-parameter transformer with a table of 6.0 billion trained numbers: one vector for each hashed bigram and trigram, added to the residual stream before every block. Its twin is the same model without the table, trained on the same 3.2 billion tokens in the same order. The two are not matched for parameters. The table costs memory more than compute: Billet's run peaked at 24.1 GiB against its twin's 12.4, and a training step takes 5.2% longer with it on.

On 7,314 held-out documents, Billet scores 0.8866 bits per byte and its twin 0.9030; lower is better. Billet minus its twin is **−0.0164 bits per byte** (95% interval −0.0172 to −0.0157). By the rule I fixed before the twin finished, the table helps.

![Training loss of Billet and its twin, step by step](https://github.com/cogvera-labs/akeratos/raw/main/results/billet/training-loss.svg)

Each model was trained once, on one MacBook Pro (M4 Max, 48 GB). [The result page](https://github.com/cogvera-labs/akeratos/blob/main/docs/billet.md) has every readout and what one seed cannot say. [The analysis plan](https://github.com/cogvera-labs/akeratos/blob/main/docs/analysis-plan.md) is the rule as I posted it. [`results/billet/`](https://github.com/cogvera-labs/akeratos/tree/main/results/billet) holds the report this number is read from. The weights are on Hugging Face: [Billet](https://huggingface.co/Cogvera-Labs/Akeratos-Billet) and [its twin](https://huggingface.co/Cogvera-Labs/Akeratos-Billet-Twin).

Hashed n-gram tables are not my idea. [Engram](https://arxiv.org/abs/2601.07372) and the [Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975) hash n-grams into embedding tables, and [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) ships 196B parameters of Engram memory. [The n-gram memory](https://github.com/cogvera-labs/akeratos/blob/main/docs/learn/n-gram-memory.md) names the rest of the work this builds on.

## Run it

```sh
git clone https://github.com/cogvera-labs/akeratos && cd akeratos
uv sync
uv run bake bench
```

You need an Apple silicon Mac and [uv](https://docs.astral.sh/uv/). `bake bench` trains a 10M model twice on synthetic tokens, table off and table on, and prints what the table costs per step on your Mac. On my M4 Max it took 74 seconds and peaked at 6.2 GiB of memory.

- `uv run bake train` and `uv run bake evaluate` rerun and score Billet: [Reproduce](https://github.com/cogvera-labs/akeratos/blob/main/docs/reproduce.md) says how long that takes and how much memory.
- `uv run bake speedrun` runs one training file for five minutes and scores the model: [the Mac speedrun](https://github.com/cogvera-labs/akeratos/blob/main/docs/speedrun.md).

## Read the code

- [`akeratos/model.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/model.py): the forward pass.
- [`akeratos/table/table.py`](https://github.com/cogvera-labs/akeratos/blob/main/akeratos/table/table.py): the table and its sparse update.
- [`speedrun/train.py`](https://github.com/cogvera-labs/akeratos/blob/main/speedrun/train.py): the whole 10M model, its optimiser and schedule, in one file.

Billet was trained with the banded attention code, not the fused Metal kernels.

## How it is made

AI coding agents (Claude) write most of the code; I review and merge. Development happens in a private repository where the agents' checks run; this one is exported from it, one commit per change.

I am Thomas Nicklin ([@BhodiSea](https://github.com/BhodiSea)), at Cogvera Labs. [CONTRIBUTING.md](https://github.com/cogvera-labs/akeratos/blob/main/CONTRIBUTING.md) says how to send a fix. [OPEN_PROBLEMS.md](https://github.com/cogvera-labs/akeratos/blob/main/OPEN_PROBLEMS.md) lists what I cannot answer alone. The licence is Apache-2.0.
