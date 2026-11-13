# Training a language model on a Mac: the options

Several open projects train or fine-tune language models on an Apple silicon Mac, and each is built for a different job. This page says what each one is for, so you can pick the one that fits. What it says about other projects comes from their own pages, read on 2026-10-29; check them for changes.

| If you want to | Start with |
|---|---|
| Adapt an existing open model to your own data | [mlx-lm fine-tuning](#mlx-lm-fine-tuning) |
| Train a small chat model from scratch, end to end, on a Mac | [nanochat-mlx](#nanochat-mlx) |
| Pretrain a standard small model that mlx-lm can load | [mlx-pretrain](#mlx-pretrain) |
| Stay in PyTorch, including Engram-style memory experiments | [PyTorch on the Mac's GPU](#pytorch-on-the-macs-gpu) |
| Train on NVIDIA GPUs, or compare with GPU speedrun records | [modded-nanogpt or nanochat](#built-for-nvidia-gpus-modded-nanogpt-and-nanochat) |
| Study a large learned n-gram table trained in a Mac's unified memory, against a twin without it | [Akeratos](#akeratos) |

## mlx-lm fine-tuning

[mlx-lm](https://github.com/ml-explore/mlx-lm) runs and fine-tunes existing language models with MLX: its `mlx_lm.lora` command does LoRA, DoRA or full fine-tuning. [Its guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md) covers models that already exist, not pretraining. Pick it to adapt one to your data.

## nanochat-mlx

[nanochat-mlx](https://github.com/scasella/nanochat-mlx) is an MLX port of Karpathy's nanochat for Apple silicon, covering the whole path from the data download to a chat command line. Models range from about 5M parameters (about a minute on an M3 Pro) to about 600M (about a day). Pick it to train your own small chat model on a Mac.

## mlx-pretrain

[mlx-pretrain](https://github.com/N8python/mlx-pretrain) pretrains language models with MLX: its example trains a 2M-parameter model on 200M tokens of FineWeb-Edu in about 2 hours on an M3 Max, and a script converts the result to mlx-lm's format. Pick it to pretrain a standard small model for mlx-lm.

## PyTorch on the Mac's GPU

PyTorch trains on a Mac's GPU through its [`mps` device](https://docs.pytorch.org/docs/stable/notes/mps.html). [nano-scalemb](https://github.com/zy-ning/nano-scalemb), a nanochat-based repository, reproduces Engram on small models and runs on MPS. Pick this route to stay in PyTorch, or to test Engram-style memory in it.

## Built for NVIDIA GPUs: modded-nanogpt and nanochat

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) is a speedrun: reach a FineWeb validation loss of 3.28 as fast as possible on 8 H100 GPUs. Record 92 (2026-08-30) takes 0.665 minutes and uses an 84.6M-row hashed bigram and trigram embedding table (65B parameters).
- [nanochat](https://github.com/karpathy/nanochat) trains a chat model end to end on one GPU node, in about 2 hours on 8 H100s. A small script shrinks it to run on a CPU or Apple silicon.

Pick them when you have NVIDIA GPUs, or to compare a method with their records.

## Akeratos

Akeratos is a research project, not a ready tool for training chat models. I pretrain small transformers from scratch with MLX, add a large learned n-gram table held in the Mac's unified memory, and train each headline run twice: with the table and without it, on the same tokens. The logs and the report behind each number are published. Pick it to test an idea against that baseline on your own Mac, or to study what a large learned memory does; the [open problems](https://github.com/cogvera-labs/akeratos/blob/main/OPEN_PROBLEMS.md) list where help is wanted.

Pick something else if you want a model to chat with or fine-tune today. Akeratos's models run in Akeratos's own code, not in mlx-lm, transformers or llama.cpp. Its smallest run peaks at about 8 GiB of memory and Billet at about 24 GiB; [Reproduce](reproduce.md) has the details.
