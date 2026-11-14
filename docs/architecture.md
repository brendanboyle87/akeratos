# Code map

Akeratos targets Apple silicon only, through MLX. This page names the files and follows one training step.

## The files

| File | What it holds |
|---|---|
| `akeratos/cli.py` | the `bake` command |
| `akeratos/configs.py` | the model's configuration, and Billet's and its twin's as trained |
| `akeratos/model.py` | the forward pass |
| `akeratos/flash.py`, `flash_metal.py` | the fused attention: three Metal kernels and their gradient |
| `akeratos/table/` | the n-gram table: hashing, shards, the sparse update |
| `akeratos/train/` | the trainer: flags, steps, optimiser, recipe, batches, checkpoints, the log |
| `akeratos/objective.py` | multi-token prediction |
| `akeratos/data/` | the token store, the packer, the chat format, the decontamination scan |
| `akeratos/eval/` | held-out bits per byte and the zero-shot tasks |
| `akeratos/export.py`, `infer.py` | the release folder, and greedy decoding from it |
| `akeratos/post/` | supervised fine-tuning |
| `akeratos/macops.py` | the guard that stops a run on battery, heat, memory pressure or low disk |
| `akeratos/bench.py`, `speedrun.py` | `bake bench`, and the speedrun's fixed code |
| the other files in `akeratos/` | checkpoints, number checks, seeds, the clock, the environment, errors, digests, typed MLX calls |
| `speedrun/` | the speedrun's training file, its slice cutter, and the results pages' checks |
| `scripts/data/` | the scripts that build the data store and the evaluation sets |
| `scripts/analysis/` | the scripts behind Billet's report |

## One training step

`Run.step` in `akeratos/train/loop.py`:

1. reads one batch per micro-batch of 2,048-token rows from the packed store;
2. with the table on, hashes each micro-batch's bigrams and trigrams into row ids and gathers those rows (`Table.gather`);
3. through `Stepper.step`, runs the forward pass, sums the gradients over micro-batches, skips the step if one is not finite, clips them and updates the dense weights;
4. updates the table rows it touched (`Table.update`);
5. updates the weight average and writes the step's line to `metrics.jsonl`.

## What runs where

- The table's hashing runs on the CPU, in NumPy. The rest is MLX.
- `bake bench` and the speedrun use the fused Metal kernels where they fit: on the GPU, in bfloat16, with a head size up to 128. `bake train`'s three configurations and scoring use the banded attention path, as Billet and its twin did.
- The tests run on Linux, on MLX's CPU backend.
