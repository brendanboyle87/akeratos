"""Post-training.

Stage 1's chat template and masks (`data`), the supervised fine-tuning run on a frozen table
(`run`, entered through `sft`) and greedy samples for the format check (`sample`).

`post` sits above `train` and below `cli`; nothing else in `akeratos/` imports it.
"""
