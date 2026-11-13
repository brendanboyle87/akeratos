# Decontamination records

These records list the zero-shot evaluation items that overlap Billet's training data: all 59 shards of the `train` split of store `9545f2d205efd574`. An item that shares any 13 consecutive token ids with a shard is dropped whole, and the zero-shot scorer leaves it out.

| Record | Items | Dropped | Unchecked |
|---|---:|---:|---:|
| `arc_easy.decontam.json` | 2,376 | 39 | 13 |
| `hellaswag.decontam.json` | 10,042 | 178 | 0 |
| `lambada_openai.decontam.json` | 5,153 | 7 | 0 |
| `piqa.decontam.json` | 1,838 | 19 | 94 |
| `winogrande.decontam.json` | 1,267 | 0 | 0 |

An unchecked item is shorter than 13 tokens, so it has no 13-gram to compare; it is kept.

Each record is one JSON object. `task_sha256` is the SHA-256 of the task file checked, which the scorer must match. `drop` lists the dropped items' `doc_id`s. `report` gives the counts above, `complete` (true when every shard was read; the scorer refuses a record without it), and each match's 13-gram, shard and document.

## How they were made

```sh
~/.venvs/mf-data/bin/python -I -B scripts/data/decontam_items.py --store ~/mf-data/9545f2d205efd574 --conversations ~/mf-sft/sft-text/2f7d3d1c06eb0e18/conversations.jsonl --tasks ~/mf-evals/6e9b59081ae31400/*.jsonl --out ~/mf-decontam/items
caffeinate -dimsu uv run --frozen --no-sync python -m akeratos.data.decontam --store ~/mf-data/9545f2d205efd574 --items ~/mf-decontam/items/*.items.jsonl --out ~/mf-decontam/records
```

To score with them, pass `--decontam results/decontamination` to the zero-shot scorer, `akeratos/eval/mc.py`.

## How to check a copy

`shasum -a 256 -c --ignore-missing SHA256SUMS` checks the five records here. `SHA256SUMS` also lists the fine-tuning set's record, which is not in this folder.
