# The Mac speedrun

`bake speedrun` trains `speedrun/train.py` for 300 seconds of step time on a fixed FineWeb-Edu slice, then scores its held-out bits per byte. Lower is better. Claims rank within a hardware class, laptops apart from desktops.

## Run it

Run from the root of a clone of `main`, with your edits to `speedrun/train.py` uncommitted.

```sh
uv run --frozen --no-sync hf download Cogvera-Labs/Akeratos-Speedrun-5min --repo-type dataset --revision 4fadb133baf45efe8dd764c2a172ff92e5ee2482 --local-dir ~/mf-speedrun-5min
uv run --frozen --no-sync bake speedrun --data ~/mf-speedrun-5min --claim "$(date -u +%F)-yourhandle" --person yourhandle --ai-system none
```

The download is about 285 MB. `--claim` is the UTC date of the claim's first run and your GitHub handle. `--ai-system` names the AI system that helped, or `none`. `--category no-table` is for a model without the n-gram table. Exit code 22 means a slice differs from its pin; exit code 78, that `akeratos/`, `pyproject.toml` or `uv.lock` differ from the commit.

## The training file

It defines `ROWS`, `SEQ_LEN` and `build(seed)`, whose object has `step`, `arrays` and `logits`. The reference is Akeratos's 10M model without the table, trained in BF16 with AdamW.

The file reads no clock, file, environment variable or network. It trains only in `step`, no more in a warm-up step than in a later one, and replaces nothing outside itself. Every random draw derives from `build`'s seed. Logits may not depend on later tokens: `bake speedrun` checks this (exit code 78).

## Timing and scoring

A step is timed from just before `step` to the end of evaluating `arrays()`. The first 10 steps count at the median of the later ones, so compilation is free. The score covers whole held-out documents until they hold 2^20 targets.

## Sending a claim

A claim is one person's runs of one training file at one commit on `main`, seeds 0 upward, at least K. Its score is the mean of seeds 0 to K − 1. It takes the record only by beating it by more than the margin m, and it is unverified until another person's rerun on the same class comes within the tolerance t. K is 5, or 3 when s is at most 0.002; m = 2·s·√(2/K) and t = 3·s·√(2/K), each rounded up to 0.001. s, the standard deviation of ten runs of the reference file, is 0.0203, so K is 5, m is 0.026 and t is 0.039.

Run seeds 0 to K − 1 under one `--claim` name, adding `--seed 1` and so on. Send the claim's folder in a pull request, with [the records page](https://github.com/cogvera-labs/akeratos/blob/main/results/speedrun/README.md) rewritten by `uv run python speedrun/records.py --write`. `--check` runs the same checks.

## `bake bench`

`uv run bake bench` trains the 10M model on synthetic tokens twice, table off, then on, at the shape [the bench rows page](https://github.com/cogvera-labs/akeratos/blob/main/results/bench/README.md) states. It prints both step times, their ratio, peak memory and the table's estimated bandwidth. It took 74 seconds on an M4 Max. Run it on mains power. It saves its row and says where. Paste it into [the bench form](https://github.com/cogvera-labs/akeratos/issues/new?template=bench.yml), or run with `--out results/bench --person HANDLE` and send it by pull request. [The open problem on that cost](https://github.com/cogvera-labs/akeratos/blob/main/OPEN_PROBLEMS.md#the-tables-cost-on-each-mac) asks for rows from other Macs.
