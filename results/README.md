# Results

`billet/` holds the report that compares Akeratos Billet with its twin, and [its README](billet/README.md), which says how to read it.

`decontamination/` holds the records of which zero-shot evaluation items overlap Billet's training data and are left out of the scores ([its README](decontamination/README.md)).

`speedrun/` holds the claims of [the Mac speedrun](../docs/speedrun.md) and [their page](speedrun/README.md).

`bench/` holds the `bake bench` rows and [their page](bench/README.md). `uv run python speedrun/records.py --write` writes both pages.
