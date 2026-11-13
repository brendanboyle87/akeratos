# Speedrun records

The claims on the 5-minute track of [the Mac speedrun](../../docs/speedrun.md): held-out bits per byte after 300 seconds of step time, lower is better. Each claim is a folder here, `<class>/<claim>/`, holding the receipt and the step log of each of its runs. Claims rank within their hardware class, laptops apart from desktops, and No-table claims rank in Open as well. A claim takes a category's record only if its mean beats the standing record's by more than the margin m, and it is unverified until another person's rerun verifies it. `uv run python speedrun/records.py --write` generates this page from the receipts.

The rules' numbers are not set: the runs a claim needs (K, 5 for now), the margin m by which a new record must beat the standing one, and the tolerance t within which another person's rerun verifies a claim. Until they are set, no claim takes a record.

No claims yet.
