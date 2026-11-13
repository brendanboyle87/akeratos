# Speedrun records

The claims on the 5-minute track of [the Mac speedrun](../../docs/speedrun.md): held-out bits per byte after 300 seconds of step time, lower is better. Each claim is a folder here, `<class>/<claim>/`, holding the receipt and the step log of each of its runs. Claims rank within their hardware class, laptops apart from desktops, and No-table claims rank in Open as well. A claim takes a category's record only if its mean beats the standing record's by more than the margin m, and it is unverified until another person's rerun verifies it. `uv run python speedrun/records.py --write` generates this page from the receipts.

A claim's score is the mean of its runs with seeds 0 to 4 (K = 5). A new record must beat the standing record's mean by more than m = 0.026, and another person's rerun verifies a claim if the rerun's mean is at most the claim's plus t = 0.039.

## Laptops

### `m4-max-40c-48gb-macos26-laptop`

Open:

|  | Mean bpb | Category | Date | Person | AI system | Commit | Status |
|---|---|---|---|---|---|---|---|
| record | 1.4922 | No table | [2026-11-13](m4-max-40c-48gb-macos26-laptop/2026-11-13-BhodiSea/) | [BhodiSea](https://github.com/BhodiSea) | Claude Code | `1de0f07` | reference |

No table:

|  | Mean bpb | Date | Person | AI system | Commit | Status |
|---|---|---|---|---|---|---|
| record | 1.4922 | [2026-11-13](m4-max-40c-48gb-macos26-laptop/2026-11-13-BhodiSea/) | [BhodiSea](https://github.com/BhodiSea) | Claude Code | `1de0f07` | reference |
