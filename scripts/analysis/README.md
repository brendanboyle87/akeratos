# The analysis scripts

These scripts made Billet's report, [`results/billet/report.json`](../../results/billet/report.json). `analysis.py` computes the readouts of [the analysis plan](../../docs/analysis-plan.md) from both models' scored records. `reads.py`, `overlap.py` and `s6.py` make three of its inputs: each target's table read counts, each held-out document's 13-gram overlap with the training data, and the memorisation documents.

Two files are never edited, and the report depends on both:

| File | SHA-256 |
|---|---|
| `analysis.py` | `7876fbb31c28c6b5a082284a3a2693f57acc30476a70b4f5eedc8372617ab85b` |
| `overlap.py` | `96e0c00ec54c3729371f9dc329fae79e0fc0844b6e2ee650132659408ec58c2f` |

`analysis.py` and `overlap.py` were written under the project's first name, MetalForge, and `overlap.py` imports the package by that name. [`firstname.py`](firstname.py) points that name at the package and runs a script: `uv run --frozen --no-sync python scripts/analysis/firstname.py scripts/analysis/overlap.py --store STORE --out OUT`.

In a report, `m1` is the analysis plan's address in the private repository where I develop Akeratos, as it was under the project's first name, MetalForge. Here it opens nothing; [the plan's page](../../docs/analysis-plan.md) holds its text.
