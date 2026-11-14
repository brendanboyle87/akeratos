# Akeratos Billet against its twin

Akeratos Billet was trained with its table on; its twin was trained the same way with the table off. This folder holds what the comparison found.

- `report.json` is the report that `scripts/analysis/analysis.py` wrote from the scored records of both models.
- `plot.py` draws the training loss of both models on one axis.
- `training-loss.svg` is the figure `plot.py` draws.

## The primary readout

On held-out text at the final checkpoints, Billet's score minus the twin's is -0.0164 bits per byte (`primary.delta`; lower is better), with a 95% interval from -0.0172 to -0.0157 bits per byte (`primary.ci95`). The report's outcome (`primary.band`) is "the table helps".

## Reading the report

The field `m1` is the analysis plan's address in the private development repository, as it was under the project's first name, MetalForge. Here it opens nothing; [the plan's page](../../docs/analysis-plan.md) holds its text. The names under `inputs` are the names the files had when the script ran, each with its SHA-256.

## Drawing the figure

```bash
uv run --extra plot python results/billet/plot.py BILLET_LOG TWIN_LOG results/billet/training-loss.svg
```

`BILLET_LOG` and `TWIN_LOG` are `logs/metrics.jsonl` in each model's release folder. The `plot` extra brings matplotlib.
