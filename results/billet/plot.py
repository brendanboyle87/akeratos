"""Draw the training loss of Akeratos Billet and its twin on one axis, as an SVG.

Usage: python results/billet/plot.py BILLET_LOG TWIN_LOG OUT.svg, where the two logs are
logs/metrics.jsonl in each model's release folder. Drawing needs matplotlib, the "plot" extra.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

WINDOW = 100  # steps per plotted mean
USAGE = "usage: python results/billet/plot.py BILLET_LOG TWIN_LOG OUT.svg\n"
N_ARGS = 3


def windowed_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean step and mean loss of each window of WINDOW steps, finite losses only."""
    windows: dict[int, list[tuple[int, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step, loss = int(record["step"]), float(record["loss"])
        if math.isfinite(loss):
            windows.setdefault(step // WINDOW, []).append((step, loss))
    rows = [np.mean(np.array(windows[key]), axis=0) for key in sorted(windows)]
    table = np.array(rows).reshape(-1, 2)
    return table[:, 0], table[:, 1]


def main(argv: list[str]) -> int:
    """Read the two logs and write the figure; return the exit status."""
    if len(argv) != N_ARGS:
        sys.stderr.write(USAGE)
        return 2
    billet, twin, out = (Path(arg) for arg in argv)
    plt.rcParams["svg.hashsalt"] = "metalforge-billet"  # fixed ids, so the same bytes each run
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for path, label in ((billet, "Billet (table on)"), (twin, "twin (table off)")):
        steps, losses = windowed_loss(path)
        ax.plot(steps, losses, label=label, linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (nats per token)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, format="svg", metadata={"Date": None})  # no date in the file
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
