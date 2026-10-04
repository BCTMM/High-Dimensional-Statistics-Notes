"""Shared matplotlib style: transparent background, dark ink (inverted by CSS in dark mode)."""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / "docs" / "notes" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PINK, GREY = "#4f6bed", "#f57c00", "#2e9d6a", "#c2399a", "#888888"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": "#222",
    "axes.labelcolor": "#222",
    "axes.edgecolor": "#444",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#ddd",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "legend.frameon": False,
    "svg.fonttype": "none",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW"):  # optional PNG copies for eyeballing
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / f"{name}.png", dpi=90, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.svg")
