"""Figure 3: scree plot with parallel analysis (NEW - justifies k=5).

Observed eigenvalues from the coded item correlation matrix against the
95th-percentile threshold from `factors.parallel_analysis`'s random-matrix
simulation, with the retained factor count marked. This is the figure that
justifies the measurement model's k=5, replacing the old brief's taxonomy
schema (which described a category scheme this manuscript no longer uses).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style


def build_axes(pa: dict, max_rank: int = 20):
    """Build (fig, ax) from a `factors.parallel_analysis(...)`-shaped dict:
    {"n_factors": int, "eigenvalues": array, "threshold": array}.
    """
    style.apply()

    eig = np.asarray(pa["eigenvalues"], dtype=float)
    thr = np.asarray(pa["threshold"], dtype=float)
    n_retained = int(pa["n_factors"])

    n_show = min(max_rank, len(eig))
    ranks = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ranks, eig[:n_show], marker="o", ms=4, lw=1.4,
           color=style.PALETTE[0], label="Observed eigenvalues")
    ax.plot(ranks, thr[:n_show], marker="s", ms=3, lw=1.1, ls="--",
           color=style.PALETTE[1], label="95th percentile, random eigenvalues")
    ax.axhline(1.0, ls=":", lw=0.7, color="#8a8d91")

    if 0 < n_retained <= n_show:
        ax.axvline(n_retained + 0.5, ls=":", lw=1.0, color="#5a6672")
        y_top = ax.get_ylim()[1]
        ax.annotate(f"{n_retained} factors retained",
                    xy=(n_retained + 0.5, y_top * 0.82),
                    xytext=(min(n_retained + 3.0, n_show - 0.5), y_top * 0.92),
                    fontsize=8.5, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.9, color="#5a6672"))

    ax.set_xlabel("Factor rank")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_title("Parallel analysis: observed vs. random eigenvalues", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return fig, ax


def render(out_path, pa: dict, max_rank: int = 20) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = build_axes(pa, max_rank=max_rank)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
