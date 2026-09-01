"""Figure 4: factor loadings heatmap (NEW - the manuscript's centrepiece).

44 items x 5 factors, grouped by primary factor then sorted by loading
within each group, on a diverging colour scale centred on zero. Loadings
below the simple-structure threshold are muted (masked out of the colour
scale, and rendered with faded, unbolded text) so the solution's simple
structure is visible at a glance; every cell still carries its two-decimal
value, since a muted loading is still a real, reportable number.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config
from . import style

# Moved to style.truncate_label (I2) so fig_descriptives can share the same
# word-boundary truncation instead of its own bare character slice. Kept as
# an alias here in case anything outside this module still imports the old
# private name.
_truncate_label = style.truncate_label


def _ordered_index(loadings: pd.DataFrame, threshold: float) -> pd.Index:
    from .. import factors  # local import: keeps this module's import surface light

    assignment = factors.assign_items(loadings, threshold=threshold)
    sort_key = pd.DataFrame({
        "factor": assignment["primary"].fillna("~unassigned"),
        "loading": assignment["primary_loading"],
    })
    return sort_key.sort_values(["factor", "loading"], ascending=[True, False]).index


def build_axes(loadings: pd.DataFrame, threshold: float = 0.40):
    """Build (fig, ax, order) without saving. `order` is the row order (item
    codes) used, grouped by primary factor and sorted by loading within
    each group.
    """
    style.apply()

    order = _ordered_index(loadings, threshold)
    L = loadings.loc[order]
    n_items, n_factors = L.shape
    values = L.to_numpy(dtype=float)
    display = np.where(np.abs(values) >= threshold, values, np.nan)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eef0f2")

    fig_h = max(4.2, 0.26 * n_items + 1.6)
    fig, ax = plt.subplots(figsize=(7.6, fig_h))
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.grid(False)

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(list(L.columns), fontsize=9)

    label_fontsize = 6.0 if n_items > 30 else 7.5
    labels = [f"{c}  {_truncate_label(config.SUBDIMENSIONS.get(c, ''))}" for c in order]
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(labels, fontsize=label_fontsize)

    ann_fontsize = 5.0 if n_items > 30 else 7.2
    for i in range(n_items):
        for j in range(n_factors):
            v = values[i, j]
            above = abs(v) >= threshold
            if not above:
                colour, weight, alpha = "#3a3f45", "normal", 0.55
            else:
                colour = "white" if abs(v) > 0.55 else "black"
                weight, alpha = "bold", 1.0
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=ann_fontsize, color=colour, weight=weight, alpha=alpha)

    ax.set_title("Rotated factor loadings (|loading| < 0.40 muted)", fontsize=10, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.5, label="loading", pad=0.03)
    return fig, ax, list(order)


def render(out_path, loadings: pd.DataFrame, threshold: float = 0.40) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, _ = build_axes(loadings, threshold=threshold)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
