"""Figure 7: Ward dendrogram and the silhouette scan (SECONDARY analysis).

Ward clustering is no longer the manuscript's taxonomy - the measurement
model (Figures 3-4) is. This figure and its silhouette scan report the
clustering result as a secondary analysis: the weak separation it finds
(silhouette argmax ~0.22, on the boundary of a monotonically rising curve)
is evidence FOR the factor model, since hard partitions are the wrong tool
for 44 ordinal indicators of overlapping latent constructs, and it
pre-empts a reviewer asking whether clustering was tried.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .. import clustering, config
from . import style

WEAK_THRESHOLD = 0.25


def build_axes(df, scan):
    style.apply()

    n_leaves = len(config.SCORE_COLS)
    fig_height = max(6.6, 0.155 * n_leaves + 1.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, fig_height),
                                   gridspec_kw={"width_ratios": [2, 1], "wspace": 0.32})

    dendrogram(clustering.linkage_matrix(df), labels=config.SCORE_COLS,
              orientation="left", ax=ax1, color_threshold=None, leaf_font_size=6.5)
    ax1.set_xlabel("Ward distance (1 - |Spearman rho|)")
    ax1.set_title("Secondary analysis: Ward clustering of the 44 items", fontsize=10)

    ax2.plot(scan["k"], scan["silhouette"], marker="o", lw=1.3, color=style.PALETTE[0])
    # `verdict_from_scan` (not `structure_verdict`) so this reads the SAME
    # scan being plotted rather than re-running `silhouette_scan` on `df` a
    # second time - avoids paying twice for a possibly-expensive scan and
    # guarantees the displayed k/silhouette can never diverge from the plot.
    verdict = clustering.verdict_from_scan(scan)
    best_k, best_sil = verdict["best_k"], verdict["best_silhouette"]
    ax2.axvline(best_k, ls="--", lw=0.9, color="#c1666b")
    ax2.set_xlabel("number of clusters k")
    # The y-axis label sits on ax2's LEFT edge by default, immediately next
    # to ax1's leaf-label column (a `orientation="left"` dendrogram draws
    # its 44 leaf labels right up against that boundary) - moved to the
    # right edge instead, well clear of ax1's content.
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_ylabel("silhouette", rotation=270, labelpad=14)

    # The real data's k=15 argmax sits at the upper edge of its tested range
    # (2..16) - `verdict["on_boundary"]` says so. Surfaced explicitly rather
    # than leaving "(weak separation)" to carry that meaning alone, since a
    # boundary argmax and a genuinely weak-but-interior optimum are
    # different findings (one says "the window may be too narrow", the
    # other says "even a wider window wouldn't help").
    notes = []
    if best_sil < WEAK_THRESHOLD:
        notes.append("weak separation")
    if verdict["on_boundary"]:
        notes.append("argmax at the edge of the tested k range")
    note = f"max silhouette = {best_sil:.3f}" + (f" ({'; '.join(notes)})" if notes else "")
    ax2.set_title(f"k = {best_k}: {note}", fontsize=9.2)

    fig.suptitle("Secondary analysis - weak cluster separation supports the\n"
                "factor model (Figures 3-4) rather than a competing taxonomy",
                fontsize=8.6, style="italic", y=1.02)
    return fig, ax1, ax2


def render(out_path, df, scan) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1, ax2 = build_axes(df, scan)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
