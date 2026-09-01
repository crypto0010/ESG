"""Figure 5: sub-dimension means with the 2020-2025 trend.

I2 fix: the left panel's y-tick labels used a bare `label[:38]` character
slice, which cut mid-word on several real labels (e.g. "...Compliance
Appr", "...core bus"). Now uses `style.truncate_label`, the same
word-boundary-with-ellipsis truncator `fig_loadings` already had.

Task 15 fixes (found by looking at the rendered figure, not by any test):

(a) 2025 is a partial year - 16 articles against 2024's 693 - so every one
    of the 11 trend lines plunges at the last point. Undisclosed, a reader
    sees ESG attention collapsing across the board. The 2024->2025 segment
    is now drawn dashed (every line, uniformly, so "dashed" always means
    "partial-year segment" and never doubles as a per-line style), and the
    x-axis ticks carry each year's article count so the thin final year is
    impossible to miss.
(b) 11 dimension lines were drawn against a 7-colour palette (index % 7),
    so H/I/J/K silently reused A/B/C/D's colours and the legend was
    unusable. Now uses the 11-colour `style.PALETTE` plus one distinct
    marker shape per dimension (`style.MARKERS`), so lines stay
    distinguishable by shape alone in greyscale or for colour-blind readers.
"""
from pathlib import Path

import matplotlib.pyplot as plt

from .. import config
from . import style

_LABEL_WIDTH = 38  # this panel is wider than fig_loadings' heatmap rows


def build_axes(sub, trend, counts=None):
    """Build (fig, (ax1, ax2)) without saving.

    `counts`: optional Series/dict of year -> number of coded articles
    (see `descriptives.yearly_counts`). When given, the trend panel's
    x-axis ticks are annotated with it so a thin final year is visible
    rather than reading as a genuine decline.
    """
    style.apply()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 7.5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    order = sub.sort_values("mean", ascending=True)
    colours = [style.PALETTE[list(config.DIMENSIONS).index(c[0]) % len(style.PALETTE)]
               for c in order.index]
    ax1.barh(range(len(order)), order["mean"], xerr=order["sd"],
             color=colours, error_kw={"lw": 0.6, "alpha": 0.5})
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(
        [f"{c}  {style.truncate_label(config.SUBDIMENSIONS[c], width=_LABEL_WIDTH)}"
         for c in order.index],
        fontsize=6.5)
    ax1.set_xlabel("Mean coded score (0-5)")
    ax1.set_xlim(0, 5)
    ax1.set_title("Attention by sub-dimension", fontsize=10)

    years = list(trend.index)
    for i, letter in enumerate(trend.columns):
        colour = style.PALETTE[i % len(style.PALETTE)]
        marker = style.MARKERS[i % len(style.MARKERS)]
        y = trend[letter]
        if len(years) >= 2:
            # Full history solid; the last (possibly partial) year's
            # segment dashed, uniformly across every line - "dashed" means
            # "incomplete year", not a per-dimension style.
            ax2.plot(years[:-1], y.iloc[:-1], marker=marker, ms=4, lw=1.2,
                      color=colour, linestyle="-", label=letter)
            ax2.plot(years[-2:], y.iloc[-2:], marker=marker, ms=4, lw=1.2,
                      color=colour, linestyle="--")
        else:
            ax2.plot(years, y, marker=marker, ms=4, lw=1.2, color=colour,
                      linestyle="-", label=letter)
    ax2.set_xlabel("Publication year")
    ax2.set_ylabel("Mean dimension score")
    ax2.set_title("Trend, 2020-2025 (final year partial, dashed)", fontsize=9.5)
    ax2.legend(ncol=2, fontsize=6.5, frameon=False)

    if counts is not None:
        counts_map = dict(counts.items()) if hasattr(counts, "items") else dict(counts)
        ax2.set_xticks(years)
        ax2.set_xticklabels(
            [f"{yr}\n(n={int(counts_map[yr])})" for yr in years], fontsize=6.5)
        last_year, last_n = years[-1], int(counts_map[years[-1]])
        ax2.annotate(f"{last_year} partial, n={last_n}",
                     xy=(years[-1], trend[trend.columns[0]].iloc[-1]),
                     xytext=(0.98, 0.02), textcoords="axes fraction",
                     ha="right", va="bottom", fontsize=7, style="italic",
                     color="#333333")

    return fig, (ax1, ax2)


def render(out_path, sub, trend, counts=None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = build_axes(sub, trend, counts=counts)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
