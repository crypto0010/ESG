"""Figure 2: methodology architecture (R3.2).

Stage list reflects the CURRENT pipeline after the measurement-model pivot
(analysis/factors.py) - the "Tier A/B/C/D" wording of the original Task 13
brief predates that pivot and described a taxonomy the manuscript no longer
claims. The seven stages here are: search, expert coding, reliability,
measurement model, predictive benchmark, convergent importance, and Study
2's firm-level correspondence check (Task 18, R1.2) -- reported as a
correspondence check against independent LSEG/Refinitiv data, never as
validation of the literature model's substantive claims.
"""
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style

STAGES = [
    ("1. Scopus search",
     "2020-2025 literature search; 1,026 articles retained"),
    ("2. Expert coding",
     "11 dimensions x 4 sub-dimensions, 0-5 ordinal scale, scored from abstracts"),
    ("3. Reliability",
     "weighted kappa and ICC(2,1) on a double-coded subsample"),
    ("4. Measurement model",
     "exploratory factor analysis: KMO, parallel analysis, 5 factors retained"),
    ("5. Predictive benchmark",
     "held-out prediction of observed coded scores against a dummy floor"),
    ("6. Convergent importance",
     "gain, permutation, SHAP and network centrality agreement"),
    ("7. Study 2: correspondence check",
     "firm-level governance vs. disclosure, independent LSEG/Refinitiv data"),
]

_WRAP_WIDTH = 30
_BOX_W = 6.6
_LINE_H = 0.34
_TITLE_H = 0.36
_TOP_PAD = 0.30
_BOTTOM_PAD = 0.26
_STAGE_GAP = 0.55
_MARGIN_TOP = 0.55
_MARGIN_BOTTOM = 0.45
_FIG_WIDTH = 7.6


def _wrap(subtitle: str) -> list[str]:
    return textwrap.wrap(subtitle, width=_WRAP_WIDTH) or [subtitle]


def _box_height(lines: list[str]) -> float:
    return _TOP_PAD + _TITLE_H + len(lines) * _LINE_H + _BOTTOM_PAD


def build_figure():
    """Build and return (fig, ax, text_artists, box_patches) without saving.

    text_artists and box_patches are parallel lists (two text artists -
    title, body - per box, so each box_patch appears twice) for geometry
    testing, mirroring fig_prisma's iter_labelled_boxes convention.
    """
    style.apply()

    wrapped = [_wrap(sub) for _, sub in STAGES]
    heights = [_box_height(lines) for lines in wrapped]

    centers = []
    y = 0.0
    for h in reversed(heights):
        y += h / 2
        centers.append(y)
        y += h / 2 + _STAGE_GAP
    centers = list(reversed(centers))

    top = centers[0] + heights[0] / 2 + _MARGIN_TOP
    bottom = centers[-1] - heights[-1] / 2 - _MARGIN_BOTTOM
    fig_height = max(6.0, 0.92 * (top - bottom))

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, fig_height))
    ax.set_xlim(0, 10)
    ax.set_ylim(bottom, top)
    ax.axis("off")

    text_artists, box_patches = [], []
    for i, ((title, _), lines, h, y) in enumerate(zip(STAGES, wrapped, heights, centers)):
        colour = style.PALETTE[i % len(style.PALETTE)]
        patch = FancyBboxPatch((5.0 - _BOX_W / 2, y - h / 2), _BOX_W, h,
                               boxstyle="round,pad=0.14", fc=colour + "1f",
                               ec=colour, lw=1.2)
        ax.add_patch(patch)

        title_artist = ax.text(5.0, y + h / 2 - _TOP_PAD, title,
                               ha="center", va="top", fontsize=9.6, weight="bold")
        body_artist = ax.text(5.0, y + h / 2 - _TOP_PAD - _TITLE_H,
                              "\n".join(lines),
                              ha="center", va="top", fontsize=7.9,
                              style="italic", linespacing=1.55)
        text_artists.extend([title_artist, body_artist])
        box_patches.extend([patch, patch])

        if i:
            prev_y, prev_h = centers[i - 1], heights[i - 1]
            ax.add_patch(FancyArrowPatch((5.0, prev_y - prev_h / 2 - 0.04),
                                         (5.0, y + h / 2 + 0.04),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color="#33475b"))

    return fig, ax, text_artists, box_patches


def render(out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, text_artists, box_patches = build_figure()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
