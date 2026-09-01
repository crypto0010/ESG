"""Shared figure styling. Springer wants embedded fonts and vector output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2f5d8a", "#c1666b", "#5b8c5a", "#d4a26a", "#6b5b95", "#4f9aa3", "#8a8d91"]

# Fig 5's trend panel plots all 11 config.DIMENSIONS letters as separate
# lines. The original 7-colour PALETTE forced a wrap (index % 7), so H, I, J,
# K silently reused A, B, C, D's colours and the legend became unusable.
# These 4 are appended (not inserted) so every existing PALETTE[i] lookup
# elsewhere (fig_architecture, fig_benchmark, fig_dendrogram, fig_scree, all
# of which index only the first couple of entries) is unchanged.
_PALETTE_EXTRA = ["#e0a52c", "#7f4f24", "#264653", "#9d4edd"]
PALETTE = PALETTE + _PALETTE_EXTRA

# 11 visually distinct marker shapes, one per config.DIMENSIONS letter, so a
# line is identifiable by shape alone (colour and hue distinctions collapse
# in greyscale or for colour-blind readers; shape does not).
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]

_LABEL_WIDTH = 34


def apply():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def truncate_label(label: str, width: int = _LABEL_WIDTH) -> str:
    """Truncate on a word boundary with an ellipsis, never mid-word.

    Shared by every figure that prints a `config.SUBDIMENSIONS` label
    alongside a fixed-width axis (originally `fig_loadings._truncate_label`,
    moved here so `fig_descriptives` - which used a bare
    `label[:38]` slice and produced visible mid-word breaks like "...Compliance
    Appr" - gets the same fix instead of a second, divergent implementation).
    """
    if len(label) <= width:
        return label
    cut = label[:width].rsplit(" ", 1)[0]
    return f"{cut}…"
