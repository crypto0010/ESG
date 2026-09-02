"""PRISMA 2020 flow diagram. Unknown counts print 'not recorded', never a guess."""
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import style

UNKNOWN = "not recorded"

# Vertical spacing between the footer lines beneath the diagram, in axis units.
FOOTER_LINE_GAP = 0.45

# Characters per footer line before wrapping. The axes are 10 units wide and
# the footer prints at fontsize 8; beyond roughly this width a line runs past
# the figure's trimmed bounding box.
FOOTER_WRAP_CHARS = 88


def load_counts(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {k: v for k, v in json.load(fh).items() if not k.startswith("_")}


def _fmt(v) -> str:
    return UNKNOWN if v is None else f"n = {v:,}"


def _format_exclusion_box(count, reasons_dict: dict) -> str:
    """Format exclusion box with count and reason breakdown when available.

    Wraps reason text to fit in box width (~20 chars with indent).
    """
    text = _fmt(count)

    if reasons_dict:
        # Get non-null reasons
        valid_reasons = [(k, v) for k, v in reasons_dict.items() if v is not None]
        if valid_reasons:
            for reason, reason_count in valid_reasons:
                # Wrap reason text to fit box (~18 chars after "  " indent)
                reason_line = f"{reason}: {reason_count:,}"
                wrapped = textwrap.fill(reason_line, width=18,
                                       initial_indent="  ",
                                       subsequent_indent="    ")
                text += f"\n{wrapped}"

    return text


STAGE_KEYS = ("duplicates_removed", "screened_title_abstract",
              "excluded_title_abstract", "fulltext_assessed", "fulltext_excluded")


def footer_lines(counts: dict) -> list[str]:
    """Provenance lines printed under the diagram.

    A bare "search date: not recorded" stamp tells a reader nothing and reads
    as a rendering fault. When the date is unavailable, state instead what the
    record does establish -- the database searched and the coverage the query
    was restricted to -- and explain the absent stage counts once, rather than
    leaving the boxes to imply it six times over.
    """
    databases = ", ".join(counts.get("databases") or ["Scopus"])
    filters = (counts.get("filters") or "").strip()

    provenance = f"Source: {databases}."
    date = counts.get("search_date")
    if date:
        provenance += f" Search date: {date}."
    if filters:
        provenance += f" Coverage: {filters}."

    lines = [provenance]
    if any(counts.get(k) is None for k in STAGE_KEYS):
        lines.append(f"Stages shown as '{UNKNOWN}' were not preserved "
                     "in the original screening record and are not estimated.")
    if not date:
        lines.append("The exact query string and search date were not archived; "
                     "the included records are listed in Supplementary Data B.")

    # The axes are 10 units wide at fontsize 8; anything much past this wraps
    # outside the figure's trimmed bounding box.
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=FOOTER_WRAP_CHARS) or [""])
    return wrapped



def validate(counts: dict) -> list[str]:
    """Check the arithmetic, but only where every term is known."""
    problems = []

    def known(*keys):
        return all(counts.get(k) is not None for k in keys)

    if known("identified", "duplicates_removed", "screened_title_abstract"):
        if counts["identified"] - counts["duplicates_removed"] != counts["screened_title_abstract"]:
            problems.append("identified - duplicates_removed != screened_title_abstract")
    if known("screened_title_abstract", "excluded_title_abstract", "fulltext_assessed"):
        if (counts["screened_title_abstract"] - counts["excluded_title_abstract"]
                != counts["fulltext_assessed"]):
            problems.append("title/abstract screening does not balance against fulltext_assessed")
    if known("fulltext_assessed", "fulltext_excluded", "included"):
        if counts["fulltext_assessed"] - counts["fulltext_excluded"] != counts["included"]:
            problems.append("fulltext_assessed - fulltext_excluded != included")
    return problems


def box_labels(counts: dict) -> list[str]:
    """Generate box labels, including exclusion reasons when available."""
    labels = [
        f"Records identified\n{_fmt(counts.get('identified'))}",
        f"Duplicates removed\n{_fmt(counts.get('duplicates_removed'))}",
        f"Records screened\n(title/abstract)\n{_fmt(counts.get('screened_title_abstract'))}",
        f"Records excluded\n{_format_exclusion_box(counts.get('excluded_title_abstract'), counts.get('excluded_title_abstract_reasons', {}))}",
        f"Full-text articles\nassessed\n{_fmt(counts.get('fulltext_assessed'))}",
        f"Full-text articles\nexcluded\n{_format_exclusion_box(counts.get('fulltext_excluded'), counts.get('fulltext_excluded_reasons', {}))}",
        f"Studies included\nand coded\n{_fmt(counts.get('included'))}",
    ]
    return labels


def _count_lines(text: str) -> int:
    """Count newlines in text to determine box height."""
    return text.count("\n") + 1


# Side boxes used a fixed height of 1.2 and fixed 2.6-unit centre spacing
# before exclusion reasons could grow a box taller. REF_SIDE_HEIGHT and
# BASE_SIDE_GAP describe that original, already-approved layout; BOX_PAD
# is the boxstyle's "round,pad=0.12" padding, which visually extends every
# box by this amount on each side beyond its nominal height/width.
REF_SIDE_HEIGHT = 1.2
BASE_SIDE_GAP = 2.6
BOX_PAD = 0.12

# Geometry of the original, already-signed-off layout: main boxes are
# fixed-height (1.4) at a fixed x/y grid; the axes span y in [0, 12] inside
# a 7.2x9.5in figure; the search-date stamp sits 0.98 units below the
# lowest box's padded bottom edge, and the axes' lower limit sits a
# further 0.6 units below the stamp. These constants let the layout grow
# downward (extending the axes/figure) only when tall side boxes actually
# need the room, while reproducing the original numbers exactly when they
# don't.
MAIN_BOX_HEIGHT = 1.4
STAMP_MARGIN = 0.98
AXIS_MARGIN = 0.6
BASE_YLIM_TOP = 12.0
BASE_YLIM_RANGE = 12.0
BASE_FIGSIZE = (7.2, 9.5)


def _side_centers(top_anchor: float, heights: list[float]) -> list[float]:
    """Stack side-box y-centres downward from ``top_anchor``.

    Adjacent boxes are spaced so that whenever a box grows past
    ``REF_SIDE_HEIGHT`` (its original fixed height), the extra half-height
    is added to the gap on that side. This keeps the clearance between any
    two padded box edges exactly what the original fixed-height, fixed-gap
    layout had (BASE_SIDE_GAP - REF_SIDE_HEIGHT - 2*BOX_PAD), regardless of
    how tall a box's wrapped reason text makes it. When every height equals
    REF_SIDE_HEIGHT this reduces to the original hardcoded centres exactly.
    """
    centers = [top_anchor]
    for prev_h, h in zip(heights, heights[1:]):
        gap = BASE_SIDE_GAP + max(0.0, prev_h - REF_SIDE_HEIGHT) / 2 + max(0.0, h - REF_SIDE_HEIGHT) / 2
        centers.append(centers[-1] - gap)
    return centers


def build_figure(counts: dict):
    """Build and return the PRISMA figure (fig, ax) without saving.

    Returns (fig, ax, text_artists, box_patches) for testing geometry.
    text_artists and box_patches are parallel lists for iter_labelled_boxes.
    """
    style.apply()
    labels = box_labels(counts)

    # Calculate heights based on content (side boxes are at indices 1, 3, 5)
    side_heights = [
        max(1.2, _count_lines(labels[1]) * 0.35),
        max(1.2, _count_lines(labels[3]) * 0.35),
        max(1.2, _count_lines(labels[5]) * 0.35),
    ]
    side_y = _side_centers(9.1, side_heights)

    main_y = [10.4, 7.8, 5.2, 2.4]
    main = [(2.6, main_y[0], labels[0]), (2.6, main_y[1], labels[2]),
            (2.6, main_y[2], labels[4]), (2.6, main_y[3], labels[6])]
    side = [(7.2, side_y[0], labels[1], side_heights[0]),
            (7.2, side_y[1], labels[3], side_heights[1]),
            (7.2, side_y[2], labels[5], side_heights[2])]

    # The bottom-most box (main or side, whichever now sits lower) sets
    # where the search-date stamp goes, and how far the axes/figure must
    # extend downward to hold it without overlapping the lowest box.
    main_bottom_edge = main_y[-1] - MAIN_BOX_HEIGHT / 2 - BOX_PAD
    side_bottom_edge = side_y[-1] - side_heights[-1] / 2 - BOX_PAD
    content_bottom = min(main_bottom_edge, side_bottom_edge)
    stamp_y = content_bottom - STAMP_MARGIN
    footer_depth = (len(footer_lines(counts)) - 1) * FOOTER_LINE_GAP
    ylim_bottom = min(0.0, stamp_y - footer_depth - AXIS_MARGIN)

    y_range = BASE_YLIM_TOP - ylim_bottom
    fig_height = BASE_FIGSIZE[1] * y_range / BASE_YLIM_RANGE
    fig, ax = plt.subplots(figsize=(BASE_FIGSIZE[0], fig_height))
    ax.set_xlim(0, 10); ax.set_ylim(ylim_bottom, BASE_YLIM_TOP); ax.axis("off")

    text_artists = []
    box_patches = []

    def box(x, y, text, w=3.6, h=1.4, fc="#eef2f7"):
        patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.12", fc=fc, ec="#33475b", lw=1.1)
        ax.add_patch(patch)
        text_artist = ax.text(x, y, text, ha="center", va="center", fontsize=9)
        return text_artist, patch

    for x, y, t in main:
        text_artist, patch = box(x, y, t)
        text_artists.append(text_artist)
        box_patches.append(patch)
    for x, y, t, h in side:
        text_artist, patch = box(x, y, t, w=3.2, h=h, fc="#f7f2ee")
        text_artists.append(text_artist)
        box_patches.append(patch)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.1, color="#33475b"))

    for (x, y, _), (_, y2, _) in zip(main, main[1:]):
        arrow(x, y - 0.72, x, y2 + 0.72)
    for (x, y, _), (sx, sy, _, _) in zip(main[:3], side):
        arrow(x + 1.85, (y + sy) / 2 + 0.35, sx - 1.65, sy)

    for i, line in enumerate(footer_lines(counts)):
        ax.text(5, stamp_y - i * FOOTER_LINE_GAP, line,
                ha="center", fontsize=8, color="#5a6672")

    return fig, ax, text_artists, box_patches


def iter_labelled_boxes(text_artists, box_patches):
    """Iterate (text_artist, box_patch) pairs for geometry testing."""
    for text_artist, patch in zip(text_artists, box_patches):
        yield text_artist, patch


def render(counts: dict, out_path) -> Path:
    """Render the PRISMA diagram and save to PDF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, text_artists, box_patches = build_figure(counts)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
