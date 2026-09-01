# tests/test_prisma.py
import itertools
import json
import pytest
from analysis.figures import fig_prisma

COMPLETE = {
    "search_string": "TITLE-ABS-KEY(esg)", "search_date": "2026-08-15", "databases": ["Scopus"],
    "filters": "English", "identified": 2198, "duplicates_removed": 198,
    "screened_title_abstract": 2000, "excluded_title_abstract": 700,
    "excluded_title_abstract_reasons": {"off topic": 700},
    "fulltext_assessed": 1300, "fulltext_excluded": 274,
    "fulltext_excluded_reasons": {"full text unavailable": 274}, "included": 1026,
}

WITH_DETAILED_REASONS = {
    "search_string": "TITLE-ABS-KEY(esg)", "search_date": "2026-08-15", "databases": ["Scopus"],
    "filters": "English", "identified": 2198, "duplicates_removed": 198,
    "screened_title_abstract": 2000, "excluded_title_abstract": 700,
    "excluded_title_abstract_reasons": {"off topic": 500, "not empirical": 200},
    "fulltext_assessed": 1300, "fulltext_excluded": 274,
    "fulltext_excluded_reasons": {"no ESG performance focus": 150, "insufficient detail to code": 124},
    "included": 1026,
}

ALL_REASONS_NULL = {
    "search_string": "TITLE-ABS-KEY(esg)", "search_date": "2026-08-15", "databases": ["Scopus"],
    "filters": "English", "identified": 2198, "duplicates_removed": 198,
    "screened_title_abstract": 2000, "excluded_title_abstract": 700,
    "excluded_title_abstract_reasons": {"off topic": None, "not empirical": None},
    "fulltext_assessed": 1300, "fulltext_excluded": 274,
    "fulltext_excluded_reasons": {"no ESG performance focus": None, "insufficient detail to code": None},
    "included": 1026,
}

def test_template_parses_and_has_the_known_endpoints():
    counts = fig_prisma.load_counts("templates/prisma_counts.json")
    assert counts["identified"] == 2198
    assert counts["included"] == 1026

def test_validate_accepts_a_consistent_set():
    assert fig_prisma.validate(COMPLETE) == []

def test_validate_catches_broken_arithmetic():
    bad = dict(COMPLETE, excluded_title_abstract=999)
    problems = fig_prisma.validate(bad)
    assert problems and any("title/abstract" in p for p in problems)

def test_validate_is_silent_about_unknowns():
    partial = {"identified": 2198, "included": 1026, "duplicates_removed": None,
               "screened_title_abstract": None, "excluded_title_abstract": None,
               "fulltext_assessed": None, "fulltext_excluded": None}
    assert fig_prisma.validate(partial) == []

def test_render_writes_a_pdf_with_unknowns_marked(tmp_path):
    partial = {"identified": 2198, "included": 1026, "duplicates_removed": None,
               "screened_title_abstract": None, "excluded_title_abstract": None,
               "fulltext_assessed": None, "fulltext_excluded": None,
               "search_string": None, "search_date": None}
    out = fig_prisma.render(partial, tmp_path / "prisma.pdf")
    assert out.exists() and out.stat().st_size > 1000

def test_render_refuses_to_invent(tmp_path):
    partial = {"identified": 2198, "included": 1026, "duplicates_removed": None,
               "screened_title_abstract": None, "excluded_title_abstract": None,
               "fulltext_assessed": None, "fulltext_excluded": None}
    labels = fig_prisma.box_labels(partial)
    assert any("not retained" in l for l in labels)
    assert not any("1172" in l for l in labels)   # 2198-1026, the tempting fabrication

def test_render_includes_exclusion_reasons_when_present(tmp_path):
    """Verify exclusion reasons are displayed in the diagram when populated."""
    out = fig_prisma.render(WITH_DETAILED_REASONS, tmp_path / "prisma_reasons.pdf")
    assert out.exists()

    # Extract text to verify reasons are present (text wrapping may break lines)
    labels = fig_prisma.box_labels(WITH_DETAILED_REASONS)
    reasons_text = labels[3] + labels[5]  # Title/abstract exclusions + fulltext exclusions
    assert "off topic" in reasons_text and "500" in reasons_text
    assert "not empirical" in reasons_text and "200" in reasons_text
    # Text wrapping may break "no ESG performance focus" across lines
    assert ("ESG" in reasons_text and "performance" in reasons_text and "150" in reasons_text)

def test_render_omits_exclusion_reasons_when_all_null(tmp_path):
    """Verify exclusion reasons are not displayed when all are null."""
    out_with_reasons = fig_prisma.render(WITH_DETAILED_REASONS, tmp_path / "with_reasons.pdf")
    out_without_reasons = fig_prisma.render(ALL_REASONS_NULL, tmp_path / "without_reasons.pdf")

    labels_with = fig_prisma.box_labels(WITH_DETAILED_REASONS)
    labels_without = fig_prisma.box_labels(ALL_REASONS_NULL)

    # When all reasons are null, the exclusion box should show only count
    assert labels_without[3].count("\n") < labels_with[3].count("\n")
    assert "off topic" not in labels_without[3]
    assert "n = 700" in labels_without[3]

def test_render_uses_embedded_fonts(tmp_path):
    """Verify PDF uses Type 42 fonts (embedded outlines, not bitmaps)."""
    out = fig_prisma.render(COMPLETE, tmp_path / "prisma_embedded.pdf")

    with open(out, "rb") as f:
        pdf_bytes = f.read()

    # Type 3 fonts are bitmap-based and will be rejected by publishers
    assert b"/Type3" not in pdf_bytes, "PDF must use embedded fonts (Type 42), not bitmap Type 3"
    # Type 42 fonts embed TrueType outlines via /FontFile2 or CIDFont structures
    assert (b"/FontFile2" in pdf_bytes or b"/CIDFontType2" in pdf_bytes), \
        "PDF should contain embedded font definitions (/FontFile2 or /CIDFontType2)"

def test_reason_text_stays_inside_its_box(tmp_path):
    """Reason labels must not overflow their box. Uses the template's own
    'no ESG performance focus' which is long enough to catch fixed-width issues."""
    import matplotlib.pyplot as plt

    counts = {
        "identified": 2198, "duplicates_removed": 198,
        "screened_title_abstract": 2000, "excluded_title_abstract": 700,
        "excluded_title_abstract_reasons": {"off topic": 500, "not empirical": 200},
        "fulltext_assessed": 1300, "fulltext_excluded": 274,
        "fulltext_excluded_reasons": {
            "no ESG performance focus": 150, "insufficient detail to code": 124,
        },
        "included": 1026, "search_string": None, "search_date": "2026-08-15",
    }
    fig, ax, text_artists, box_patches = fig_prisma.build_figure(counts)
    renderer = fig.canvas.get_renderer()

    # Check each text stays within its box
    for text_artist, patch in fig_prisma.iter_labelled_boxes(text_artists, box_patches):
        t_extent = text_artist.get_window_extent(renderer)
        b_extent = patch.get_window_extent(renderer)
        # Text x-extent must be within box x-extent
        assert t_extent.x0 >= b_extent.x0, \
            f"Text '{text_artist.get_text()!r}' overflows left of box"
        assert t_extent.x1 <= b_extent.x1, \
            f"Text '{text_artist.get_text()!r}' overflows right of box"
        # Text y-extent must be within box y-extent
        assert t_extent.y0 >= b_extent.y0, \
            f"Text '{text_artist.get_text()!r}' overflows bottom of box"
        assert t_extent.y1 <= b_extent.y1, \
            f"Text '{text_artist.get_text()!r}' overflows top of box"
    plt.close(fig)


# Minimum visible gap, in rendered display pixels at the figure's dpi, that
# must separate any two box patches sharing a column. The old hardcoded
# side-box centres produced an overlap of about -16px (negative, i.e. the
# boxes crossed); the current layout keeps roughly 212px of clearance
# between adjacent side boxes. 20px sits well clear of both: comfortably
# below the real design's margin, and comfortably above "just barely not
# touching".
MIN_BOX_GAP_PX = 20.0


def _assert_no_box_overlaps(counts):
    import matplotlib.pyplot as plt

    fig, ax, text_artists, box_patches = fig_prisma.build_figure(counts)
    renderer = fig.canvas.get_renderer()
    try:
        extents = [(i, p.get_window_extent(renderer)) for i, p in enumerate(box_patches)]
        for (i, e1), (j, e2) in itertools.combinations(extents, 2):
            assert not e1.overlaps(e2), (
                f"box_patches[{i}] and box_patches[{j}] overlap: "
                f"{tuple(e1.extents)} vs {tuple(e2.extents)}"
            )
            # Boxes that share an x-range (i.e. sit in the same column) must
            # keep a visible vertical gap between their nearest edges.
            shares_x = e1.x0 < e2.x1 and e2.x0 < e1.x1
            if shares_x:
                gap = e1.y0 - e2.y1 if e1.y0 >= e2.y1 else e2.y0 - e1.y1
                assert gap >= MIN_BOX_GAP_PX, (
                    f"box_patches[{i}] and box_patches[{j}] are only {gap:.1f}px "
                    f"apart (minimum {MIN_BOX_GAP_PX}px)"
                )
    finally:
        plt.close(fig)


def test_side_boxes_never_overlap_with_populated_reasons():
    """Reviewer-reported bug: variable-height side boxes ('Records excluded',
    'Full-text articles excluded') were positioned at hardcoded fixed
    centres and their rounded borders collided when reasons made them tall."""
    counts = {
        "identified": 2198, "duplicates_removed": 198,
        "screened_title_abstract": 2000, "excluded_title_abstract": 700,
        "excluded_title_abstract_reasons": {"off topic": 500, "not empirical": 200},
        "fulltext_assessed": 1300, "fulltext_excluded": 274,
        "fulltext_excluded_reasons": {
            "no ESG performance focus": 150, "insufficient detail to code": 124,
        },
        "included": 1026, "search_string": None, "search_date": "2026-08-15",
    }
    _assert_no_box_overlaps(counts)


def test_side_boxes_never_overlap_with_all_null_reasons():
    """The all-null (real template) case must also keep its boxes separated."""
    counts = fig_prisma.load_counts("templates/prisma_counts.json")
    _assert_no_box_overlaps(counts)
