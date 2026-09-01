"""Tests for the manuscript's figure-rendering modules (Task 13, revised: the
measurement model - not clustering - is the primary taxonomy result).

Figure numbering: 1 PRISMA (untouched, tested in test_prisma.py), 2 architecture,
3 scree/parallel analysis, 4 loadings heatmap, 5 descriptives, 6 benchmark,
7 clustering (secondary analysis), 8 external (Task 18, Study 2 firm-level
correspondence check, gated on author input I1).
"""
import numpy as np
import pandas as pd
import pytest

from analysis import clustering, config, descriptives, external, factors, loading, quality
from analysis.figures import (fig_architecture, fig_benchmark, fig_dendrogram,
                              fig_descriptives, fig_external, fig_loadings,
                              fig_scree, style)

DF = quality.clean(loading.load_scoring())

# Loaded once, at module scope, like DF above - `load_firm_data` is cheap
# (merges three CSVs), but `correspondence_test`'s bootstrap is not, so the
# n_boot=100 result is computed once here and reused by every fig_external
# test rather than paying for a fresh bootstrap in each one.
EXTERNAL_DF = external.load_firm_data()
EXTERNAL_COVERAGE = external.coverage_report(EXTERNAL_DF)
EXTERNAL_CORRESPONDENCE = external.correspondence_test(EXTERNAL_DF, n_boot=100)


def test_style_applies_without_error():
    style.apply()
    import matplotlib.pyplot as plt
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


# --------------------------------------------------------------------------
# style.truncate_label (I2): fig_loadings already had a word-boundary
# truncator (`_truncate_label`) that fig_descriptives needed and never got -
# moved to style.py so both figures share exactly one implementation.
# --------------------------------------------------------------------------

def test_truncate_label_never_splits_a_word():
    long_label = "Alignment with international standards (GRI, SASB, etc.), a very long label"
    out = style.truncate_label(long_label, width=34)
    assert out.endswith("…")
    prefix = out[:-1].rstrip()
    assert long_label.startswith(prefix)
    # the character immediately after the truncated prefix, in the ORIGINAL
    # label, must be a space (or nothing) - never mid-word
    assert long_label[len(prefix):len(prefix) + 1] in (" ", "")


def test_truncate_label_leaves_short_labels_unchanged():
    short_label = "Depth of ESG due diligence processes"
    assert style.truncate_label(short_label, width=200) == short_label


def test_truncate_label_default_width_matches_fig_loadings_original():
    # fig_loadings.py's own truncator defaulted to width=34; style.truncate_label
    # must default identically so fig_loadings' visual output is unchanged.
    long_label = "Alignment with international standards (GRI, SASB, etc.)"
    assert style.truncate_label(long_label) == style.truncate_label(long_label, width=34)


# --------------------------------------------------------------------------
# Figure 2: architecture
# --------------------------------------------------------------------------

def test_architecture_renders(tmp_path):
    p = fig_architecture.render(tmp_path / "f2.pdf")
    assert p.exists() and p.stat().st_size > 1000


def test_architecture_uses_embedded_fonts(tmp_path):
    p = fig_architecture.render(tmp_path / "f2.pdf")
    pdf_bytes = p.read_bytes()
    assert b"/Type3" not in pdf_bytes


def test_architecture_has_seven_stages_reflecting_the_current_pipeline():
    assert len(fig_architecture.STAGES) == 7
    text = " ".join(f"{t} {s}" for t, s in fig_architecture.STAGES).lower()
    for keyword in ("search", "coding", "reliability", "measurement model",
                    "predictive", "convergent importance", "study 2"):
        assert keyword in text


def test_architecture_stage_seven_is_never_called_validation():
    """Task 18 (Study 2) landed: the seventh stage is a firm-level
    correspondence check against independent LSEG/Refinitiv data, never
    described as validating the literature model's substantive claims --
    the same discipline enforced on the manuscript prose
    (tests/test_manuscript_claims.py::test_convergent_importance_is_never_called_validation)
    and on analysis/external.py itself."""
    stage_seven_text = " ".join(fig_architecture.STAGES[6]).lower()
    assert "valid" not in stage_seven_text


def test_architecture_does_not_use_the_old_tier_wording():
    text = " ".join(f"{t} {s}" for t, s in fig_architecture.STAGES).lower()
    assert "tier a" not in text and "tier b" not in text
    assert "tier c" not in text and "tier d" not in text


def test_architecture_box_text_stays_inside_its_box():
    import matplotlib.pyplot as plt
    fig, ax, text_artists, box_patches = fig_architecture.build_figure()
    renderer = fig.canvas.get_renderer()
    for text_artist, patch in zip(text_artists, box_patches):
        t = text_artist.get_window_extent(renderer)
        b = patch.get_window_extent(renderer)
        assert t.x0 >= b.x0, f"{text_artist.get_text()!r} overflows left of its box"
        assert t.x1 <= b.x1, f"{text_artist.get_text()!r} overflows right of its box"
        assert t.y0 >= b.y0, f"{text_artist.get_text()!r} overflows bottom of its box"
        assert t.y1 <= b.y1, f"{text_artist.get_text()!r} overflows top of its box"
    plt.close(fig)


def test_architecture_boxes_never_overlap():
    import itertools
    import matplotlib.pyplot as plt
    fig, ax, text_artists, box_patches = fig_architecture.build_figure()
    renderer = fig.canvas.get_renderer()
    unique_patches = list({id(p): p for p in box_patches}.values())
    extents = [p.get_window_extent(renderer) for p in unique_patches]
    for e1, e2 in itertools.combinations(extents, 2):
        assert not e1.overlaps(e2)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: scree plot with parallel analysis (NEW, justifies k=5)
# --------------------------------------------------------------------------

def test_scree_renders(tmp_path):
    pa = factors.parallel_analysis(DF, n_iter=25, seed=config.SEED)
    p = fig_scree.render(tmp_path / "f3.pdf", pa)
    assert p.exists() and p.stat().st_size > 1000


def test_scree_uses_embedded_fonts(tmp_path):
    pa = factors.parallel_analysis(DF, n_iter=25, seed=config.SEED)
    p = fig_scree.render(tmp_path / "f3.pdf", pa)
    assert b"/Type3" not in p.read_bytes()


def test_scree_marks_the_retained_count():
    pa = {"n_factors": 5,
          "eigenvalues": np.array([14.7, 5.2, 3.1, 2.4, 1.9, 1.1, 0.9]),
          "threshold": np.array([1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0])}
    import matplotlib.pyplot as plt
    fig, ax = fig_scree.build_axes(pa)
    texts = [t.get_text() for t in ax.texts]
    assert any("5" in t and "factor" in t.lower() for t in texts)
    plt.close(fig)


def test_scree_plots_both_observed_and_random_threshold_series():
    pa = {"n_factors": 3, "eigenvalues": np.array([5.0, 3.0, 2.0, 0.8, 0.5]),
          "threshold": np.array([1.2, 1.1, 1.0, 0.9, 0.8])}
    import matplotlib.pyplot as plt
    fig, ax = fig_scree.build_axes(pa)
    assert len(ax.lines) >= 2
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 4: factor loadings heatmap (NEW, the manuscript's centrepiece)
# --------------------------------------------------------------------------

def test_loadings_renders(tmp_path):
    r = factors.fit_efa(DF, n_factors=5)
    p = fig_loadings.render(tmp_path / "f4.pdf", r.loadings)
    assert p.exists() and p.stat().st_size > 1000


def test_loadings_uses_embedded_fonts(tmp_path):
    r = factors.fit_efa(DF, n_factors=5)
    p = fig_loadings.render(tmp_path / "f4.pdf", r.loadings)
    assert b"/Type3" not in p.read_bytes()


def test_loadings_covers_all_44_items_grouped_by_primary_factor():
    r = factors.fit_efa(DF, n_factors=5)
    import matplotlib.pyplot as plt
    fig, ax, order = fig_loadings.build_axes(r.loadings)
    assert len(order) == 44
    assert sorted(order) == sorted(r.loadings.index)

    # Grouped: each primary-factor label forms one contiguous run down the
    # row order (a label may not reappear once the run has moved past it).
    a = factors.assign_items(r.loadings, threshold=0.40)
    primaries = [a.loc[c, "primary"] for c in order]
    runs = []
    for p in primaries:
        if not runs or runs[-1] != p:
            runs.append(p)
    assert len(runs) == len(set(primaries))
    plt.close(fig)


def test_loadings_never_hardcodes_the_seven_original_esg_factor_names():
    import inspect
    src = inspect.getsource(__import__("analysis.figures.fig_loadings", fromlist=["x"]))
    banned = ["Board Independence", "Carbon Emissions", "Employee Training",
              "Water Management", "Executive Compensation", "Renewable Energy Use",
              "Community Engagement"]
    for name in banned:
        assert name not in src


def test_loadings_annotations_stay_inside_their_cells():
    loadings = pd.DataFrame({
        "F1": [0.82, 0.05, -0.71, 0.30],
        "F2": [0.10, 0.88, 0.15, -0.65],
    }, index=["A-D1", "A-D2", "E-D1", "E-D2"])
    import matplotlib.pyplot as plt
    fig, ax, order = fig_loadings.build_axes(loadings)
    renderer = fig.canvas.get_renderer()
    n_items, n_factors = len(order), loadings.shape[1]
    checked = 0
    for txt in ax.texts:
        tx, ty = txt.get_position()
        j, i = round(tx), round(ty)
        if not (0 <= j < n_factors and 0 <= i < n_items):
            continue
        x0d, x1d = sorted(ax.transData.transform((j - 0.5, i))[:1].tolist() +
                          ax.transData.transform((j + 0.5, i))[:1].tolist())
        y0d, y1d = sorted([ax.transData.transform((j, i - 0.5))[1],
                           ax.transData.transform((j, i + 0.5))[1]])
        ext = txt.get_window_extent(renderer)
        assert ext.x0 >= x0d - 1, f"annotation at ({i},{j}) overflows left of its cell"
        assert ext.x1 <= x1d + 1, f"annotation at ({i},{j}) overflows right of its cell"
        assert ext.y0 >= y0d - 1, f"annotation at ({i},{j}) overflows below its cell"
        assert ext.y1 <= y1d + 1, f"annotation at ({i},{j}) overflows above its cell"
        checked += 1
    assert checked == n_items * n_factors
    plt.close(fig)


def test_loadings_mutes_cells_below_the_simple_structure_threshold():
    """Below-threshold loadings must not render with the same saturated
    diverging colour as above-threshold ones - otherwise simple structure
    is not visible at a glance."""
    loadings = pd.DataFrame({"F1": [0.90, 0.10], "F2": [0.05, 0.85]},
                            index=["A-D1", "A-D2"])
    import matplotlib.pyplot as plt
    fig, ax, order = fig_loadings.build_axes(loadings, threshold=0.40)
    im = [c for c in ax.images][0]
    arr = im.get_array()
    # matplotlib converts NaN entries in the imshow array into a masked
    # array (rendered with cmap.set_bad, i.e. blanked) - the below-threshold
    # cells (0.10, 0.05) must be masked, the above-threshold ones (0.90,
    # 0.85) must not.
    mask = np.ma.getmaskarray(arr)
    assert mask.sum() == 2
    assert not mask[0, 0] and not mask[1, 1]
    assert mask[0, 1] and mask[1, 0]
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 5: descriptives (I2: fixed - word-boundary truncation, not a bare
# character slice)
# --------------------------------------------------------------------------

def test_descriptives_renders(tmp_path):
    p = fig_descriptives.render(tmp_path / "f5.pdf",
                                sub=descriptives.subdimension_table(DF),
                                trend=descriptives.yearly_trend(DF),
                                counts=descriptives.yearly_counts(DF))
    assert p.exists() and p.stat().st_size > 1000


def test_descriptives_uses_embedded_fonts(tmp_path):
    p = fig_descriptives.render(tmp_path / "f5.pdf",
                                sub=descriptives.subdimension_table(DF),
                                trend=descriptives.yearly_trend(DF),
                                counts=descriptives.yearly_counts(DF))
    assert b"/Type3" not in p.read_bytes()


# --------------------------------------------------------------------------
# Task 15 fixes: the 2025-partial-year trend collapse and the 11-line/
# 7-colour palette collision, both found by looking at the rendered figure.
# --------------------------------------------------------------------------

def test_yearly_counts_matches_the_real_year_distribution():
    counts = descriptives.yearly_counts(DF)
    assert int(counts.loc[2025]) < int(counts.loc[2024])
    assert int(counts.sum()) == len(DF)


def test_descriptives_trend_palette_has_11_distinct_colours_for_11_dimensions():
    assert len(config.DIMENSIONS) == 11
    assert len(set(style.PALETTE)) >= 11
    assert len(set(style.MARKERS)) >= 11


def test_descriptives_trend_lines_use_11_distinct_colour_marker_pairs():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = fig_descriptives.build_axes(
        descriptives.subdimension_table(DF), descriptives.yearly_trend(DF),
        counts=descriptives.yearly_counts(DF))
    lines = [ln for ln in ax2.get_lines()]
    # two Line2D objects per dimension (solid history + dashed final
    # segment); colour+marker pairs must be unique per dimension.
    solid = [ln for ln in lines if ln.get_linestyle() in ("-", "solid")]
    pairs = {(ln.get_color(), ln.get_marker()) for ln in solid}
    assert len(pairs) == len(config.DIMENSIONS)
    plt.close(fig)


def test_descriptives_trend_final_segment_is_dashed_and_earlier_segments_are_not():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = fig_descriptives.build_axes(
        descriptives.subdimension_table(DF), descriptives.yearly_trend(DF),
        counts=descriptives.yearly_counts(DF))
    dashed = [ln for ln in ax2.get_lines() if ln.get_linestyle() in ("--", "dashed")]
    solid = [ln for ln in ax2.get_lines() if ln.get_linestyle() in ("-", "solid")]
    assert len(dashed) == len(config.DIMENSIONS)
    assert len(solid) == len(config.DIMENSIONS)
    for ln in dashed:
        xdata = list(ln.get_xdata())
        assert len(xdata) == 2  # only the final year-pair segment
    plt.close(fig)


def test_descriptives_trend_panel_discloses_the_partial_final_year():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = fig_descriptives.build_axes(
        descriptives.subdimension_table(DF), descriptives.yearly_trend(DF),
        counts=descriptives.yearly_counts(DF))
    fig.canvas.draw()  # tick/annotation text is not finalised before draw()
    counts = descriptives.yearly_counts(DF)
    last_year, last_n = counts.index[-1], int(counts.iloc[-1])
    texts = " ".join(t.get_text() for t in ax2.texts)
    xticklabels = " ".join(t.get_text() for t in ax2.get_xticklabels())
    combined = f"{ax2.get_title()} {texts} {xticklabels}"
    assert str(last_year) in combined
    assert f"n={last_n}" in combined
    plt.close(fig)


# The exact mid-word breaks the prior implementation produced from
# `config.SUBDIMENSIONS[c][:38]` on the real labels - a visible defect a
# reader would see immediately, and the concrete regression this guards.
_KNOWN_MID_WORD_BREAKS = [
    "Proactive vs. Reactive Compliance Appr",
    "Comprehensiveness of risk identificati",
    "Integration of ESG risks into core bus",
    "Data collection and analysis technolog",
    "Stakeholder accessibility of informati",
]


def test_descriptives_yticklabels_never_truncate_mid_word():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = fig_descriptives.build_axes(
        descriptives.subdimension_table(DF), descriptives.yearly_trend(DF))
    all_labels = " | ".join(t.get_text() for t in ax1.get_yticklabels())
    for broken in _KNOWN_MID_WORD_BREAKS:
        assert broken not in all_labels
    plt.close(fig)


def test_descriptives_yticklabels_that_truncate_end_with_an_ellipsis_at_a_word_boundary():
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = fig_descriptives.build_axes(
        descriptives.subdimension_table(DF), descriptives.yearly_trend(DF))
    for tick in ax1.get_yticklabels():
        code, _, label_part = tick.get_text().partition("  ")
        full = config.SUBDIMENSIONS[code]
        if label_part == full:
            continue  # short enough: not truncated
        assert label_part.endswith("…")
        prefix = label_part[:-1].rstrip()
        assert full.startswith(prefix)
        assert full[len(prefix):len(prefix) + 1] in (" ", "")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 6: model benchmark (unchanged from the Task 13 brief)
# --------------------------------------------------------------------------

def test_benchmark_renders_with_intervals(tmp_path):
    summary = pd.DataFrame({"mae": [0.4, 0.6], "mae_lo": [0.35, 0.55],
                            "mae_hi": [0.45, 0.65]}, index=["RandomForest", "Dummy"])
    p = fig_benchmark.render(tmp_path / "f6.pdf", summary=summary)
    assert p.exists() and p.stat().st_size > 1000


def test_benchmark_marks_the_dummy_floor():
    summary = pd.DataFrame({"mae": [0.4, 0.6], "mae_lo": [0.35, 0.55],
                            "mae_hi": [0.45, 0.65]}, index=["RandomForest", "Dummy"])
    labels = fig_benchmark.bar_labels(summary)
    assert any("floor" in l.lower() for l in labels)


# --------------------------------------------------------------------------
# Figure 7: clustering, now a SECONDARY analysis (weak separation)
# --------------------------------------------------------------------------

def test_dendrogram_renders(tmp_path):
    p = fig_dendrogram.render(tmp_path / "f7.pdf", df=DF,
                              scan=clustering.silhouette_scan(DF, range(2, 8)))
    assert p.exists() and p.stat().st_size > 1000


def test_dendrogram_uses_embedded_fonts(tmp_path):
    p = fig_dendrogram.render(tmp_path / "f7.pdf", df=DF,
                              scan=clustering.silhouette_scan(DF, range(2, 8)))
    assert b"/Type3" not in p.read_bytes()


def test_dendrogram_is_labelled_as_a_secondary_analysis():
    scan = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.15, 0.18, 0.20, 0.224]})
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_dendrogram.build_axes(DF, scan)
    all_text = " ".join(t.get_text() for ax in (ax1, ax2) for t in ax.texts)
    all_text += " " + (ax1.get_title() or "") + " " + (ax2.get_title() or "")
    if fig._suptitle is not None:
        all_text += " " + fig._suptitle.get_text()
    assert "secondary" in all_text.lower()
    plt.close(fig)


def test_dendrogram_panels_do_not_collide():
    """Regression guard: the right panel's 'silhouette' y-axis label was
    found (by visual inspection) overlapping the left panel's leaf tick
    labels, since a `orientation='left'` dendrogram draws its 44 leaf
    labels right up against the panel boundary."""
    import matplotlib.pyplot as plt
    scan = clustering.silhouette_scan(DF, range(2, 8))
    fig, ax1, ax2 = fig_dendrogram.build_axes(DF, scan)
    fig.canvas.draw()  # tick/axis-label positions are only finalised on draw
    renderer = fig.canvas.get_renderer()
    ylabel_ext = ax2.yaxis.label.get_window_extent(renderer)
    for tick_label in ax1.get_yticklabels():
        leaf_ext = tick_label.get_window_extent(renderer)
        assert not ylabel_ext.overlaps(leaf_ext), (
            f"ax2's y-axis label overlaps ax1 leaf label {tick_label.get_text()!r}")
    plt.close(fig)


def test_dendrogram_reports_the_weak_separation_finding():
    scan = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.15, 0.18, 0.20, 0.224]})
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_dendrogram.build_axes(DF, scan)
    all_text = " ".join(t.get_text() for ax in (ax1, ax2) for t in ax.texts)
    all_text += " " + (ax1.get_title() or "") + " " + (ax2.get_title() or "")
    assert "0.22" in all_text or "weak" in all_text.lower()
    plt.close(fig)


def test_dendrogram_reports_the_boundary_argmax_using_clustering_verdict_from_scan():
    """Also (I2-adjacent): the real data's k=15 argmax sits at the edge of
    its tested range (2..16) - `clustering.verdict_from_scan`'s
    `on_boundary` flag says so, and the figure must surface it rather than
    leaving '(weak separation)' to carry that meaning alone."""
    import matplotlib.pyplot as plt
    scan = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.10, 0.15, 0.18, 0.224]})
    assert clustering.verdict_from_scan(scan)["on_boundary"] is True   # sanity check the fixture
    fig, ax1, ax2 = fig_dendrogram.build_axes(DF, scan)
    title = ax2.get_title().lower()
    assert "edge" in title or "boundary" in title
    plt.close(fig)


def test_dendrogram_does_not_claim_a_boundary_argmax_when_the_optimum_is_interior():
    import matplotlib.pyplot as plt
    scan = pd.DataFrame({"k": [2, 3, 4, 5, 6], "silhouette": [0.10, 0.15, 0.30, 0.20, 0.12]})
    assert clustering.verdict_from_scan(scan)["on_boundary"] is False  # sanity check the fixture
    fig, ax1, ax2 = fig_dendrogram.build_axes(DF, scan)
    title = ax2.get_title().lower()
    assert "edge" not in title and "boundary" not in title
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 8: external (Task 18, Study 2 firm-level correspondence check)
# --------------------------------------------------------------------------

def test_external_renders(tmp_path):
    p = fig_external.render(tmp_path / "f8.pdf", EXTERNAL_DF, EXTERNAL_COVERAGE,
                            EXTERNAL_CORRESPONDENCE)
    assert p.exists() and p.stat().st_size > 1000


def test_external_uses_embedded_fonts(tmp_path):
    p = fig_external.render(tmp_path / "f8.pdf", EXTERNAL_DF, EXTERNAL_COVERAGE,
                            EXTERNAL_CORRESPONDENCE)
    assert b"/Type3" not in p.read_bytes()


def test_external_left_panel_shows_every_sector_in_the_coverage_report():
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_external.build_axes(EXTERNAL_DF, EXTERNAL_COVERAGE, EXTERNAL_CORRESPONDENCE)
    assert len(ax1.get_yticklabels()) == len(EXTERNAL_COVERAGE["sector_coverage"])
    plt.close(fig)


def test_external_right_panel_scatter_covers_only_covered_firms_with_a_governance_index():
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_external.build_axes(EXTERNAL_DF, EXTERNAL_COVERAGE, EXTERNAL_CORRESPONDENCE)
    scatter = [c for c in ax2.collections][0]
    covered = EXTERNAL_DF.loc[EXTERNAL_DF["esg_covered"].astype(bool)]
    expected_n = int((external.governance_index(covered).notna()
                      & external.disclosure_completeness(covered).notna()).sum())
    assert scatter.get_offsets().shape[0] == expected_n
    plt.close(fig)


def test_external_annotation_stays_inside_the_right_panel():
    """Nothing clipped: the stats annotation box must not overflow ax2's
    own extent. `get_window_extent()` is stale for auto-positioned artists
    until `fig.canvas.draw()`, so draw before measuring."""
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_external.build_axes(EXTERNAL_DF, EXTERNAL_COVERAGE, EXTERNAL_CORRESPONDENCE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    annotation = [t for t in ax2.texts if "partial r" in t.get_text()][0]
    t_ext = annotation.get_window_extent(renderer)
    ax_ext = ax2.get_window_extent(renderer)
    assert t_ext.x0 >= ax_ext.x0, "annotation overflows left of the right panel"
    assert t_ext.x1 <= ax_ext.x1, "annotation overflows right of the right panel"
    assert t_ext.y0 >= ax_ext.y0, "annotation overflows below the right panel"
    assert t_ext.y1 <= ax_ext.y1, "annotation overflows above the right panel"
    plt.close(fig)


def test_external_reports_correspondence_not_validation():
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = fig_external.build_axes(EXTERNAL_DF, EXTERNAL_COVERAGE, EXTERNAL_CORRESPONDENCE)
    all_text = " ".join(t.get_text() for ax in (ax1, ax2) for t in ax.texts)
    all_text += " " + (ax1.get_title() or "") + " " + (ax2.get_title() or "")
    assert "not validation" in all_text.lower() or "correspondence" in all_text.lower()
    plt.close(fig)


def test_external_synthetic_sector_coverage_matches_bar_heights():
    """Regression guard against a stale/re-derived coverage series: the bar
    heights must be exactly the `coverage['sector_coverage']` values passed
    in, not something re-computed from `df` inside the figure."""
    import matplotlib.pyplot as plt
    synthetic_df = EXTERNAL_DF.copy()
    synthetic_coverage = dict(EXTERNAL_COVERAGE)
    synthetic_coverage["sector_coverage"] = {"Alpha": 0.25, "Beta": 0.75}
    fig, ax1, ax2 = fig_external.build_axes(synthetic_df, synthetic_coverage, EXTERNAL_CORRESPONDENCE)
    heights = sorted(p.get_width() for p in ax1.patches)
    assert heights == pytest.approx([25.0, 75.0])
    plt.close(fig)


# --------------------------------------------------------------------------
# Cross-cutting global constraints
# --------------------------------------------------------------------------

BANNED_FACTOR_NAMES = ["Board Independence", "Carbon Emissions", "Employee Training",
                       "Water Management", "Executive Compensation", "Renewable Energy Use",
                       "Community Engagement"]


def test_no_banned_esg_factor_names_in_any_figure_module():
    import inspect
    from analysis.figures import fig_architecture as a
    from analysis.figures import fig_benchmark as e
    from analysis.figures import fig_dendrogram as f
    from analysis.figures import fig_descriptives as d
    from analysis.figures import fig_external as g
    from analysis.figures import fig_loadings as c
    from analysis.figures import fig_scree as b
    src = "".join(inspect.getsource(m) for m in (a, b, c, d, e, f, g))
    for name in BANNED_FACTOR_NAMES:
        assert name not in src


def test_factor_labels_are_never_hardcoded_only_f1_through_f5():
    r = factors.fit_efa(DF, n_factors=5)
    assert list(r.loadings.columns) == ["F1", "F2", "F3", "F4", "F5"]
