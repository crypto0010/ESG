"""Data-quality audit and cleaning tests."""
import numpy as np
from analysis import config, loading, quality


def test_audit_finds_exactly_nine_missing_cells():
    rep = quality.audit(loading.load_scoring())
    assert rep.missing == [
        (45, "H-D1"), (166, "E-D2"), (241, "G-D3"), (291, "I-D4"), (476, "K-D1"),
        (569, "D-D4"), (668, "I-D3"), (671, "C-D1"), (973, "J-D3"),
    ]


def test_audit_finds_the_two_out_of_range_values():
    rep = quality.audit(loading.load_scoring())
    assert rep.out_of_range == [(409, "J-D1", 9.0), (710, "D-D4", -2.0)]


def test_audit_finds_no_duplicate_titles():
    assert quality.audit(loading.load_scoring()).duplicate_titles == []


def test_audit_counts_articles_and_years():
    rep = quality.audit(loading.load_scoring())
    assert rep.n_articles == 1026
    assert rep.year_counts[2024] == 693


def test_audit_surfaces_the_one_missing_article_title():
    """SR.NO 336 has no title in the source. Task 1's loader preserves the gap;
    the audit is what discloses it to the manuscript's Methods section."""
    rep = quality.audit(loading.load_scoring())
    assert rep.missing_metadata == [(336, "title")]


def test_clean_puts_every_score_inside_the_scale():
    clean = quality.clean(loading.load_scoring())
    vals = clean[config.SCORE_COLS].to_numpy()
    assert np.isfinite(vals).all()
    assert vals.min() >= config.SCALE_MIN and vals.max() <= config.SCALE_MAX


def test_clean_preserves_shape_and_leaves_valid_scores_untouched():
    raw = loading.load_scoring()
    clean = quality.clean(raw)
    assert len(clean) == len(raw)
    untouched = raw.loc[raw["sr_no"] == 1, config.SCORE_COLS].to_numpy()
    assert (clean.loc[clean["sr_no"] == 1, config.SCORE_COLS].to_numpy() == untouched).all()


def test_clean_imputes_the_known_bad_cells_to_the_column_median():
    raw = loading.load_scoring()
    clean = quality.clean(raw)
    expected = raw.loc[(raw["J-D1"] >= 0) & (raw["J-D1"] <= 5), "J-D1"].median()
    assert clean.loc[clean["sr_no"] == 409, "J-D1"].iloc[0] == expected


def test_report_markdown_mentions_every_finding():
    text = quality.report_markdown(quality.audit(loading.load_scoring()))
    assert "9" in text and "409" in text and "710" in text


def test_audit_flags_whitespace_only_metadata_as_missing():
    """Whitespace-only metadata cells (after stripping at load time) are reported
    as missing, not silently passed as clean."""
    df = loading.load_scoring().head(3).copy()
    df.loc[df.index[1], "authors"] = "   "
    rep = quality.audit(df)
    assert (int(df.loc[df.index[1], "sr_no"]), "authors") in rep.missing_metadata


def test_clean_leaves_the_missing_title_missing():
    """clean() operates on the 44 score columns only; metadata gaps are
    disclosed by audit(), never repaired."""
    clean = quality.clean(loading.load_scoring())
    assert clean.loc[clean["sr_no"] == 336, "title"].isna().all()
