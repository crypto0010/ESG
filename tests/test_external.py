"""Tests for Study 2: external validation against the LSEG/Refinitiv
firm-level extract (Task 18, gated on author input I1, now arrived).

R1.2 offered two remedies for the "coded literature attention, presented as
firm ESG performance" objection: reframe (done, Task 16), or validate against
independent firm-level data (this module). The two constructs -- what the
literature attends to (`analysis.factors`) and what firms actually do and
disclose (this module) -- are not the same thing, so every finding here is
worded as a CORRESPONDENCE check between two independently derived
measurement structures, never as validation of the literature model's
substantive claims (mirrors `tests/test_importance.py`'s
`test_no_output_label_claims_validation` pattern for the same reason).

Coverage is not missing at random (Ruling AF): 390 of 500 firms have at
least two of the 17 measured ESG indicators. The authors restrict to those
390 and disclose the restriction; every test that touches coverage checks
against the audited numbers, not invented ones. The proxy scores
(`4_top500_ESG_proxy_scores.csv`) are excluded by ruling and must never be
read by this module for inference.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from analysis import config, external

REAL_DATA = config.DATA


# --------------------------------------------------------------------------
# is_available / load_firm_data
# --------------------------------------------------------------------------

def test_is_available_is_true_for_the_real_data_directory():
    assert external.is_available(REAL_DATA) is True


def test_is_available_is_false_without_the_files(tmp_path):
    assert external.is_available(tmp_path) is False


def test_is_available_default_argument_finds_the_real_data():
    assert external.is_available() is True


def test_load_firm_data_returns_one_row_per_firm():
    df = external.load_firm_data(REAL_DATA)
    assert len(df) == 500
    assert df["RIC"].is_unique


def test_load_firm_data_flags_esg_covered_matching_the_audited_count():
    """Ruling AF's audit: 390 of 500 firms have any (>=2 of 17) ESG data."""
    df = external.load_firm_data(REAL_DATA)
    assert int(df["esg_covered"].sum()) == 390
    assert int((~df["esg_covered"]).sum()) == 110


def test_load_firm_data_never_reads_the_proxy_score_file():
    """Ruling AF: ESG_Proxy_Score / E_Score / S_Score / G_Score are author-
    derived and excluded by ruling -- must never be used for inference. The
    module may still NAME the proxy file in prose explaining the exclusion
    (the dispatch explicitly allows a Methods sentence to that effect) --
    what is actually forbidden is `_FILES` (what `load_firm_data` reads)
    pointing at it, or a loaded frame carrying its columns."""
    assert "4_top500_ESG_proxy_scores.csv" not in external._FILES.values()
    assert not any("proxy" in str(v).lower() for v in external._FILES.values())
    df = external.load_firm_data(REAL_DATA)
    assert "ESG_Proxy_Score" not in df.columns


def test_load_firm_data_derives_the_financial_controls():
    df = external.load_firm_data(REAL_DATA)
    for col in ("log_total_assets", "roa", "leverage"):
        assert col in df.columns
    have_assets = df["TotalAssets_INR"].notna()
    assert np.allclose(
        df.loc[have_assets, "log_total_assets"], np.log(df.loc[have_assets, "TotalAssets_INR"]))
    have_ni = have_assets & df["NetIncome_INR"].notna()
    assert np.allclose(
        df.loc[have_ni, "roa"], df.loc[have_ni, "NetIncome_INR"] / df.loc[have_ni, "TotalAssets_INR"])
    have_eq = have_assets & df["TotalEquity_INR"].notna()
    assert np.allclose(
        df.loc[have_eq, "leverage"],
        (df.loc[have_eq, "TotalAssets_INR"] - df.loc[have_eq, "TotalEquity_INR"]) / df.loc[have_eq, "TotalAssets_INR"])


def test_load_firm_data_prefers_fy2024_25_and_discloses_fallback_use():
    """Falls back to the most recent non-missing year per firm; the number
    of firms that needed a fallback must be countable, not hidden."""
    df = external.load_firm_data(REAL_DATA)
    assert "fundamentals_year" in df.columns
    assert "fundamentals_is_fallback" in df.columns
    have_year = df["fundamentals_year"].notna()
    assert (df.loc[have_year & ~df["fundamentals_is_fallback"], "fundamentals_year"]
           == "FY2024-25").all()
    assert (df.loc[have_year & df["fundamentals_is_fallback"], "fundamentals_year"]
           != "FY2024-25").all()
    # Audited: only 2 of 500 firms need a fallback year, 1 has no usable year at all.
    assert int(df["fundamentals_is_fallback"].sum()) == 2
    assert int(df["fundamentals_year"].isna().sum()) == 1


# --------------------------------------------------------------------------
# validate_schema
# --------------------------------------------------------------------------

def test_schema_validation_names_every_missing_column():
    problems = external.validate_schema(pd.DataFrame({"RIC": ["A"]}))
    assert problems
    assert any("esg_covered" in p for p in problems)
    assert any("MarketCap_USD" in p for p in problems)
    assert any("G_BoardIndependence_pct" in p for p in problems)


def test_schema_validation_passes_the_loaded_frame():
    assert external.validate_schema(external.load_firm_data(REAL_DATA)) == []


# --------------------------------------------------------------------------
# coverage_report
# --------------------------------------------------------------------------

def test_coverage_report_matches_the_audited_counts():
    df = external.load_firm_data(REAL_DATA)
    rep = external.coverage_report(df)
    assert rep["n_covered"] == 390
    assert rep["n_uncovered"] == 110


def test_coverage_report_matches_the_audited_size_bias():
    df = external.load_firm_data(REAL_DATA)
    rep = external.coverage_report(df)
    assert rep["median_marketcap_covered"] > rep["median_marketcap_uncovered"]
    assert 5.0e9 < rep["median_marketcap_covered"] < 5.4e9
    assert 2.0e9 < rep["median_marketcap_uncovered"] < 2.6e9
    assert rep["marketcap_mannwhitney_p"] < 1e-10
    assert rep["spearman_rank_vs_n_indicators"] < -0.35
    assert rep["spearman_rank_vs_n_indicators_p"] < 1e-15


def test_coverage_report_per_indicator_coverage_matches_the_audit():
    df = external.load_firm_data(REAL_DATA)
    rep = external.coverage_report(df)
    per = rep["per_indicator_coverage"]
    assert len(per) == 17
    assert abs(per["S_WomenManagers_pct"] - 0.254) < 0.01
    assert abs(per["S_Employees"] - 0.820) < 0.01
    assert min(per.values()) == pytest.approx(per["S_WomenManagers_pct"], abs=1e-9)
    assert max(per.values()) == pytest.approx(per["S_Employees"], abs=1e-9)


def test_coverage_report_sector_coverage_matches_the_audit():
    df = external.load_firm_data(REAL_DATA)
    rep = external.coverage_report(df)
    sector = rep["sector_coverage"]
    assert abs(sector["Technology"] - 0.698) < 0.02
    assert abs(sector["Consumer Non-Cyclicals"] - 0.871) < 0.02


def test_coverage_report_synthetic_frame_is_exact():
    df = pd.DataFrame({
        "esg_covered": [True, True, False, False],
        "MarketCap_USD": [10.0, 8.0, 2.0, 1.0],
        "n_esg_indicators": [10, 12, 0, 1],
        "Sector": ["A", "A", "A", "B"],
        "fundamentals_is_fallback": [False, True, False, False],
        "fundamentals_year": ["FY2024-25", "FY2023-24", None, "FY2024-25"],
    })
    df["E_CO2_Total"] = [1.0, np.nan, np.nan, np.nan]
    rep = external.coverage_report(df)
    assert rep["n_covered"] == 2
    assert rep["n_uncovered"] == 2
    assert rep["per_indicator_coverage"]["E_CO2_Total"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# governance_index
# --------------------------------------------------------------------------

def _governance_frame():
    return pd.DataFrame({
        "G_BoardIndependence_pct": [80.0, 40.0, np.nan, 60.0, 50.0],
        "G_WomenOnBoard_pct": [30.0, 10.0, 20.0, np.nan, 25.0],
        "G_AuditCommInd_pct": [100.0, 60.0, np.nan, np.nan, 75.0],
        "G_BoardSize": [5.0, 30.0, 9.0, 11.0, 7.0],
        "G_CEOPayRatio": [500.0, 1.0, 200.0, 300.0, 50.0],
    })


def test_governance_index_is_nan_with_fewer_than_two_items_present():
    df = _governance_frame()
    idx = external.governance_index(df)
    # row 2 (index 2) has only G_BoardIndependence_pct missing... check exact pattern:
    # row 2: BoardIndependence=NaN, WomenOnBoard=20.0, AuditCommInd=NaN -> only 1 present -> NaN
    # row 3: BoardIndependence=60.0, WomenOnBoard=NaN, AuditCommInd=NaN -> only 1 present -> NaN
    assert pd.isna(idx.iloc[2])
    assert pd.isna(idx.iloc[3])
    assert idx.iloc[0] == idx.iloc[0]  # not NaN
    assert idx.iloc[4] == idx.iloc[4]  # not NaN


def test_governance_index_ignores_board_size_and_pay_ratio():
    """G_BoardSize has no monotone quality interpretation and G_CEOPayRatio's
    direction is contested -- neither belongs in the index (dispatch)."""
    df = _governance_frame()
    perturbed = df.copy()
    perturbed["G_BoardSize"] = [999.0, -999.0, 0.0, 1e6, -1.0]
    perturbed["G_CEOPayRatio"] = [-1.0, 1e9, -1e9, 0.0, 42.0]
    idx1 = external.governance_index(df)
    idx2 = external.governance_index(perturbed)
    pd.testing.assert_series_equal(idx1, idx2)


def test_governance_index_is_standardised_with_zero_mean_when_complete():
    df = pd.DataFrame({
        "G_BoardIndependence_pct": np.linspace(20, 90, 50),
        "G_WomenOnBoard_pct": np.linspace(5, 45, 50)[::-1],
        "G_AuditCommInd_pct": np.linspace(50, 100, 50),
    })
    idx = external.governance_index(df)
    assert idx.notna().all()
    assert abs(idx.mean()) < 1e-8


def test_governance_index_on_real_data_has_the_expected_coverage():
    df = external.load_firm_data(REAL_DATA)
    idx = external.governance_index(df)
    assert idx.notna().sum() >= 380  # nearly all covered firms report >=2 board items


# --------------------------------------------------------------------------
# disclosure_completeness
# --------------------------------------------------------------------------

def test_disclosure_completeness_is_fraction_of_seventeen():
    row = {c: 1.0 for c in external.ESG_INDICATORS[:5]}
    for c in external.ESG_INDICATORS[5:]:
        row[c] = np.nan
    df = pd.DataFrame([row])
    out = external.disclosure_completeness(df)
    assert out.iloc[0] == pytest.approx(5 / 17)


def test_disclosure_completeness_bounds_are_zero_and_one():
    full = pd.DataFrame([{c: 1.0 for c in external.ESG_INDICATORS}])
    empty = pd.DataFrame([{c: np.nan for c in external.ESG_INDICATORS}])
    assert external.disclosure_completeness(full).iloc[0] == pytest.approx(1.0)
    assert external.disclosure_completeness(empty).iloc[0] == pytest.approx(0.0)


def test_disclosure_completeness_matches_n_esg_indicators_on_real_data():
    df = external.load_firm_data(REAL_DATA)
    out = external.disclosure_completeness(df)
    assert np.allclose(out * 17, df["n_esg_indicators"])


# --------------------------------------------------------------------------
# correspondence_test
# --------------------------------------------------------------------------

def test_correspondence_test_returns_the_documented_keys():
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=200)
    for key in ("partial_correlation", "ci_lo", "ci_hi", "n",
               "literature_model_value"):
        assert key in out


def test_correspondence_test_n_matches_the_complete_case_sample():
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=200)
    assert out["n"] == 388


def test_correspondence_test_partial_correlation_is_close_to_the_audited_value():
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=200)
    assert abs(out["partial_correlation"] - 0.186) < 0.02
    assert out["ci_lo"] < out["partial_correlation"] < out["ci_hi"]
    assert np.isfinite(out["ci_lo"]) and np.isfinite(out["ci_hi"])


def test_correspondence_test_literature_value_is_computed_not_hardcoded():
    """Never invent data: the 0.14 comparator must come from a live
    `analysis.factors.fit_efa` call, not a literal constant in this module."""
    src = inspect.getsource(external)
    assert "0.14" not in src
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=50)
    assert abs(out["literature_model_value"] - 0.1392) < 0.01


def test_correspondence_test_accepts_an_injected_literature_value():
    """run_all.py already fits the literature EFA once for the main pipeline
    -- correspondence_test must accept that value rather than refitting."""
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=50, literature_value=0.99)
    assert out["literature_model_value"] == 0.99


def test_correspondence_test_reports_fallback_use_in_the_sample():
    df = external.load_firm_data(REAL_DATA)
    out = external.correspondence_test(df, n_boot=50)
    assert out["n_fallback_in_sample"] == 1


def test_correspondence_test_raises_on_too_few_complete_rows():
    df = external.load_firm_data(REAL_DATA).iloc[:5].copy()
    df["esg_covered"] = True
    with pytest.raises(ValueError):
        external.correspondence_test(df, n_boot=50)


def test_correspondence_test_is_reproducible_under_the_global_seed():
    df = external.load_firm_data(REAL_DATA)
    a = external.correspondence_test(df, n_boot=200)
    b = external.correspondence_test(df, n_boot=200)
    assert a["ci_lo"] == b["ci_lo"]
    assert a["ci_hi"] == b["ci_hi"]


# --------------------------------------------------------------------------
# Wording / naming constraints
# --------------------------------------------------------------------------

BANNED_FACTOR_NAMES = ["Board Independence", "Carbon Emissions", "Employee Training",
                       "Water Management", "Executive Compensation", "Renewable Energy Use",
                       "Community Engagement"]


def test_no_withdrawn_factor_names_reintroduced_as_variables_or_labels():
    src = inspect.getsource(external)
    for name in BANNED_FACTOR_NAMES:
        assert name not in src


def test_no_output_label_claims_validation_of_the_literature_model():
    """Mirrors test_importance.py's rule: report a correspondence check
    between two independently derived measurement structures, never
    validation of the literature model's substantive claims.

    Uses `ast` (not a hand-rolled quote regex) to extract only genuine
    docstrings -- module, class and function -- since a regex built for
    small modules like `importance.py` mis-parses across the many
    ordinary double-quoted string literals a merge/derive-heavy module
    like this one contains.
    """
    import ast
    source = inspect.getsource(external)
    tree = ast.parse(source)
    allowed = {"never as validation"}
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    hits = []
    for node in nodes:
        doc = ast.get_docstring(node)
        if doc and "valid" in doc.lower():
            hits.append(" ".join(doc.split()))  # collapse newlines/indentation
    unexplained = [h for h in hits if not any(a in h.lower() for a in allowed)]
    assert not unexplained, f"unexplained validation language: {unexplained}"
