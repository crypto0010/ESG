"""Study 2: correspondence between the literature measurement model and
independent, firm-level ESG/financial data (Task 18, gated on author input
I1, arrived as the LSEG/Refinitiv extract under ``data/``).

R1.2 gave two remedies for the objection that the study's outcome variable
is expert-coded literature attention presented as firm ESG performance:
reframe the contribution (done, Task 16), or validate against independent
firm-level data. This module does the second half. The literature model
(``analysis.factors``) describes what the ESG literature attends to; this
module describes what firms actually do and disclose. These are different
constructs measured on different populations by different methods, so
every result here is reported as a correspondence check between two
independently derived measurement structures, never as validation of the
literature model's substantive claims -- the same discipline
``analysis.importance`` applies to its own internal-consistency check (see
``tests/test_importance.py::test_no_output_label_claims_validation`` and
this module's own mirror of that test).

Coverage is not missing at random (author audit, Ruling AF): of the 500
firms in the extract, 390 report at least two of the 17 measured ESG
indicators (``esg_covered``); covered firms are systematically larger
(median market cap USD 5.19bn vs 2.28bn, Mann-Whitney p = 3.8e-14) and
better-covered sectors skew toward Consumer Non-Cyclicals over Technology.
The authors restrict every Study 2 finding to the 390 covered firms and
disclose the restriction throughout, rather than only in a limitations
paragraph: findings generalise to well-disclosing large-cap Indian listed
firms, not to the full market.

The fourth file, ``4_top500_ESG_proxy_scores.csv`` (``ESG_Proxy_Score`` and
its E/S/G pillars), is excluded from inference by the same ruling: it is
exactly the mean of three pillar scores the author never documented the
construction of, its E pillar ranks absolute emissions with no intensity or
peer-group adjustment (so it tracks sector rather than performance), and
two of its G indicators are inert against their own pillar. This module
never reads that file, and nothing here depends on it -- see
``coverage_report``'s docstring for the one place its existence is
mentioned, in a Methods-facing sentence explaining why it was not used.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config, stats

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

# The 17 measured ESG indicators in `3_top500_ESG_raw_indicators.csv`.
ESG_INDICATORS = [
    "E_CO2_Total", "E_CO2_Scope1", "E_CO2_Scope2", "E_EnergyUse_Total",
    "E_WaterWithdrawal_Total", "E_Waste_Total", "S_Employees",
    "S_WomenEmployees_pct", "S_WomenManagers_pct", "S_InjuryRate_TIR",
    "S_AvgTrainingHours", "S_HumanRightsPolicy", "G_BoardIndependence_pct",
    "G_BoardSize", "G_WomenOnBoard_pct", "G_AuditCommInd_pct", "G_CEOPayRatio",
]

# The three board indicators with a direct measured analogue to the
# literature's governance items (a priori dimension I, most of factor F1).
# `G_BoardSize` and `G_CEOPayRatio` are measured but deliberately excluded:
# board size has no monotone quality interpretation (neither a large nor a
# small board is unambiguously "better governed") and the pay ratio's
# direction is contested in the literature (high pay-for-performance
# alignment vs. excessive extraction read the same ratio oppositely). Both
# are reported separately (`coverage_report`), never silently dropped.
GOVERNANCE_ITEMS = ["G_BoardIndependence_pct", "G_WomenOnBoard_pct", "G_AuditCommInd_pct"]
GOVERNANCE_EXCLUDED_ITEMS = ["G_BoardSize", "G_CEOPayRatio"]

# A firm counts as ESG-covered once at least this many of the 17 indicators
# are reported. Matches the author audit exactly: 390 of 500 firms clear
# this bar (Mann-Whitney p = 3.8e-14 on market cap between the two groups,
# Spearman(market-cap rank, n indicators) = -0.393, p = 7e-20).
MIN_ESG_INDICATORS_FOR_COVERAGE = 2

# Fundamentals columns needed for the firm-level controls (log assets, ROA,
# leverage). Revenue is carried through but not required for any control.
_CONTROL_FUNDAMENTALS = ["NetIncome_INR", "TotalAssets_INR", "TotalEquity_INR"]
_FUNDAMENTALS_COLUMNS = ["Revenue_INR"] + _CONTROL_FUNDAMENTALS

# Most recent fiscal year first: `load_firm_data` prefers FY2024-25 and
# falls back to the most recent year with complete control fundamentals.
FISCAL_YEARS_DESC = ["FY2024-25", "FY2023-24", "FY2022-23", "FY2021-22", "FY2020-21"]

_CONTROL_COLUMNS = ["log_total_assets", "roa", "leverage"]

REQUIRED_COLUMNS = (
    ["RIC", "CompanyName", "Sector", "MarketCap_USD",
     "esg_covered", "n_esg_indicators",
     "fundamentals_year", "fundamentals_is_fallback"]
    + ESG_INDICATORS + GOVERNANCE_EXCLUDED_ITEMS + _CONTROL_COLUMNS
)

_FILES = {
    "companies": "1_top500_indian_companies_by_marketcap.csv",
    "fundamentals": "2_top500_fundamentals_FY2020-21_to_FY2024-25.csv",
    "esg": "3_top500_ESG_raw_indicators.csv",
}
# Deliberately excluded from `_FILES`: "4_top500_ESG_proxy_scores.csv"
# (Ruling AF -- author-derived, not an independent criterion, and its E
# pillar measures sector rather than performance; see module docstring).


def is_available(data_dir=None) -> bool:
    """True when the three files `load_firm_data` needs are present."""
    d = Path(data_dir) if data_dir is not None else config.DATA
    return all((d / name).exists() for name in _FILES.values())


def _select_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """One row per RIC: FY2024-25's Revenue/NetIncome/TotalAssets/TotalEquity
    where all three control fundamentals are non-missing, else the most
    recent earlier fiscal year that has them complete. `fundamentals_year`
    is None and every value NaN for a firm with no usable year at all;
    `fundamentals_is_fallback` marks a firm that needed a year other than
    FY2024-25, so how many firms rely on a fallback is always countable.
    """
    rows = []
    for ric, grp in fundamentals.groupby("RIC"):
        by_year = grp.set_index("FiscalYear")
        chosen = None
        for year in FISCAL_YEARS_DESC:
            if year in by_year.index and by_year.loc[year, _CONTROL_FUNDAMENTALS].notna().all():
                chosen = year
                break
        row = {"RIC": ric}
        if chosen is None:
            row["fundamentals_year"] = None
            row["fundamentals_is_fallback"] = False
            row.update({c: np.nan for c in _FUNDAMENTALS_COLUMNS})
        else:
            src = by_year.loc[chosen]
            row["fundamentals_year"] = chosen
            row["fundamentals_is_fallback"] = chosen != FISCAL_YEARS_DESC[0]
            row.update({c: src[c] for c in _FUNDAMENTALS_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def load_firm_data(data_dir=None) -> pd.DataFrame:
    """Merge the three usable LSEG extract files on RIC into one row per firm.

    Never reads the proxy-score file (Ruling AF). Adds:
    - `n_esg_indicators` / `esg_covered`: how many of the 17 indicators a
      firm reports, and whether that clears `MIN_ESG_INDICATORS_FOR_COVERAGE`.
    - `fundamentals_year` / `fundamentals_is_fallback`: which fiscal year's
      fundamentals were used for the controls below, and whether that was a
      fallback from FY2024-25 (see `_select_fundamentals`).
    - `log_total_assets`, `roa`, `leverage`: the firm-level controls used by
      `correspondence_test`, derived from the selected fundamentals year.
    """
    d = Path(data_dir) if data_dir is not None else config.DATA
    companies = pd.read_csv(d / _FILES["companies"])
    fundamentals = pd.read_csv(d / _FILES["fundamentals"])
    esg = pd.read_csv(d / _FILES["esg"])

    df = companies.merge(esg[["RIC"] + ESG_INDICATORS], on="RIC", how="left")
    df = df.merge(_select_fundamentals(fundamentals), on="RIC", how="left")

    n_ind = df[ESG_INDICATORS].notna().sum(axis=1)
    df["n_esg_indicators"] = n_ind
    df["esg_covered"] = n_ind >= MIN_ESG_INDICATORS_FOR_COVERAGE

    df["log_total_assets"] = np.log(df["TotalAssets_INR"])
    df["roa"] = df["NetIncome_INR"] / df["TotalAssets_INR"]
    df["leverage"] = (df["TotalAssets_INR"] - df["TotalEquity_INR"]) / df["TotalAssets_INR"]

    return df


def validate_schema(df: pd.DataFrame) -> list[str]:
    """A complaint per column `load_firm_data` promises but `df` lacks; empty when complete."""
    return [f"missing required column: {col!r}" for col in REQUIRED_COLUMNS if col not in df.columns]


# --------------------------------------------------------------------------
# coverage_report
# --------------------------------------------------------------------------

def coverage_report(df: pd.DataFrame) -> dict:
    """Coverage is not missing at random: this is the evidence for that
    claim, computed fresh from `df` rather than asserted.

    Reports, over the FULL frame passed in (typically all 500 firms, not
    the 390-firm restriction Study 2's other functions apply): the covered
    / uncovered split, per-indicator coverage, the market-cap size bias
    between the two groups (Mann-Whitney U, and Spearman between market-cap
    rank and the number of indicators reported), sector-level coverage, and
    how many firms needed a fallback fiscal year for their controls.
    """
    covered = df["esg_covered"].astype(bool)
    n_covered = int(covered.sum())
    n_uncovered = int((~covered).sum())

    present_indicators = [c for c in ESG_INDICATORS if c in df.columns]
    per_indicator = {c: float(df[c].notna().mean()) for c in present_indicators}

    mcap_covered = df.loc[covered, "MarketCap_USD"].dropna().to_numpy(dtype=float)
    mcap_uncovered = df.loc[~covered, "MarketCap_USD"].dropna().to_numpy(dtype=float)
    if mcap_covered.size and mcap_uncovered.size:
        _, mwu_p = sps.mannwhitneyu(mcap_covered, mcap_uncovered)
        median_covered = float(np.median(mcap_covered))
        median_uncovered = float(np.median(mcap_uncovered))
    else:
        mwu_p = float("nan")
        median_covered = float(np.median(mcap_covered)) if mcap_covered.size else float("nan")
        median_uncovered = float(np.median(mcap_uncovered)) if mcap_uncovered.size else float("nan")

    rank = df["MarketCap_USD"].rank(ascending=False)
    n_ind = df["n_esg_indicators"]
    valid = rank.notna() & n_ind.notna()
    if valid.sum() >= 2:
        rho, rho_p = sps.spearmanr(rank[valid], n_ind[valid])
    else:
        rho, rho_p = float("nan"), float("nan")

    sector_coverage = df.groupby("Sector")["esg_covered"].mean().astype(float).to_dict()

    n_fallback = int(df["fundamentals_is_fallback"].fillna(False).sum()) \
        if "fundamentals_is_fallback" in df.columns else 0
    n_no_fundamentals = int(df["fundamentals_year"].isna().sum()) \
        if "fundamentals_year" in df.columns else 0

    return {
        "n_covered": n_covered,
        "n_uncovered": n_uncovered,
        "per_indicator_coverage": per_indicator,
        "median_marketcap_covered": median_covered,
        "median_marketcap_uncovered": median_uncovered,
        "marketcap_mannwhitney_p": float(mwu_p),
        "spearman_rank_vs_n_indicators": float(rho),
        "spearman_rank_vs_n_indicators_p": float(rho_p),
        "sector_coverage": sector_coverage,
        "n_fallback_fundamentals_year": n_fallback,
        "n_no_usable_fundamentals": n_no_fundamentals,
    }


# --------------------------------------------------------------------------
# Firm-level constructs
# --------------------------------------------------------------------------

def governance_index(df: pd.DataFrame) -> pd.Series:
    """Standardised mean of the three board indicators with a direct
    literature analogue (`GOVERNANCE_ITEMS`); NaN where a firm reports
    fewer than two of them. Each item is z-scored across `df`'s rows before
    averaging, so the three differently-scaled percentages are commensurable.
    `G_BoardSize` and `G_CEOPayRatio` are read by nothing in this function
    (see `GOVERNANCE_EXCLUDED_ITEMS`'s docstring for why).
    """
    items = df[GOVERNANCE_ITEMS]
    z = (items - items.mean()) / items.std(ddof=0)
    n_present = items.notna().sum(axis=1)
    index = z.mean(axis=1, skipna=True)
    index[n_present < 2] = np.nan
    index.name = "governance_index"
    return index


def disclosure_completeness(df: pd.DataFrame) -> pd.Series:
    """Fraction of the 17 ESG indicators a firm reports -- the measured,
    firm-level analogue of what the literature's factor F2 (Disclosure
    Quality and Reporting Standardisation) captures at the corpus level.
    """
    out = df[ESG_INDICATORS].notna().sum(axis=1) / len(ESG_INDICATORS)
    out.name = "disclosure_completeness"
    return out


# --------------------------------------------------------------------------
# correspondence_test
# --------------------------------------------------------------------------

def _design_matrix(sub: pd.DataFrame) -> np.ndarray:
    controls = sub[_CONTROL_COLUMNS].to_numpy(dtype=float)
    dummies = pd.get_dummies(sub["Sector"], drop_first=True, dtype=float).to_numpy()
    intercept = np.ones((len(sub), 1))
    return np.hstack([intercept, controls, dummies])


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _partial_correlation(sub: pd.DataFrame) -> float:
    g = governance_index(sub).to_numpy(dtype=float)
    d = disclosure_completeness(sub).to_numpy(dtype=float)
    X = _design_matrix(sub)
    rg = _residualize(g, X)
    rd = _residualize(d, X)
    if np.std(rg) == 0 or np.std(rd) == 0:
        return float("nan")
    return float(np.corrcoef(rg, rd)[0, 1])


def literature_governance_disclosure_correlation() -> float:
    """The literature measurement model's F1 (Strategic Integration & Risk
    Governance) - F2 (Disclosure Quality & Reporting Standardisation)
    factor correlation, fit fresh from `analysis.factors.fit_efa` on the
    coded literature corpus -- this is the comparator `correspondence_test`
    reports against, never a literal constant, so it always reflects the
    currently fitted model rather than a number transcribed once and left
    to drift.
    """
    from . import factors, loading, quality  # local import: keeps this module import-light
    df_lit = quality.clean(loading.load_scoring())
    efa = factors.fit_efa(df_lit, n_factors=5)
    return float(efa.factor_correlations.loc["F1", "F2"])


def correspondence_test(df: pd.DataFrame, n_boot: int = 2000, seed=None,
                        literature_value: float = None) -> dict:
    """Partial correlation between the firm-level governance index and
    disclosure completeness, controlling for log total assets, ROA,
    leverage and sector, with a bootstrap interval -- compared against the
    literature model's F1-F2 factor correlation.

    This is a correspondence check between two independently derived
    measurement structures, never as validation of the literature model's
    substantive claims: the literature model describes what the ESG
    literature attends to, this test describes what firms actually do and
    disclose, and the two are commensurable only at this one governance /
    disclosure seam.

    Restricted to `esg_covered` firms with a complete governance index,
    disclosure completeness, sector and control set -- coverage is not
    missing at random (`coverage_report`), so this restriction is disclosed
    via `n` and `n_fallback_in_sample` rather than left implicit.

    `literature_value` lets a caller (`analysis.run_all`, which already
    fits the literature EFA once for the main pipeline) supply the F1-F2
    correlation instead of paying to refit it here; the default (None)
    computes it via `literature_governance_disclosure_correlation`.
    """
    if "esg_covered" in df.columns:
        covered = df.loc[df["esg_covered"].astype(bool)].copy()
    else:
        covered = df.copy()

    g = governance_index(covered)
    d = disclosure_completeness(covered)
    mask = g.notna() & d.notna()
    for col in ["Sector"] + _CONTROL_COLUMNS:
        mask &= covered[col].notna() if col in covered.columns else False
    sample = covered.loc[mask].reset_index(drop=True)
    n = len(sample)
    if n < 10:
        raise ValueError(
            f"correspondence_test needs at least 10 complete firm-rows "
            f"(governance index, disclosure completeness, sector and all three "
            f"controls all present) after restricting to ESG-covered firms; got {n}"
        )

    def statistic(idx_float):
        idx = idx_float.astype(int)
        return _partial_correlation(sample.iloc[idx])

    point, lo, hi = stats.bca_bootstrap(
        np.arange(n), statistic, n_boot=n_boot,
        seed=config.SEED if seed is None else seed,
    )

    lit_value = (literature_governance_disclosure_correlation()
                if literature_value is None else float(literature_value))

    n_fallback = int(sample["fundamentals_is_fallback"].fillna(False).sum()) \
        if "fundamentals_is_fallback" in sample.columns else None

    return {
        "partial_correlation": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "n": n,
        "n_fallback_in_sample": n_fallback,
        "literature_model_value": lit_value,
        "literature_model_label": "literature F1 (governance) - F2 (disclosure) factor correlation",
    }
