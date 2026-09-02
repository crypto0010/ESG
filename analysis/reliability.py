"""Inter-rater reliability for R1.3 / R2.2 (Task 12). Gated on author input
I2 -- the completed `templates/IRR_double_coding.xlsx` (two independent
experts, 51 articles stratified by publication year, the same 0-5 abstract-
based coding used for the original 1,026-article corpus).

`is_available` and `load_double_coding` are the gate: they raise, or report
unavailable, rather than ever substituting a zero, an imputed value, or a
simulated coder judgement for a cell that has not actually been coded.
Fabricating a reliability statistic from placeholder data would be worse
than reporting none -- as of this writing both expert sheets are entirely
blank, so `is_available()` is False and every function below that needs the
coded data still raises when called on the real workbook.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import config, stats

_SHEETS = ("Expert A", "Expert B")
_LABELS = list(range(config.SCALE_MIN, config.SCALE_MAX + 1))
# Coders are asked (see `sampling._INSTRUCTIONS`) to write "RECALL" in the
# Notes column when they recognise an article and remember its original
# score; "remember" is accepted too since free text does not always match
# the instructed keyword exactly.
_RECALL_PATTERN = re.compile(r"recall|remember", re.IGNORECASE)


def _default_path(path) -> Path:
    return Path(path) if path is not None else config.TEMPLATES / "IRR_double_coding.xlsx"


def _read_score_sheet(path: Path, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    missing = [c for c in config.SCORE_COLS if c not in raw.columns]
    if missing:
        raise KeyError(f"{sheet!r} in {path.name} is missing sub-dimension columns: {missing}")
    if "SR.NO" not in raw.columns:
        raise KeyError(f"{sheet!r} in {path.name} is missing the SR.NO column")

    raw = raw.dropna(subset=["SR.NO"]).copy()
    raw["SR.NO"] = raw["SR.NO"].astype(int)
    if raw["SR.NO"].duplicated().any():
        dupes = sorted(raw.loc[raw["SR.NO"].duplicated(), "SR.NO"].unique().tolist())
        raise ValueError(f"{sheet} has duplicate SR.NO values: {dupes}")

    scores = raw[config.SCORE_COLS].apply(pd.to_numeric, errors="coerce")
    scores.index = pd.Index(raw["SR.NO"].to_numpy(), name="sr_no")
    return scores.sort_index()


def load_double_coding(path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both experts' score frames (columns = `config.SCORE_COLS`,
    aligned row-for-row by SR.NO).

    Raises -- never returns zeros, NaN-as-zero, or any other manufactured
    value -- when a sheet is uncoded, partially coded, holds a value outside
    0-5, or when the two sheets do not cover the same set of articles. Each
    error names the offending sheet and how many cells are affected.
    """
    path = _default_path(path)
    if not path.exists():
        raise FileNotFoundError(f"double-coding workbook not found at {path}")

    frames = {}
    for sheet in _SHEETS:
        scores = _read_score_sheet(path, sheet)
        total = int(scores.size)
        n_blank = int(scores.isna().sum().sum())
        if n_blank == total:
            raise ValueError(
                f"{sheet} has not been coded: all {total} score cells are blank"
            )
        if n_blank > 0:
            raise ValueError(
                f"{sheet} is only partially coded: {n_blank} of {total} score cells "
                "are blank; every sub-dimension must be scored for every article"
            )
        out_of_range = (scores < config.SCALE_MIN) | (scores > config.SCALE_MAX)
        n_bad = int(out_of_range.sum().sum())
        if n_bad:
            raise ValueError(
                f"{sheet} has {n_bad} score cell(s) outside the "
                f"{config.SCALE_MIN}-{config.SCALE_MAX} range"
            )
        frames[sheet] = scores

    a, b = frames["Expert A"], frames["Expert B"]
    if set(a.index) != set(b.index):
        only_a = sorted(set(a.index) - set(b.index))
        only_b = sorted(set(b.index) - set(a.index))
        raise ValueError(
            "Expert A and Expert B coded different articles (by SR.NO): "
            f"{len(only_a)} article(s) only in Expert A {only_a}, "
            f"{len(only_b)} article(s) only in Expert B {only_b}"
        )
    b = b.loc[a.index]
    return a.reset_index(drop=True), b.reset_index(drop=True)


def is_available(path=None) -> bool:
    """True only when the workbook exists and both sheets are fully, validly coded."""
    path = _default_path(path)
    if not path.exists():
        return False
    try:
        load_double_coding(path)
    except (ValueError, KeyError, FileNotFoundError):
        return False
    return True


def weighted_kappa(a, b, weights="quadratic") -> float:
    a = np.rint(np.asarray(a, dtype=float)).astype(int)
    b = np.rint(np.asarray(b, dtype=float)).astype(int)
    if np.array_equal(a, b):
        # cohen_kappa_score returns nan (with a RuntimeWarning) when
        # observed and expected agreement are both 1 -- the po == pe == 1
        # degenerate ratio that arises for exact agreement. Perfect
        # agreement is unambiguously kappa = 1 regardless of that ratio, so
        # this is handled directly rather than by suppressing the warning.
        return 1.0
    return float(cohen_kappa_score(a, b, weights=weights, labels=_LABELS))


def icc21(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater
    (Shrout & Fleiss, 1979; McGraw & Wong, 1996), computed directly from the
    standard two-way ANOVA variance-component formula for the k=2-rater
    case this module always has (Expert A, Expert B).

    Computed in closed form rather than by calling out to a general stats
    library on every invocation: pingouin 0.6.1's `intraclass_corr` (whose
    "ICC(A,1)" row is this same absolute-agreement estimate) costs roughly
    half a second per call at the pooled size (44 sub-dimensions x n
    articles) `bootstrap_overall` resamples over -- at 2000 resamples that
    would take upwards of an hour for one bootstrap. The formula below is
    microseconds and, for k=2, mathematically identical; verified against
    pingouin directly (to floating-point precision, on identical, random,
    systematic-offset and noisy fixtures) in
    `tests/test_reliability.py::test_icc21_matches_pingouin_directly`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.column_stack([a, b])
    n, k = x.shape
    if n < 2:
        raise ValueError(f"icc21 needs at least 2 targets, got {n}")

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)
    sst = float(np.sum((x - grand) ** 2))
    ssr = float(k * np.sum((row_means - grand) ** 2))
    ssc = float(n * np.sum((col_means - grand) ** 2))
    sse = sst - ssr - ssc

    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = sse / ((n - 1) * (k - 1))

    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    value = (msr - mse) / denom if denom != 0 else float("nan")
    if not np.isfinite(value):
        raise ValueError(
            "icc21 produced a non-finite result; the input is likely degenerate "
            "(e.g. zero variance across targets or raters)"
        )
    return float(value)


def percent_agreement(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(a == b))


def _summary_row(label: str, a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "code": label,
        "n": int(len(a)),
        "percent_agreement": percent_agreement(a, b),
        "kappa": weighted_kappa(a, b),
        "icc": icc21(a, b),
    }


def reliability_table(expert_a: pd.DataFrame, expert_b: pd.DataFrame) -> pd.DataFrame:
    """One row per sub-dimension code, one row per dimension letter (pooled
    over its 4 sub-dimensions), and one `OVERALL` row (pooled across all 44
    x n_articles item-article pairs). Columns: `n`, `percent_agreement`,
    `kappa`, `icc`.
    """
    rows = []
    for code in config.SCORE_COLS:
        a = expert_a[code].to_numpy(dtype=float)
        b = expert_b[code].to_numpy(dtype=float)
        rows.append(_summary_row(code, a, b))
    for letter in config.DIMENSIONS:
        cols = [c for c in config.SCORE_COLS if c.startswith(f"{letter}-")]
        a = expert_a[cols].to_numpy(dtype=float).ravel()
        b = expert_b[cols].to_numpy(dtype=float).ravel()
        rows.append(_summary_row(letter, a, b))
    a = expert_a[config.SCORE_COLS].to_numpy(dtype=float).ravel()
    b = expert_b[config.SCORE_COLS].to_numpy(dtype=float).ravel()
    rows.append(_summary_row("OVERALL", a, b))
    return pd.DataFrame(rows).set_index("code")[["n", "percent_agreement", "kappa", "icc"]]


def bootstrap_overall(expert_a: pd.DataFrame, expert_b: pd.DataFrame, n_boot: int = 2000) -> dict:
    """BCa interval on the pooled weighted kappa and ICC(2,1), resampling
    ARTICLES (rows), not individual cells. The 44 sub-dimension scores
    within one article are not independent observations -- they share
    whatever made that particular article easy or hard to code -- so
    cell-level resampling would understate the true sampling uncertainty.
    """
    a = expert_a[config.SCORE_COLS].to_numpy(dtype=float)
    b = expert_b[config.SCORE_COLS].to_numpy(dtype=float)
    n_articles = a.shape[0]
    if n_articles < 2:
        raise ValueError(f"bootstrap_overall needs at least 2 articles, got {n_articles}")

    def _kappa_stat(idx_float):
        idx = idx_float.astype(int)
        return weighted_kappa(a[idx].ravel(), b[idx].ravel())

    def _icc_stat(idx_float):
        idx = idx_float.astype(int)
        return icc21(a[idx].ravel(), b[idx].ravel())

    article_idx = np.arange(n_articles)
    k_point, k_lo, k_hi = stats.bca_bootstrap(
        article_idx, _kappa_stat, n_boot=n_boot, seed=config.SEED)
    i_point, i_lo, i_hi = stats.bca_bootstrap(
        article_idx, _icc_stat, n_boot=n_boot, seed=config.SEED)

    return {
        "n_articles": int(n_articles),
        "n_boot": int(n_boot),
        "kappa": k_point, "kappa_lo": k_lo, "kappa_hi": k_hi,
        "icc": i_point, "icc_lo": i_lo, "icc_hi": i_hi,
    }


def disagreement_profile(expert_a: pd.DataFrame, expert_b: pd.DataFrame) -> pd.DataFrame:
    """Per sub-dimension: mean absolute difference and the proportion of
    exact agreements, sorted worst (largest mean absolute difference)
    first. Which items proved hardest to code consistently is useful to
    anyone reusing the instrument, independent of the pooled statistics.
    """
    rows = []
    for code in config.SCORE_COLS:
        a = expert_a[code].to_numpy(dtype=float)
        b = expert_b[code].to_numpy(dtype=float)
        diff = np.abs(a - b)
        rows.append({
            "code": code,
            "mean_abs_diff": float(diff.mean()),
            "exact_agreement": float(np.mean(diff == 0)),
        })
    return pd.DataFrame(rows).set_index("code").sort_values("mean_abs_diff", ascending=False)


def recall_flags(path=None) -> dict:
    """Count and list the articles either coder flagged, in their Notes
    column, as one whose original score they recalled.

    Memory contamination is a disclosed limitation of re-using the original
    coders for double coding (see `sampling._INSTRUCTIONS`) -- this
    function only surfaces what the coders themselves flagged; it does not,
    and cannot, correct for unflagged recall.
    """
    path = _default_path(path)
    per_sheet = {}
    all_flagged = set()
    for sheet in _SHEETS:
        raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        if "Notes" not in raw.columns or "SR.NO" not in raw.columns:
            per_sheet[sheet] = {"count": 0, "articles": []}
            continue
        notes = raw["Notes"].fillna("").astype(str)
        flagged = raw.loc[notes.str.contains(_RECALL_PATTERN), "SR.NO"]
        srnos = sorted(int(v) for v in flagged.dropna().tolist())
        per_sheet[sheet] = {"count": len(srnos), "articles": srnos}
        all_flagged.update(srnos)
    return {
        "per_sheet": per_sheet,
        "total_flags": sum(v["count"] for v in per_sheet.values()),
        "articles_flagged_by_either": sorted(all_flagged),
    }
