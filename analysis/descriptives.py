"""Tier A: directly inspectable statistics, answering R1.7."""
import numpy as np
import pandas as pd

from . import config


def cronbach_alpha(frame: pd.DataFrame) -> float:
    k = frame.shape[1]
    if k < 2:
        return float("nan")
    item_var = frame.var(axis=0, ddof=1).sum()
    total_var = frame.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - item_var / total_var))


def subdimension_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in config.SCORE_COLS:
        s = df[code]
        rows.append({
            "code": code,
            "label": config.SUBDIMENSIONS[code],
            "dimension": config.DIMENSIONS[code[0]],
            "mean": s.mean(),
            "sd": s.std(ddof=1),
            "median": s.median(),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "zero_rate": float((s == 0).mean()),
        })
    return pd.DataFrame(rows).set_index("code")


def dimension_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for letter, name in config.DIMENSIONS.items():
        cols = [c for c in config.SCORE_COLS if c.startswith(letter + "-")]
        block = df[cols]
        rows.append({
            "dimension": letter,
            "name": name,
            "mean": block.to_numpy().mean(),
            "sd": block.to_numpy().std(ddof=1),
            "cronbach_alpha": cronbach_alpha(block),
        })
    return pd.DataFrame(rows).set_index("dimension")


def yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for letter in config.DIMENSIONS:
        cols = [c for c in config.SCORE_COLS if c.startswith(letter + "-")]
        out[letter] = df.groupby("year")[cols].mean().mean(axis=1)
    return pd.DataFrame(out).sort_index()


def yearly_counts(df: pd.DataFrame) -> pd.Series:
    """Number of coded articles contributing to each year of `yearly_trend`.

    The final year is typically a partial year (indexing runs ahead of
    full-year coverage), so any figure built on `yearly_trend` should show
    this alongside it rather than let a thin final year read as a genuine
    decline.
    """
    return df.groupby("year").size().sort_index()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[config.SCORE_COLS].corr(method="spearman")
