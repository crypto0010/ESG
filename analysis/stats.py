"""Statistical protocol for R1.5 / R2.9. Every reported number comes from here."""
import warnings
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from . import config


@dataclass
class FriedmanResult:
    statistic: float
    df: int
    p_value: float
    kendalls_w: float


def bca_bootstrap(data, statistic, n_boot=10000, alpha=0.05, seed=None):
    """Bias-corrected and accelerated bootstrap interval."""
    x = np.asarray(data, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError(f"bca_bootstrap needs at least 2 observations, got {n}")
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    point = float(statistic(x))

    boot = np.array([float(statistic(x[rng.integers(0, n, n)])) for _ in range(n_boot)])

    prop = np.mean(boot < point)
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = sps.norm.ppf(prop)

    jack = np.array([float(statistic(np.delete(x, i))) for i in range(n)])
    dev = jack.mean() - jack
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    a = 0.0 if denom == 0 else float(np.sum(dev ** 3) / denom)
    if not (np.isfinite(z0) and np.isfinite(a)):
        raise ValueError(
            f"bca_bootstrap produced a non-finite bias-correction (z0={z0}, a={a}); "
            "the statistic is likely degenerate on this data (e.g. zero variance)"
        )

    def endpoint(z_alpha):
        adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(np.clip(sps.norm.cdf(adj) * 100, 0, 100))

    lo = np.percentile(boot, endpoint(sps.norm.ppf(alpha / 2)))
    hi = np.percentile(boot, endpoint(sps.norm.ppf(1 - alpha / 2)))
    return point, float(lo), float(hi)


def friedman(matrix) -> FriedmanResult:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("friedman needs a (n_targets, n_models) matrix with >= 2 models")
    with warnings.catch_warnings():
        # scipy warns (RuntimeWarning: invalid value in scalar divide) and returns NaN
        # for a zero-variance matrix; we convert that into a clear exception below
        # instead of letting the bare warning leak past this wrapper.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        stat, p = sps.friedmanchisquare(*m.T)
    if not (np.isfinite(stat) and np.isfinite(p)):
        raise ValueError(
            "friedman produced a non-finite result, which scipy returns when there is "
            "zero variance across models (e.g. every model scores identically on every "
            "target); the test is undefined on such a matrix"
        )
    n, k = m.shape
    return FriedmanResult(float(stat), k - 1, float(p), float(stat / (n * (k - 1))))


def nemenyi(matrix) -> pd.DataFrame:
    return sp.posthoc_nemenyi_friedman(np.asarray(matrix, dtype=float))


def wilcoxon_pairs(matrix, labels=None) -> pd.DataFrame:
    m = np.asarray(matrix, dtype=float)
    labels = list(labels) if labels is not None else [f"m{i}" for i in range(m.shape[1])]
    rows = []
    for i, j in combinations(range(m.shape[1]), 2):
        d = m[:, i] - m[:, j]
        if not np.any(d):
            # Identical predictions on every target: no evidence of a difference.
            rows.append({"a": labels[i], "b": labels[j], "statistic": 0.0,
                         "p_value": 1.0, "rank_biserial": 0.0})
            continue
        stat, p = sps.wilcoxon(m[:, i], m[:, j])
        nz = d[d != 0]
        r = np.sign(nz) * sps.rankdata(np.abs(nz)) if nz.size else np.array([0.0])
        total = np.abs(r).sum()
        rows.append({
            "a": labels[i], "b": labels[j], "statistic": float(stat), "p_value": float(p),
            "rank_biserial": float(r.sum() / total) if total else 0.0,
        })
    return pd.DataFrame(rows)


def bh_fdr(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError(
            "bh_fdr received non-finite p-values; statsmodels would propagate NaN "
            "across the whole family and blank out valid comparisons. "
            f"Offending indices: {np.flatnonzero(~np.isfinite(p)).tolist()}"
        )
    return multipletests(p, method="fdr_bh")[1]
