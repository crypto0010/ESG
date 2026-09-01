"""Measurement model: exploratory factor analysis (Task 10B, R2.11).

Supersedes Task 10's Ward clustering as the manuscript's primary taxonomy
result. Task 10's negative finding (weak silhouette separation) was a method
artefact of hard clustering forcing one box per item onto 44 ordinal
indicators of overlapping latent constructs; EFA models that overlap
directly via oblique factor correlations instead of forcing a partition.

`analysis/clustering.py` is unmodified and stays in the codebase as a
reported secondary analysis.

`factorability` and `parallel_analysis` take an explicit `basis` parameter
("pearson" | "spearman" | "polychoric") so the diagnostics always describe
the same matrix as extraction, rather than silently disagreeing with it.
Both default to "pearson", matching `fit_efa`: extraction treats the items
as continuous inputs to `factor_analyzer`'s minres estimator (its own
internal, Pearson-based correlation), the standard applied practice for
Likert-type data.

That choice is justified by the polychoric robustness check, not by any
reference figure: the Pearson solution agrees with the polychoric solution
(the methodologically correct treatment for these ordinal items) at
Tucker's congruence >= 0.99 on every factor, so Pearson is the basis
closest to the correct one for this data - see `polychoric_matrix`,
`fit_efa_polychoric`, and `sensitivity_by_correlation_basis`, which reports
how the shipped Pearson solution compares to both the Spearman and the
polychoric alternatives, including where they disagree, instead of
resolving the disagreement silently.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from scipy import stats as sps
from scipy.optimize import minimize_scalar

from . import config, descriptives

_EPS = 1e-6


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The 44 coded items, in `config.SCORE_COLS` order, when present.

    Falls back to every numeric column for frames that are not the coded
    dataset (e.g. synthetic noise in the factorability/parallel-analysis
    discrimination tests), so the diagnostics are usable on arbitrary data.
    """
    cols = [c for c in config.SCORE_COLS if c in df.columns]
    if cols:
        return df[cols]
    return df.select_dtypes(include=[np.number])


def _spearman_corr(X: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix, computed as Pearson correlation of ranks.

    Exactly matches `df.corr(method="spearman")` (verified to float
    precision) but is column-agnostic and fast enough to call ~500-700 times
    per parallel-analysis / bootstrap run.
    """
    ranks = np.apply_along_axis(sps.rankdata, 0, X)
    return np.corrcoef(ranks, rowvar=False)


def _raw_corr(X: np.ndarray, basis: str) -> np.ndarray:
    """Correlation matrix of a raw (n, p) array under `basis`.

    'pearson' or 'spearman' only - a polychoric correlation needs each
    column's category thresholds, which this raw-array entry point does not
    carry; see `_correlation_matrix` for the polychoric case.
    """
    if basis == "pearson":
        return np.corrcoef(X, rowvar=False)
    if basis == "spearman":
        return _spearman_corr(X)
    raise ValueError(f"_raw_corr supports basis='pearson' or 'spearman', got {basis!r}")


def _correlation_matrix(df: pd.DataFrame, basis: str) -> np.ndarray:
    """Correlation matrix for `df`'s coded items under `basis`:

    'pearson' (matches `fit_efa`'s own extraction), 'spearman', or
    'polychoric' (see `polychoric_matrix`).
    """
    if basis == "polychoric":
        return polychoric_matrix(df).to_numpy()
    return _raw_corr(_item_matrix(df).to_numpy(dtype=float), basis)


def tucker_congruence(x, y) -> float:
    """Tucker's phi between two loading vectors: sum(xy) / sqrt(sum(x^2) sum(y^2))."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))


def _greedy_match(ref: np.ndarray, other: np.ndarray) -> dict:
    """Match each column of `other` to its best-congruence column of `ref`.

    Greedy on absolute congruence (largest |phi| pairs claimed first, each
    factor used at most once on either side), per the brief's matching rule.
    Returns {ref_col_index: (other_col_index, signed_phi)}.
    """
    k, kb = ref.shape[1], other.shape[1]
    phi = np.empty((k, kb))
    for i in range(k):
        for j in range(kb):
            phi[i, j] = tucker_congruence(ref[:, i], other[:, j])
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(phi), axis=None), phi.shape))[0]
    used_ref, used_other, pairs = set(), set(), {}
    for i, j in order:
        i, j = int(i), int(j)
        if i in used_ref or j in used_other:
            continue
        pairs[i] = (j, float(phi[i, j]))
        used_ref.add(i)
        used_other.add(j)
        if len(used_ref) == min(k, kb):
            break
    return pairs


# --------------------------------------------------------------------------
# Factorability diagnostics
# --------------------------------------------------------------------------

def factorability(df: pd.DataFrame, basis: str = "pearson") -> dict:
    """Kaiser-Meyer-Olkin adequacy and Bartlett's test of sphericity.

    `basis` selects the correlation matrix: 'pearson' (default, matching
    `fit_efa`'s own extraction), 'spearman', or 'polychoric'. See the module
    docstring for why 'pearson' is the default.
    """
    X = _item_matrix(df)
    R = _correlation_matrix(df, basis)
    n, p = X.shape

    Rinv = np.linalg.pinv(R)
    d = np.sqrt(np.diag(Rinv))
    partial = -Rinv / np.outer(d, d)
    Rz = R.copy()
    np.fill_diagonal(Rz, 0.0)
    np.fill_diagonal(partial, 0.0)
    r2, p2 = Rz ** 2, partial ** 2

    kmo_per_item = pd.Series(r2.sum(axis=0) / (r2.sum(axis=0) + p2.sum(axis=0)), index=X.columns)
    kmo_overall = float(r2.sum() / (r2.sum() + p2.sum()))

    _, logdet = np.linalg.slogdet(R)
    chi2 = float(-(n - 1 - (2 * p + 5) / 6) * logdet)
    dfree = int(p * (p - 1) / 2)
    p_value = float(sps.chi2.sf(chi2, dfree))

    return {
        "kmo_overall": kmo_overall,
        "kmo_per_item": kmo_per_item,
        "bartlett_chi2": chi2,
        "bartlett_df": dfree,
        "bartlett_p": p_value,
        "is_factorable": bool(kmo_overall >= 0.6 and p_value < 0.05),
    }


def parallel_analysis(df: pd.DataFrame, n_iter=500, percentile=95, seed=None,
                       basis="pearson") -> dict:
    """Horn's parallel analysis.

    Retain factors whose observed eigenvalue exceeds the `percentile`-th
    percentile of eigenvalues from `n_iter` random matrices of the same
    shape (i.i.d. normal columns, same n and p). Retention stops at the
    first observed eigenvalue that does not clear its threshold (eigenvalues
    and thresholds are both effectively decreasing in rank, so this
    coincides with a plain count in practice).

    `basis` selects the correlation matrix, matching `factorability`:
    'pearson' (default, matching `fit_efa`'s own extraction) or 'spearman'.
    'polychoric' is not supported here - each of the `n_iter` null matrices
    would need its own polychoric correlation matrix (~8s each on this
    data), making a default-`n_iter` run take well over an hour. Use
    `factorability(df, basis="polychoric")` for the observed-side check.
    """
    if basis not in ("pearson", "spearman"):
        raise ValueError(
            f"parallel_analysis supports basis='pearson' or 'spearman' only "
            f"(got {basis!r}): simulating n_iter null matrices under the "
            "polychoric basis is computationally prohibitive. Use "
            "factorability(df, basis='polychoric') for the observed-side check."
        )
    X = _item_matrix(df).to_numpy(dtype=float)
    n, p = X.shape
    eigenvalues = np.linalg.eigvalsh(_raw_corr(X, basis))[::-1]

    rng = np.random.default_rng(config.SEED if seed is None else seed)
    random_eigs = np.empty((n_iter, p))
    for i in range(n_iter):
        random_eigs[i] = np.linalg.eigvalsh(_raw_corr(rng.standard_normal((n, p)), basis))[::-1]
    threshold = np.percentile(random_eigs, percentile, axis=0)

    exceeds = eigenvalues > threshold
    if exceeds.all():
        n_factors = p
    elif not exceeds.any():
        n_factors = 0
    else:
        n_factors = int(np.argmax(~exceeds))

    return {"n_factors": n_factors, "eigenvalues": eigenvalues, "threshold": threshold}


# --------------------------------------------------------------------------
# EFA extraction
# --------------------------------------------------------------------------

@dataclass
class EfaResult:
    loadings: pd.DataFrame
    communalities: pd.Series
    uniquenesses: pd.Series
    variance: pd.DataFrame
    factor_correlations: pd.DataFrame
    n_factors: int
    # Not part of the documented interface: the fitted estimator, kept so
    # `factor_scores` can call its `.transform()` without refitting or
    # re-deriving a regression-weight matrix from the rotated loadings.
    model: object = field(default=None, repr=False, compare=False)


def _run_fa(X: np.ndarray, n_factors: int, rotation: str, method: str,
            is_corr_matrix: bool, index) -> EfaResult:
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method,
                         is_corr_matrix=is_corr_matrix)
    fa.fit(X)

    cols = [f"F{i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=index, columns=cols)
    communalities = pd.Series(fa.get_communalities(), index=index)
    uniquenesses = pd.Series(fa.get_uniquenesses(), index=index)

    ss_loadings, prop_var, cum_var = fa.get_factor_variance()
    variance = pd.DataFrame([ss_loadings, prop_var, cum_var],
                             index=["ss_loadings", "prop_var", "cum_var"], columns=cols)

    phi = getattr(fa, "phi_", None)
    factor_correlations = pd.DataFrame(phi if phi is not None else np.eye(n_factors),
                                        index=cols, columns=cols)

    return EfaResult(loadings, communalities, uniquenesses, variance,
                      factor_correlations, n_factors, model=fa)


def fit_efa(df: pd.DataFrame, n_factors: int, rotation="oblimin", method="minres") -> EfaResult:
    items = _item_matrix(df)
    return _run_fa(items.to_numpy(dtype=float), n_factors, rotation, method,
                    is_corr_matrix=False, index=list(items.columns))


def assign_items(loadings: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """Primary-factor assignment by largest absolute loading, gated at `threshold`."""
    abs_loadings = loadings.abs().to_numpy()
    best = np.argmax(abs_loadings, axis=1)
    primary_loading = loadings.to_numpy()[np.arange(len(loadings)), best]
    n_above = (abs_loadings >= threshold).sum(axis=1)
    is_unassigned = n_above == 0
    is_cross_loading = n_above >= 2

    primary = pd.Series(loadings.columns[best], index=loadings.index, dtype=object)
    primary[is_unassigned] = None

    return pd.DataFrame({
        "primary": primary,
        "primary_loading": primary_loading,
        "n_loadings_above_threshold": n_above.astype(int),
        "is_cross_loading": is_cross_loading,
        "is_unassigned": is_unassigned,
    }, index=loadings.index)


def factor_scores(df: pd.DataFrame, result: EfaResult) -> pd.DataFrame:
    items = _item_matrix(df)
    scores = result.model.transform(items.to_numpy(dtype=float))
    return pd.DataFrame(scores, index=df.index, columns=list(result.loadings.columns))


def reliability(df: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha (on raw item scores) and McDonald's omega (on primary
    loadings, treating each factor's assigned items as a congeneric subscale:
    uniqueness_i = 1 - primary_loading_i^2) per factor.
    """
    items = _item_matrix(df)
    factor_names = sorted(assignment["primary"].dropna().unique(), key=lambda f: int(f[1:]))

    rows = []
    for f in factor_names:
        members = assignment.index[assignment["primary"] == f]
        L = assignment.loc[members, "primary_loading"].to_numpy(dtype=float)
        alpha = descriptives.cronbach_alpha(items[members])
        ss = L.sum() ** 2
        denom = ss + np.sum(1.0 - L ** 2)
        omega = float(ss / denom) if denom > 0 else float("nan")
        rows.append({"factor": f, "n_items": len(members),
                      "cronbach_alpha": alpha, "mcdonald_omega": omega})

    return pd.DataFrame(rows).set_index("factor")


# --------------------------------------------------------------------------
# Robustness checks
# --------------------------------------------------------------------------

def bootstrap_stability(df: pd.DataFrame, n_factors: int, n_boot=200, seed=None) -> pd.DataFrame:
    """Non-parametric bootstrap over articles (rows).

    Refits the EFA on each resample and matches its factors back to the
    full-sample reference solution (greedy, absolute congruence; column sign
    aligned to positive). `prop_replicated` is the share of matched
    replicates reaching phi >= 0.85.
    """
    ref = fit_efa(df, n_factors)
    ref_loadings = ref.loadings.to_numpy()
    cols = list(ref.loadings.columns)
    n = len(df)

    rng = np.random.default_rng(config.SEED if seed is None else seed)
    congruences = {c: [] for c in cols}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = fit_efa(df.iloc[idx], n_factors)
        pairs = _greedy_match(ref_loadings, boot.loadings.to_numpy())
        for i, (_, phi) in pairs.items():
            congruences[cols[i]].append(abs(phi))

    rows = []
    for c in cols:
        v = np.asarray(congruences[c], dtype=float)
        if v.size == 0:
            rows.append({"factor": c, "mean_congruence": np.nan, "lo": np.nan,
                          "hi": np.nan, "prop_replicated": np.nan})
            continue
        rows.append({
            "factor": c,
            "mean_congruence": float(v.mean()),
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
            "prop_replicated": float(np.mean(v >= 0.85)),
        })
    return pd.DataFrame(rows).set_index("factor")


def split_half_congruence(df: pd.DataFrame, n_factors: int, seed=None) -> pd.DataFrame:
    """Tucker's congruence between EFAs fit on two random, non-overlapping halves.

    Same matching rule as `bootstrap_stability`; the result is indexed by
    the first half's factor labels.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = len(df)
    perm = rng.permutation(n)
    half = n // 2

    fa1 = fit_efa(df.iloc[perm[:half]], n_factors)
    fa2 = fit_efa(df.iloc[perm[half:]], n_factors)
    pairs = _greedy_match(fa1.loadings.to_numpy(), fa2.loadings.to_numpy())

    cols = list(fa1.loadings.columns)
    rows = [{"factor": cols[i],
             "congruence": abs(pairs[i][1]) if i in pairs else float("nan")}
            for i in range(len(cols))]
    return pd.DataFrame(rows).set_index("factor")


# --------------------------------------------------------------------------
# Polychoric robustness check
# --------------------------------------------------------------------------
#
# The items are ordinal (0-5), so Pearson/Spearman correlations attenuate the
# latent association. semopy (installed) exposes `polycorr.polychoric_corr`,
# but calling it raises `AttributeError: scipy.stats.mvn has no attribute
# mvnun` - that function was removed from the installed scipy (1.17); semopy
# 2.3.11 still calls it. That is a hard failure, not a warning, so there is
# no usable polychoric routine to reuse. What follows is the standard
# two-step estimator named in the brief: thresholds from each item's
# marginal cumulative proportions, then a bounded 1-D MLE over rho per item
# pair using the bivariate normal CDF (`scipy.stats.multivariate_normal`).

def _ordinal_thresholds(x: np.ndarray):
    """Latent-normal cut points implied by an item's observed category proportions."""
    cats = np.sort(np.unique(x))
    cum = np.cumsum([np.mean(x == c) for c in cats])[:-1]
    cum = np.clip(cum, _EPS, 1 - _EPS)
    z = sps.norm.ppf(cum)
    return np.concatenate([[-np.inf], z, [np.inf]]), cats


def _contingency(x, y, catx, caty) -> np.ndarray:
    xi = np.searchsorted(catx, x)
    yi = np.searchsorted(caty, y)
    N = np.zeros((len(catx), len(caty)))
    np.add.at(N, (xi, yi), 1)
    return N


def _bvn_corner_grid(tx: np.ndarray, ty: np.ndarray, rho: float) -> np.ndarray:
    """P(X <= tx[i], Y <= ty[j]) for every threshold pair.

    Standard bivariate normal with correlation `rho`, handling the +/-inf
    edges analytically (scipy's `multivariate_normal.cdf` does not accept
    infinite bounds).
    """
    p, m = len(tx) - 1, len(ty) - 1
    corners = np.zeros((p + 1, m + 1))
    fin_x = [i for i in range(p + 1) if np.isfinite(tx[i])]
    fin_y = [j for j in range(m + 1) if np.isfinite(ty[j])]

    if fin_x and fin_y:
        pts = np.array([[tx[i], ty[j]] for i in fin_x for j in fin_y])
        vals = np.atleast_1d(sps.multivariate_normal.cdf(
            pts, mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]))
        for (i, j), v in zip(((i, j) for i in fin_x for j in fin_y), vals):
            corners[i, j] = v

    for i in range(p + 1):
        if not np.isfinite(tx[i]):
            if tx[i] > 0:
                for j in fin_y:
                    corners[i, j] = sps.norm.cdf(ty[j])
            # tx[i] == -inf leaves corners[i, :] at 0
    for j in range(m + 1):
        if not np.isfinite(ty[j]):
            if ty[j] > 0:
                for i in fin_x:
                    corners[i, j] = sps.norm.cdf(tx[i])
            # ty[j] == -inf leaves corners[:, j] at 0
    if not np.isfinite(tx[-1]) and not np.isfinite(ty[-1]):
        corners[-1, -1] = 1.0
    return corners


def _polychoric_pair(x, y, tx, catx, ty, caty) -> float:
    N = _contingency(x, y, catx, caty)

    def neg_log_likelihood(rho):
        corners = _bvn_corner_grid(tx, ty, rho)
        probs = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
        probs = np.clip(probs, 1e-12, None)
        return -float(np.sum(N * np.log(probs)))

    res = minimize_scalar(neg_log_likelihood, bounds=(-0.999, 0.999),
                           method="bounded", options={"xatol": 1e-3})
    return float(res.x)


def polychoric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full polychoric correlation matrix over the coded items.

    44x44 = 946 pairs; ~8 seconds measured on the real dataset (well under
    the five-minute budget), each pair an independent bounded 1-D MLE.
    """
    items = _item_matrix(df)
    cols = list(items.columns)
    arrs = {c: items[c].to_numpy(dtype=float) for c in cols}
    thresholds = {c: _ordinal_thresholds(arrs[c]) for c in cols}

    p = len(cols)
    R = np.eye(p)
    for i, j in combinations(range(p), 2):
        tx, catx = thresholds[cols[i]]
        ty, caty = thresholds[cols[j]]
        rho = _polychoric_pair(arrs[cols[i]], arrs[cols[j]], tx, catx, ty, caty)
        R[i, j] = R[j, i] = rho
    return pd.DataFrame(R, index=cols, columns=cols)


def _fit_efa_on_basis(df: pd.DataFrame, n_factors: int, basis: str,
                       rotation="oblimin", method="minres") -> EfaResult:
    """`fit_efa`, but extracted from an explicit correlation `basis`.

    Used instead of `fit_efa`'s own raw-item-matrix path for the 'spearman'
    and 'polychoric' alternatives in `sensitivity_by_correlation_basis`; not
    used for 'pearson' itself, since `fit_efa` is the tested, bit-for-bit
    reproducible reference path for the shipped solution and the two are
    numerically equivalent to ~1e-8 (verified), not worth risking drift on.
    """
    R = _correlation_matrix(df, basis)
    index = list(_item_matrix(df).columns)
    return _run_fa(R, n_factors, rotation, method, is_corr_matrix=True, index=index)


def fit_efa_polychoric(df: pd.DataFrame, n_factors: int,
                        rotation="oblimin", method="minres") -> EfaResult:
    """Same EFA, fit on the polychoric rather than the raw item matrix.

    This is the robustness check for whether the continuous treatment in
    `fit_efa` matters for this data.
    """
    return _fit_efa_on_basis(df, n_factors, "polychoric", rotation, method)


# --------------------------------------------------------------------------
# Sensitivity across correlation bases
# --------------------------------------------------------------------------

def _simple_structure_counts(loadings: pd.DataFrame, communalities: pd.Series,
                              threshold: float = 0.40) -> dict:
    a = assign_items(loadings, threshold=threshold)
    return {
        "cross_loading": int(a["is_cross_loading"].sum()),
        "unassigned": int(a["is_unassigned"].sum()),
        "low_communality": int((communalities < 0.30).sum()),
    }


def sensitivity_by_correlation_basis(df: pd.DataFrame, n_factors: int = 5) -> pd.DataFrame:
    """Per-factor Tucker congruence of the Pearson solution against Spearman and
    polychoric extractions. Disagreement is a result to report, not to resolve.

    Also carries the simple-structure counts (cross-loading, unassigned,
    low-communality items) produced under each of the three bases, so a
    reader can see not just whether the factors line up but whether the
    overall solution's cleanliness holds up under each treatment.
    """
    pearson = fit_efa(df, n_factors)
    spearman = _fit_efa_on_basis(df, n_factors, "spearman")
    polychoric = _fit_efa_on_basis(df, n_factors, "polychoric")

    pairs_spearman = _greedy_match(pearson.loadings.to_numpy(), spearman.loadings.to_numpy())
    pairs_polychoric = _greedy_match(pearson.loadings.to_numpy(), polychoric.loadings.to_numpy())

    counts = {
        "pearson": _simple_structure_counts(pearson.loadings, pearson.communalities),
        "spearman": _simple_structure_counts(spearman.loadings, spearman.communalities),
        "polychoric": _simple_structure_counts(polychoric.loadings, polychoric.communalities),
    }

    cols = list(pearson.loadings.columns)
    rows = []
    for i, factor in enumerate(cols):
        row = {
            "factor": factor,
            "vs_spearman": abs(pairs_spearman[i][1]) if i in pairs_spearman else float("nan"),
            "vs_polychoric": abs(pairs_polychoric[i][1]) if i in pairs_polychoric else float("nan"),
        }
        for basis_name, c in counts.items():
            row[f"{basis_name}_cross_loading"] = c["cross_loading"]
            row[f"{basis_name}_unassigned"] = c["unassigned"]
            row[f"{basis_name}_low_communality"] = c["low_communality"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("factor")
