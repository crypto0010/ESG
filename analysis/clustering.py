"""Derive categories from the data (R2.11). Ward linkage on correlation distance,
with k chosen by silhouette and corroborated by the gap statistic.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config, descriptives


def distance_matrix(df) -> np.ndarray:
    corr = descriptives.correlation_matrix(df).to_numpy()
    D = 1.0 - np.abs(corr)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def linkage_matrix(df) -> np.ndarray:
    return linkage(squareform(distance_matrix(df), checks=False), method="ward")


def silhouette_scan(df, k_range=range(2, 11)) -> pd.DataFrame:
    D = distance_matrix(df)
    Z = linkage_matrix(df)
    rows = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = silhouette_score(D, labels, metric="precomputed") if len(set(labels)) > 1 else np.nan
        rows.append({"k": k, "silhouette": float(score)})
    return pd.DataFrame(rows)


def gap_statistic(df, k_range=range(2, 11), n_ref=50) -> pd.DataFrame:
    """Tibshirani gap statistic against uniform references in feature space."""
    X = df[config.SCORE_COLS].to_numpy(float).T      # cluster the 44 variables
    rng = np.random.default_rng(config.SEED)
    lo, hi = X.min(axis=0), X.max(axis=0)

    def dispersion(data, k):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(data)
        return np.log(km.inertia_ + 1e-12)

    rows = []
    for k in k_range:
        obs = dispersion(X, k)
        refs = np.array([dispersion(rng.uniform(lo, hi, size=X.shape), k) for _ in range(n_ref)])
        rows.append({"k": k, "gap": float(refs.mean() - obs),
                     "sk": float(refs.std(ddof=1) * np.sqrt(1 + 1 / n_ref))})
    return pd.DataFrame(rows)


def choose_k(df, k_range=range(2, 16)) -> int:
    scan = silhouette_scan(df, k_range)
    return int(scan.loc[scan["silhouette"].idxmax(), "k"])


def assign(df, k=None) -> pd.DataFrame:
    k = k or choose_k(df)
    labels = fcluster(linkage_matrix(df), t=k, criterion="maxclust")
    return pd.DataFrame({
        "label": [config.SUBDIMENSIONS[c] for c in config.SCORE_COLS],
        "dimension": [config.DIMENSIONS[c[0]] for c in config.SCORE_COLS],
        "cluster": labels,
    }, index=pd.Index(config.SCORE_COLS, name="code"))


def name_clusters(assignment: pd.DataFrame) -> dict:
    """Name each cluster after the dimensions its members actually come from.

    Descriptive, not editorial - the name is a readout of membership.
    """
    names = {}
    for cid, grp in assignment.groupby("cluster"):
        counts = Counter(grp["dimension"])
        top = [d for d, _ in counts.most_common(2)]
        names[int(cid)] = " / ".join(top)
    return names


def verdict_from_scan(scan: pd.DataFrame) -> dict:
    """The interior-optimum verdict computed directly from an already-run
    silhouette scan. `structure_verdict` is exactly `silhouette_scan` (a
    real, possibly expensive re-scan of the data) followed by this - factored
    out so a caller that already has a scan in hand (e.g.
    `analysis.figures.fig_dendrogram`, which is handed the same `scan` its
    caller computed) can read the verdict without paying for, or risking a
    silent divergence from, a second `silhouette_scan` run.
    """
    s = scan["silhouette"].to_numpy()
    ks = scan["k"].to_numpy()
    best_i = int(np.argmax(s))
    return {
        "best_k": int(ks[best_i]),
        "best_silhouette": float(s[best_i]),
        "on_boundary": best_i in (0, len(s) - 1),
        "monotonic_increasing": bool(np.all(np.diff(s) > -1e-9)),
        "max_silhouette_is_weak": bool(s[best_i] < 0.25),
    }


def structure_verdict(df, k_range=range(2, 16)) -> dict:
    """Whether the data supports ANY well-separated clustering.

    A silhouette argmax on the boundary of the search window, or a curve that
    rises monotonically, indicates no interior optimum - the argmax is then an
    artefact of the window rather than a property of the data.
    """
    scan = silhouette_scan(df, k_range)
    verdict = verdict_from_scan(scan)
    verdict["scan"] = scan
    return verdict


def dimension_recovery(assignment) -> dict:
    """How far the derived clusters depart from the a priori 11 dimensions.
    A high ARI means clustering mostly recovers the coding framework rather
    than discovering anything new."""
    counts = assignment.groupby("cluster")["dimension"].agg(
        lambda g: g.value_counts().iloc[0] / len(g))
    sizes = assignment.groupby("cluster").size()
    return {
        "adjusted_rand_index": float(adjusted_rand_score(
            assignment["dimension"], assignment["cluster"])),
        "weighted_mean_purity": float((counts * sizes).sum() / sizes.sum()),
    }
