"""Tier C: agreement among importance methods.

This is an internal consistency check, not validation - no external criterion
is involved. Reported as such throughout (R1.1).
"""
import networkx as nx
import numpy as np
import pandas as pd
import shap
from scipy import stats as sps
from sklearn.inspection import permutation_importance

from . import benchmark, config, descriptives

METHODS = ["gain", "permutation", "shap", "centrality"]


def _normalise(values, index) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=index)
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else s * 0.0


def _fit_gbm(df, target):
    X, y, names = benchmark.build_xy(df, target)
    model = benchmark.make_estimators()["LightGBM"]
    model.fit(X, y)
    return model, X, y, names


def gain_importance(df, target="A-D1") -> pd.Series:
    model, _, _, names = _fit_gbm(df, target)
    return _normalise(model.feature_importances_, names)


def permutation_importance_scores(df, target="A-D1") -> pd.Series:
    model, X, y, names = _fit_gbm(df, target)
    res = permutation_importance(model, X, y, n_repeats=10,
                                 random_state=config.SEED, n_jobs=-1)
    return _normalise(res.importances_mean, names)


def shap_importance(df, target="A-D1") -> pd.Series:
    model, X, _, names = _fit_gbm(df, target)
    values = shap.TreeExplainer(model).shap_values(X)
    return _normalise(np.abs(values).mean(axis=0), names)


def build_graph(df, threshold=0.3) -> nx.Graph:
    corr = descriptives.correlation_matrix(df).abs()
    G = nx.Graph()
    G.add_nodes_from(config.SCORE_COLS)
    for i, a in enumerate(config.SCORE_COLS):
        for b in config.SCORE_COLS[i + 1:]:
            w = float(corr.loc[a, b])
            if w >= threshold:
                G.add_edge(a, b, weight=w)
    return G


def network_centrality(df, threshold=0.3) -> pd.DataFrame:
    G = build_graph(df, threshold)
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in G}
    return pd.DataFrame({
        "pagerank": pd.Series(nx.pagerank(G, weight="weight")),
        "betweenness": pd.Series(nx.betweenness_centrality(G, weight="weight")),
        "eigenvector": pd.Series(eig),
        "degree": pd.Series(dict(G.degree(weight="weight"))),
    }).reindex(config.SCORE_COLS).fillna(0.0)


def convergence_table(df, target="A-D1") -> pd.DataFrame:
    cent = network_centrality(df)
    table = pd.DataFrame(index=config.SCORE_COLS)
    table["gain"] = gain_importance(df, target)
    table["permutation"] = permutation_importance_scores(df, target)
    table["shap"] = shap_importance(df, target)
    table["centrality"] = _normalise(cent["pagerank"].values, cent.index)
    # The target cannot be a predictor of itself, so three of its four method
    # values are undefined rather than measured. Drop the row entirely: every
    # remaining cell is a real measurement, and no consumer can mistake a
    # construction artefact for a finding.
    table = table.drop(index=target)
    table = table.fillna(0.0)
    table["mean_rank"] = table[METHODS].rank(ascending=False).mean(axis=1)
    return table.sort_values("mean_rank")


def concordance(table: pd.DataFrame):
    """Kendall's W across the four importance methods. High W means the methods
    agree with each other - an internal consistency result, not validation."""
    ranks = table[METHODS].rank(ascending=False).to_numpy()
    n, k = ranks.shape  # n = sub-dimensions, k = methods (raters)
    stat, p = sps.friedmanchisquare(*ranks)  # each row is one object's ranks across methods
    return float(stat / (k * (n - 1))), float(p)
