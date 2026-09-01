"""Tier B: leave-one-sub-dimension-out prediction of OBSERVED coded scores.

The target is always a column of the coded datasheet. No model output is ever
a target anywhere in this module - that is the fix for R1.1 / R2.5.
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats as sps
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config, stats
from .models_nn import AutoencoderRegressor, GCNRegressor, MLPRegressorTorch

PARAM_GRIDS = {
    "ElasticNet": {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.2, 0.5, 0.8]},
    "RandomForest": {"model__max_depth": [None, 8, 16]},
    "XGBoost": {"model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.1]},
    "LightGBM": {"model__num_leaves": [15, 31], "model__learning_rate": [0.05, 0.1]},
    "CatBoost": {"model__depth": [4, 6]},
}
# PARAM_GRIDS.get(name) returning None means "fit directly, no grid search" -
# by design for estimators with no hyperparameters worth tuning here (Dummy;
# Task 8's neural estimators), not an oversight.


def make_estimators(seed=config.SEED) -> dict:
    return {
        "Dummy": DummyRegressor(strategy="mean"),
        "ElasticNet": ElasticNet(random_state=seed, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=seed, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=seed, n_jobs=-1, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=300, random_seed=seed, verbose=0,
                                      allow_writing_files=False),
        "MLP": MLPRegressorTorch(max_epochs=200, random_state=seed),
        "Autoencoder": AutoencoderRegressor(max_epochs=200, random_state=seed),
        "GNN": GCNRegressor(max_epochs=200, random_state=seed),
    }


def build_xy(df: pd.DataFrame, target: str):
    """Predictors are the other 43 sub-dimensions plus publication year.

    `target` must be an observed column of the coded datasheet (one of
    config.SCORE_COLS). This is the single guard that makes the R1.1/R2.5
    circularity impossible to express in this module: there is no code path
    by which a model's own output can become `y`.
    """
    if target not in config.SCORE_COLS:
        raise KeyError(
            f"{target!r} is not an observed coded column; targets must be one of "
            f"config.SCORE_COLS. Model outputs are never valid targets."
        )
    predictors = [c for c in config.SCORE_COLS if c != target] + ["year"]
    return df[predictors].to_numpy(float), df[target].to_numpy(float), predictors


def _metrics(y_true, y_pred):
    with warnings.catch_warnings():
        # A constant prediction (e.g. the Dummy floor, or a degenerate fold)
        # makes Spearman's rho undefined; scipy warns and returns NaN. We
        # already convert that NaN to 0.0 below, so the warning is silenced
        # here rather than leaking into every caller's output.
        warnings.filterwarnings("ignore", category=sps.ConstantInputWarning)
        rho = sps.spearmanr(y_true, y_pred).statistic
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "spearman": 0.0 if np.isnan(rho) else float(rho),
    }


def evaluate_target(df, target, estimators=None, n_outer=5, n_inner=3, seed=config.SEED):
    estimators = make_estimators(seed) if estimators is None else estimators
    X, y, _ = build_xy(df, target)
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=seed)

    rows = []
    for fold, (tr, te) in enumerate(outer.split(X)):
        for name, est in estimators.items():
            pipe = Pipeline([("scale", StandardScaler()), ("model", est)])
            grid = PARAM_GRIDS.get(name)
            if grid:
                inner = KFold(n_splits=n_inner, shuffle=True, random_state=seed)
                search = GridSearchCV(pipe, grid, cv=inner,
                                      scoring="neg_mean_absolute_error", n_jobs=-1)
                with warnings.catch_warnings():
                    # GridSearchCV pre-allocates each param's MaskedArray from
                    # uninitialized memory (np.empty) with mask=True before
                    # filling in real values (sklearn _search.py,
                    # _yield_masked_array_for_each_param). Casting that
                    # uninitialized garbage into an int dtype can occasionally
                    # overflow and warn, even though every entry is masked out
                    # and never read. A scikit-learn/numpy internal artifact,
                    # not a signal about our data or results.
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning,
                        message="invalid value encountered in cast",
                    )
                    search.fit(X[tr], y[tr])
                fitted = search.best_estimator_
            else:
                fitted = pipe.fit(X[tr], y[tr])
            rows.append({"target": target, "model": name, "fold": fold,
                         **_metrics(y[te], fitted.predict(X[te]))})
    return pd.DataFrame(rows)


def run_sweep(df, targets=None, estimators=None, quick=False, seed=config.SEED):
    targets = list(targets) if targets is not None else list(config.SCORE_COLS)
    if quick:
        targets = targets[:3]
    n_outer, n_inner = (2, 2) if quick else (5, 3)
    frames = [evaluate_target(df, t, estimators, n_outer, n_inner, seed) for t in targets]
    return pd.concat(frames, ignore_index=True)


def summarise(results: pd.DataFrame, n_boot=10000) -> pd.DataFrame:
    rows = []
    for model, grp in results.groupby("model"):
        point, lo, hi = stats.bca_bootstrap(grp["mae"].to_numpy(), np.mean, n_boot=n_boot)
        rows.append({"model": model, "mae": point, "mae_lo": lo, "mae_hi": hi,
                     "rmse": grp["rmse"].mean(), "r2": grp["r2"].mean(),
                     "spearman": grp["spearman"].mean()})
    return pd.DataFrame(rows).set_index("model").sort_values("mae")
