import numpy as np
import pandas as pd
import pytest
from analysis import benchmark, config, loading, quality

DF = quality.clean(loading.load_scoring())
SMALL = {k: v for k, v in benchmark.make_estimators(config.SEED).items()
         if k in ("Dummy", "ElasticNet", "RandomForest")}

def test_estimator_set_includes_the_dummy_floor():
    est = benchmark.make_estimators(config.SEED)
    assert "Dummy" in est
    assert {"ElasticNet", "RandomForest", "XGBoost", "LightGBM", "CatBoost"} <= set(est)

def test_evaluate_target_returns_one_row_per_model_per_fold():
    out = benchmark.evaluate_target(DF, "A-D1", SMALL, n_outer=3, n_inner=2)
    assert len(out) == 3 * 3
    assert set(out["model"]) == set(SMALL)
    assert out["target"].unique().tolist() == ["A-D1"]

def test_target_is_never_among_the_predictors():
    X, y, names = benchmark.build_xy(DF, "A-D1")
    assert "A-D1" not in names
    assert len(names) == 43 + 1          # 43 other sub-dimensions + year
    assert len(y) == len(DF)

def test_harness_rejects_a_target_that_is_not_an_observed_column():
    with pytest.raises(KeyError):
        benchmark.build_xy(DF, "GBM_feature_importance")

def test_real_models_beat_the_dummy_floor_on_a_well_covered_target():
    out = benchmark.evaluate_target(DF, "A-D1", SMALL, n_outer=3, n_inner=2)
    mae = out.groupby("model")["mae"].mean()
    assert mae["RandomForest"] < mae["Dummy"]

def test_metrics_are_finite_and_sane():
    out = benchmark.evaluate_target(DF, "C-D1", SMALL, n_outer=3, n_inner=2)
    assert np.isfinite(out[["mae", "rmse", "r2", "spearman"]].to_numpy()).all()
    assert (out["mae"] >= 0).all() and (out["rmse"] >= out["mae"] - 1e-9).all()
    assert out["spearman"].between(-1, 1).all()

def test_sweep_covers_every_requested_target():
    out = benchmark.run_sweep(DF, ["A-D1", "B-D1"], SMALL, quick=True)
    assert set(out["target"]) == {"A-D1", "B-D1"}

def test_summarise_reports_bootstrap_intervals_around_mae():
    out = benchmark.run_sweep(DF, ["A-D1", "B-D1"], SMALL, quick=True)
    s = benchmark.summarise(out)
    assert set(s.index) == set(SMALL)
    assert (s["mae_lo"] <= s["mae"]).all() and (s["mae"] <= s["mae_hi"]).all()

def test_results_are_reproducible():
    a = benchmark.run_sweep(DF, ["A-D1"], SMALL, quick=True)
    b = benchmark.run_sweep(DF, ["A-D1"], SMALL, quick=True)
    pd.testing.assert_frame_equal(a, b)

def test_results_are_reproducible_for_the_full_estimator_set():
    """SMALL (Dummy/ElasticNet/RandomForest) can't catch a CatBoost-specific
    reproducibility bug: parallel GridSearchCV workers racing to create the same
    catboost_info/ working directory silently changes which hyperparameters get
    selected. This must run the full six-estimator set that actually ships."""
    est = benchmark.make_estimators(config.SEED)
    a = benchmark.run_sweep(DF, ["A-D1"], est, quick=True)
    b = benchmark.run_sweep(DF, ["A-D1"], est, quick=True)
    pd.testing.assert_frame_equal(a, b)
