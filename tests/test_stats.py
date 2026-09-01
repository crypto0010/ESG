import numpy as np
import pytest
from analysis import config, stats


def _rng(offset: int) -> np.random.Generator:
    """Per-test generator so tests never depend on execution order."""
    return np.random.default_rng(config.SEED + offset)


def test_bca_interval_achieves_near_nominal_coverage():
    """The real property of a 95% interval: it covers the truth ~95% of the time.
    A single interval missing proves nothing - a coverage rate far from nominal does.
    """
    rng = _rng(1)
    hits = 0
    reps = 150
    for _ in range(reps):
        data = rng.normal(loc=5.0, scale=1.0, size=200)
        _, lo, hi = stats.bca_bootstrap(
            data, np.mean, n_boot=400, seed=int(rng.integers(1_000_000))
        )
        hits += lo < 5.0 < hi
    coverage = hits / reps
    assert 0.88 <= coverage <= 0.99, f"coverage {coverage:.3f} far from nominal 0.95"

def test_bca_interval_brackets_the_point_estimate():
    rng = _rng(2)
    data = rng.normal(loc=5.0, scale=1.0, size=300)
    point, lo, hi = stats.bca_bootstrap(data, np.mean, n_boot=2000, seed=1)
    assert lo < point < hi
    assert abs(point - data.mean()) < 1e-12

def test_bca_narrows_as_n_grows():
    rng = _rng(3)
    small = rng.normal(size=50)
    large = rng.normal(size=2000)
    _, l1, h1 = stats.bca_bootstrap(small, np.mean, n_boot=1500, seed=3)
    _, l2, h2 = stats.bca_bootstrap(large, np.mean, n_boot=1500, seed=3)
    assert (h2 - l2) < (h1 - l1)

def test_bca_is_reproducible_under_a_seed():
    rng = _rng(4)
    data = rng.normal(size=200)
    assert stats.bca_bootstrap(data, np.mean, n_boot=800, seed=11) == \
           stats.bca_bootstrap(data, np.mean, n_boot=800, seed=11)

def test_friedman_detects_a_real_difference():
    rng = _rng(5)
    m = np.column_stack([rng.normal(0, .1, 40), rng.normal(1, .1, 40), rng.normal(2, .1, 40)])
    res = stats.friedman(m)
    assert res.p_value < 0.001
    assert res.df == 2
    assert res.kendalls_w > 0.8

def test_friedman_finds_nothing_in_noise():
    rng = _rng(6)
    m = rng.normal(size=(60, 3))
    assert stats.friedman(m).p_value > 0.05

def test_friedman_rejects_a_single_column():
    rng = _rng(7)
    with pytest.raises(ValueError):
        stats.friedman(rng.normal(size=(20, 1)))

def test_nemenyi_returns_a_square_symmetric_pvalue_frame():
    rng = _rng(8)
    m = np.column_stack([rng.normal(0, .1, 40), rng.normal(1, .1, 40), rng.normal(2, .1, 40)])
    p = stats.nemenyi(m)
    assert p.shape == (3, 3)
    assert np.allclose(p.to_numpy(), p.to_numpy().T)

def test_wilcoxon_pairs_covers_every_pair():
    rng = _rng(9)
    m = np.column_stack([rng.normal(0, .3, 50), rng.normal(1, .3, 50), rng.normal(2, .3, 50)])
    out = stats.wilcoxon_pairs(m)
    assert len(out) == 3                      # 3 choose 2
    assert out["rank_biserial"].between(-1, 1).all()

def test_bh_fdr_leaves_a_single_pvalue_alone():
    assert np.isclose(stats.bh_fdr([0.03])[0], 0.03)

def test_bh_fdr_is_monotone_and_never_shrinks_a_pvalue():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.2, 0.7])
    adj = stats.bh_fdr(p)
    assert (adj >= p - 1e-12).all()
    assert (np.diff(adj) >= -1e-12).all()
    assert (adj <= 1.0).all()

def test_bh_fdr_matches_a_hand_computed_case():
    """Benjamini-Hochberg: adjusted p_i = min over j>=i of (p_j * n / j).
    For p = [0.01, 0.02, 0.03, 0.04] every term is 0.04, so all four adjust to 0.04."""
    adj = stats.bh_fdr([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.04, 0.04, 0.04])

def test_a_tied_pair_does_not_blank_the_rest_of_the_family():
    """One pair tying everywhere must not destroy other pairs' adjusted p-values."""
    rng = _rng(10)
    a = rng.normal(size=40)
    m = np.column_stack([a, a, a + 3.0])        # cols 0 and 1 are identical
    out = stats.wilcoxon_pairs(m, labels=["x", "y", "z"])
    assert np.isfinite(out["p_value"]).all()
    adj = stats.bh_fdr(out["p_value"].to_numpy())
    assert np.isfinite(adj).all()
    tied = out[(out["a"] == "x") & (out["b"] == "y")].iloc[0]
    assert tied["p_value"] == 1.0 and tied["rank_biserial"] == 0.0

def test_bh_fdr_rejects_non_finite_pvalues():
    """A NaN p-value must not silently propagate across the whole family via statsmodels."""
    with pytest.raises(ValueError):
        stats.bh_fdr([0.01, np.nan, 0.03])

def test_bca_bootstrap_rejects_a_single_observation():
    """n=1 makes the jackknife variance zero and would otherwise crash deep inside
    np.percentile with an opaque error; reject it up front with a clear message."""
    with pytest.raises(ValueError):
        stats.bca_bootstrap([1.0], np.mean)

def test_friedman_rejects_a_constant_matrix():
    """Zero variance across models makes scipy's statistic NaN; that must not flow
    silently into a FriedmanResult and on into a results table."""
    m = np.full((10, 3), 2.0)
    with pytest.raises(ValueError):
        stats.friedman(m)
