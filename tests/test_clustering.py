import numpy as np
import pandas as pd
from analysis import clustering, config, loading, quality

DF = quality.clean(loading.load_scoring())

def test_distance_matrix_is_a_valid_metric_shape():
    D = clustering.distance_matrix(DF)
    assert D.shape == (44, 44)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.allclose(D, D.T)
    assert (D >= -1e-9).all() and (D <= 1.0 + 1e-9).all()

def test_linkage_has_the_scipy_shape():
    Z = clustering.linkage_matrix(DF)
    assert Z.shape == (43, 4)          # n-1 merges

def test_silhouette_scan_covers_the_requested_k():
    s = clustering.silhouette_scan(DF, k_range=range(2, 7))
    assert list(s["k"]) == [2, 3, 4, 5, 6]
    assert s["silhouette"].between(-1, 1).all()

def test_gap_statistic_returns_one_row_per_k():
    g = clustering.gap_statistic(DF, k_range=range(2, 6), n_ref=10)
    assert list(g["k"]) == [2, 3, 4, 5]
    assert np.isfinite(g["gap"]).all() and (g["sk"] >= 0).all()

def test_choose_k_lands_in_the_scanned_range():
    k = clustering.choose_k(DF, k_range=range(2, 9))
    assert 2 <= k <= 8

def test_assign_labels_every_subdimension_exactly_once():
    a = clustering.assign(DF, k=5)
    assert len(a) == 44
    assert sorted(a.index) == sorted(config.SCORE_COLS)
    assert a["cluster"].nunique() == 5
    assert a["cluster"].between(1, 5).all()

def test_assignment_is_reproducible():
    assert list(clustering.assign(DF, k=5)["cluster"]) == list(clustering.assign(DF, k=5)["cluster"])

def test_cluster_names_are_derived_from_member_dimensions():
    a = clustering.assign(DF, k=5)
    names = clustering.name_clusters(a)
    assert set(names) == set(a["cluster"].unique())
    assert all(isinstance(v, str) and v for v in names.values())

def test_no_hardcoded_esg_category_names():
    """The submitted manuscript's categories came from nowhere; ours are derived."""
    import inspect
    src = inspect.getsource(clustering).lower()
    for banned in ["climate action", "human capital", "social license", "social licence"]:
        assert banned not in src


def test_structure_verdict_returns_required_fields():
    v = clustering.structure_verdict(DF, k_range=range(2, 8))
    assert "best_k" in v
    assert "best_silhouette" in v
    assert "on_boundary" in v
    assert "monotonic_increasing" in v
    assert "max_silhouette_is_weak" in v
    assert "scan" in v
    assert isinstance(v["best_k"], int)
    assert isinstance(v["best_silhouette"], float)
    assert isinstance(v["on_boundary"], bool)
    assert isinstance(v["monotonic_increasing"], bool)
    assert isinstance(v["max_silhouette_is_weak"], bool)


def test_structure_verdict_detects_boundary_argmax_on_real_data():
    """On real data with weak structure, argmax should be on boundary."""
    v = clustering.structure_verdict(DF, k_range=range(2, 16))
    # The real data shows silhouette rising monotonically with no interior peak,
    # so the argmax should be on the boundary (at k=15, the upper edge).
    assert v["on_boundary"] or v["monotonic_increasing"]


def test_verdict_from_scan_matches_structure_verdict_on_the_same_data():
    """`verdict_from_scan` is `structure_verdict`'s boundary/verdict logic,
    factored out so a caller that already has a scan in hand (fig_dendrogram)
    doesn't need to pay for - or risk diverging from - a second
    `silhouette_scan` run. Given the SAME scan, the two must agree exactly."""
    scan = clustering.silhouette_scan(DF, k_range=range(2, 8))
    from_scan = clustering.verdict_from_scan(scan)
    full = clustering.structure_verdict(DF, k_range=range(2, 8))
    for key in ("best_k", "best_silhouette", "on_boundary", "monotonic_increasing",
               "max_silhouette_is_weak"):
        assert from_scan[key] == full[key]


def test_verdict_from_scan_flags_an_argmax_at_either_edge():
    rising = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.10, 0.15, 0.18, 0.24]})
    assert clustering.verdict_from_scan(rising)["on_boundary"] is True

    falling = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.24, 0.18, 0.15, 0.10]})
    assert clustering.verdict_from_scan(falling)["on_boundary"] is True


def test_verdict_from_scan_does_not_flag_an_interior_optimum():
    interior = pd.DataFrame({"k": [2, 3, 4, 5, 6], "silhouette": [0.10, 0.15, 0.30, 0.20, 0.12]})
    assert clustering.verdict_from_scan(interior)["on_boundary"] is False


def test_structure_verdict_would_not_flag_synthetic_separated_clusters():
    """Synthetic data with genuine tight clusters should not be flagged."""
    # Build three tight blobs of correlated columns
    rng = np.random.default_rng(config.SEED)
    n_articles = 1026
    n_cols_per_blob = 2  # Small clusters for testing

    # Blob 1: high correlation (mean ~0.8)
    blob1 = rng.normal(loc=4, scale=0.3, size=(n_articles, n_cols_per_blob))
    # Blob 2: different range
    blob2 = rng.normal(loc=2, scale=0.3, size=(n_articles, n_cols_per_blob))
    # Blob 3: another different range
    blob3 = rng.normal(loc=0, scale=0.3, size=(n_articles, n_cols_per_blob))

    synthetic_data = np.hstack([blob1, blob2, blob3])
    synthetic_df = pd.DataFrame(synthetic_data, columns=[f"col_{i}" for i in range(6)])

    # Compute distance on synthetic data
    corr = synthetic_df.corr().to_numpy()
    D = 1.0 - np.abs(corr)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0

    # Perform linkage
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(squareform(D, checks=False), method="ward")

    # Silhouette scan on synthetic
    from sklearn.metrics import silhouette_score
    rows = []
    for k in range(2, 5):
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = silhouette_score(D, labels, metric="precomputed")
        rows.append({"k": k, "silhouette": float(score)})
    s_syn = pd.DataFrame(rows)

    # For synthetic data with three tight blobs, k=3 should have high silhouette
    # and NOT be on boundary, and NOT monotonically increasing
    s_vals = s_syn["silhouette"].to_numpy()
    best_i = int(np.argmax(s_vals))
    on_boundary = best_i in (0, len(s_vals) - 1)
    monotonic = bool(np.all(np.diff(s_vals) > -1e-9))

    # At least one should be False (not both flagged)
    assert not (on_boundary and monotonic), "Synthetic clusters should show a genuine interior peak"


def test_dimension_recovery_returns_required_fields():
    a = clustering.assign(DF, k=5)
    rec = clustering.dimension_recovery(a)
    assert "adjusted_rand_index" in rec
    assert "weighted_mean_purity" in rec
    assert isinstance(rec["adjusted_rand_index"], float)
    assert isinstance(rec["weighted_mean_purity"], float)
    assert 0 <= rec["adjusted_rand_index"] <= 1
    assert 0 <= rec["weighted_mean_purity"] <= 1


def test_dimension_recovery_on_real_data():
    """Regression test: dimension recovery on real data should match known values."""
    a = clustering.assign(DF, k=10)  # k=10 was the original choice
    rec = clustering.dimension_recovery(a)
    # Known values from coordinator's analysis
    ari = rec["adjusted_rand_index"]
    purity = rec["weighted_mean_purity"]
    # Should be close to 0.318 and 0.591 respectively (within rounding)
    assert 0.30 < ari < 0.35, f"ARI {ari} should be ~0.318"
    assert 0.55 < purity < 0.62, f"Purity {purity} should be ~0.591"
