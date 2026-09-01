from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analysis import config, factors, loading, quality

DF = quality.clean(loading.load_scoring())
REFERENCE_LOADINGS_PATH = Path(__file__).parent / "data" / "factors_reference_loadings.csv"


def test_data_is_factorable():
    """KMO 0.94 and a highly significant Bartlett test on the real data, on

    the default ('pearson') basis - the same basis `fit_efa` extracts on, so
    this diagnostic describes the matrix the model actually uses. These are
    regression values measured from the datasheet, not targets."""
    f = factors.factorability(DF)
    assert round(f["kmo_overall"], 2) == 0.94
    assert f["bartlett_p"] < 1e-10
    assert f["bartlett_df"] == 946          # 44*43/2
    assert f["is_factorable"] is True
    assert len(f["kmo_per_item"]) == 44


def test_kmo_rejects_unfactorable_noise():
    """The detector must discriminate: i.i.d. noise is not factorable."""
    rng = np.random.default_rng(config.SEED)
    noise = pd.DataFrame(rng.normal(size=(400, 12)),
                         columns=[f"X{i}" for i in range(12)])
    assert factors.factorability(noise)["kmo_overall"] < 0.6


def test_factorability_basis_parameter_changes_the_correlation_matrix():
    """The `basis` diagnostics and extraction must describe the same matrix

    when neither is overridden, but the parameter itself must actually
    change the number: Pearson (the default, matching `fit_efa`) and
    Spearman disagree on this data (0.94 vs 0.95), so a `basis` that quietly
    did nothing would pass `test_data_is_factorable` without being real."""
    assert round(factors.factorability(DF, basis="pearson")["kmo_overall"], 2) == 0.94
    assert round(factors.factorability(DF, basis="spearman")["kmo_overall"], 2) == 0.95


def test_factorability_polychoric_basis_runs_on_real_data():
    f = factors.factorability(DF, basis="polychoric")
    assert 0 <= f["kmo_overall"] <= 1
    assert f["bartlett_df"] == 946
    assert len(f["kmo_per_item"]) == 44


def test_parallel_analysis_retains_five_factors():
    pa = factors.parallel_analysis(DF, n_iter=200, seed=config.SEED)
    assert pa["n_factors"] == 5
    assert 14 < pa["eigenvalues"][0] < 15        # pearson basis (default)
    assert pa["eigenvalues"][0] > pa["threshold"][0]


def test_parallel_analysis_basis_parameter_changes_the_correlation_matrix():
    """Same discrimination check as `factorability`'s: Spearman's first

    eigenvalue on this data (~16.02) is a different number from Pearson's
    (~14.74), and both must still retain 5 factors."""
    pa_pearson = factors.parallel_analysis(DF, n_iter=200, seed=config.SEED, basis="pearson")
    pa_spearman = factors.parallel_analysis(DF, n_iter=200, seed=config.SEED, basis="spearman")
    assert pa_pearson["n_factors"] == 5
    assert pa_spearman["n_factors"] == 5
    assert pa_spearman["eigenvalues"][0] > 15
    assert pa_pearson["eigenvalues"][0] < pa_spearman["eigenvalues"][0]


def test_parallel_analysis_rejects_polychoric_basis():
    """Simulating n_iter null matrices under polychoric correlation is

    computationally prohibitive (~8s per null matrix on this data); this is
    an explicit, documented refusal rather than a silent hour-long run."""
    with pytest.raises(ValueError):
        factors.parallel_analysis(DF, n_iter=5, seed=config.SEED, basis="polychoric")


def test_parallel_analysis_retains_nothing_from_noise():
    rng = np.random.default_rng(config.SEED)
    noise = pd.DataFrame(rng.normal(size=(400, 12)))
    assert factors.parallel_analysis(noise, n_iter=100, seed=config.SEED)["n_factors"] <= 1


def test_efa_produces_the_expected_shape():
    r = factors.fit_efa(DF, n_factors=5)
    assert r.loadings.shape == (44, 5)
    assert list(r.loadings.index) == config.SCORE_COLS
    assert r.factor_correlations.shape == (5, 5)
    assert np.allclose(np.diag(r.factor_correlations.to_numpy()), 1.0)
    assert (r.communalities.between(0, 1)).all()
    assert np.allclose(r.communalities + r.uniquenesses, 1.0, atol=1e-6)


def test_efa_reproduces_the_reference_solution():
    """Congruence against a stored reference, not aggregate counts. Counts are

    invariant to which items move and invite tuning; congruence is neither.
    (This replaces a prior version of this test that asserted three
    aggregate counts - cross-loading/unassigned/low-communality totals - out
    of the 44x5 matrix. Those counts are invariant to *which* items land
    where, so a solution could drift item-by-item while still passing; they
    were also the numbers a correlation-basis choice got quietly tuned
    against during development, which is exactly the failure mode a
    congruence check against a fixed, committed reference avoids.)
    """
    r = factors.fit_efa(DF, n_factors=5)
    ref = pd.read_csv(REFERENCE_LOADINGS_PATH, index_col=0)
    for f in ref.columns:
        assert factors.tucker_congruence(
            r.loadings[f].to_numpy(), ref[f].to_numpy()) >= 0.95


def test_every_a_priori_dimension_block_stays_together_where_expected():
    """Stakeholder Engagement (E) and Technological Integration (J) each form a

    clean factor, and Risk Management Frameworks (D) another - the solution
    must recover all three intact, and the three factors carrying them must
    be pairwise distinct (not just "some single factor each", which would
    also be satisfied by all three collapsing onto one factor)."""
    r = factors.fit_efa(DF, n_factors=5)
    a = factors.assign_items(r.loadings, threshold=0.40)

    def block_factor(letter):
        items = [c for c in config.SCORE_COLS if c.startswith(letter + "-")]
        primaries = set(a.loc[items, "primary"])
        assert len(primaries) == 1 and None not in primaries
        return primaries.pop()

    factor_e = block_factor("E")
    factor_j = block_factor("J")
    factor_d = block_factor("D")
    assert len({factor_e, factor_j, factor_d}) == 3


def test_factor_scores_have_one_row_per_article():
    r = factors.fit_efa(DF, n_factors=5)
    s = factors.factor_scores(DF, r)
    assert len(s) == len(DF)
    assert list(s.columns) == list(r.loadings.columns)
    assert np.isfinite(s.to_numpy()).all()


def test_reliability_is_reported_per_factor():
    r = factors.fit_efa(DF, n_factors=5)
    a = factors.assign_items(r.loadings, threshold=0.40)
    rel = factors.reliability(DF, a)
    assert len(rel) == 5
    assert rel["cronbach_alpha"].between(-1, 1).all()
    assert rel["mcdonald_omega"].between(0, 1).all()
    # F1 and F2 are large, high-loading factors and must be reliable
    assert rel["mcdonald_omega"].max() > 0.80


def test_congruence_is_one_for_identical_loadings():
    x = np.array([0.8, 0.6, 0.2, -0.4])
    assert abs(factors.tucker_congruence(x, x) - 1.0) < 1e-12
    assert abs(factors.tucker_congruence(x, -x) + 1.0) < 1e-12


def test_greedy_match_recovers_a_known_permutation():
    """Known-answer test for `_greedy_match`'s contract: greedy assignment on

    |congruence|, sign-aligned, one factor used at most once on either side.
    `other`'s columns are an explicit permutation of `ref`'s, sign-flipped
    on some columns; every reference column has an exact (|phi| = 1.0)
    match, so a correct matcher must recover the permutation exactly. This
    was checked by hand during review but was not in the suite; a
    regression that drops the sign alignment or the used-once constraint
    would previously have passed silently."""
    rng = np.random.default_rng(config.SEED)
    ref = rng.normal(size=(44, 5))
    order = [3, 1, 4, 0, 2]      # other[:, i] is ref[:, order[i]], sign-flipped per column
    signs = [1, -1, 1, 1, -1]
    other = np.column_stack([ref[:, order[i]] * signs[i] for i in range(5)])

    pairs = factors._greedy_match(ref, other)

    assert len(pairs) == 5
    for ref_idx, (other_idx, phi) in pairs.items():
        assert order[other_idx] == ref_idx
        assert abs(abs(phi) - 1.0) < 1e-9
        assert np.sign(phi) == signs[other_idx]


def test_bootstrap_stability_reports_congruence_per_factor():
    st = factors.bootstrap_stability(DF, n_factors=5, n_boot=25, seed=config.SEED)
    assert len(st) == 5
    assert st["mean_congruence"].between(-1, 1).all()
    assert (st["lo"] <= st["mean_congruence"]).all()
    assert (st["mean_congruence"] <= st["hi"]).all()


def test_split_half_congruence_runs_on_real_data():
    sh = factors.split_half_congruence(DF, n_factors=5, seed=config.SEED)
    assert len(sh) == 5
    assert sh["congruence"].between(-1, 1).all()


def test_results_are_reproducible():
    a = factors.fit_efa(DF, n_factors=5).loadings
    b = factors.fit_efa(DF, n_factors=5).loadings
    pd.testing.assert_frame_equal(a, b)


def test_sensitivity_by_correlation_basis_reports_per_factor_congruence():
    """Documents the actual reason the shipped solution is Pearson-based:

    polychoric (the correct treatment for ordinal items) agrees with it more
    closely than Spearman does, on every factor. Disagreement is reported,
    not resolved - the simple-structure counts under each basis are carried
    alongside the congruence numbers rather than collapsed into one figure.
    """
    sens = factors.sensitivity_by_correlation_basis(DF, n_factors=5)
    assert len(sens) == 5
    assert sens["vs_spearman"].between(-1, 1).all()
    assert sens["vs_polychoric"].between(-1, 1).all()
    assert (sens["vs_polychoric"] >= 0.95).all()
    # The finding that justifies defaulting to Pearson: polychoric (the
    # methodologically correct basis for ordinal items) tracks the shipped
    # solution more closely than Spearman does, on the worst-matching factor.
    assert sens["vs_polychoric"].min() > sens["vs_spearman"].min()
    for basis in ("pearson", "spearman", "polychoric"):
        for count in ("cross_loading", "unassigned", "low_communality"):
            assert (sens[f"{basis}_{count}"] >= 0).all()
