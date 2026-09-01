import numpy as np
import pytest
from analysis import config, importance, loading, quality

DF = quality.clean(loading.load_scoring())

def test_network_centrality_covers_all_44_nodes():
    c = importance.network_centrality(DF)
    assert len(c) == 44
    assert list(c.index) == config.SCORE_COLS
    assert {"pagerank", "betweenness", "eigenvector", "degree"} <= set(c.columns)

def test_pagerank_sums_to_one():
    c = importance.network_centrality(DF)
    assert abs(c["pagerank"].sum() - 1.0) < 1e-6

def test_gain_importance_is_normalised_and_excludes_the_target():
    s = importance.gain_importance(DF, "A-D1")
    assert "A-D1" not in s.index
    assert s.min() >= 0 and abs(s.max() - 1.0) < 1e-9

def test_permutation_importance_is_normalised():
    s = importance.permutation_importance_scores(DF, "A-D1")
    assert s.min() >= 0 and abs(s.max() - 1.0) < 1e-9

def test_shap_importance_is_normalised():
    s = importance.shap_importance(DF, "A-D1")
    assert s.min() >= 0 and abs(s.max() - 1.0) < 1e-9

def test_convergence_table_excludes_the_undefined_target_row():
    """The target's gain/permutation/shap are undefined by construction, so the
    row is dropped rather than zero-filled - every remaining cell is measured."""
    t = importance.convergence_table(DF, target="A-D1")
    assert len(t) == 43
    assert "A-D1" not in t.index
    assert {"gain", "permutation", "shap", "centrality", "mean_rank"} <= set(t.columns)

def test_concordance_is_one_for_perfect_agreement():
    """Four methods ranking six items identically is perfect concordance.
    The transposed orientation returns NaN here, so this test has real teeth."""
    import pandas as pd
    ranks = pd.DataFrame(np.tile(np.arange(1, 7).reshape(-1, 1), (1, 4)),
                         columns=importance.METHODS)
    w, p = importance.concordance(ranks)
    assert abs(w - 1.0) < 1e-9
    assert p < 0.01

def test_concordance_matches_the_textbook_formula_on_real_data():
    """W = 12S / (m^2 (n^3 - n)), computed independently."""
    table = importance.convergence_table(DF)
    # Target is now excluded by convergence_table, so all 43 rows are measured
    ranks = table[importance.METHODS].rank(ascending=False).to_numpy()
    n, m = ranks.shape
    rsum = ranks.sum(axis=1)
    expected = 12 * ((rsum - rsum.mean()) ** 2).sum() / (m ** 2 * (n ** 3 - n))
    w, _ = importance.concordance(table)
    assert abs(w - expected) < 1e-4

def test_no_output_label_claims_validation():
    """R1.1 required this tier be disclosed as an internal consistency check,
    never as validation. Any new use of 'valid*' must be added here deliberately.
    """
    import inspect, re
    source = inspect.getsource(importance)
    allowed = {
        "internal consistency check, not validation",
        "an internal consistency result, not validation",
    }
    # Extract quoted strings (docstrings and string literals)
    quoted = re.findall(r'"""([^"]*(?:"[^"]*)*?)"""|\'\'\'([^\']*)\'\'\'|"([^"]*(?:\\"[^"]*)?)"|\'([^\']*(?:\\\'[^\']*)?)', source, re.DOTALL)
    hits = []
    for match in quoted:
        text = ''.join(match)  # one of the groups will have content
        if 'valid' in text.lower():
            hits.append(text.strip())
    unexplained = [h for h in hits
                   if not any(a in h.lower() for a in allowed)]
    assert not unexplained, f"unexplained validation language: {unexplained}"
