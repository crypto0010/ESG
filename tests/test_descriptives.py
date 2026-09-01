import numpy as np
import pandas as pd
from analysis import config, descriptives, loading, quality

DF = quality.clean(loading.load_scoring())

def test_subdimension_table_covers_all_44():
    t = descriptives.subdimension_table(DF)
    assert len(t) == 44
    assert list(t.index) == config.SCORE_COLS
    assert t.loc["A-D1", "label"] == "Reporting Comprehensiveness"

def test_subdimension_means_are_inside_the_scale():
    t = descriptives.subdimension_table(DF)
    assert t["mean"].between(0, 5).all()
    assert t["zero_rate"].between(0, 1).all()

def test_known_descriptive_values_reproduce():
    t = descriptives.subdimension_table(DF)
    # computed from the cleaned datasheet; guards against silent data drift
    assert round(t.loc["A-D1", "mean"], 2) == 2.59
    assert round(t.loc["J-D3", "mean"], 2) == 0.50

def test_dimension_table_has_eleven_rows_with_alpha():
    t = descriptives.dimension_table(DF)
    assert len(t) == 11
    assert t["cronbach_alpha"].between(-1, 1).all()
    assert t.loc["A", "name"] == "Sustainability Reporting"

def test_cronbach_alpha_is_one_for_identical_items():
    frame = pd.DataFrame({"a": [1, 2, 3, 4.0], "b": [1, 2, 3, 4.0], "c": [1, 2, 3, 4.0]})
    assert descriptives.cronbach_alpha(frame) > 0.99

def test_cronbach_alpha_is_low_for_noise():
    rng = np.random.default_rng(config.SEED)
    frame = pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd"))
    assert descriptives.cronbach_alpha(frame) < 0.2

def test_yearly_trend_spans_2020_to_2025():
    t = descriptives.yearly_trend(DF)
    assert list(t.index) == [2020, 2021, 2022, 2023, 2024, 2025]
    assert list(t.columns) == list(config.DIMENSIONS)

def test_correlation_matrix_is_square_and_symmetric():
    m = descriptives.correlation_matrix(DF)
    assert m.shape == (44, 44)
    assert np.allclose(m.to_numpy(), m.to_numpy().T, atol=1e-9)
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
