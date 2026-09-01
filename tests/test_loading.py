import pandas as pd
from analysis import config, loading


def test_loads_all_articles():
    df = loading.load_scoring()
    assert len(df) == 1026
    assert df["sr_no"].min() == 1 and df["sr_no"].max() == 1026


def test_has_all_score_columns_as_float():
    df = loading.load_scoring()
    for col in config.SCORE_COLS:
        assert col in df.columns
        assert pd.api.types.is_float_dtype(df[col])


def test_metadata_columns_present():
    df = loading.load_scoring()
    for col in ["year", "authors", "title", "source_title", "abstract"]:
        assert col in df.columns


def test_the_one_known_missing_title_is_preserved_not_filled():
    """SR.NO 336 has no title in the source. The loader must not invent one -
    Task 2's audit discloses it instead."""
    df = loading.load_scoring()
    assert df["title"].isna().sum() == 1
    assert df.loc[df["sr_no"] == 336, "title"].isna().all()
    assert df.loc[df["sr_no"] != 336, "title"].notna().all()


def test_year_distribution_matches_source():
    df = loading.load_scoring()
    counts = df["year"].value_counts().sort_index().to_dict()
    assert counts == {2020: 19, 2021: 25, 2022: 69, 2023: 204, 2024: 693, 2025: 16}


def test_empty_metadata_columns_are_dropped():
    # RESEARCH DOMAIN and GEOGRAPHIC DOMAIN are empty for all 1026 rows (spec 2.4)
    df = loading.load_scoring()
    assert "research_domain" not in df.columns
    assert "geographic_domain" not in df.columns
