from analysis import config


def test_eleven_dimensions():
    assert len(config.DIMENSIONS) == 11
    assert config.DIMENSIONS["A"] == "Sustainability Reporting"
    assert config.DIMENSIONS["K"] == "Due Diligence"


def test_forty_four_subdimensions():
    assert len(config.SUBDIMENSIONS) == 44
    assert len(config.SCORE_COLS) == 44
    assert config.SCORE_COLS[0] == "A-D1"
    assert config.SCORE_COLS[-1] == "K-D4"
    assert config.SUBDIMENSIONS["A-D1"] == "Reporting Comprehensiveness"
    assert config.SUBDIMENSIONS["K-D4"] == "Continuous improvement frameworks"


def test_every_dimension_has_four_subdimensions():
    for letter in config.DIMENSIONS:
        assert sum(c.startswith(letter + "-") for c in config.SCORE_COLS) == 4


def test_normalise_code_accepts_both_spellings():
    # datasheet uses "A-D1", the Table A.1 codebook uses "AD1"
    assert config.normalise_code("AD1") == "A-D1"
    assert config.normalise_code("A-D1") == "A-D1"
    assert config.normalise_code(" a-d1 ") == "A-D1"


def test_no_esg_factor_names_leak_into_schema():
    banned = ["board", "carbon", "water", "renewable", "compensation", "community", "training"]
    blob = " ".join(config.SUBDIMENSIONS.values()).lower()
    assert not any(b in blob for b in banned)
