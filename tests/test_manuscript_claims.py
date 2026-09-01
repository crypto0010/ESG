# tests/test_manuscript_claims.py
"""Task 16: the manuscript's prose satisfies the reviewer-facing content
requirements (construct stated, circular-validation language gone, every
results number traceable to a table or figure). Complements
tests/test_manuscript_build.py, which checks the LaTeX plumbing rather
than the content of the prose.
"""
import re
from pathlib import Path

SECTIONS = Path(__file__).resolve().parent.parent / "manuscript" / "sections"


def _all_text():
    return "\n".join(p.read_text(encoding="utf-8") for p in SECTIONS.glob("*.tex"))


def test_no_circular_validation_language_survives():
    text = _all_text().lower()
    assert "ground truth" not in text
    for bad in ["gradient boosting feature importance were used as the reference",
                "predict the benchmark gbm", "predict these benchmark scores"]:
        assert bad not in text


def test_no_firm_level_causal_claims_survive():
    text = _all_text().lower()
    for bad in ["directly and quantitatively affects organisational performance",
                "directly and quantitatively affects organizational performance",
                "drivers of firm esg performance"]:
        assert bad not in text


def test_construct_is_stated_explicitly_in_abstract_and_methods():
    for name in ("abstract", "methodology"):
        text = (SECTIONS / f"{name}.tex").read_text(encoding="utf-8").lower()
        assert "literature" in text and ("attention" in text or "coded" in text)


def test_tabnet_is_gone_and_lightgbm_is_introduced():
    text = _all_text()
    assert "TabNet" not in text
    assert "LightGBM" in (SECTIONS / "methodology.tex").read_text(encoding="utf-8")


def test_ontology_claims_are_moderated():
    text = _all_text().lower()
    assert "ai reasoning" not in text
    assert "standardized esg scoring ontology" not in text


def test_methods_names_the_statistical_tests():
    text = (SECTIONS / "methodology.tex").read_text(encoding="utf-8")
    for term in ["Friedman", "Benjamini", "bootstrap", "Wilcoxon"]:
        assert term in text


def test_limitations_appear_before_the_conclusion():
    """R1.8: flag limitations where the claims are made, not only at the end."""
    disc = (SECTIONS / "discussion.tex").read_text(encoding="utf-8").lower()
    assert "limitation" in disc


def test_every_results_number_cites_a_table_or_figure():
    text = (SECTIONS / "results.tex").read_text(encoding="utf-8")
    for para in [p for p in text.split("\n\n") if re.search(r"\d+\.\d{2}", p)]:
        assert re.search(r"\\(ref|autoref)\{(tab|fig):", para), \
            f"numeric claim with no table/figure reference:\n{para[:200]}"


def test_convergent_importance_is_never_called_validation():
    """R1.1: convergent importance is an internal consistency check; 'validate'
    or 'validation' must never be used to describe that specific result."""
    text = _all_text().lower()
    for bad in ["validates the convergent importance", "validates the importance",
                "validated by the convergent", "kendall's w validates",
                "kendall's w validated"]:
        assert bad not in text


def test_no_prescriptive_firm_level_language():
    """No board-composition thresholds, KPI targets, or compensation
    redesign prescriptions -- the specific over-reach the submitted
    manuscript made on top of a literature-attention construct."""
    text = _all_text().lower()
    for bad in ["firms should", "companies should", "organizations should implement",
                "organisations should implement", "board composition threshold",
                "compensation redesign"]:
        assert bad not in text
