"""Task 15: the Springer scaffold compiles and the bibliography is correct.

Four checks that don't need a LaTeX toolchain (class/column, declarations
subheadings, bibliography corrections, no phantom factor names anywhere
under manuscript/), plus one end-to-end compile check that does. The
compile check is skipped (not failed) when latexmk/pdflatex/the sn-jnl
class aren't available, so this file doesn't break CI environments without
a LaTeX install - but it is the check that actually proves the manuscript
builds, so it must run wherever the toolchain exists.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "manuscript"
MAIN = MANUSCRIPT / "main.tex"
BIB = MANUSCRIPT / "refs.bib"
SECTIONS = MANUSCRIPT / "sections"

_LATEX_AVAILABLE = shutil.which("latexmk") is not None and shutil.which("pdflatex") is not None


def test_main_uses_the_springer_single_column_class():
    text = MAIN.read_text(encoding="utf-8")
    assert r"\documentclass" in text and "sn-jnl" in text
    assert "sn-basic" in text          # single column, author-year
    assert "twocolumn" not in text


def test_main_carries_title_authors_affiliations_and_orcids_above_the_sections():
    text = MAIN.read_text(encoding="utf-8")
    assert r"\title" in text
    assert "Sharma" in text and "Lamkuche" in text
    assert "Symbiosis" in text and "VIT Bhopal" in text
    assert "0000-0003-0819-4340" in text  # Ravi Sharma ORCID
    assert "0000-0002-8354-6898" in text  # Hemraj Lamkuche ORCID
    # the ORCID text must appear before the first body-section \input (the
    # abstract's own \input{sections/abstract} sits inside \abstract{...} in
    # the title block, above \maketitle, so it doesn't count here)
    orcid_pos = text.find("0000-0003-0819-4340")
    first_input_pos = text.find(r"\input{sections/introduction}")
    assert orcid_pos != -1 and first_input_pos != -1
    assert orcid_pos < first_input_pos


def test_main_inputs_every_section_file():
    text = MAIN.read_text(encoding="utf-8")
    for section in ["introduction", "literature", "methodology", "results",
                     "discussion", "conclusion", "declarations"]:
        assert f"sections/{section}" in text, f"main.tex never \\input's {section}.tex"
    assert (SECTIONS / "abstract.tex").exists()  # pulled in via \abstract{\input{...}}


def test_main_includes_at_least_one_real_figure_and_one_real_table():
    text = "".join((SECTIONS / f"{s}.tex").read_text(encoding="utf-8")
                    for s in ["introduction", "literature", "methodology",
                              "results", "discussion", "conclusion"])
    assert r"\includegraphics" in text
    assert r"\input{tables/" in text or r"\input{../manuscript/tables/" in text
    # the referenced figure and table must be real, generated artefacts
    fig_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    assert fig_match is not None
    fig_name = Path(fig_match.group(1)).name
    assert (ROOT / "figures" / f"{fig_name}.pdf").exists() or \
        (ROOT / "figures" / fig_name).with_suffix(".pdf").exists()
    tab_match = re.search(r"\\input\{tables/([^}]+)\}", text)
    assert tab_match is not None
    assert (MANUSCRIPT / "tables" / f"{tab_match.group(1)}.tex").exists()


def test_declarations_section_has_all_required_subheadings():
    text = (SECTIONS / "declarations.tex").read_text(encoding="utf-8")
    for heading in ["Ethical approval", "Consent to participate", "Consent to publish"]:
        assert heading in text
    assert "Data availability" in text or "Data Availability" in text
    # substantive content, not a Task-16-style placeholder sentence
    assert "not applicable" in text.lower()
    assert "Supplementary Data B" in text
    assert "Funding" in text and "Conflict of interest" in text


def test_bibliography_fixes_the_four_citation_problems():
    bib = BIB.read_text(encoding="utf-8")
    assert "fama1983" in bib                                   # AF1: was missing entirely
    assert re.search(r"kotsantonis.*?year\s*=\s*\{2019\}", bib, re.S | re.I)
    assert re.search(r"matacera.*?year\s*=\s*\{2025\}", bib, re.S | re.I)
    assert re.search(r"damato.*?year\s*=\s*\{2024\}", bib, re.S | re.I)


def test_bibliography_has_all_35_transcribed_references_plus_fama():
    bib = BIB.read_text(encoding="utf-8")
    entries = re.findall(r"^@\w+\{([a-zA-Z0-9]+),", bib, re.M)
    assert len(entries) == 36
    assert len(set(entries)) == 36  # no duplicate keys


def test_no_phantom_esg_factors_survive_anywhere_in_the_manuscript():
    banned = ["Board Independence", "Carbon Emissions", "Water Management",
              "Renewable Energy Use", "Executive Compensation",
              "Community Engagement", "Employee Training"]
    for tex in MANUSCRIPT.rglob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        for term in banned:
            assert term not in text, f"{term} still present in {tex}"


@pytest.mark.skipif(not _LATEX_AVAILABLE, reason="latexmk/pdflatex not installed")
def test_sn_jnl_class_is_installed_beside_main_tex():
    # Task 15 requires the real Springer Nature class, not a substitute -
    # kpsewhich alone isn't enough evidence since a stray same-named file
    # would also satisfy it, so also check it resolves relative to main.tex.
    assert (MANUSCRIPT / "sn-jnl.cls").exists()
    assert (MANUSCRIPT / "sn-basic.bst").exists()


@pytest.mark.skipif(not _LATEX_AVAILABLE, reason="latexmk/pdflatex not installed")
def test_manuscript_compiles():
    r = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=MANUSCRIPT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-4000:]
    assert (MANUSCRIPT / "main.pdf").exists()


@pytest.mark.skipif(not _LATEX_AVAILABLE, reason="latexmk/pdflatex not installed")
def test_compiled_pdf_has_no_type3_fonts():
    # Springer preflight fails a submission that embeds Type 3 (bitmap)
    # fonts; the class/body text must use real outline fonts throughout.
    pdf_bytes = (MANUSCRIPT / "main.pdf").read_bytes()
    assert b"/Type3" not in pdf_bytes


@pytest.mark.skipif(not _LATEX_AVAILABLE, reason="latexmk/pdflatex not installed")
def test_compiled_bibliography_has_the_corrected_years_in_the_pdf_text():
    # end-to-end proof, not just of the .bib source: bibtex + sn-basic.bst
    # must actually TYPESET the corrected years, not silently drop a field.
    pytest.importorskip("fitz", reason="PyMuPDF not installed")
    import fitz
    doc = fitz.open(MANUSCRIPT / "main.pdf")
    text = "\n".join(page.get_text() for page in doc)
    assert re.search(r"Fama.*?\(1983\)", text, re.S)
    assert re.search(r"Kotsantonis.*?\(2019\)", text, re.S)
    assert re.search(r"Matacera.*?\(2025\)", text, re.S)
    assert re.search(r"Amato.*?\(2024\)", text, re.S)
