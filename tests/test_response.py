# tests/test_response.py
"""Task 17: the 32-item point-by-point response to reviewers is complete,
in order, non-empty, names concrete changes, addresses the two central
Reviewer 1 concerns head-on, discloses AF3 without softening it, and
compiles to a PDF.
"""
import re
import subprocess
from pathlib import Path

from response import response_items

ROOT = Path(__file__).resolve().parent.parent
IDS = (["E1", "E2"]
       + [f"R1.{i}" for i in range(1, 9)]
       + [f"R2.{i}" for i in range(1, 14)]
       + [f"R3.{i}" for i in range(1, 5)]
       + [f"AF{i}" for i in range(1, 6)])


def test_every_spec_item_has_a_response():
    assert [i["id"] for i in response_items.ITEMS] == IDS
    assert len(response_items.ITEMS) == 32


def test_no_response_is_empty_or_a_placeholder():
    for item in response_items.ITEMS:
        assert len(item["response"].strip()) > 40, item["id"]
        low = item["response"].lower()
        assert not any(p in low for p in ["tbd", "todo", "placeholder", "as discussed above"])


def test_every_item_points_at_a_concrete_change():
    for item in response_items.ITEMS:
        assert item["changes"].strip(), item["id"]
        assert any(k in item["changes"] for k in
                   ["Section", "Table", "Figure", "Supplementary", "reference"]), item["id"]


def test_reject_recommendation_is_addressed_head_on():
    r11 = next(i for i in response_items.ITEMS if i["id"] == "R1.1")
    r12 = next(i for i in response_items.ITEMS if i["id"] == "R1.2")
    assert "circular" in r11["response"].lower()
    assert "reframe" in r12["response"].lower() or "reframed" in r12["response"].lower()


def test_audit_items_are_disclosed_not_buried():
    af3 = next(i for i in response_items.ITEMS if i["id"] == "AF3")
    assert "44" in af3["response"]
    assert "regenerat" in af3["response"].lower()


def test_pending_items_are_not_claimed_as_resolved():
    """R1.3, R1.4, R2.2 and R2.4 are pending -- the response must say so,
    not imply completion."""
    for item_id in ["R1.3", "R1.4", "R2.2", "R2.4"]:
        item = next(i for i in response_items.ITEMS if i["id"] == item_id)
        low = item["response"].lower()
        assert "not yet" in low or "pending" in low, item_id


def test_convergent_importance_never_called_validation():
    """Binding rule: never describe the convergent-importance result as
    validation, anywhere in the response document."""
    for item in response_items.ITEMS:
        low = item["response"].lower()
        for bad in ["validates the convergent importance", "validated by the convergent",
                    "kendall's w validates", "kendall's w validated"]:
            assert bad not in low, item["id"]


def test_seven_phantom_factor_names_appear_only_in_af3():
    """The seven phantom ESG factor names may appear only inside AF3."""
    phantom_terms = ["board independence", "carbon emissions", "employee training",
                      "water management", "executive compensation",
                      "community engagement", "renewable energy use"]
    for item in response_items.ITEMS:
        haystack = (item["comment"] + " " + item["response"]).lower()
        for term in phantom_terms:
            if term in haystack:
                assert item["id"] == "AF3", f"{term!r} found outside AF3, in {item['id']}"


def test_response_figure_numbers_match_compiled_manuscript_order():
    """The figure *filenames* (fig1_prisma, fig2_architecture, ...) encode
    an intended order that the manuscript prose does not follow: what
    actually determines each figure's printed number is the order
    \\includegraphics commands appear across manuscript/sections/*.tex, in
    the order main.tex \\input's those files. response_items.py must cite
    that compiled number, not the filename's number.

    This derives the true order directly from the manuscript source (not
    from a hardcoded table) and checks every "Figure N" mentioned in
    response_items.ITEMS against it, identified by a descriptive keyword
    that must appear in the same sentence/clause as the "Figure N" text --
    which is also why every figure reference in response_items.py should
    read as "Figure N (the ... diagram)" rather than a bare "Figure N": a
    bare number gives this test nothing to anchor the check to.
    """
    manuscript_dir = ROOT / "manuscript"
    main_text = (manuscript_dir / "main.tex").read_text(encoding="utf-8")

    # 1. main.tex's \input order for body sections (the abstract is pulled
    #    in separately via \abstract{\input{...}}, above \maketitle, and
    #    carries no figures, so it is not part of the float ordering).
    body_sections = re.findall(r"\\input\{sections/(\w+)\}", main_text)
    assert body_sections, "main.tex does not \\input any section files"

    # 2. \includegraphics order across those files, each mapped to the
    #    \label{fig:...} inside the same figure environment.
    fig_env = re.compile(
        r"\\begin\{figure\}.*?\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
        r".*?\\label\{(fig:[^}]+)\}.*?\\end\{figure\}",
        re.DOTALL,
    )
    compiled_order = []  # [(label, filename), ...] in compiled order
    for section in body_sections:
        path = manuscript_dir / "sections" / f"{section}.tex"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in fig_env.finditer(text):
            filename, label = match.group(1), match.group(2)
            compiled_order.append((label, filename))

    assert len(compiled_order) == 8, (
        f"expected 8 figures in the compiled manuscript, found "
        f"{len(compiled_order)}: {compiled_order}")
    compiled_number = {label: i + 1 for i, (label, _) in enumerate(compiled_order)}

    # Sanity-check against the known-correct compiled order (task table),
    # so a bug in the parser above doesn't silently validate itself.
    expected = {
        "fig:architecture": 1, "fig:prisma": 2, "fig:descriptives": 3,
        "fig:scree": 4, "fig:loadings": 5, "fig:benchmark": 6,
        "fig:external": 7, "fig:dendrogram": 8,
    }
    assert compiled_number == expected, compiled_number

    # 3. A descriptive keyword, unique per figure, that must appear in the
    #    parenthetical directly following a "Figure N" mention -- i.e. every
    #    reference in response_items.py must read "Figure N (the ... "
    #    diagram/plot/heatmap ...)" rather than a bare "Figure N", which is
    #    also why that style was required when fixing the numbers: a bare
    #    number gives this test nothing to anchor the check to, and scoping
    #    to the parenthetical (rather than the whole sentence) avoids an
    #    unrelated word in the same sentence -- e.g. "the predictive
    #    benchmark" as a subsection name -- being mistaken for a reference
    #    to Figure 6, the benchmark *figure*.
    KEYWORDS = {
        "fig:architecture": ("architecture", "pipeline diagram"),
        "fig:prisma": ("prisma",),
        "fig:descriptives": ("descriptives",),
        "fig:scree": ("scree",),
        "fig:loadings": ("loadings",),
        "fig:benchmark": ("benchmark",),
        "fig:external": ("external",),
        "fig:dendrogram": ("dendrogram",),
    }

    def _bounded(keyword, text):
        """keyword occurs in text and isn't part of a longer lowercase
        word (but *is* allowed to sit next to '_', '.', '/', digits, etc.,
        so it still matches inside a filename like fig8_external.pdf)."""
        return re.search(r"(?<![a-z])" + re.escape(keyword) + r"(?![a-z])", text) is not None

    figure_mention = re.compile(r"Figure (\d+)\s*\(([^()]*)\)")

    checked = 0
    for item in response_items.ITEMS:
        haystack = item["response"] + " " + item["changes"]
        for match in figure_mention.finditer(haystack):
            stated_number = int(match.group(1))
            descriptor = match.group(2).lower()
            named = [label for label, kws in KEYWORDS.items()
                     if any(_bounded(kw, descriptor) for kw in kws)]
            assert named, (
                f"{item['id']}: 'Figure {stated_number}' has no "
                f"recognisable figure name in its parenthetical: "
                f"{match.group(0)!r}")
            assert len(named) == 1, (
                f"{item['id']}: ambiguous figure name(s) {named} in "
                f"{match.group(0)!r}")
            label = named[0]
            assert compiled_number[label] == stated_number, (
                f"{item['id']}: says 'Figure {stated_number}' for "
                f"{label}, but {label} is compiled as Figure "
                f"{compiled_number[label]} (filename order does not "
                f"match compiled order): {match.group(0)!r}")
            checked += 1

    # Every figure this response letter explicitly discusses (all but the
    # descriptives/dendrogram figures, which no item happens to number).
    assert checked == 11, f"expected 11 'Figure N' mentions, checked {checked}"


def test_response_compiles():
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "response.tex"], cwd=ROOT / "response", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:]
    assert (ROOT / "response" / "response.pdf").exists()
