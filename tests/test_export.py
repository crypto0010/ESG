"""Tests for the manuscript's LaTeX table export (Task 14, revised for the
EFA-primary / clustering-secondary taxonomy: 9 tables, not 6)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import benchmark, export


# --------------------------------------------------------------------------
# Pure formatting: escape_latex / write_table / write_all
# --------------------------------------------------------------------------

def test_escapes_latex_special_characters():
    assert export.escape_latex("a_b & c% d#e") == r"a\_b \& c\% d\#e"
    assert export.escape_latex("GRI, SASB (etc.)") == "GRI, SASB (etc.)"


def test_escape_latex_handles_backslash_and_angle_brackets():
    """I1: `escape_latex` omitted '\\', '<' and '>'. The prior fix worded
    around the single call site that needed '<'/'>' (`_fmt_p`) instead of
    closing the hole in the general escaper every table cell flows through -
    so any OTHER content containing these characters (a free-text label, a
    future data value) was still unescaped."""
    assert export.escape_latex("a\\b") == r"a\textbackslash{}b"
    assert export.escape_latex("x < y") == r"x \textless{} y"
    assert export.escape_latex("x > y") == r"x \textgreater{} y"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_escaped_backslash_lt_gt_round_trip_through_pdflatex(tmp_path):
    """Each of the three newly-escaped characters must actually compile -
    the '<'/'>' fix history in this module shows a plausible-looking escape
    can still break under a real compiler (OT1 encoding upside-down
    punctuation), so this is checked with a real pdflatex, not just string
    assertions."""
    raw = r"a\b < c > d"
    escaped = export.escape_latex(raw)
    doc = tmp_path / "doc.tex"
    doc.write_text(
        "\\documentclass{article}\n\\begin{document}\n" + escaped + "\n\\end{document}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout[-3000:]
    assert (tmp_path / "doc.pdf").exists()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_descriptives_dimension_column_does_not_overflow_the_page(tmp_path):
    """Regression guard, found by compiling the REAL tab_descriptives.tex
    (all 44 rows, every column) with a real pdflatex: the 'dimension'
    column's wrap width, computed from `longest * _WRAP_CM_PER_CHAR`,
    rounded to 2.9cm for the real data's longest dimension name
    ('Standardization and Benchmarking', 32 chars) - 2.44pt too narrow,
    producing 5 overfull \\hbox warnings. A simplified single-numeric-column
    fixture did NOT reproduce it (the exact row/column composition of the
    real table matters here), so this uses the real pipeline end to end."""
    from analysis import descriptives, export as export_mod, loading, quality

    df = quality.clean(loading.load_scoring())
    sub = descriptives.subdimension_table(df)
    table = export_mod.descriptives_table(sub)
    text = export.write_table(table, tmp_path / "t.tex", caption="c", label="l",
                              longtable=True).read_text(encoding="utf-8")

    doc = tmp_path / "doc.tex"
    doc.write_text(
        "\\documentclass[11pt,letterpaper]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\usepackage{booktabs}\n\\usepackage{longtable}\n\\usepackage{array}\n"
        "\\begin{document}\n" + text + "\n\\end{document}\n",
        encoding="utf-8",
    )
    for _ in range(2):  # longtable needs two passes to converge on column widths
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
            cwd=tmp_path, capture_output=True, text=True, timeout=60,
        )
    assert result.returncode == 0, result.stdout[-3000:]
    log = (tmp_path / "doc.log").read_text(encoding="utf-8", errors="replace")
    overfull = [line for line in log.splitlines() if "Overfull" in line]
    assert not overfull, f"overfull hbox warnings: {overfull}"


def test_write_table_emits_a_labelled_table_environment(tmp_path):
    df = pd.DataFrame({"mean": [1.234], "sd": [0.5]}, index=["A-D1"])
    p = export.write_table(df, tmp_path / "t.tex", caption="Descriptives", label="tab:desc")
    text = p.read_text(encoding="utf-8")
    assert r"\begin{table}" in text and r"\end{table}" in text
    assert r"\caption{Descriptives}" in text
    assert r"\label{tab:desc}" in text
    assert "1.23" in text


def test_write_table_escapes_content(tmp_path):
    df = pd.DataFrame({"label": ["cost_of_compliance & more"]}, index=["F-D3"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l").read_text(encoding="utf-8")
    assert r"\_" in text and r"\&" in text


def test_write_table_refuses_an_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        export.write_table(pd.DataFrame(), tmp_path / "t.tex", caption="c", label="l")


def test_write_table_longtable_uses_the_longtable_environment(tmp_path):
    df = pd.DataFrame({"mean": list(range(50))}, index=[f"item{i}" for i in range(50)])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l",
                              longtable=True).read_text(encoding="utf-8")
    assert r"\begin{longtable}" in text and r"\end{longtable}" in text
    assert r"\begin{table}" not in text


def test_write_table_longtable_is_set_in_a_reduced_font_size(tmp_path):
    """Every table (longtable or not) is set in \\footnotesize with a
    reduced \\tabcolsep, not the document's normal 10pt/6pt: measured by
    compiling the real manuscript with pdflatex against sn-basic's actual
    372pt \\textwidth (narrower than the 6.5in placeholder page these
    tables were first tuned against), tab_descriptives, tab_loadings,
    tab_factor_reliability and tab_convergence all overflowed at \\small.
    \\footnotesize plus a 2pt \\tabcolsep clears every one with margin."""
    df = pd.DataFrame({"mean": list(range(50))}, index=[f"item{i}" for i in range(50)])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l",
                              longtable=True).read_text(encoding="utf-8")
    assert r"\footnotesize" in text
    assert r"\setlength{\tabcolsep}{2pt}" in text


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_all_manuscript_tables_fit_the_real_sn_basic_textwidth(tmp_path):
    """Regression guard for the sn-basic pivot: sn-jnl's sn-basic option
    gives a real \\textwidth of 372pt (measured via \\typeout on the real
    manuscript), 96.75pt narrower than the 6.5in/468.75pt placeholder page
    these tables were originally tuned against. Compiles every one of the
    nine real, currently-committed manuscript tables together against a
    372pt-textwidth page and asserts none produces an Overfull \\hbox."""
    tables_dir = Path(__file__).resolve().parent.parent / "manuscript" / "tables"
    bodies = "\n".join(p.read_text(encoding="utf-8")
                       for p in sorted(tables_dir.glob("tab_*.tex")))

    doc = tmp_path / "doc.tex"
    doc.write_text(
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[textwidth=372pt,textheight=9in]{geometry}\n"
        "\\usepackage{booktabs}\n\\usepackage{longtable}\n\\usepackage{array}\n"
        "\\begin{document}\n" + bodies + "\n\\end{document}\n",
        encoding="utf-8",
    )
    result = None
    for _ in range(2):  # longtable needs two passes to converge on column widths
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", doc.name],
            cwd=tmp_path, capture_output=True, text=True, timeout=120,
        )
    assert result.returncode == 0, result.stdout[-3000:]
    log = (tmp_path / "doc.log").read_text(encoding="utf-8", errors="replace")
    overfull = [line for line in log.splitlines() if "Overfull" in line]
    assert not overfull, f"overfull hbox warnings: {overfull}"


def test_write_table_left_aligns_text_columns_and_right_aligns_numeric_ones(tmp_path):
    """A table mixing a text column (task) with a numeric one should not
    right-align prose - that is what earlier made tab_model_spec.tex read
    oddly (every text cell flush against the far right)."""
    df = pd.DataFrame({"task": ["Regression"], "n_items": [13]}, index=["F1"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l").read_text(encoding="utf-8")
    assert r"\begin{tabular}{lrl}" not in text  # not naively "l" then all "r"
    assert r"\begin{tabular}{llr}" in text


def test_write_table_wraps_a_long_text_column_in_a_paragraph_column(tmp_path):
    """A long free-text column (e.g. a sub-dimension label or a model's
    'target' description) must not be left as a plain 'l' column - LaTeX
    never wraps those, so one long cell forces the whole table past the
    page's textwidth (verified by compiling a real table with pdflatex:
    an unwrapped 'label' column produced a 127pt overfull \\hbox)."""
    long_text = "Alignment with international standards (GRI, SASB, etc.), a very long label indeed"
    df = pd.DataFrame({"label": [long_text], "mean": [1.5]}, index=["A-D2"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l").read_text(encoding="utf-8")
    assert "p{" in text


def test_write_table_does_not_wrap_a_short_text_column(tmp_path):
    df = pd.DataFrame({"dimension": ["A"], "mean": [1.5]}, index=["A-D1"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l").read_text(encoding="utf-8")
    assert "p{" not in text
    assert r"\begin{tabular}{llr}" in text


def test_write_table_never_double_escapes_already_escaped_content(tmp_path):
    """A regression guard: to_latex must be called with escape=False once we've
    pre-escaped, or '\\_' would become '\\\\_'."""
    df = pd.DataFrame({"label": ["a_b"]}, index=["x"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l").read_text(encoding="utf-8")
    assert r"\\_" not in text
    assert r"\_" in text


def test_write_table_supports_a_per_column_float_format_dict(tmp_path):
    """C2: congruence columns need 3dp while the rest of the manuscript's
    tables stay at 2dp - `float_format` must be settable per column, not
    only globally."""
    df = pd.DataFrame({"alpha": [0.912345], "congruence": [0.9956]}, index=["F1"])
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l",
                              float_format={"congruence": "{:.3f}"}).read_text(encoding="utf-8")
    assert "0.91" in text and "0.912" not in text     # alpha: falls back to the 2dp default
    assert "0.996" in text                            # congruence: 3dp as requested


def test_write_table_supports_index_false_and_drops_the_index_column(tmp_path):
    """The stats table's bare `reset_index(drop=True)` RangeIndex printed as
    an unlabelled leading column of sequential integers - a rendering
    artefact, not data. `index=False` must omit it entirely."""
    df = pd.DataFrame({"a": ["X", "Y"], "b": ["Y", "Z"], "statistic": [1.0, 2.0]})
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l",
                              index=False).read_text(encoding="utf-8")
    assert "X & Y & 1.00" in text
    assert r"\begin{tabular}{llr}" in text     # a, b (text), statistic (numeric) - no extra index column
    # no bare unlabelled integer-index row like "0 & X & Y & 1.00"
    assert "0 & X" not in text and "1 & Y" not in text


def test_write_table_float_format_accepts_a_callable_for_conditional_markup(tmp_path):
    """I3: bolding a loading at/above the simple-structure threshold needs
    cell-specific LaTeX markup (\\textbf{...}), not just a fixed decimal
    format - a per-column callable formatter must be supported, and its
    output must NOT be re-escaped (it is trusted markup we generated, not
    untrusted text)."""
    df = pd.DataFrame({"F1": [0.82, 0.10]}, index=["a", "b"])
    fmt = lambda v: (f"\\textbf{{{v:.2f}}}" if abs(v) >= 0.40 else f"{v:.2f}")
    text = export.write_table(df, tmp_path / "t.tex", caption="c", label="l",
                              float_format={"F1": fmt}).read_text(encoding="utf-8")
    assert r"\textbf{0.82}" in text
    assert r"\textbf{0.10}" not in text
    assert "0.10" in text


def test_specs_covers_exactly_the_eleven_manuscript_tables():
    """Eleven, not ten: 'reliability' (tab_reliability.tex, Task 12) was
    added alongside `analysis.reliability` - the module exists now, even
    though its own gate input (the completed double-coding workbook) has
    not arrived yet, so the table is written only once that data lands."""
    assert set(export.SPECS) == {
        "factorability", "loadings", "factor_reliability", "sensitivity",
        "descriptives", "model_spec", "benchmark", "stats", "convergence",
        "external", "reliability",
    }


def test_specs_filenames_match_the_task_14_dispatch():
    expected_files = {
        "factorability": "tab_factorability.tex",
        "loadings": "tab_loadings.tex",
        "factor_reliability": "tab_factor_reliability.tex",
        "sensitivity": "tab_sensitivity.tex",
        "descriptives": "tab_descriptives.tex",
        "model_spec": "tab_model_spec.tex",
        "benchmark": "tab_benchmark.tex",
        "stats": "tab_stats.tex",
        "convergence": "tab_convergence.tex",
        "external": "tab_external.tex",
        "reliability": "tab_reliability.tex",
    }
    for key, filename in expected_files.items():
        assert export.SPECS[key][2] == filename


def test_convergence_spec_caption_says_internal_consistency_check_not_validation():
    caption = export.SPECS["convergence"][0]
    assert "internal consistency check" in caption
    assert "valid" not in caption.lower()


def test_write_all_writes_every_specified_table_when_all_present(tmp_path):
    ctx = {k: pd.DataFrame({"x": [1.0]}, index=["r1"]) for k in export.SPECS}
    written = export.write_all(ctx, out_dir=tmp_path)
    assert set(written) == set(export.SPECS)
    for _, (_, _, filename) in export.SPECS.items():
        assert (tmp_path / filename).exists()


def test_write_all_reports_what_it_skipped(tmp_path):
    ctx = {"descriptives": pd.DataFrame({"mean": [1.0]}, index=["A-D1"])}
    written = export.write_all(ctx, out_dir=tmp_path)
    assert "descriptives" in written
    assert "factor_reliability" not in written    # absent from context, so not invented
    assert "model_spec" not in written


def test_write_all_never_invents_an_unrequested_key(tmp_path):
    ctx = {"descriptives": pd.DataFrame({"mean": [1.0]}, index=["A-D1"]),
           "some_other_table": pd.DataFrame({"y": [1.0]}, index=["z"])}
    written = export.write_all(ctx, out_dir=tmp_path)
    assert "some_other_table" not in written


def test_loadings_table_is_written_as_longtable(tmp_path):
    ctx = {"loadings": pd.DataFrame({"F1": [0.7]}, index=["A-D1"])}
    export.write_all(ctx, out_dir=tmp_path)
    text = (tmp_path / "tab_loadings.tex").read_text(encoding="utf-8")
    assert r"\begin{longtable}" in text


# --------------------------------------------------------------------------
# factorability_table
# --------------------------------------------------------------------------

def test_factorability_table_reports_kmo_and_bartlett():
    f = {"kmo_overall": 0.9376, "bartlett_chi2": 30544.53, "bartlett_df": 946,
         "bartlett_p": 0.0, "is_factorable": True}
    t = export.factorability_table(f)
    assert "0.938" in t.loc["Kaiser-Meyer-Olkin (overall)", "value"]
    assert "below 0.001" in t.loc["Bartlett's p", "value"]
    matches = [i for i in t.index if "KMO" in i and "p" in i]
    assert len(matches) == 1
    assert t.loc[matches[0], "value"] == "Yes"


def test_factorability_and_stats_never_use_bare_lt_gt_characters():
    """A literal '<' or '>' renders as an upside-down punctuation mark under
    the default OT1 font encoding - confirmed by compiling a real table
    with pdflatex ('< 0.001' rendered as inverted-exclamation '0.001').
    None of the generated table CONTENT may use these characters; wording
    them out ("below"/"at least") is robust to whatever font encoding the
    final manuscript's preamble ends up using."""
    f = {"kmo_overall": 0.94, "bartlett_chi2": 100.0, "bartlett_df": 10,
         "bartlett_p": 1e-20, "is_factorable": True}
    t = export.factorability_table(f)
    for col in t.reset_index().columns:
        for v in t.reset_index()[col]:
            assert "<" not in str(v) and ">" not in str(v)


def test_factorability_table_does_not_collapse_a_tiny_but_nonzero_p():
    f = {"kmo_overall": 0.7, "bartlett_chi2": 12.0, "bartlett_df": 10,
         "bartlett_p": 0.002, "is_factorable": True}
    t = export.factorability_table(f)
    assert t.loc["Bartlett's p", "value"] == "0.002"


# --------------------------------------------------------------------------
# loadings_table
# --------------------------------------------------------------------------

def test_loadings_table_groups_by_primary_factor_and_sorts_by_loading():
    loadings = pd.DataFrame({
        "F1": [0.80, 0.10, 0.60],
        "F2": [0.05, 0.75, 0.20],
    }, index=["x1", "x2", "x3"])
    communalities = pd.Series([0.65, 0.58, 0.40], index=loadings.index)
    uniquenesses = 1 - communalities
    t = export.loadings_table(loadings, communalities, uniquenesses)
    assert list(t.index) == ["x1", "x3", "x2"]   # F1 group (desc loading), then F2 group
    assert "communality" in t.columns and "uniqueness" in t.columns


def test_loadings_table_includes_a_readable_label_column():
    loadings = pd.DataFrame({"F1": [0.8]}, index=["A-D1"])
    communalities = pd.Series([0.6], index=["A-D1"])
    uniquenesses = pd.Series([0.4], index=["A-D1"])
    t = export.loadings_table(loadings, communalities, uniquenesses)
    assert "label" in t.columns
    assert t.loc["A-D1", "label"]


def test_loadings_table_does_not_crash_on_a_code_with_no_known_label():
    loadings = pd.DataFrame({"F1": [0.8]}, index=["Z-D9"])
    communalities = pd.Series([0.6], index=["Z-D9"])
    uniquenesses = pd.Series([0.4], index=["Z-D9"])
    t = export.loadings_table(loadings, communalities, uniquenesses)
    assert t.loc["Z-D9", "label"] == ""


def test_loadings_table_bolds_loadings_at_or_above_the_simple_structure_threshold(tmp_path):
    """I3: a reader with only the table (grayscale print, supplementary
    material) cannot see the simple structure Figure 4 makes obvious
    without SOME visual distinction for |loading| >= 0.40."""
    loadings = pd.DataFrame({"F1": [0.82, 0.10], "F2": [0.05, 0.71]}, index=["A-D1", "A-D2"])
    communalities = pd.Series([0.68, 0.52], index=loadings.index)
    uniquenesses = 1 - communalities
    t = export.loadings_table(loadings, communalities, uniquenesses)
    export.write_all({"loadings": t}, out_dir=tmp_path)
    text = (tmp_path / "tab_loadings.tex").read_text(encoding="utf-8")
    assert r"\textbf{0.82}" in text
    assert r"\textbf{0.71}" in text
    assert r"\textbf{0.10}" not in text
    assert r"\textbf{0.05}" not in text


# --------------------------------------------------------------------------
# factor_reliability_table
# --------------------------------------------------------------------------

def test_factor_reliability_table_merges_the_three_sources():
    reliability = pd.DataFrame({"n_items": [13, 11], "cronbach_alpha": [0.91, 0.91],
                                "mcdonald_omega": [0.90, 0.89]},
                               index=pd.Index(["F1", "F2"], name="factor"))
    bootstrap = pd.DataFrame({"mean_congruence": [0.996, 0.996]},
                             index=pd.Index(["F1", "F2"], name="factor"))
    split_half = pd.DataFrame({"congruence": [0.988, 0.982]},
                              index=pd.Index(["F1", "F2"], name="factor"))
    t = export.factor_reliability_table(reliability, bootstrap, split_half)
    assert list(t.columns) == ["n_items", "cronbach_alpha", "mcdonald_omega",
                                "bootstrap_congruence", "split_half_congruence"]
    assert t.loc["F1", "bootstrap_congruence"] == 0.996
    assert t.loc["F2", "split_half_congruence"] == 0.982


def test_factor_reliability_congruence_columns_render_at_three_decimals(tmp_path):
    """C2: the same over-rounding bug as tab_sensitivity - 0.9956/0.9983-style
    bootstrap/split-half congruence values must not collapse to '1.00'."""
    reliability = pd.DataFrame({"n_items": [13], "cronbach_alpha": [0.9113],
                                "mcdonald_omega": [0.9003]},
                               index=pd.Index(["F1"], name="factor"))
    bootstrap = pd.DataFrame({"mean_congruence": [0.9956797179165667]},
                             index=pd.Index(["F1"], name="factor"))
    split_half = pd.DataFrame({"congruence": [0.9884262939155627]},
                              index=pd.Index(["F1"], name="factor"))
    t = export.factor_reliability_table(reliability, bootstrap, split_half)
    export.write_all({"factor_reliability": t}, out_dir=tmp_path)
    text = (tmp_path / "tab_factor_reliability.tex").read_text(encoding="utf-8")
    assert "0.996" in text
    assert "0.988" in text
    assert "1.00" not in text


# --------------------------------------------------------------------------
# sensitivity_table (C1: restructured to congruence-only; counts move to the
# caption via sensitivity_caption, since they are per-basis, not per-factor,
# and forcing them into every factor row falsely reads as "F1 has 5
# cross-loaders, F2 has 5 cross-loaders, ...")
# --------------------------------------------------------------------------

_SENS = pd.DataFrame({
    "vs_spearman": [0.9956460540191907, 0.891120583478223],
    "vs_polychoric": [0.9982524882942204, 0.9904459034877507],
    "pearson_cross_loading": [5, 5], "pearson_unassigned": [5, 5], "pearson_low_communality": [3, 3],
    "spearman_cross_loading": [1, 1], "spearman_unassigned": [4, 4], "spearman_low_communality": [1, 1],
    "polychoric_cross_loading": [5, 5], "polychoric_unassigned": [4, 4], "polychoric_low_communality": [1, 1],
}, index=pd.Index(["F1", "F3"], name="factor"))


def test_sensitivity_table_is_congruence_only_no_broadcast_counts():
    """C1: the old table printed the same '5 / 5 / 3' triple on every
    factor row - a per-basis (whole-solution) count broadcast to look like
    a per-factor one. The counts must not appear as factor-row columns."""
    t = export.sensitivity_table(_SENS)
    assert list(t.columns) == ["vs Spearman", "vs polychoric"]
    assert "cross-load. (n)" not in t.columns


def test_sensitivity_caption_reports_all_three_bases_counts():
    """C1: Spearman (1/4/1) and polychoric (5/4/1) counts were absent
    entirely from the old table - the comparison the table exists to show.
    All three bases must appear, read from the data, not hardcoded."""
    caption = export.sensitivity_caption(_SENS)
    assert "pearson" in caption.lower()
    assert "5 / 5 / 3" in caption
    assert "spearman" in caption.lower()
    assert "1 / 4 / 1" in caption
    assert "polychoric" in caption.lower()
    assert "5 / 4 / 1" in caption


def test_sensitivity_congruence_survives_at_three_decimals(tmp_path):
    """C2: 0.9956 and 0.9983 both rounded to '1.00' at 2dp, collapsing the
    exact distinction the column exists to convey. Three decimals."""
    t = export.sensitivity_table(_SENS)
    export.write_all({"sensitivity": t, "sensitivity_caption": export.sensitivity_caption(_SENS)},
                     out_dir=tmp_path)
    text = (tmp_path / "tab_sensitivity.tex").read_text(encoding="utf-8")
    assert "0.996" in text
    assert "0.998" in text
    assert "1.00" not in text
    assert "0.891" in text


# --------------------------------------------------------------------------
# model_spec_table
# --------------------------------------------------------------------------

def test_model_spec_table_covers_every_estimator_with_a_single_task_and_metric_set():
    t = export.model_spec_table()
    assert set(t.index) == set(benchmark.make_estimators().keys())
    assert (t["task"] == "Regression").all()
    assert t.loc["RandomForest", "tuned"] == "Yes"
    assert t.loc["Dummy", "tuned"] == "No"


def test_model_spec_table_never_mixes_regression_and_classification():
    """R2.6: task/metric inconsistency. A single task column, one value only."""
    t = export.model_spec_table()
    assert t["task"].nunique() == 1


# --------------------------------------------------------------------------
# benchmark_table
# --------------------------------------------------------------------------

def test_benchmark_table_folds_the_interval_into_one_column():
    summary = pd.DataFrame({"mae": [0.38], "mae_lo": [0.35], "mae_hi": [0.42],
                            "rmse": [0.5], "r2": [0.2], "spearman": [0.4]}, index=["RandomForest"])
    t = export.benchmark_table(summary)
    assert "95% CI" in t.columns
    assert "[0.35, 0.42]" in t.loc["RandomForest", "95% CI"]


# --------------------------------------------------------------------------
# stats_table
# --------------------------------------------------------------------------

def test_stats_table_reports_bh_adjusted_p_for_every_model_pair():
    rng = np.random.default_rng(0)
    results = pd.DataFrame({
        "target": np.repeat([f"T{i}" for i in range(10)], 3),
        "model": list(["A", "B", "C"]) * 10,
        "mae": rng.uniform(0.2, 0.6, 30),
    })
    t = export.stats_table(results)
    assert len(t) == 3   # C(3,2) pairs
    assert "p_adj" in t.columns


def test_stats_table_formats_tiny_pvalues_as_a_threshold_string(tmp_path):
    """A raw p-value like 1.1e-13 must never render as the misleading '0.00'
    a bare 2-decimal float format would produce (verified against the real
    44-target sweep, whose closest pairwise comparisons are ~1e-13) - it
    must show as the same 'below 0.001' string factorability_table uses."""
    rng = np.random.default_rng(2)
    results = pd.DataFrame({
        "target": np.repeat([f"T{i}" for i in range(30)], 2),
        "model": list(["A", "B"]) * 30,
        "mae": np.tile([0.2, 0.9], 30) + rng.normal(0, 0.001, 60),
    })
    t = export.stats_table(results)
    assert t.loc[0, "p_value"] == "below 0.001"
    # the formatted table must still compile/serialise as plain text columns
    text = export.write_table(t, tmp_path / "t.tex", caption="c", label="l",
                              longtable=True).read_text(encoding="utf-8")
    assert "below 0.001" in text
    assert "e-" not in text  # no leaked scientific notation
    assert "<" not in text and ">" not in text  # renders as upside-down punctuation in OT1


def test_stats_table_columns_are_manuscript_ready():
    rng = np.random.default_rng(1)
    results = pd.DataFrame({
        "target": np.repeat([f"T{i}" for i in range(12)], 2),
        "model": list(["A", "B"]) * 12,
        "mae": rng.uniform(0.2, 0.6, 24),
    })
    t = export.stats_table(results)
    assert {"a", "b", "statistic", "p_value", "p_adj", "rank_biserial"} <= set(t.columns)


def test_write_all_drops_the_unlabelled_index_column_from_tab_stats(tmp_path):
    """`stats_table`'s `reset_index(drop=True)` leaves a meaningless 0..n-1
    RangeIndex that, once rendered, appears as an unlabelled leading column
    of sequential integers - not data, a serialisation artefact."""
    rng = np.random.default_rng(1)
    results = pd.DataFrame({
        "target": np.repeat([f"T{i}" for i in range(12)], 2),
        "model": list(["A", "B"]) * 12,
        "mae": rng.uniform(0.2, 0.6, 24),
    })
    t = export.stats_table(results)
    export.write_all({"stats": t}, out_dir=tmp_path)
    text = (tmp_path / "tab_stats.tex").read_text(encoding="utf-8")
    assert "\\begin{longtable}{llr" in text        # a, b, statistic - no extra leading index column
    assert "\\begin{longtable}{lllr" not in text   # would mean an index column survived
    assert " & a & b & statistic" not in text      # the old blank-header artefact


# --------------------------------------------------------------------------
# convergence_table / convergence_caption (C3: convergent importance had no
# table at all, despite importance.convergence_table existing and being
# ready to feed one)
# --------------------------------------------------------------------------

def _synthetic_convergence_table(n=20, seed=0):
    from analysis import importance
    rng = np.random.default_rng(seed)
    idx = [f"X-D{i}" for i in range(n)]
    t = pd.DataFrame({m: rng.uniform(0, 1, n) for m in importance.METHODS}, index=idx)
    t["mean_rank"] = t[importance.METHODS].rank(ascending=False).mean(axis=1)
    return t.sort_values("mean_rank")


def test_convergence_table_takes_the_top_15_by_mean_rank():
    from analysis import importance
    full = _synthetic_convergence_table(n=25)
    t = export.convergence_table(full, top_n=15)
    assert len(t) == 15
    assert list(t.index) == list(full.index[:15])           # most convergently important first
    assert set(importance.METHODS) <= set(t.columns)
    assert "mean_rank" in t.columns


def test_convergence_table_includes_a_readable_label_column():
    full = _synthetic_convergence_table(n=20)
    t = export.convergence_table(full, top_n=5)
    assert "label" in t.columns


def test_convergence_caption_reports_kendalls_w_and_p_read_not_hardcoded():
    from analysis import importance
    full = _synthetic_convergence_table(n=20)
    caption = export.convergence_caption(full)
    w, p = importance.concordance(full)
    assert f"{w:.3f}" in caption or f"{w:.4f}" in caption
    assert "internal consistency check" in caption
    assert "valid" not in caption.lower()


def test_convergence_caption_never_hardcodes_a_prior_sessions_w_and_p():
    """Regression guard against copying the 0.4595 / 0.00076 numbers quoted
    in the finding text verbatim instead of reading `importance.concordance`
    on the data actually passed in."""
    from analysis import importance
    full = _synthetic_convergence_table(n=20, seed=99)   # different data => different W
    caption = export.convergence_caption(full)
    w, _ = importance.concordance(full)
    assert "0.4595" not in caption or abs(w - 0.4595) < 1e-6


# --------------------------------------------------------------------------
# external_table / external_caption (Task 18: Study 2, tab_external)
# --------------------------------------------------------------------------

def _synthetic_coverage():
    return {
        "n_covered": 390, "n_uncovered": 110,
        "per_indicator_coverage": {"S_WomenManagers_pct": 0.254, "S_Employees": 0.820,
                                   "E_CO2_Total": 0.750},
        "median_marketcap_covered": 5.19e9, "median_marketcap_uncovered": 2.28e9,
        "marketcap_mannwhitney_p": 3.8e-14,
        "spearman_rank_vs_n_indicators": -0.393, "spearman_rank_vs_n_indicators_p": 7.1e-20,
        "sector_coverage": {"Technology": 0.698, "Consumer Non-Cyclicals": 0.871},
        "n_fallback_fundamentals_year": 2, "n_no_usable_fundamentals": 1,
    }


def _synthetic_correspondence():
    return {
        "partial_correlation": 0.186, "ci_lo": 0.089, "ci_hi": 0.281, "n": 388,
        "n_fallback_in_sample": 1, "literature_model_value": 0.1392,
        "literature_model_label": "literature F1 (governance) - F2 (disclosure) factor correlation",
    }


def test_external_table_reports_coverage_and_correspondence():
    t = export.external_table(_synthetic_coverage(), _synthetic_correspondence())
    assert "390" in t.loc["Firms with ESG data (of 500)", "value"]
    assert "110" in t.loc["Firms without ESG data", "value"]
    assert "388" in t.loc["Firms in correspondence sample", "value"]
    assert "0.186" in t.loc["Governance-disclosure partial correlation", "value"]
    assert "0.089" in t.loc["95% bootstrap CI"]["value"]
    assert "0.281" in t.loc["95% bootstrap CI"]["value"]
    assert "0.139" in t.loc["Literature model comparator", "value"]


def test_external_table_reports_tiny_pvalues_as_a_threshold_string():
    t = export.external_table(_synthetic_coverage(), _synthetic_correspondence())
    assert "below 0.001" in t.loc["Coverage vs. size, Mann-Whitney p", "value"]


def test_external_table_names_the_lowest_and_highest_coverage_indicators():
    t = export.external_table(_synthetic_coverage(), _synthetic_correspondence())
    text = " ".join(t.index)
    assert "S_WomenManagers_pct" in text
    assert "S_Employees" in text


def test_external_spec_registered_with_the_right_filename_and_label():
    assert export.SPECS["external"][2] == "tab_external.tex"
    assert export.SPECS["external"][1] == "tab:external"


def test_external_caption_reports_actual_numbers_not_hardcoded():
    caption = export.external_caption(_synthetic_correspondence())
    assert "388" in caption
    assert "0.186" in caption or "0.19" in caption


def test_external_caption_never_claims_validation_of_the_literature_model():
    caption = export.external_caption(_synthetic_correspondence())
    low = caption.lower()
    assert "correspondence" in low
    assert "validates" not in low and "validated" not in low
    if "valid" in low:
        assert "never as validation" in low or "not validation" in low


def test_external_caption_differs_when_the_correspondence_data_differs():
    """Regression guard against a static, copy-pasted caption: two
    different correspondence results must produce two different captions."""
    a = export.external_caption(_synthetic_correspondence())
    other = dict(_synthetic_correspondence())
    other["partial_correlation"] = 0.02
    other["ci_lo"], other["ci_hi"] = -0.05, 0.09
    b = export.external_caption(other)
    assert a != b


# --------------------------------------------------------------------------
# reliability_table / reliability_caption (Task 12, tab_reliability)
# --------------------------------------------------------------------------

def _synthetic_reliability_table():
    from analysis import config as _config, reliability
    n = 20
    rng = np.random.default_rng(_config.SEED)
    ea = pd.DataFrame({c: rng.integers(0, 6, n).astype(float) for c in _config.SCORE_COLS})
    eb = ea.copy()
    return reliability.reliability_table(ea, eb)


def _synthetic_reliability_bootstrap():
    return {"n_articles": 51, "n_boot": 2000, "kappa": 0.812, "kappa_lo": 0.741,
           "kappa_hi": 0.869, "icc": 0.855, "icc_lo": 0.793, "icc_hi": 0.901}


def _synthetic_recall():
    return {"per_sheet": {"Expert A": {"count": 1, "articles": [69]},
                          "Expert B": {"count": 0, "articles": []}},
           "total_flags": 1, "articles_flagged_by_either": [69]}


def test_reliability_table_adds_a_readable_label_for_subdimension_rows():
    from analysis import config as _config
    t = export.reliability_table(_synthetic_reliability_table())
    assert "label" in t.columns
    assert t.loc["A-D1", "label"] == _config.SUBDIMENSIONS["A-D1"]
    assert t.loc["OVERALL", "label"] == ""
    assert t.loc["A", "label"] == ""


def test_reliability_table_keeps_the_original_columns():
    t = export.reliability_table(_synthetic_reliability_table())
    assert list(t.columns) == ["label", "n", "percent_agreement", "kappa", "icc"]


def test_reliability_spec_registered_with_the_right_filename_and_label():
    assert export.SPECS["reliability"][2] == "tab_reliability.tex"
    assert export.SPECS["reliability"][1] == "tab:reliability"


def test_reliability_spec_caption_names_weighted_kappa_and_icc():
    caption = export.SPECS["reliability"][0].lower()
    assert "kappa" in caption
    assert "icc" in caption


def test_reliability_caption_reports_actual_numbers_not_hardcoded():
    caption = export.reliability_caption(_synthetic_reliability_bootstrap(), _synthetic_recall())
    assert "0.812" in caption
    assert "0.855" in caption
    assert "51" in caption
    assert "1 article" in caption or " 1 " in caption


def test_reliability_caption_differs_when_the_bootstrap_data_differs():
    a = export.reliability_caption(_synthetic_reliability_bootstrap(), _synthetic_recall())
    other = dict(_synthetic_reliability_bootstrap())
    other["kappa"], other["kappa_lo"], other["kappa_hi"] = 0.30, 0.10, 0.48
    b = export.reliability_caption(other, _synthetic_recall())
    assert a != b
