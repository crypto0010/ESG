"""Generate the manuscript's LaTeX tables. Never hand-edit the output files.

Revised for the measurement-model pivot (see analysis/factors.py): the
manuscript now reports 9 tables, not the original brief's 6 -
factorability, loadings, factor_reliability, sensitivity and convergence
replace the old reliability/categories pair that described Ward clustering
as the primary taxonomy.

Two layers:

- ``escape_latex`` / ``write_table`` / ``write_all`` are pure LaTeX
  serialisation: they take an already-built, presentation-ready DataFrame
  and never compute anything from raw analysis output. ``write_all`` writes
  only the tables present in its ``context`` dict, so an absent result is
  skipped, never fabricated.
- The ``*_table`` builder functions (``factorability_table``,
  ``loadings_table``, ``factor_reliability_table``, ``sensitivity_table``,
  ``model_spec_table``, ``benchmark_table``, ``stats_table``,
  ``convergence_table``) turn the analysis modules' native return shapes
  into that presentation-ready form.
  They take already-computed objects (dicts, Series, DataFrames) as
  arguments rather than a raw coded-score DataFrame, so they stay cheap and
  testable with small synthetic fixtures - the expensive computation (EFA
  bootstrap, the 44-target benchmark sweep, ...) happens once, elsewhere,
  and is cached.
"""
import re
from pathlib import Path

import pandas as pd

from . import benchmark, config, stats

_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
            # I1: the general escaper every table cell flows through omitted
            # these three. The '<'/'>' compile defect (renders as upside-down
            # punctuation under OT1) was previously worked around only at its
            # single call site (_fmt_p, by wording "below"/"at least" instead
            # of using the literal character) rather than closed here - so
            # any OTHER cell containing '\', '<' or '>' was still unescaped.
            "\\": r"\textbackslash{}", "<": r"\textless{}", ">": r"\textgreater{}"}

# key -> (caption, label, filename). Nine tables: the Task 14 dispatch's
# eight, plus "convergence" (C3) - the convergent-importance internal
# consistency check had `importance.convergence_table` ready to feed a
# table, but no SPECS entry ever called it. The largest tables (loadings,
# descriptives, stats) are written as longtable so none overflows a page.
SPECS = {
    "external": ("Study 2: firm-level ESG coverage and the governance-disclosure "
                "correspondence check (LSEG extract, restricted to well-disclosing firms)",
                "tab:external", "tab_external.tex"),
    "factorability": ("Sampling adequacy and sphericity",
                      "tab:factorability", "tab_factorability.tex"),
    "loadings": ("Rotated factor loadings, communalities and uniquenesses",
                "tab:loadings", "tab_loadings.tex"),
    "factor_reliability": ("Factor reliability and stability",
                           "tab:factorreliability", "tab_factor_reliability.tex"),
    "sensitivity": ("Sensitivity of the solution to correlation basis",
                    "tab:sensitivity", "tab_sensitivity.tex"),
    "descriptives": ("Descriptive statistics for the 44 coded sub-dimensions",
                     "tab:descriptives", "tab_descriptives.tex"),
    "model_spec": ("Model, task, target and metric specification",
                   "tab:modelspec", "tab_model_spec.tex"),
    "benchmark": ("Held-out predictive accuracy with 95\\% BCa intervals",
                  "tab:benchmark", "tab_benchmark.tex"),
    "stats": ("Statistical comparison with Benjamini--Hochberg adjusted $p$",
              "tab:stats", "tab_stats.tex"),
    "convergence": ("Convergence of importance methods (internal consistency check)",
                    "tab:convergence", "tab_convergence.tex"),
}

LONGTABLE_KEYS = {"loadings", "descriptives", "stats"}

# `stats_table`'s row order carries no meaning (it is sorted by p_adj) once
# the (a, b) model-pair columns already identify each row, so its DataFrame
# index is a bare `reset_index(drop=True)` RangeIndex - rendering it produced
# an unlabelled leading column of sequential integers. Written with
# `index=False` instead of trying to invent a meaningful label for it.
NO_INDEX_KEYS = {"stats"}


def _bold_at_or_above(threshold: float):
    """A float formatter (not a format string) that wraps the value in
    \\textbf{} when |v| >= threshold. Used for `loadings`' F1..Fk columns
    (I3) so a reader with only the table - grayscale print, supplementary
    material - can still see the simple structure Figure 4 makes visible.
    The returned markup is trusted (we generated it, not escape_latex'd
    user content) and bypasses escaping, exactly like every other numeric
    cell in this module.
    """
    def fmt(v):
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if abs(v) >= threshold else s
    return fmt


_FACTOR_COL = re.compile(r"^F\d+$")


def _loadings_float_format(df: pd.DataFrame, threshold: float = 0.40) -> dict:
    return {c: _bold_at_or_above(threshold) for c in df.columns if _FACTOR_COL.match(c)}


# key -> float_format for write_table, resolved against the actual DataFrame
# at write_all time. A plain dict is used as-is; a callable receives the
# DataFrame (so, e.g., "loadings" can bold whichever F1..Fk columns are
# actually present) and must return a dict. Keys absent here keep
# write_table's global "{:.2f}" default.
FLOAT_FORMATS = {
    # C2: congruence values (0.9956 vs 0.9983) were both printing as "1.00"
    # at the module default of 2dp, collapsing the distinction the column
    # exists to convey.
    "sensitivity": {"vs Spearman": "{:.3f}", "vs polychoric": "{:.3f}"},
    "factor_reliability": {"bootstrap_congruence": "{:.3f}", "split_half_congruence": "{:.3f}"},
    "loadings": _loadings_float_format,
}


def escape_latex(s) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in str(s))


# A plain 'l' column never wraps in LaTeX, so one long free-text cell (a
# sub-dimension label, a "target" description, ...) forces the WHOLE table
# past the page's textwidth - confirmed by compiling a real generated table
# with pdflatex, where an unwrapped "label" column produced a 127pt overfull
# \hbox. Any text column whose longest cell or header clears this threshold
# is instead given a `p{width}` paragraph column, sized from its content and
# capped so no single column can dominate the page.
# Widths were originally tuned against a 6.5in textwidth, 1in-margin US
# letter page - explicitly flagged in this module as "a representative but
# not necessarily final page width; Task 15/16 fixes the manuscript's real
# document class". Task 15 fixed the document class (sn-jnl, sn-basic),
# whose real \textwidth is 372pt (12.9cm) - 96.75pt narrower than the page
# these caps were tuned against, confirmed by compiling the real manuscript
# with pdflatex: tab_descriptives, tab_loadings, tab_factor_reliability and
# tab_convergence all overflowed (81.4/65.2/41.5/14.7pt respectively). Two
# changes close the gap: WRITE_TABLE_FONT drops from \small to \footnotesize
# (shrinks every natural-width l/r column, which is most of the deficit for
# tab_factor_reliability's un-wrappable "bootstrap_congruence"-style
# headers), and _TABCOLSEP shrinks the fixed per-column padding that scales
# with column count (a 9-column table like tab_loadings pays for 9 pairs of
# it). Re-verified clear of overfull warnings after this change.
_WRAP_CHAR_THRESHOLD = 18
_WRAP_MIN_CM = 2.2
# See the _WRAP_CHAR_THRESHOLD comment above: 4.0 (the prior tuning, for the
# wider placeholder page) left the real sn-basic page's tab_descriptives and
# tab_loadings longtables overfull even after the font/tabcolsep fix alone;
# capping the "label" column at 3.4cm instead closes the remainder with
# margin to spare, confirmed by compiling the real manuscript with pdflatex.
_WRAP_MAX_CM = 3.2
_WRAP_CM_PER_CHAR = 0.09

# Table-wide font and inter-column padding, applied by `write_table` to
# every generated table (see _WRAP_CHAR_THRESHOLD's comment for why: the
# real sn-basic \textwidth is 372pt, narrower than these were first tuned
# against). \small was the prior font; tabcolsep's LaTeX default is 6pt.
# 3pt (tried first) still left tab_descriptives 16.4pt and
# tab_factor_reliability 5.5pt overfull when every one of the nine real
# manuscript tables was compiled together against sn-basic's real 372pt
# textwidth; 2pt clears both with margin (confirmed the same way).
_TABLE_FONT = r"\footnotesize"
_TABCOLSEP = "2pt"


def _fmt_p(p: float) -> str:
    # Worded rather than "< 0.001": a literal '<' renders as an upside-down
    # punctuation mark under the default OT1 font encoding - confirmed by
    # compiling a real generated table with pdflatex.
    return "below 0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------
# Pure LaTeX serialisation
# --------------------------------------------------------------------------

def _column_formatter(float_format, col):
    """Resolve `float_format` (a format string, a {column: spec} dict, or a
    per-column callable) into a single callable for `col`.

    A dict entry - or the bare `float_format` itself - may be either a
    "{:.Nf}"-style format string or a callable `value -> str` (used by
    `loadings` to bold at-or-above-threshold cells with \\textbf{...}; see
    `_bold_at_or_above`). Columns absent from a dict fall back to the
    module's plain 2dp default, not whatever the dict's other columns use.
    """
    spec = float_format.get(col, "{:.2f}") if isinstance(float_format, dict) else float_format
    return spec if callable(spec) else (lambda v, spec=spec: spec.format(v))


def write_table(df: pd.DataFrame, path, caption: str, label: str,
                float_format="{:.2f}", longtable=False, index=True) -> Path:
    if df is None or df.empty:
        raise ValueError(f"refusing to write an empty table to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    body = df.copy()
    # Column alignment: numeric/boolean columns stay right-aligned. Text
    # columns are left-aligned UNLESS their longest cell or header clears
    # _WRAP_CHAR_THRESHOLD, in which case they become a `p{width}`
    # paragraph column (see _WRAP_* above) so long prose wraps instead of
    # pushing the whole table past the page.
    col_specs = ["l"] if index else []  # the index column, when rendered
    for col in body.columns:
        if pd.api.types.is_bool_dtype(body[col]):
            body[col] = body[col].map(lambda v: "Yes" if v else "No")
            col_specs.append("r")
        elif pd.api.types.is_numeric_dtype(body[col]):
            if pd.api.types.is_float_dtype(body[col]):
                fmt = _column_formatter(float_format, col)
                body[col] = body[col].map(lambda v: "" if pd.isna(v) else fmt(v))
            else:
                body[col] = body[col].map(lambda v: "" if pd.isna(v) else escape_latex(v))
            col_specs.append("r")
        else:
            body[col] = body[col].map(lambda v: "" if pd.isna(v) else escape_latex(v))
            longest = max([len(str(v)) for v in body[col]] + [len(str(col))])
            if longest > _WRAP_CHAR_THRESHOLD:
                width = min(_WRAP_MAX_CM, max(_WRAP_MIN_CM, longest * _WRAP_CM_PER_CHAR))
                col_specs.append(f"p{{{width:.1f}cm}}")
            else:
                col_specs.append("l")
    if index:
        body.index = [escape_latex(i) for i in body.index]
    body.columns = [escape_latex(c) for c in body.columns]

    text = body.to_latex(escape=False, index=index, longtable=longtable,
                         caption=caption, label=label, column_format="".join(col_specs))
    if longtable:
        # A longtable is only used for the manuscript's largest tables
        # (many rows and/or many columns); left at the document's normal
        # 10pt it was measured, by compiling a real table with pdflatex, to
        # overflow the page width even after wrapping the long text
        # column. Wrapped in a group so the font/tabcolsep change never
        # leaks past \end{longtable}.
        text = f"{{{_TABLE_FONT}\n\\setlength{{\\tabcolsep}}{{{_TABCOLSEP}}}\n{text}}}\n"
    else:
        # pandas emits a bare "\begin{table}\n\caption{...}"; add the
        # centring/font/tabcolsep wrapper the manuscript's other tables use.
        text = text.replace(
            "\\begin{table}\n",
            f"\\begin{{table}}[htbp]\n\\centering\n{_TABLE_FONT}\n"
            f"\\setlength{{\\tabcolsep}}{{{_TABCOLSEP}}}\n",
            1,
        )
    path.write_text(text, encoding="utf-8")
    return path


def write_all(context: dict, out_dir=None) -> dict:
    """Write only the tables whose data is present. Absent data is skipped, never faked.

    A caller may supply `context[f"{key}_caption"]` to override a table's
    static SPECS caption with one built from the actual data - used for
    "sensitivity" (the three correlation bases' simple-structure counts, C1)
    and "convergence" (Kendall's W and p, C3), neither of which is a fixed
    string known ahead of the real analysis run.
    """
    out_dir = Path(out_dir or config.TABLES)
    written = {}
    for key, (default_caption, label, filename) in SPECS.items():
        df = context.get(key)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        caption = context.get(f"{key}_caption", default_caption)
        float_format = FLOAT_FORMATS.get(key, "{:.2f}")
        if callable(float_format):
            float_format = float_format(df)
        written[key] = write_table(df, out_dir / filename, caption, label,
                                   float_format=float_format,
                                   longtable=(key in LONGTABLE_KEYS),
                                   index=(key not in NO_INDEX_KEYS))
    return written


# --------------------------------------------------------------------------
# Table builders: analysis output -> presentation-ready DataFrame
# --------------------------------------------------------------------------

def factorability_table(f: dict) -> pd.DataFrame:
    """`factors.factorability(df)`'s dict -> a one-column summary table."""
    rows = [
        ("Kaiser-Meyer-Olkin (overall)", f"{f['kmo_overall']:.3f}"),
        ("Bartlett's chi-square", f"{f['bartlett_chi2']:.1f}"),
        ("Bartlett's df", f"{int(f['bartlett_df']):d}"),
        ("Bartlett's p", _fmt_p(f["bartlett_p"])),
        ("Sampling adequate (KMO at least .60, p below .05)",
         "Yes" if f["is_factorable"] else "No"),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"]).set_index("statistic")


def loadings_table(loadings: pd.DataFrame, communalities: pd.Series,
                   uniquenesses: pd.Series, threshold: float = 0.40) -> pd.DataFrame:
    """44 items x 5 factors -> one row per item, grouped by primary factor
    (unassigned items last) and sorted by loading within each group, with a
    readable sub-dimension label, communality and uniqueness alongside the
    F1..F5 loadings themselves.
    """
    from . import factors  # local import: keeps write_table/write_all import-light

    assignment = factors.assign_items(loadings, threshold=threshold)
    sort_key = pd.DataFrame({
        "factor": assignment["primary"].fillna("~unassigned"),
        "loading": assignment["primary_loading"],
    })
    order = sort_key.sort_values(["factor", "loading"], ascending=[True, False]).index

    out = loadings.loc[order].copy()
    out.insert(0, "label", [config.SUBDIMENSIONS.get(c, "") for c in order])
    out["communality"] = communalities.loc[order]
    out["uniqueness"] = uniquenesses.loc[order]
    return out


def factor_reliability_table(reliability: pd.DataFrame, bootstrap: pd.DataFrame,
                             split_half: pd.DataFrame) -> pd.DataFrame:
    """Merge `factors.reliability`, `.bootstrap_stability` and
    `.split_half_congruence` (each indexed by factor) into one table:
    item count, Cronbach's alpha, McDonald's omega, bootstrap congruence,
    split-half congruence.
    """
    out = reliability.copy()
    out["bootstrap_congruence"] = bootstrap["mean_congruence"]
    out["split_half_congruence"] = split_half["congruence"]
    return out[["n_items", "cronbach_alpha", "mcdonald_omega",
               "bootstrap_congruence", "split_half_congruence"]]


def sensitivity_table(sens: pd.DataFrame) -> pd.DataFrame:
    """`factors.sensitivity_by_correlation_basis`'s wide per-basis-count
    table -> congruence only (C1).

    The simple-structure counts (cross-loading / unassigned / low-
    communality item counts) that used to sit alongside these columns
    describe the WHOLE solution under one basis, not any individual factor -
    `sensitivity_by_correlation_basis` computes them once per basis and
    broadcasts the same triple onto every factor row, so keeping them as
    factor-row columns reads as "F1 has 5 cross-loaders, F2 has 5
    cross-loaders, ..." when it means "the Pearson solution has 5 in total".
    They belong in the caption instead - see `sensitivity_caption`, which
    also surfaces the Spearman and polychoric counts this table previously
    dropped entirely.
    """
    return sens.rename(columns={
        "vs_spearman": "vs Spearman",
        "vs_polychoric": "vs polychoric",
    })[["vs Spearman", "vs polychoric"]]


def sensitivity_caption(sens: pd.DataFrame) -> str:
    """Build tab_sensitivity's caption from the actual data: the three
    correlation bases' simple-structure counts (cross-loading / unassigned /
    low-communality items, out of 44), read from `sens` rather than
    hardcoded, since they are identical across every row by construction
    (see `sensitivity_table`'s docstring) - row 0 carries them for all three
    bases.
    """
    row = sens.iloc[0]
    parts = []
    for basis in ("pearson", "spearman", "polychoric"):
        c = int(row[f"{basis}_cross_loading"])
        u = int(row[f"{basis}_unassigned"])
        lc = int(row[f"{basis}_low_communality"])
        parts.append(f"{basis} {c} / {u} / {lc}")
    return ("Sensitivity of the solution to correlation basis: Tucker congruence "
           "(3 decimal places) of the shipped Pearson solution against Spearman- "
           "and polychoric-basis extractions, by factor. Simple-structure counts "
           "(cross-loading / unassigned / low-communality items, of 44) under "
           "each basis: " + "; ".join(parts) + ".")


def descriptives_table(sub: pd.DataFrame) -> pd.DataFrame:
    """`descriptives.subdimension_table`'s output, column order tidied for
    the manuscript (label first, zero_rate renamed for readability)."""
    out = sub[["label", "dimension", "mean", "sd", "median", "iqr", "zero_rate"]].copy()
    return out.rename(columns={"zero_rate": "share coded 0"})


def model_spec_table() -> pd.DataFrame:
    """Model / task / target / metric specification (R2.6): every model
    in `benchmark.make_estimators` does the SAME task on the SAME target with
    the SAME metrics - regression and classification are never mixed. Every
    field is read from the actual pipeline code, not typed by hand.
    """
    estimators = benchmark.make_estimators()
    rows = []
    for name in estimators:
        rows.append({
            "model": name,
            "task": "Regression",
            "target": "Observed coded sub-dimension score (0-5)",
            "metric": "MAE, RMSE, R2, Spearman rho",
            "tuned": "Yes" if name in benchmark.PARAM_GRIDS else "No",
        })
    return pd.DataFrame(rows).set_index("model")


def benchmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    """`benchmark.summarise`'s output, folding the BCa interval into one
    "95% CI" column for compact presentation.
    """
    out = pd.DataFrame(index=summary.index)
    out["mae"] = summary["mae"]
    out["95% CI"] = [f"[{lo:.2f}, {hi:.2f}]"
                     for lo, hi in zip(summary["mae_lo"], summary["mae_hi"])]
    out["rmse"] = summary["rmse"]
    out["r2"] = summary["r2"]
    out["spearman"] = summary["spearman"]
    return out


def stats_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank comparison of every pair of models'
    per-target mean MAE (from `benchmark.run_sweep`'s fold-level output),
    with Benjamini-Hochberg adjusted p across all pairs. p-values are
    formatted as "below 0.001" below that threshold rather than as a bare
    2-decimal float, which would otherwise misleadingly print "0.00" for a
    value like 1.1e-13.
    """
    matrix = results.groupby(["target", "model"])["mae"].mean().unstack()
    labels = list(matrix.columns)
    pairs = stats.wilcoxon_pairs(matrix.to_numpy(), labels=labels)
    pairs["p_adj"] = stats.bh_fdr(pairs["p_value"].to_numpy())
    pairs = pairs.sort_values("p_adj").reset_index(drop=True)
    pairs["p_value"] = pairs["p_value"].map(_fmt_p)
    pairs["p_adj"] = pairs["p_adj"].map(_fmt_p)
    return pairs


# --------------------------------------------------------------------------
# convergence_table / convergence_caption (C3)
# --------------------------------------------------------------------------

def convergence_table(table: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """`importance.convergence_table(df)`'s output -> the top-N sub-dimensions
    by `mean_rank` (most convergently important first - the input is already
    sorted ascending by `mean_rank`, so this is a `head`), with a readable
    label column alongside the four `importance.METHODS` scores.
    """
    from . import importance  # local import: keeps write_table/write_all import-light

    top = table.head(top_n)[importance.METHODS + ["mean_rank"]].copy()
    top.insert(0, "label", [config.SUBDIMENSIONS.get(c, "") for c in top.index])
    return top


def convergence_caption(full_table: pd.DataFrame) -> str:
    """Build tab_convergence's caption, including Kendall's W and p, read
    from `importance.concordance(full_table)` rather than hardcoded - `W`
    and `p` are properties of the actual run's data, not fixed text.

    `full_table` is the FULL `importance.convergence_table(df)` output (not
    the top-N slice `convergence_table` above writes), since concordance is
    computed across every ranked sub-dimension.

    Says "internal consistency check", never "validation" (R1.1) - the same
    rule `tests/test_importance.py::test_no_output_label_claims_validation`
    enforces on `analysis/importance.py` itself.
    """
    from . import importance  # local import: keeps write_table/write_all import-light

    w, p = importance.concordance(full_table)
    return (f"Convergence of importance methods (internal consistency check): "
           f"Kendall's W = {w:.3f} across the four ranking methods "
           f"(gain, permutation, SHAP, network centrality), "
           f"p {_fmt_p(p) if p < 0.001 else '= ' + _fmt_p(p)}.")


# --------------------------------------------------------------------------
# external_table / external_caption (Task 18, R1.2: Study 2)
# --------------------------------------------------------------------------

def external_table(coverage: dict, correspondence: dict) -> pd.DataFrame:
    """`external.coverage_report` and `external.correspondence_test`'s dicts
    -> a one-column summary table for Study 2: firm-level ESG coverage
    (LSEG extract, 500 Indian listed firms) and the governance-disclosure
    correspondence check against the literature measurement model.

    A correspondence check between two independently derived measurement
    structures, never validation of the literature model's substantive
    claims (see `analysis/external.py`'s module docstring) - the caption
    (`external_caption`) states this explicitly using the actual numbers,
    not just this table's row labels.
    """
    per_ind = coverage["per_indicator_coverage"]
    lo_name = min(per_ind, key=per_ind.get)
    hi_name = max(per_ind, key=per_ind.get)

    rows = [
        ("Firms with ESG data (of 500)", f"{coverage['n_covered']}"),
        ("Firms without ESG data", f"{coverage['n_uncovered']}"),
        ("Median market cap, firms with ESG data (USD bn)",
         f"{coverage['median_marketcap_covered'] / 1e9:.2f}"),
        ("Median market cap, firms without ESG data (USD bn)",
         f"{coverage['median_marketcap_uncovered'] / 1e9:.2f}"),
        ("Coverage vs. size, Mann-Whitney p", _fmt_p(coverage["marketcap_mannwhitney_p"])),
        ("Coverage vs. market-cap rank, Spearman rho",
         f"{coverage['spearman_rank_vs_n_indicators']:.3f}"),
        ("Coverage vs. market-cap rank, p", _fmt_p(coverage["spearman_rank_vs_n_indicators_p"])),
        (f"Lowest-coverage indicator: {lo_name}", f"{per_ind[lo_name] * 100:.1f}%"),
        (f"Highest-coverage indicator: {hi_name}", f"{per_ind[hi_name] * 100:.1f}%"),
        ("Firms in correspondence sample", f"{correspondence['n']}"),
        ("Governance-disclosure partial correlation",
         f"{correspondence['partial_correlation']:.3f}"),
        ("95% bootstrap CI",
         f"[{correspondence['ci_lo']:.3f}, {correspondence['ci_hi']:.3f}]"),
        ("Literature model comparator", f"{correspondence['literature_model_value']:.3f}"),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"]).set_index("statistic")


def external_caption(correspondence: dict) -> str:
    """Build tab_external's caption from the actual correspondence result
    (n, partial correlation, bootstrap CI, literature comparator) rather
    than a fixed string, matching `sensitivity_caption` /
    `convergence_caption`'s pattern. States the correspondence-not-
    validation framing explicitly, in the caption itself rather than only
    in a limitations paragraph elsewhere.
    """
    r = correspondence["partial_correlation"]
    lo, hi = correspondence["ci_lo"], correspondence["ci_hi"]
    n = correspondence["n"]
    lit = correspondence["literature_model_value"]
    return (
        f"Study 2, firm-level correspondence check (n = {n} well-disclosing firms): "
        f"partial correlation between the governance index and disclosure "
        f"completeness, controlling for firm size, ROA, leverage and sector, "
        f"is {r:.3f} (95\\% bootstrap CI [{lo:.3f}, {hi:.3f}]), against the "
        f"literature model's F1-F2 (governance-disclosure) factor correlation of "
        f"{lit:.3f}. Reported as a correspondence check between two "
        f"independently derived measurement structures, never as validation of "
        f"the literature model's substantive claims."
    )
