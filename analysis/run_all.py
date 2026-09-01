"""One-command, reproducible pipeline: load -> audit -> clean -> descriptives
-> factor model -> benchmark -> importance -> clustering -> figures -> tables.

I4: before this module existed, `export.write_all()` and every
`fig_*.render()` had zero non-test callers - the committed tables and
figures were produced by scratchpad scripts that were never committed, and
`analysis/_outputs/` is gitignored, so a fresh checkout could not reproduce
a single artefact the manuscript cites. `main()` is that missing entry
point: every number and picture the manuscript uses is reachable by running
this one function.

Gated stages - inter-rater reliability (`analysis.reliability`, Task 12,
waits on input I2) and external validation (`analysis.external`, Task 18,
waits on input I1) - are each skipped, not failed, when their upstream
module is absent or its own `is_available()` says its gate input has not
arrived. `analysis.reliability` still does not exist (I2 has not arrived),
so that stage is still reported in `skipped`. `analysis.external` landed
once I1 (the LSEG/Refinitiv firm-level extract under `data/`) arrived: when
`data/` is present, Study 2's firm-level coverage report and
governance-disclosure correspondence check (against the literature model's
F1-F2 factor correlation - a correspondence check, never validation of the
literature model's substantive claims) run as part of this pipeline,
producing `tab_external` and `fig8_external.pdf`; when `data/` is absent,
external validation is skipped exactly like reliability.

Caching: the genuinely expensive stages (the EFA bootstrap/parallel-analysis
robustness checks, the polychoric-basis sensitivity check, the predictive
benchmark sweep, the SHAP-based convergent-importance table) are cached to
`analysis/_outputs/` - written after computing, read instead of recomputing
when present. The cache is a pure optimisation: a fresh checkout with no
cache at all still runs `main()` end to end, it is simply slower.

`quick=True` is for development: a handful of targets and estimators, few
CV folds, small bootstrap counts. A quick run NEVER reads or writes the
shared cache - reusing a low-fidelity quick result as if it were a real one
(silently, in a later full run) would be a correctness bug, not a
convenience, so quick runs always compute fresh.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import (benchmark, clustering, config, descriptives, export, factors,
              importance, loading, quality)
from .figures import (fig_architecture, fig_benchmark, fig_dendrogram,
                      fig_descriptives, fig_external, fig_loadings, fig_prisma,
                      fig_scree)

N_FACTORS = 5
K_RANGE = range(2, 16)


def _roots(out_root) -> dict:
    """Where figures/tables/cache go. `out_root=None` (the default, used by
    the real one-shot invocation) resolves to the actual repository paths
    (`config.FIGURES`, `config.TABLES`, `config.OUTPUTS`); any other
    `out_root` redirects the whole tree under it, which is how tests stay
    isolated from the committed manuscript.
    """
    root = Path(out_root) if out_root is not None else config.ROOT
    return {
        "figures": root / "figures",
        "tables": root / "manuscript" / "tables",
        "cache": root / "analysis" / "_outputs",
    }


def _cache_csv(cache_dir, name, compute, use_cache, index_col=0, write_index=True):
    path = cache_dir / f"{name}.csv"
    if use_cache and path.exists():
        return pd.read_csv(path, index_col=index_col)
    df = compute()
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=write_index)
    return df


def _cache_parallel_analysis(cache_dir, compute, use_cache) -> dict:
    path = cache_dir / "factors_parallel.npz"
    if use_cache and path.exists():
        z = np.load(path)
        return {"n_factors": int(z["n_factors"]), "eigenvalues": z["eigenvalues"],
               "threshold": z["threshold"]}
    pa = compute()
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, n_factors=np.asarray(pa["n_factors"]),
                 eigenvalues=pa["eigenvalues"], threshold=pa["threshold"])
    return pa


def _gated_stage(module_name: str, human_name: str, gate_check, skipped: list):
    """Try a gated upstream module (`analysis.reliability`, `analysis.external`).

    `analysis.reliability` still does not exist (Task 12 is pending on
    input I2), so the ImportError branch is its common case today.
    `analysis.external` (Task 18) has landed: once its own `is_available()`
    reports its data directory is present, the `gate_check` branch returns
    the module and the caller runs Study 2; when the data directory is
    absent, `gate_check` returns False and the stage is skipped exactly
    like an unimplemented module, matching each gated module's own
    `is_available(path)` contract either way.
    """
    try:
        module = __import__(f"analysis.{module_name}", fromlist=[module_name])
    except ImportError:
        skipped.append(f"{human_name}: analysis.{module_name} is not yet implemented")
        return None
    try:
        available = bool(gate_check(module))
    except Exception as exc:  # unknown future is_available() signature - skip, don't crash
        skipped.append(f"{human_name}: could not check availability ({exc})")
        return None
    if not available:
        skipped.append(f"{human_name}: analysis.{module_name} is implemented but its gate input has not arrived")
        return None
    return module


def main(quick=False, out_root=None) -> dict:
    roots = _roots(out_root)
    # A quick run is for development speed; it must never poison the shared
    # cache with low-fidelity results a later full run could silently reuse.
    use_cache = not quick
    skipped = []

    # ---- load -> audit -> clean ----
    raw = loading.load_scoring()
    quality.audit(raw)  # findings are disclosed in Methods; run here so an
                        # audit failure surfaces even though nothing
                        # downstream consumes the report object directly.
    df = quality.clean(raw)

    # ---- descriptives ----
    sub = descriptives.subdimension_table(df)
    trend = descriptives.yearly_trend(df)
    year_counts = descriptives.yearly_counts(df)
    # A hash of a deterministic, RNG-free computation - the same input data
    # and code always produce the same subdimension_table, so this is what
    # makes "two runs give the same checksum" a meaningful determinism check
    # rather than a tautology.
    descriptives_checksum = hashlib.sha256(sub.to_csv().encode("utf-8")).hexdigest()

    # ---- factor model ----
    n_iter = 25 if quick else 500
    n_boot = 15 if quick else 200
    fb = factors.factorability(df)
    pa = _cache_parallel_analysis(
        roots["cache"],
        lambda: factors.parallel_analysis(df, n_iter=n_iter, seed=config.SEED),
        use_cache,
    )
    efa = factors.fit_efa(df, n_factors=N_FACTORS)
    assignment = factors.assign_items(efa.loadings)
    reliability_tbl = factors.reliability(df, assignment)
    bootstrap = _cache_csv(
        roots["cache"], "factors_bootstrap",
        lambda: factors.bootstrap_stability(df, N_FACTORS, n_boot=n_boot, seed=config.SEED),
        use_cache,
    )
    splithalf = factors.split_half_congruence(df, N_FACTORS, seed=config.SEED)
    sensitivity = _cache_csv(
        roots["cache"], "factors_sensitivity",
        lambda: factors.sensitivity_by_correlation_basis(df, N_FACTORS),
        use_cache,
    )

    # ---- predictive benchmark ----
    if quick:
        full = benchmark.make_estimators()
        estimators = {k: full[k] for k in ("Dummy", "ElasticNet") if k in full}
        targets = list(config.SCORE_COLS[:3])
    else:
        estimators = None
        targets = None
    bench_results = _cache_csv(
        roots["cache"], "benchmark_results",
        lambda: benchmark.run_sweep(df, targets=targets, estimators=estimators,
                                    quick=quick, seed=config.SEED),
        use_cache, index_col=None, write_index=False,
    )
    bench_summary = benchmark.summarise(bench_results, n_boot=200 if quick else 10000)

    # ---- convergent importance ----
    convergence_full = _cache_csv(
        roots["cache"], "importance_convergence",
        lambda: importance.convergence_table(df),
        use_cache,
    )

    # ---- clustering (secondary analysis; closed-form, always fresh) ----
    scan = clustering.silhouette_scan(df, K_RANGE)

    # ---- gated stages ----
    _gated_stage("reliability", "reliability",
                lambda m: m.is_available(config.TEMPLATES / "IRR_double_coding.xlsx"),
                skipped)
    external_module = _gated_stage("external", "external validation",
                                   lambda m: m.is_available(),
                                   skipped)

    external_firm_df = external_coverage = external_correspondence = None
    if external_module is not None:
        # The literature F1-F2 (governance-disclosure) factor correlation is
        # already fitted above (`efa`) for the main pipeline - passed in
        # rather than paying to refit it a second time inside
        # `correspondence_test`.
        external_firm_df = external_module.load_firm_data()
        external_coverage = external_module.coverage_report(external_firm_df)
        external_correspondence = external_module.correspondence_test(
            external_firm_df,
            n_boot=100 if quick else 2000,
            literature_value=float(efa.factor_correlations.loc["F1", "F2"]),
        )

    # ---- figures ----
    figures_dir = roots["figures"]
    figures = {
        "prisma": fig_prisma.render(
            fig_prisma.load_counts(config.TEMPLATES / "prisma_counts.json"),
            figures_dir / "fig1_prisma.pdf"),
        "architecture": fig_architecture.render(figures_dir / "fig2_architecture.pdf"),
        "scree": fig_scree.render(figures_dir / "fig3_scree.pdf", pa),
        "loadings": fig_loadings.render(figures_dir / "fig4_loadings.pdf", efa.loadings),
        "descriptives": fig_descriptives.render(figures_dir / "fig5_descriptives.pdf", sub, trend,
                                                counts=year_counts),
        "benchmark": fig_benchmark.render(figures_dir / "fig6_benchmark.pdf", bench_summary),
        "dendrogram": fig_dendrogram.render(figures_dir / "fig7_dendrogram.pdf", df, scan),
    }
    if external_module is not None:
        figures["external"] = fig_external.render(
            figures_dir / "fig8_external.pdf",
            external_firm_df, external_coverage, external_correspondence)

    # ---- tables ----
    context = {
        "factorability": export.factorability_table(fb),
        "loadings": export.loadings_table(efa.loadings, efa.communalities, efa.uniquenesses),
        "factor_reliability": export.factor_reliability_table(reliability_tbl, bootstrap, splithalf),
        "sensitivity": export.sensitivity_table(sensitivity),
        "sensitivity_caption": export.sensitivity_caption(sensitivity),
        "descriptives": export.descriptives_table(sub),
        "model_spec": export.model_spec_table(),
        "benchmark": export.benchmark_table(bench_summary),
        "stats": export.stats_table(bench_results),
        "convergence": export.convergence_table(convergence_full),
        "convergence_caption": export.convergence_caption(convergence_full),
    }
    if external_module is not None:
        context["external"] = export.external_table(external_coverage, external_correspondence)
        context["external_caption"] = export.external_caption(external_correspondence)
    tables = export.write_all(context, out_dir=roots["tables"])

    return {
        "figures": figures,
        "tables": tables,
        "skipped": skipped,
        "descriptives_checksum": descriptives_checksum,
    }


def _cli():
    parser = argparse.ArgumentParser(
        description="Regenerate every manuscript figure and table from the real dataset.")
    parser.add_argument("--quick", action="store_true",
                        help="fast development run: few targets/estimators, few CV folds, "
                             "small bootstrap counts - never reads or writes the shared cache")
    args = parser.parse_args()

    result = main(quick=args.quick)
    printable = {
        "figures": {k: str(v) for k, v in result["figures"].items()},
        "tables": {k: str(v) for k, v in result["tables"].items()},
        "skipped": result["skipped"],
        "descriptives_checksum": result["descriptives_checksum"],
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    _cli()
