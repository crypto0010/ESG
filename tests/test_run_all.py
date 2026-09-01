"""Tests for the one-command reproducible pipeline (I4).

Before this module existed, `export.write_all()` and every `fig_*.render()`
had zero non-test callers - the eight committed tables and six figures were
produced by uncommitted scratchpad scripts, and `analysis/_outputs/` is
gitignored, so a fresh checkout could not reproduce any artifact the
manuscript cites. `run_all.main()` is the single entry point that can.

These tests always pass `out_root=tmp_path` (or a subdirectory of it) so a
test run never writes into the real `figures/`, `manuscript/tables/` or
`analysis/_outputs/` - only the real, explicit, one-time invocation
(`python -m analysis.run_all`, run manually per the dispatch, not by the
test suite) writes there.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import run_all


# --------------------------------------------------------------------------
# Caching primitives, tested directly against cheap synthetic `compute`
# functions - NOT through the full pipeline, since the read-hit path
# ("cache expensive stages") can never be exercised by a quick=True run
# (quick never touches the cache, by design - see run_all.py's docstring)
# and quick=False runs the real 35-minute benchmark sweep, unusable in a
# unit test.
# --------------------------------------------------------------------------

def test_cache_csv_reads_back_an_existing_cache_without_recomputing(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])

    first = run_all._cache_csv(tmp_path, "demo", compute, use_cache=True)
    second = run_all._cache_csv(tmp_path, "demo", compute, use_cache=True)
    assert len(calls) == 1   # the second call read the cache file, it did not recompute
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "demo.csv").exists()


def test_cache_csv_with_use_cache_false_never_reads_or_writes(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return pd.DataFrame({"x": [1.0]}, index=["a"])

    run_all._cache_csv(tmp_path, "demo", compute, use_cache=False)
    run_all._cache_csv(tmp_path, "demo", compute, use_cache=False)
    assert len(calls) == 2                          # recomputed both times
    assert not (tmp_path / "demo.csv").exists()      # and never wrote a cache file either


def test_cache_csv_never_requires_the_cache_to_exist(tmp_path):
    """I4: 'never require the cache to exist' - a completely absent
    cache_dir must not raise, the first call must create it."""
    missing_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not missing_dir.exists()
    out = run_all._cache_csv(missing_dir, "demo", lambda: pd.DataFrame({"x": [1.0]}), use_cache=True)
    assert not out.empty
    assert (missing_dir / "demo.csv").exists()


def test_cache_parallel_analysis_round_trips_through_npz_without_recomputing(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"n_factors": 5, "eigenvalues": np.array([3.0, 2.0, 1.0]),
               "threshold": np.array([1.5, 1.4, 1.3])}

    first = run_all._cache_parallel_analysis(tmp_path, compute, use_cache=True)
    second = run_all._cache_parallel_analysis(tmp_path, compute, use_cache=True)
    assert len(calls) == 1
    assert second["n_factors"] == 5
    assert isinstance(second["n_factors"], int)
    assert np.allclose(second["eigenvalues"], first["eigenvalues"])
    assert np.allclose(second["threshold"], first["threshold"])


def test_quick_run_produces_non_empty_figures_and_tables(tmp_path):
    out = run_all.main(quick=True, out_root=tmp_path)
    assert out["figures"] and out["tables"]
    # Figures are always > 10KB (vector PDF with embedded fonts). Tables vary
    # a lot by legitimate row count - tab_factorability is a real, correct,
    # 5-row table at 399 bytes; tab_benchmark's quick 2-estimator subset is
    # 370. 200 bytes comfortably rules out an empty/near-empty file (a bare
    # \begin{table}...\end{table} with no rows) without penalising a
    # genuinely small table.
    for p in out["figures"].values():
        assert Path(p).exists() and Path(p).stat().st_size > 1000
    for p in out["tables"].values():
        assert Path(p).exists() and Path(p).stat().st_size > 200


def test_quick_run_writes_under_figures_and_manuscript_tables(tmp_path):
    """I4: 'must write every figure to figures/ and every table to
    manuscript/tables/' - checked against the actual paths returned, not
    just that files exist somewhere."""
    out = run_all.main(quick=True, out_root=tmp_path)
    for p in out["figures"].values():
        assert Path(p).parent == tmp_path / "figures"
    for p in out["tables"].values():
        assert Path(p).parent == tmp_path / "manuscript" / "tables"


def test_run_reports_skipped_gated_stages(tmp_path):
    """Reliability (Task 12, gated on input I2) still has no upstream module
    and must be disclosed in `skipped`, not silently absent. External
    validation (Task 18, gated on input I1) is no longer skipped now that
    I1 has arrived (`data/` is committed) - see
    `test_run_runs_external_validation_now_that_i1_has_arrived` below."""
    out = run_all.main(quick=True, out_root=tmp_path)
    assert "skipped" in out
    assert all(isinstance(s, str) for s in out["skipped"])
    joined = " ".join(out["skipped"]).lower()
    assert "reliability" in joined


def test_run_runs_external_validation_now_that_i1_has_arrived(tmp_path):
    """Task 18: `data/` (the LSEG/Refinitiv extract) is present in this
    repository, so `analysis.external.is_available()` is True and Study 2
    must actually run as part of the pipeline, not be skipped."""
    out = run_all.main(quick=True, out_root=tmp_path)
    joined = " ".join(out["skipped"]).lower()
    assert "external" not in joined
    assert "external" in out["tables"]
    assert "external" in out["figures"]
    assert Path(out["tables"]["external"]).exists()
    assert Path(out["figures"]["external"]).exists()


def test_run_does_not_fail_when_gated_stages_are_unavailable(tmp_path):
    """The whole point of C1/I4's 'append to skipped rather than failing':
    a missing analysis.reliability / analysis.external module must not
    raise, it must be recorded and the run must still complete."""
    out = run_all.main(quick=True, out_root=tmp_path)
    assert out["tables"]   # the run reached the table-writing stage at all


def test_run_is_deterministic_under_the_global_seed(tmp_path):
    a = run_all.main(quick=True, out_root=tmp_path / "a")
    b = run_all.main(quick=True, out_root=tmp_path / "b")
    assert a["descriptives_checksum"] == b["descriptives_checksum"]


def test_full_specs_table_set_is_reachable_in_the_returned_dict_keys(tmp_path):
    """Not every SPECS table needs data in a quick run, but every table that
    IS produced must be one export.SPECS actually knows about - run_all must
    never invent an unrequested table key."""
    from analysis import export
    out = run_all.main(quick=True, out_root=tmp_path)
    assert set(out["tables"]) <= set(export.SPECS)


def test_convergence_and_loadings_tables_are_always_produced_even_when_quick(tmp_path):
    """These two are the fix targets of C3 and I3 - a quick run must still
    exercise them, not just the cheapest tables."""
    out = run_all.main(quick=True, out_root=tmp_path)
    assert "convergence" in out["tables"]
    assert "loadings" in out["tables"]


def test_cache_is_not_required_to_exist(tmp_path):
    """A fresh out_root has no analysis/_outputs cache at all - the run must
    not assume one is there."""
    cache_dir = tmp_path / "analysis" / "_outputs"
    assert not cache_dir.exists()
    out = run_all.main(quick=True, out_root=tmp_path)
    assert out["tables"]


def test_quick_run_never_touches_the_real_repository_output_directories(tmp_path):
    """Regression guard: `out_root` must actually redirect every write - a
    bug that silently fell back to `config.FIGURES`/`config.TABLES` would
    pollute the committed manuscript on every test run."""
    from analysis import config
    before_figs = set(config.FIGURES.glob("*")) if config.FIGURES.exists() else set()
    before_tabs = set(config.TABLES.glob("*")) if config.TABLES.exists() else set()
    run_all.main(quick=True, out_root=tmp_path)
    after_figs = set(config.FIGURES.glob("*")) if config.FIGURES.exists() else set()
    after_tabs = set(config.TABLES.glob("*")) if config.TABLES.exists() else set()
    assert before_figs == after_figs
    assert before_tabs == after_tabs
