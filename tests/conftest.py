"""Collection guards for running this package without its data.

Most test modules load the coded matrix (`Datasheet.xlsx`) at import time, so
they cannot be collected when it is absent. The matrix is supplied as
supplementary material with the manuscript rather than in this repository, and
the firm-level LSEG extract is licensed and not redistributable (see README).

A third group of tests checks the manuscript and response-letter sources
themselves. Those live with the paper, not in this repository, so they are
skipped here too.

Rather than erroring out, collection skips the modules that need data the
checkout does not have, and reports why. Everything that can run, runs -
`pytest` on a bare clone still exercises the statistics toolkit, the schema, the
neural estimator contracts, the reliability estimators and the PRISMA figure
logic.
"""
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DATASHEET = _ROOT / "Datasheet.xlsx"
_FIRM_DATA = _ROOT / "data"
_MANUSCRIPT = _ROOT / "manuscript"

_NEEDS_DATASHEET = [
    "test_benchmark.py",
    "test_clustering.py",
    "test_descriptives.py",
    "test_export.py",
    "test_factors.py",
    "test_figures.py",
    "test_importance.py",
    "test_loading.py",
    "test_models_nn.py",
    "test_quality.py",
    "test_reliability.py",
    "test_run_all.py",
    "test_sampling.py",
]

_NEEDS_FIRM_DATA = ["test_external.py"]

# These read manuscript/sections/*.tex, manuscript/main.tex and
# response/response_items.py, none of which ship in this repository.
_NEEDS_MANUSCRIPT = [
    "test_manuscript_build.py",
    "test_manuscript_claims.py",
    "test_response.py",
]

collect_ignore = []
if not _DATASHEET.exists():
    collect_ignore += _NEEDS_DATASHEET
if not _FIRM_DATA.is_dir():
    collect_ignore += _NEEDS_FIRM_DATA
if not _MANUSCRIPT.is_dir():
    collect_ignore += _NEEDS_MANUSCRIPT


def pytest_report_header(config):
    lines = []
    if not _DATASHEET.exists():
        lines.append(
            f"Datasheet.xlsx not found at {_ROOT} - skipping "
            f"{len(_NEEDS_DATASHEET)} data-dependent modules (see README)"
        )
    if not _FIRM_DATA.is_dir():
        lines.append(
            "data/ not found - skipping the firm-level external check "
            "(LSEG extract is licensed and not redistributable)"
        )
    if not _MANUSCRIPT.is_dir():
        lines.append(
            f"manuscript/ not found - skipping {len(_NEEDS_MANUSCRIPT)} modules "
            "that check the manuscript and response-letter sources"
        )
    return lines or None
