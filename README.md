# ESG literature-attention measurement model — analysis pipeline

Reproducible analysis code for a systematic review of 1,026 ESG research articles
(Scopus, 2020–2025), scored by two domain experts on a 0–5 scale across 44
sub-dimensions nested in 11 dimensions.

**Status:** the associated manuscript is under review. Results produced by this code
should be treated as under peer review, not as published findings.

## What this measures

The outcome variable is an expert-coded measure of **how prominently each ESG
sub-dimension is signalled in an article's title and abstract** — a literature-attention
construct. It is not a measure of any firm's ESG performance, and nothing in this
pipeline should be read as estimating firm-level ESG outcomes.

## What the pipeline does

| Module | Purpose |
|---|---|
| `loading`, `quality` | Read the coded matrix; audit and clean it (missing, out-of-range) |
| `descriptives` | Per-sub-dimension statistics, Cronbach's α, temporal trends |
| `factors` | Measurement model: factorability, parallel analysis, EFA with oblique rotation, reliability, bootstrap and split-half stability, polychoric robustness |
| `clustering` | Ward clustering, reported as a secondary analysis |
| `benchmark` | Held-out prediction of observed coded scores across nine estimators against a mean-predictor floor |
| `models_nn` | MLP, autoencoder-with-head and a dense GCN, in plain PyTorch |
| `importance` | Cross-method importance agreement (gain, permutation, SHAP, network centrality) |
| `stats` | BCa bootstrap, Friedman, Nemenyi, Wilcoxon, Benjamini–Hochberg FDR |
| `external` | Firm-level correspondence check against an LSEG extract |
| `figures/`, `export` | Publication figures (vector PDF, embedded fonts) and LaTeX tables |
| `run_all` | Regenerates every figure and table in one command |

## Two design constraints worth knowing

**No model output is ever a prediction target.** `benchmark.build_xy` raises if handed
anything that is not an observed column of the coded matrix. An earlier version of this
work used one model's feature importances as the reference target for the others; that
design is not expressible in this code.

**Cross-method importance agreement is an internal consistency check, not validation.**
`importance` reports whether four ways of ranking importance agree with each other. It
says nothing about whether the ranked drivers are real, and the module's own tests
enforce that its labels never claim otherwise.

## Data not included

- **The coded matrix** (`Datasheet.xlsx`, 1,026 × 44) is supplied as supplementary
  material with the manuscript. Place it at the repository root to run the pipeline.
- **Firm-level ESG and financial data** were obtained from LSEG/Refinitiv under
  institutional licence and cannot be redistributed. `analysis/external.py` documents
  the extract specification; any licensed user can reproduce it. Without a `data/`
  directory, `run_all` reports the external stage as skipped rather than failing.

## Running it

```bash
pip install numpy pandas scipy scikit-learn statsmodels networkx torch \
            xgboost lightgbm catboost shap pingouin scikit-posthocs \
            factor-analyzer matplotlib openpyxl pytest

python -m pytest tests/ -q      # full suite
python -m analysis.run_all      # regenerate every figure and table
python -m analysis.run_all --quick   # fast path for development
```

The full run trains nine estimators under nested cross-validation across 44 targets and
takes roughly 35 minutes. Everything is seeded (`config.SEED = 20260831`) and
deterministic.

## Tests

The suite covers behaviour rather than implementation. Several tests exist specifically
to stop known failure modes recurring: that the benchmark cannot accept a model output as
a target; that a fitted GCN's adjacency is not the identity; that figure text stays inside
its box when measured on a drawn canvas; that BCa intervals achieve near-nominal empirical
coverage; and that the response letter's figure numbers match the manuscript's compiled
figure order.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The coded matrix and any
third-party data are not covered by it.
