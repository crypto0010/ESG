"""Figure 8: Study 2, firm-level ESG coverage and the governance-disclosure
correspondence check (Task 18, R1.2's firm-level validation remedy).

Two panels on the same LSEG/Refinitiv extract:

- Left: coverage by sector -- the evidence that coverage is NOT missing at
  random (author audit, Ruling AF), which is why every Study 2 number is
  restricted to the covered firms rather than all 500.
- Right: the firm-level governance index against disclosure completeness
  among covered firms, with the partial correlation (controlling for size,
  ROA, leverage and sector) and its bootstrap interval set against the
  literature model's F1-F2 factor correlation.

Reported as a correspondence check between two independently derived
measurement structures, never as validation of the literature model's
substantive claims -- the literature model describes what the ESG
literature attends to, this figure describes what firms actually do and
disclose.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import external
from . import style


def build_axes(df: pd.DataFrame, coverage: dict, correspondence: dict):
    style.apply()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.6, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.2], "wspace": 0.42})

    # ---- Left: coverage by sector ----
    sector_cov = pd.Series(coverage["sector_coverage"]).sort_values()
    y = np.arange(len(sector_cov))
    ax1.barh(y, sector_cov.to_numpy() * 100, color=style.PALETTE[0])
    ax1.set_yticks(y)
    ax1.set_yticklabels([style.truncate_label(s, width=26) for s in sector_cov.index], fontsize=7.6)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Firms reporting at least 2 of 17 ESG indicators (%)", fontsize=8.5)
    n_total = coverage["n_covered"] + coverage["n_uncovered"]
    ax1.set_title(f"Coverage is not missing at random\n(n = {coverage['n_covered']} of {n_total} firms)",
                  fontsize=9)

    # ---- Right: governance index vs. disclosure completeness ----
    covered = df.loc[df["esg_covered"].astype(bool)] if "esg_covered" in df.columns else df
    gov = external.governance_index(covered)
    disc = external.disclosure_completeness(covered)
    mask = gov.notna() & disc.notna()
    ax2.scatter(disc[mask], gov[mask], s=16, alpha=0.5, color=style.PALETTE[0],
               edgecolor="none", zorder=2)

    if mask.sum() >= 3:
        coeffs = np.polyfit(disc[mask].to_numpy(dtype=float), gov[mask].to_numpy(dtype=float), 1)
        xs = np.linspace(disc[mask].min(), disc[mask].max(), 50)
        ax2.plot(xs, np.polyval(coeffs, xs), color=style.PALETTE[1], lw=1.4, zorder=3)

    ax2.set_xlabel("Disclosure completeness (fraction of 17 indicators)", fontsize=8.5)
    ax2.set_ylabel("Governance index (standardised)", fontsize=8.5)
    ax2.set_title("Governance-disclosure correspondence\n(controlling for size, ROA, leverage, sector)",
                  fontsize=9)

    r, lo, hi = correspondence["partial_correlation"], correspondence["ci_lo"], correspondence["ci_hi"]
    lit = correspondence["literature_model_value"]
    n = correspondence["n"]
    annotation = (f"partial r = {r:.2f}\n95% CI [{lo:.2f}, {hi:.2f}], n = {n}\n"
                 f"literature model: {lit:.2f}\n(correspondence, not validation)")
    ax2.text(0.03, 0.97, annotation, transform=ax2.transAxes, ha="left", va="top",
            fontsize=7.6, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                    edgecolor="#8a8d91", linewidth=0.7, alpha=0.92))

    fig.suptitle("Study 2: firm-level correspondence against the LSEG/Refinitiv extract",
                fontsize=9.5, style="italic", y=1.03)
    return fig, ax1, ax2


def render(out_path, df: pd.DataFrame, coverage: dict, correspondence: dict) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1, ax2 = build_axes(df, coverage, correspondence)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
