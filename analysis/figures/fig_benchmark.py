"""Figure 6: model benchmark with bootstrap intervals and the dummy floor
(unchanged from the Task 13 brief)."""
from pathlib import Path

import matplotlib.pyplot as plt

from . import style


def bar_labels(summary) -> list:
    return [f"{m} (floor)" if m == "Dummy" else m for m in summary.index]


def render(out_path, summary) -> Path:
    style.apply()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s = summary.sort_values("mae")
    lo = (s["mae"] - s["mae_lo"]).clip(lower=0)
    hi = (s["mae_hi"] - s["mae"]).clip(lower=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = ["#8a8d91" if m == "Dummy" else style.PALETTE[0] for m in s.index]
    ax.bar(range(len(s)), s["mae"], yerr=[lo, hi], color=colours,
          capsize=3, error_kw={"lw": 0.9})
    if "Dummy" in s.index:
        ax.axhline(float(s.loc["Dummy", "mae"]), ls="--", lw=0.9, color="#8a8d91")
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(bar_labels(s), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MAE against observed coded score (95% BCa CI)")
    ax.set_title("Held-out predictive accuracy", fontsize=10)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
