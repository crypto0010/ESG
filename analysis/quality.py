"""Data-quality audit and cleaning. Findings are disclosed in Methods (spec 2.4)."""
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class QualityReport:
    n_articles: int
    missing: list = field(default_factory=list)
    out_of_range: list = field(default_factory=list)
    duplicate_titles: list = field(default_factory=list)
    missing_metadata: list = field(default_factory=list)
    year_counts: dict = field(default_factory=dict)
    empty_columns: list = field(default_factory=list)

    @property
    def n_cells(self) -> int:
        return self.n_articles * len(config.SCORE_COLS)


def audit(df: pd.DataFrame) -> QualityReport:
    rep = QualityReport(n_articles=len(df))

    for _, row in df.iterrows():
        sr = int(row["sr_no"])
        for code in config.SCORE_COLS:
            v = row[code]
            if pd.isna(v):
                rep.missing.append((sr, code))
            elif v < config.SCALE_MIN or v > config.SCALE_MAX:
                rep.out_of_range.append((sr, code, float(v)))

    # Check metadata columns for null or whitespace-only values
    metadata_cols = ["title", "authors", "source_title", "abstract"]
    for _, row in df.iterrows():
        sr = int(row["sr_no"])
        for col in metadata_cols:
            val = row[col]
            # Catch both pd.isna() (for StringDtype <NA>) and empty strings (whitespace stripped at load time)
            if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                rep.missing_metadata.append((sr, col))

    titles = df["title"].str.strip().str.lower()
    rep.duplicate_titles = sorted(titles[titles.duplicated()].unique().tolist())
    rep.year_counts = df["year"].value_counts().sort_index().to_dict()
    rep.empty_columns = ["RESEARCH DOMAIN", "GEOGRAPHIC DOMAIN"]
    return rep


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-range values become missing, then all gaps take the column median.

    Median rather than mean because the scale is ordinal and the distributions
    are strongly zero-inflated (spec 2.4).
    """
    out = df.copy()
    for code in config.SCORE_COLS:
        col = out[code]
        col = col.where((col >= config.SCALE_MIN) & (col <= config.SCALE_MAX))
        out[code] = col.fillna(col.median())
    return out


def report_markdown(rep: QualityReport) -> str:
    pct = 100 * len(rep.missing) / rep.n_cells
    lines = [
        f"- Articles: {rep.n_articles}; coded cells: {rep.n_cells}",
        f"- Missing cells: {len(rep.missing)} ({pct:.2f}%) at "
        + ", ".join(f"SR.NO {s} {c}" for s, c in rep.missing),
        "- Out-of-range values: "
        + ", ".join(f"SR.NO {s} {c} = {v:g}" for s, c, v in rep.out_of_range),
        f"- Duplicate titles: {len(rep.duplicate_titles)}",
        "- Missing metadata: "
        + (", ".join(f"SR.NO {s} {c}" for s, c in rep.missing_metadata) if rep.missing_metadata else "none"),
        "- Empty metadata columns: " + ", ".join(rep.empty_columns),
        "- Year distribution: "
        + ", ".join(f"{y}: {n}" for y, n in sorted(rep.year_counts.items())),
    ]
    return "\n".join(lines)
