"""Read the SCORING sheet into a tidy frame. Header is two rows deep."""
import numpy as np
import pandas as pd

from . import config

_META = {
    0: "sr_no", 2: "year", 3: "authors", 4: "title", 7: "source_title", 8: "abstract",
}
# Columns 5 (RESEARCH DOMAIN) and 6 (GEOGRAPHIC DOMAIN) are empty for all rows
# and are dropped; column 7 is labelled DOCUMENT TYPE in the sheet but holds
# journal names, so it is renamed source_title (spec 2.4, AF5).
_FIRST_SCORE_COL = 9


def load_scoring(path=None) -> pd.DataFrame:
    path = path or config.DATASHEET
    raw = pd.read_excel(path, sheet_name="SCORING", header=None, engine="openpyxl")

    codes = [config.normalise_code(raw.iat[1, j])
             for j in range(_FIRST_SCORE_COL, _FIRST_SCORE_COL + 44)]
    if codes != config.SCORE_COLS:
        raise ValueError(f"unexpected score columns in datasheet: {codes}")

    body = raw.iloc[2:].reset_index(drop=True)
    body = body[body[0].notna()]

    out = pd.DataFrame({name: body[j].values for j, name in _META.items()})
    out["sr_no"] = out["sr_no"].astype(int)
    out["year"] = out["year"].astype(int)
    for name in ("authors", "title", "source_title", "abstract"):
        out[name] = out[name].astype("string").str.strip()

    for offset, code in enumerate(config.SCORE_COLS):
        out[code] = pd.to_numeric(body[_FIRST_SCORE_COL + offset], errors="coerce").astype(float)

    return out.reset_index(drop=True)
