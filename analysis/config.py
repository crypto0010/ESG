"""Schema constants and paths. Single source of truth for the 11x4 framework."""
from pathlib import Path

SEED = 20260831

ROOT = Path(__file__).resolve().parent.parent
DATASHEET = ROOT / "Datasheet.xlsx"
FIGURES = ROOT / "figures"
TABLES = ROOT / "manuscript" / "tables"
TEMPLATES = ROOT / "templates"
OUTPUTS = ROOT / "analysis" / "_outputs"
# Task 18 (I1): the LSEG/Refinitiv firm-level extract (analysis/external.py).
DATA = ROOT / "data"

DIMENSIONS = {
    "A": "Sustainability Reporting",
    "B": "ESG Ratings",
    "C": "Transparency",
    "D": "Risk Management Frameworks",
    "E": "Stakeholder Engagement",
    "F": "Regulatory Compliance",
    "G": "ESG Disclosure",
    "H": "Standardization and Benchmarking",
    "I": "Corporate Governance Practices",
    "J": "Technological Integration",
    "K": "Due Diligence",
}

# Labels transcribed verbatim from Supplementary Table A.1.
SUBDIMENSIONS = {
    "A-D1": "Reporting Comprehensiveness",
    "A-D2": "Alignment with international standards (GRI, SASB, etc.)",
    "A-D3": "Quantitative vs. Qualitative reporting approaches",
    "A-D4": "Reporting frequency and consistency",
    "B-D1": "Methodology of rating assessment",
    "B-D2": "Comparative analysis of different rating agencies",
    "B-D3": "Correlation between rating agencies",
    "B-D4": "Transparency of rating methodologies",
    "C-D1": "Disclosure depth and quality",
    "C-D2": "Stakeholder accessibility of information",
    "C-D3": "Consistency of reported information",
    "C-D4": "Independent verification mechanisms",
    "D-D1": "Comprehensiveness of risk identification",
    "D-D2": "Integration of ESG risks into core business strategy",
    "D-D3": "Proactive vs. Reactive Risk Management Approaches",
    "D-D4": "Quantification and mitigation strategies",
    "E-D1": "Engagement mechanisms and channels",
    "E-D2": "Depth of stakeholder consultation",
    "E-D3": "Impact of stakeholder feedback on corporate strategies",
    "E-D4": "Diversity and inclusivity of engagement processes",
    "F-D1": "Compliance breadth across jurisdictions",
    "F-D2": "Proactive vs. Reactive Compliance Approaches",
    "F-D3": "Cost of compliance",
    "F-D4": "Impact on organizational performance",
    "G-D1": "Comprehensiveness of disclosure",
    "G-D2": "Standardization of disclosure metrics",
    "G-D3": "Voluntary vs. Mandatory disclosures",
    "G-D4": "Financial and non-financial disclosure integration",
    "H-D1": "Alignment with international standards",
    "H-D2": "Cross-industry comparability",
    "H-D3": "Benchmarking methodologies",
    "H-D4": "Evolution of standardization frameworks",
    "I-D1": "Strategic ESG integration",
    "I-D2": "Leadership commitment",
    "I-D3": "Cultural transformation",
    "I-D4": "Innovation in ESG implementation",
    "J-D1": "Digital tools for ESG management",
    "J-D2": "Data collection and analysis technologies",
    "J-D3": "Blockchain and AI in ESG reporting",
    "J-D4": "Automation of ESG processes",
    "K-D1": "Depth of ESG due diligence processes",
    "K-D2": "Supply chain ESG assessment",
    "K-D3": "Third-party verification mechanisms",
    "K-D4": "Continuous improvement frameworks",
}

SCORE_COLS = list(SUBDIMENSIONS.keys())

SCALE_ANCHORS = {
    0: "Not addressed",
    1: "Minimal implementation",
    2: "Partial implementation",
    3: "Substantial implementation",
    4: "Comprehensive implementation",
    5: "Best-in-class/exemplary",
}
SCALE_MIN, SCALE_MAX = 0, 5


def normalise_code(raw: str) -> str:
    """Accept 'AD1', 'A-D1', ' a-d1 ' and return the canonical 'A-D1'."""
    s = str(raw).strip().upper().replace("-", "").replace(" ", "")
    if len(s) < 2:
        raise ValueError(f"unrecognised sub-dimension code: {raw!r}")
    return f"{s[0]}-{s[1:]}"


def dimension_of(code: str) -> str:
    return normalise_code(code)[0]
