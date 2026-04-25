"""
ASEAN Policy Dashboard — Streamlit Application

Clean, professional, interactive dashboard for ASEAN socioeconomic analysis.
Provides descriptive, predictive, and prescriptive analytics with contextual
explanations accessible to both technical and non-technical audiences.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np
import pandas as pd
import streamlit as st

# ── Path Setup ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "panel" / "Final_Panel_Dataset_with_DTM_ETM.csv"
MODELING_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "panel_cleaned.csv"
CORRELATION_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "correlation_matrix.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

RECOMMENDATION_RANKED_PATH = RESULTS_DIR / "policy_recommendations_ranked.csv"
RECOMMENDATION_EVIDENCE_PATH = RESULTS_DIR / "policy_recommendation_evidence.csv"
SCENARIO_RESULTS_PATH = RESULTS_DIR / "scenario_results.csv"
SCENARIO_SUMMARY_PATH = RESULTS_DIR / "scenario_summary.csv"

TARGET_LABELS: Dict[str, str] = {
    "gdp_per_capita": "GDP per Capita",
    "log_gdp_per_capita": "Log GDP per Capita",
    "gini_index": "Gini Index",
    "life_expectancy": "Life Expectancy",
    "infant_mortality": "Infant Mortality",
}

METRIC_GLOSSARY: Dict[str, str] = {
    "RMSE": "Root Mean Squared Error — average prediction error in the same units as the target. Lower is better.",
    "MAE": "Mean Absolute Error — average of absolute differences between predictions and actuals. Lower is better.",
    "R\u00b2": "Coefficient of Determination — proportion of variance explained by the model (0\u20131). Higher is better.",
    "MAPE": "Mean Absolute Percentage Error — average percentage off from actual values. Lower is better.",
    "Gini Index": "A measure of income inequality (0 = perfect equality, 100 = maximum inequality).",
    "VIF": "Variance Inflation Factor — detects multicollinearity among predictors. Values above 10 suggest concern.",
}

USER_STORIES: List[Dict[str, str]] = [
    {
        "id": "Story 1",
        "type": "Descriptive",
        "title": "Country-Level Socioeconomic Comparison for ASEAN Progress",
        "persona": "ASEAN Economic Community Department",
        "goal": "Summarize inequality and key socioeconomic indicators by country to compare trends in support of ASEAN goals.",
        "description": "As a member of ASEAN’s Economic Community Department, I want to summarize inequality and key socioeconomic indicators by country to compare trends in order to support ASEAN’s goals.",
    },
    {
        "id": "Story 2",
        "type": "Descriptive",
        "title": "Trade-Inequality Correlation Monitoring",
        "persona": "ASEAN Trade Facilitation Division",
        "goal": "Observe how inequality moves with trade openness so policy adjustments can be prioritized.",
        "description": "As a member of ASEAN’s Trade Facilitation Division, I want to observe how the Gini index is correlated with international trade so that we can identify what policies need to be modified or implemented.",
    },
    {
        "id": "Story 3",
        "type": "Predictive",
        "title": "Predict GDP Growth Competitiveness Signals",
        "persona": "ASEAN country government",
        "goal": "Predict GDP growth pathways to improve competitiveness against larger economies.",
        "description": "As a member of the government of one of ASEAN’s member countries, I want to predict how to grow GDP so that our country can grow economically and compete alongside larger countries.",
    },
    {
        "id": "Story 4",
        "type": "Predictive",
        "title": "Predictive Reform Impact Signals",
        "persona": "ASEAN country economic planner",
        "goal": "Simulate how reforms in trade, governance proxies, and demographics affect GDP and inequality outcomes.",
        "description": "As an economic planner in an ASEAN member country, I want to simulate how changes in trade openness, governance, or demographic indicators would affect inequality and GDP outcomes so that we can implement data-driven reforms with economic impact.",
    },
    {
        "id": "Story 5",
        "type": "Prescriptive",
        "title": "Country Inequality Reduction for Sustainable Growth",
        "persona": "Government member of an ASEAN country",
        "goal": "Identify policies to reduce inequality while sustaining economic development.",
        "description": "As a member of the government for one of ASEAN’s member countries, I want to identify policies to reduce inequality so that we can continue to develop economically.",
    },
    {
        "id": "Story 6",
        "type": "Prescriptive",
        "title": "Ranked Policy Recommendations",
        "persona": "ASEAN Economic Research policy advisor",
        "goal": "Prioritize high-impact reforms using modeled relationships and current risk conditions.",
        "description": "As a policy advisor within ASEAN’s Economic Research Institute, I want to receive ranked policy recommendations based on modeled relationships between inequality, trade, governance, and demographic stage so that we can prioritize the most impactful policy reforms for member countries.",
    },
]

# ── Styling ─────────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#1B2A4A",
    "secondary": "#2E86AB",
    "accent": "#E8913A",
    "bg": "#F5F7FA",
    "card": "#FFFFFF",
    "text": "#2D3748",
    "muted": "#718096",
    "success": "#38A169",
    "warning": "#D69E2E",
    "danger": "#E53E3E",
    "border": "#E2E8F0",
}

CUSTOM_CSS = f"""
<style>
/* == Global == */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

.main .block-container {{
    max-width: 1420px;
    padding-top: 1.1rem;
    padding-bottom: 2.2rem;
}}

h2 {{
    letter-spacing: -0.01em;
    margin-bottom: 0.45rem;
}}

h3 {{
    margin-top: 1.1rem;
    margin-bottom: 0.4rem;
}}

h4, h5 {{
    margin-bottom: 0.3rem;
}}

/* == Sidebar (always dark, self-contained) == */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {COLORS["primary"]} 0%, #0f1d36 100%);
}}
section[data-testid="stSidebar"] * {{
    color: #CBD5E0 !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: #FFFFFF !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12);
}}
section[data-testid="stSidebar"] .stMarkdown a {{
    color: #63B3ED !important;
}}

/* == Metric Cards (theme-aware) == */
[data-testid="stMetric"] {{
    background: var(--secondary-background-color);
    padding: 0.9rem 1rem;
    border-radius: 8px;
    border-left: 4px solid {COLORS["secondary"]};
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    min-height: 6.6rem;
    align-items: flex-start !important;
}}
[data-testid="stMetric"] > div {{
    width: 100%;
}}
[data-testid="stMetric"] label {{
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    line-height: 1.25;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-weight: 700;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    line-height: 1.25;
}}
[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    overflow-wrap: anywhere;
    word-break: break-word;
    line-height: 1.25;
}}

/* == Tabs == */
.stTabs [data-baseweb="tab-list"] {{
    gap: 2px;
    background: var(--secondary-background-color);
    border-radius: 8px 8px 0 0;
    padding: 4px 4px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px 6px 0 0;
    padding: 10px 18px;
    font-weight: 500;
    font-size: 0.85rem;
}}
.stTabs [aria-selected="true"] {{
    background: var(--background-color) !important;
    font-weight: 600;
}}

/* == DataFrames == */
[data-testid="stDataFrame"] {{
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}

/* == Expanders == */
.streamlit-expanderHeader {{
    font-weight: 600;
    font-size: 0.9rem;
}}

/* == Dashboard Header (always dark, self-contained) == */
.dash-header {{
    background: linear-gradient(135deg, {COLORS["primary"]} 0%, #16213e 60%, {COLORS["secondary"]} 100%);
    color: white;
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(27,42,74,0.25);
}}
.dash-header h1 {{
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: white !important;
}}
.dash-header p {{
    margin: 0.5rem 0 0;
    color: #a0b4c8;
    font-size: 0.95rem;
}}

/* == Section Intro (theme-aware) == */
.section-intro {{
    color: var(--text-color);
    opacity: 0.8;
    font-size: 0.92rem;
    line-height: 1.65;
    padding: 0.8rem 1rem;
    background: var(--secondary-background-color);
    border-left: 3px solid {COLORS["secondary"]};
    border-radius: 0 6px 6px 0;
    margin-bottom: 1.2rem;
}}

/* == Insight Callout == */
.insight-callout {{
    background: var(--secondary-background-color);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-left: 4px solid {COLORS["secondary"]};
    border-radius: 8px;
    padding: 0.7rem 0.95rem;
    margin: 0.45rem 0 1rem;
}}

.insight-callout .insight-title {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    opacity: 0.75;
    margin-bottom: 0.25rem;
}}

.insight-callout .insight-text {{
    font-size: 0.9rem;
    line-height: 1.45;
    color: var(--text-color);
}}

.insight-neutral {{ border-left-color: {COLORS["secondary"]}; }}
.insight-positive {{ border-left-color: {COLORS["success"]}; }}
.insight-risk {{ border-left-color: {COLORS["danger"]}; }}

/* == KPI Grid (custom cards for predictive analytics) == */
.kpi-grid {{
    display: grid;
    grid-template-columns: minmax(260px, 1.8fr) repeat(3, minmax(120px, 1fr));
    gap: 0.75rem;
    margin: 0.5rem 0 1.5rem;
}}
.kpi-card {{
    background: var(--secondary-background-color);
    padding: 0.8rem 0.95rem;
    border-radius: 8px;
    border-bottom: 3px solid transparent;
}}
.kpi-card-highlight {{
    border-bottom-color: {COLORS["secondary"]};
}}
.kpi-label {{
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-color);
    opacity: 0.55;
    margin-bottom: 0.35rem;
}}
.kpi-value {{
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text-color);
    line-height: 1.3;
    white-space: normal;
    word-wrap: break-word;
    overflow-wrap: break-word;
    overflow: visible;
    text-overflow: clip;
}}
.kpi-value-sm {{
    font-size: 1.0rem;
}}

@media (max-width: 1100px) {{
    .kpi-grid {{
        grid-template-columns: repeat(2, minmax(160px, 1fr));
    }}
}}

@media (max-width: 680px) {{
    .kpi-grid {{
        grid-template-columns: 1fr;
    }}
}}

/* == Story Cards (theme-aware) == */
.story-card {{
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-left: 4px solid {COLORS["accent"]};
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 1.05rem;
}}
.story-card-top {{
    display: flex;
    gap: 0.45rem;
    align-items: center;
    margin-bottom: 0.55rem;
    flex-wrap: wrap;
}}
.story-id {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(148, 163, 184, 0.16);
    color: var(--text-color);
}}
.story-card .story-type {{
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
}}
.story-type-descriptive {{ background: rgba(56,178,172,0.15); color: {COLORS["secondary"]}; }}
.story-type-predictive {{ background: rgba(49,130,206,0.15); color: #3182CE; }}
.story-type-prescriptive {{ background: rgba(229,62,62,0.15); color: {COLORS["danger"]}; }}

.story-card h4 {{ margin: 0.2rem 0 0.35rem; color: var(--text-color); font-size: 1.06rem; }}
.story-card .persona {{ font-size: 0.82rem; color: var(--text-color); opacity: 0.66; margin-bottom: 0.5rem; }}
.story-card .goal {{ font-size: 0.89rem; color: var(--text-color); opacity: 0.87; line-height: 1.5; }}

/* == Story Layout Blocks == */
.story-section {{
    margin-top: 0.55rem;
    margin-bottom: 0.35rem;
}}
.story-section .story-section-label {{
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-color);
    opacity: 0.62;
    margin-bottom: 0.15rem;
}}
.story-section .story-section-title {{
    font-size: 1rem;
    font-weight: 650;
    color: var(--text-color);
}}
.story-section .story-section-subtitle {{
    font-size: 0.86rem;
    color: var(--text-color);
    opacity: 0.72;
    margin-top: 0.14rem;
    margin-bottom: 0.3rem;
}}

.story-placeholder {{
    background: var(--secondary-background-color);
    border: 1px dashed rgba(148, 163, 184, 0.45);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 0.35rem;
    margin-bottom: 0.55rem;
}}
.story-placeholder strong {{
    font-size: 0.86rem;
    letter-spacing: 0.01em;
}}

/* == Recommendation Pills (theme-aware) == */
.rec-pill {{
    display: inline-block;
    background: rgba(46,134,171,0.12);
    color: var(--text-color);
    font-size: 0.82rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px 4px 3px 0;
    border: 1px solid rgba(46,134,171,0.25);
}}
.rec-pill-scenario {{
    background: rgba(56,161,105,0.12);
    border-color: rgba(56,161,105,0.25);
}}

/* == Footer == */
.footer {{
    text-align: center;
    color: var(--text-color);
    opacity: 0.45;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--secondary-background-color);
    margin-top: 3rem;
}}
</style>
"""



def _register_altair_theme() -> None:
    """Register a consistent, professional Altair theme for all charts."""

    theme_base = str(st.get_option("theme.base") or "light").lower()
    is_dark = theme_base == "dark"

    axis_title = "#E2E8F0" if is_dark else "#4A5568"
    axis_label = "#CBD5E0" if is_dark else "#718096"
    grid_color = "#334155" if is_dark else "#EDF2F7"
    domain_color = "#64748B" if is_dark else "#CBD5E0"
    title_color = "#F1F5F9" if is_dark else "#2D3748"

    @alt.theme.register("asean", enable=True)
    def _asean_theme():
        return alt.theme.ThemeConfig({
            "config": {
                "background": "transparent",
                "view": {"stroke": "transparent"},
                "range": {
                    "category": [
                        "#2E86AB", "#A23B72", "#E8913A", "#38A169",
                        "#E53E3E", "#805AD5", "#DD6B20", "#3182CE",
                        "#D69E2E", "#319795",
                    ],
                },
                "axis": {
                    "labelFont": "Inter, system-ui, sans-serif",
                    "titleFont": "Inter, system-ui, sans-serif",
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "titleColor": axis_title,
                    "labelColor": axis_label,
                    "gridColor": grid_color,
                    "domainColor": domain_color,
                    "tickColor": domain_color,
                },
                "legend": {
                    "labelFont": "Inter, system-ui, sans-serif",
                    "titleFont": "Inter, system-ui, sans-serif",
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "titleColor": axis_title,
                    "labelColor": axis_label,
                },
                "title": {
                    "font": "Inter, system-ui, sans-serif",
                    "fontSize": 14,
                    "fontWeight": 600,
                    "color": title_color,
                },
                "bar": {"cornerRadiusEnd": 3},
                "line": {"strokeWidth": 2.5},
                "point": {"size": 60},
            }
        })


# ── Data Loading ────────────────────────────────────────────────────────────

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace("%", "percent", regex=False)
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return out


@st.cache_data
def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_columns(df)


@st.cache_data
def load_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_model_result_tables(results_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for file_path in sorted(results_dir.glob("model_results_*.csv")):
        target = file_path.stem.replace("model_results_", "")
        tables[target] = pd.read_csv(file_path)
    return tables


@st.cache_data
def load_target_tables(results_dir: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for file_path in sorted(results_dir.glob(f"{prefix}*.csv")):
        target = file_path.stem.replace(prefix, "")
        tables[target] = pd.read_csv(file_path)
    return tables


# ── Helpers ─────────────────────────────────────────────────────────────────

def _pretty(name: str) -> str:
    return TARGET_LABELS.get(name, name.replace("_", " ").title())


def _insight_callout(title: str, text: str, tone: str = "neutral") -> None:
    tone_class = tone if tone in {"neutral", "positive", "risk"} else "neutral"
    st.markdown(
        f'<div class="insight-callout insight-{tone_class}">'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-text">{text}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _risk_scale() -> alt.Scale:
    return alt.Scale(range=["#FECACA", COLORS["danger"]])


def _neutral_scale() -> alt.Scale:
    return alt.Scale(range=["#BFDBFE", COLORS["secondary"]])


def _story_section(title: str, subtitle: str = "", label: str = "Narrative") -> None:
    subtitle_html = f'<div class="story-section-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="story-section">'
        f'<div class="story-section-label">{label}</div>'
        f'<div class="story-section-title">{title}</div>'
        f"{subtitle_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _story_takeaway(text: str, tone: str = "neutral") -> None:
    _insight_callout("Analyst Takeaway", text, tone=tone)


def _section_intro(text: str) -> None:
    """Render a styled introductory paragraph for a dashboard section."""
    st.markdown(f'<div class="section-intro">{text}</div>', unsafe_allow_html=True)


def _glossary_expander(terms: List[str]) -> None:
    """Show a collapsible glossary for the given metric names."""
    items = {k: v for k, v in METRIC_GLOSSARY.items() if k in terms}
    if not items:
        return
    with st.expander("Glossary — what do these metrics mean?", expanded=False):
        for term, defn in items.items():
            st.markdown(f"**{term}:** {defn}")


@st.cache_data
def _parse_ols_coefficients(path: Path) -> pd.DataFrame:
    """Extract coefficient rows from a statsmodels OLS summary text file."""
    if not path.exists():
        return pd.DataFrame(columns=["feature", "ols_coef"])

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, object]] = []
    in_table = False
    numeric_re = re.compile(r"^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", flags=re.IGNORECASE)

    for line in lines:
        if "coef" in line and "std err" in line and "P>|t|" in line:
            in_table = True
            continue
        if in_table and line.strip().startswith("==="):
            break
        if not in_table:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        feature, coef_raw = parts[0], parts[1]
        if not numeric_re.match(coef_raw):
            continue
        rows.append({"feature": feature, "ols_coef": float(coef_raw)})

    if not rows:
        return pd.DataFrame(columns=["feature", "ols_coef"])
    return pd.DataFrame(rows, columns=["feature", "ols_coef"])


@st.cache_data
def _parse_panel_ols_coefficients(path: Path) -> pd.DataFrame:
    """Extract coefficient rows from a linearmodels PanelOLS summary text file."""
    if not path.exists():
        return pd.DataFrame(columns=["feature", "panel_ols_coef"])

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: List[Dict[str, object]] = []
    in_section = False
    in_rows = False
    numeric_re = re.compile(r"^-?\d+(?:\.\d+)?(?:e[-+]?\d+)?$", flags=re.IGNORECASE)

    for line in lines:
        if "Parameter Estimates" in line:
            in_section = True
            continue
        if not in_section:
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if "Parameter" in line and "Std. Err." in line:
            in_rows = True
            continue
        if not in_rows:
            continue
        if stripped.startswith("="):
            break
        if set(stripped) == {"-"}:
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        feature, coef_raw = parts[0], parts[1]
        if not numeric_re.match(coef_raw):
            continue
        rows.append({"feature": feature, "panel_ols_coef": float(coef_raw)})

    if not rows:
        return pd.DataFrame(columns=["feature", "panel_ols_coef"])
    return pd.DataFrame(rows, columns=["feature", "panel_ols_coef"])


def _render_panel_ols_coefficients(fe_path: Path) -> None:
    """Render Panel OLS coefficient bar chart and table for presentation."""
    panel_coef_df = _parse_panel_ols_coefficients(fe_path)
    if "feature" not in panel_coef_df.columns:
        panel_coef_df = pd.DataFrame(columns=["feature", "panel_ols_coef"])

    if panel_coef_df.empty:
        st.info("Panel OLS coefficients are unavailable because no coefficient rows were parsed from the model summary.")
        return

    coeff_df = panel_coef_df.copy()
    coeff_df = coeff_df[coeff_df["feature"].str.lower() != "const"].copy()
    if coeff_df.empty:
        st.info("Panel OLS coefficients are unavailable after excluding intercept (`const`).")
        return
    coeff_df["max_abs_coef"] = coeff_df["panel_ols_coef"].abs()
    coeff_df = coeff_df.sort_values("max_abs_coef", ascending=False).reset_index(drop=True)
    coeff_df["feature_label"] = coeff_df["feature"].map(_pretty)

    st.markdown("#### Panel OLS Coefficients")
    st.caption(
        "Fixed-effects estimates show within-country relationships after absorbing time-invariant country differences."
    )

    chart_df = (
        coeff_df[["feature", "feature_label", "panel_ols_coef"]]
        .melt(
            id_vars=["feature", "feature_label"],
            value_vars=["panel_ols_coef"],
            var_name="model",
            value_name="coefficient",
        )
        .dropna(subset=["coefficient"])
    )
    feature_sort = coeff_df["feature_label"].tolist()

    chart_height = max(260, len(feature_sort) * 26)
    bars = (
        alt.Chart(chart_df)
        .mark_bar(color=COLORS["accent"])
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient Value"),
            y=alt.Y("feature_label:N", sort=feature_sort, title=""),
            tooltip=[
                alt.Tooltip("feature_label:N", title="Feature"),
                alt.Tooltip("coefficient:Q", title="Coefficient", format=".6f"),
            ],
        )
        .properties(title="Panel OLS (Fixed Effects) Coefficients", height=chart_height)
    )
    zero_line = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color=COLORS["muted"], strokeDash=[4, 4]
    ).encode(x="x:Q")
    st.altair_chart(bars + zero_line, width="stretch")

    table_df = coeff_df[
        ["feature_label", "panel_ols_coef"]
    ].rename(
        columns={
            "feature_label": "Feature",
            "panel_ols_coef": "Panel OLS Coef",
        }
    )
    st.dataframe(table_df, width="stretch", hide_index=True)


def _render_executive_summary(df: pd.DataFrame) -> None:
    st.header("Executive Summary")
    _section_intro(
        "High-level view of descriptive trend direction and predictive model performance. "
        "Use this tab for stakeholder updates and sprint-level decisions."
    )

    model_tables = load_model_result_tables(RESULTS_DIR)
    modeled_targets = len(model_tables)

    c1, c2, c3, c4 = st.columns(4)
    countries = int(df["country"].nunique())
    rows = int(len(df))
    min_year = int(df["year"].min()) if "year" in df.columns and not df.empty else None
    max_year = int(df["year"].max()) if "year" in df.columns and not df.empty else None
    c1.metric("Countries", countries)
    if min_year is not None and max_year is not None:
        c2.metric("Year Range", f"{min_year}\u2013{max_year}")
    else:
        c2.metric("Year Range", "N/A")
    c3.metric("Observations", f"{rows:,}")
    c4.metric("Targets Modeled", modeled_targets)

    _insight_callout(
        "Executive Readout",
        "This summary is optimized for presentation: key trend shifts, best model performance, and current project scope.",
        tone="neutral",
    )

    st.markdown("### Descriptive Trend Highlights")
    trend_rows: List[Dict[str, object]] = []
    for metric in ["gini_index", "gdp_per_capita", "life_expectancy", "infant_mortality"]:
        if metric not in df.columns:
            continue
        by_year = df.groupby("year", as_index=False)[metric].mean().dropna(subset=[metric]).sort_values("year")
        if len(by_year) < 2:
            continue
        start_val = float(by_year.iloc[0][metric])
        end_val = float(by_year.iloc[-1][metric])
        trend_rows.append(
            {
                "Metric": _pretty(metric),
                "Start": round(start_val, 3),
                "Latest": round(end_val, 3),
                "Delta": round(end_val - start_val, 3),
            }
        )
    if trend_rows:
        st.dataframe(pd.DataFrame(trend_rows), width="stretch", hide_index=True)
    else:
        st.info("Insufficient data to compute descriptive trend highlights.")

    st.markdown("### Predictive Model Snapshot")
    summary_rows: List[Dict[str, object]] = []
    for target, table in model_tables.items():
        ranked = table.sort_values("rmse")
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        summary_rows.append(
            {
                "Target": _pretty(target),
                "Best Model": str(best["model"]).replace("_", " ").title(),
                "RMSE": round(float(best["rmse"]), 4),
                "MAE": round(float(best["mae"]), 4),
                "R\u00b2": round(float(best["r2"]), 4),
            }
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("Target")
        st.dataframe(summary_df, width="stretch", hide_index=True)

        strong = summary_df.sort_values("R\u00b2", ascending=False).iloc[0]
        weak = summary_df.sort_values("R\u00b2", ascending=True).iloc[0]
        tone = "positive" if float(weak["R\u00b2"]) >= 0.4 else "risk"
        strong_r2 = strong["R\u00b2"]
        weak_r2 = weak["R\u00b2"]
        _insight_callout(
            "Performance Spread",
            f"Strongest target fit: {strong['Target']} (R² {strong_r2:.3f}). "
            f"Weakest target fit: {weak['Target']} (R² {weak_r2:.3f}).",
            tone=tone,
        )
    else:
        st.info("No predictive model result tables found yet.")

    st.markdown("### Current Scope Note")
    rec_df = load_optional_csv(RECOMMENDATION_RANKED_PATH)
    sim_df = load_optional_csv(SCENARIO_RESULTS_PATH)
    if rec_df is not None and not rec_df.empty and sim_df is not None and not sim_df.empty:
        latest_year = int(rec_df["year"].max()) if "year" in rec_df.columns else int(df["year"].max())
        st.markdown(
            f"- Recommendation Engine and What-If Simulation are active for the latest-year snapshot ({latest_year}).\n"
            "- Prescriptive outputs are reported as normalized composite score indices and stage-template policy bundles."
        )
    else:
        st.markdown(
            "- Prescriptive artifacts are not available yet. Run `python scripts/run_pipeline.py --stage models` "
            "to generate recommendation and simulation outputs."
        )


# ── Sidebar ─────────────────────────────────────────────────────────────────

def _render_sidebar(df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## ASEAN Policy Dashboard")
        st.caption("Data-driven decision support for ASEAN socioeconomic policy")

        st.markdown("---")

        # Data summary
        st.markdown("### Data Overview")
        n_countries = int(df["country"].nunique())
        min_y, max_y = int(df["year"].min()), int(df["year"].max())
        st.markdown(f"**Countries:** {n_countries}")
        st.markdown(f"**Time span:** {min_y}\u2013{max_y}")
        st.markdown(f"**Observations:** {len(df):,}")
        st.markdown(f"**Variables:** {len(df.columns)}")

        st.markdown("---")

        # Quick guide
        st.markdown("### Quick Guide")
        st.markdown(
            "**Intro** \u2014 Team/project overview, ASEAN context, and policy-transfer framing.\n\n"
            "**Executive Summary** \u2014 High-level trends and model performance snapshot.\n\n"
            "**Story Mode** \u2014 Narrative walkthroughs tied to real policy questions.\n\n"
            "**Data Explorer** \u2014 Filter, browse, and download the panel dataset.\n\n"
            "**Descriptive Analytics** \u2014 Cross-country comparisons and trend analysis.\n\n"
            "**Econometric Results** \u2014 Panel OLS regression results and diagnostics.\n\n"
            "**Predictive Analytics** \u2014 ML model performance and feature importance.\n\n"
            "**Policy Recommendations** \u2014 Ranked country priorities and stage-template policy bundles.\n\n"
            "**What-If Simulation** \u2014 Policy-impact scenario analysis aligned with recommendation pathways."
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.75rem; color:#CBD5E0; opacity:0.85;">'
            "Built for academic evaluation and portfolio demonstration. "
            "Data sourced from World Bank, WHO, and UN indicators."
            "</div>",
        unsafe_allow_html=True,
    )


# ── Tab: Intro ─────────────────────────────────────────────────────────────

def _render_intro(df: pd.DataFrame) -> None:
    st.header("Intro")
    _section_intro(
        "Project orientation for presentation context: who ASEAN is, what this dashboard does, "
        "and why cross-country evidence (including non-ASEAN comparators) strengthens policy design."
    )

    countries = int(df["country"].nunique()) if "country" in df.columns else 0
    min_year = int(df["year"].min()) if "year" in df.columns and not df.empty else None
    max_year = int(df["year"].max()) if "year" in df.columns and not df.empty else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Countries Covered", countries)
    c2.metric("Panel Window", f"{min_year}\u2013{max_year}" if min_year is not None and max_year is not None else "N/A")
    c3.metric("Observations", f"{len(df):,}")

    st.markdown("### Team and Objective")
    st.markdown(
        "- Build a decision-support dashboard for ASEAN-focused inequality and development analysis.\n"
        "- Combine descriptive, econometric, predictive, and prescriptive workflows in one interface.\n"
        "- Turn analysis outputs into actionable policy recommendations and scenario comparisons."
    )

    st.markdown("### Who and What is ASEAN")
    st.markdown(
        "- ASEAN is the Association of Southeast Asian Nations, a regional bloc focused on economic growth, social progress, and cooperation.\n"
        "- This project supports ASEAN-oriented planning by translating socioeconomic indicators into policy signals.\n"
        "- Country-level comparisons help surface where targeted interventions may be most impactful."
    )

    st.markdown("### Why Include Non-ASEAN Countries")
    st.markdown(
        "- Non-ASEAN countries are included as comparator signals for trend and policy analysis.\n"
        "- This broader reference set helps identify potentially transferable policy patterns.\n"
        "- Recommended actions are still interpreted through ASEAN country context before adoption."
    )


# ── Header ──────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown(
        '<div class="dash-header">'
        "<h1>ASEAN Policy Dashboard</h1>"
        "<p>Socioeconomic analytics, predictive modeling, and policy simulation across ASEAN member states</p>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Tab: Data Explorer ─────────────────────────────────────────────────────

def _render_data_explorer(df: pd.DataFrame) -> None:
    st.header("Data Explorer")
    _section_intro(
        "Browse and filter the full panel dataset. Select countries and a year range "
        "to narrow the view, choose which columns to display, and download the "
        "filtered result as CSV. The trend chart at the bottom lets you track any "
        "numeric indicator over time for the selected countries."
    )

    countries = sorted(df["country"].dropna().astype(str).unique().tolist())
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    col1, col2 = st.columns([2, 2])
    with col1:
        selected_countries = st.multiselect(
            "Countries",
            options=countries,
            default=countries[:3] if len(countries) > 3 else countries,
        )
    with col2:
        year_range = st.slider(
            "Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )

    filtered = df[
        df["country"].isin(selected_countries)
        & (df["year"] >= year_range[0])
        & (df["year"] <= year_range[1])
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Countries in View", int(filtered["country"].nunique()))
    c2.metric("Rows in View", f"{len(filtered):,}")
    if filtered.empty:
        c3.metric("Year Span", "—")
        _insight_callout(
            "Filter Result",
            "No rows match the current filters. Adjust country selection or year range.",
            tone="risk",
        )
        st.warning("No data available for the selected filters.")
        return
    c3.metric("Year Span", f"{int(filtered['year'].min())}\u2013{int(filtered['year'].max())}")
    _insight_callout(
        "Data Explorer Insight",
        f"Current view includes {filtered['country'].nunique()} countries across "
        f"{int(filtered['year'].min())}\u2013{int(filtered['year'].max())}, "
        f"with {len(filtered):,} observations ready for export.",
        tone="neutral",
    )

    display_columns = st.multiselect(
        "Columns to Display",
        options=list(filtered.columns),
        default=["country", "year", "gini_index", "life_expectancy", "gdp_per_capita"],
    )
    if display_columns:
        st.dataframe(filtered[display_columns], width="stretch", hide_index=True)
    else:
        st.dataframe(filtered, width="stretch", hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_panel_data.csv",
        mime="text/csv",
    )

    numeric_columns = [c for c in filtered.select_dtypes(include="number").columns if c != "year"]
    if numeric_columns and not filtered.empty:
        st.markdown("#### Trend Over Time")
        metric = st.selectbox("Trend Metric", numeric_columns, index=0)
        trend = (
            filtered[["country", "year", metric]]
            .pivot_table(index="year", columns="country", values=metric, aggfunc="mean")
            .sort_index()
        )
        chart_data = trend.reset_index().melt("year", var_name="Country", value_name=_pretty(metric)).dropna(subset=[_pretty(metric)])
        line = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y(f"{_pretty(metric)}:Q", title=_pretty(metric)),
                color=alt.Color("Country:N"),
                tooltip=["Country:N", "year:O", alt.Tooltip(f"{_pretty(metric)}:Q", format=",.2f")],
            )
            .interactive()
            .properties(height=380)
        )
        st.altair_chart(line, width="stretch")


# ── Tab: Descriptive Analytics ──────────────────────────────────────────────

def _render_descriptive_analytics(df: pd.DataFrame) -> None:
    st.header("Descriptive Analytics")
    _section_intro(
        "Explore cross-sectional snapshots and long-run trends across ASEAN and peer countries. "
        "Use the year slider to see how countries compare at a point in time, then "
        "examine the ASEAN-group trend for any indicator. The correlation heatmap "
        "reveals which variables move together\u200a\u2014\u200ahover over cells for exact values."
    )

    latest_year = int(df["year"].max())
    selected_year = st.slider(
        "Cross-Section Year",
        min_value=int(df["year"].min()),
        max_value=latest_year,
        value=latest_year,
    )
    year_df = df[df["year"] == selected_year].copy()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Countries", int(year_df["country"].nunique()))
    c2.metric("Observations", int(len(year_df)))
    if "gini_index" in year_df.columns:
        c3.metric("Avg Gini Index", f"{year_df['gini_index'].mean():.2f}")
    if "life_expectancy" in year_df.columns:
        c4.metric("Avg Life Expectancy", f"{year_df['life_expectancy'].mean():.1f} yrs")

    insight_parts: List[str] = [f"{selected_year} snapshot: {len(year_df)} observations"]
    if "gini_index" in year_df.columns and year_df["gini_index"].notna().any():
        insight_parts.append(f"avg Gini {year_df['gini_index'].mean():.2f}")
    if "life_expectancy" in year_df.columns and year_df["life_expectancy"].notna().any():
        insight_parts.append(f"avg life expectancy {year_df['life_expectancy'].mean():.1f} years")
    _insight_callout(
        "Snapshot Insight",
        "; ".join(insight_parts) + ".",
        tone="neutral",
    )

    numeric_columns = [c for c in df.select_dtypes(include="number").columns if c != "year"]
    if not numeric_columns:
        st.info("No numeric columns available for descriptive analysis.")
        return

    # Country comparison bar charts
    st.markdown("#### Country Comparison")
    metric = st.selectbox(
        "Metric",
        options=numeric_columns,
        index=numeric_columns.index("gdp_per_capita") if "gdp_per_capita" in numeric_columns else 0,
        key="desc_metric",
    )

    comp_df = year_df[["country", metric]].dropna().sort_values(metric, ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Top 10 \u2014 {_pretty(metric)}**")
        top_df = comp_df.head(10).copy()
        top_chart = (
            alt.Chart(top_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=_pretty(metric)),
                y=alt.Y("country:N", sort="-x", title=""),
                color=alt.value(COLORS["secondary"]),
                tooltip=["country:N", alt.Tooltip(f"{metric}:Q", format=",.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(top_chart, width="stretch")
    with c2:
        st.markdown(f"**Bottom 10 \u2014 {_pretty(metric)}**")
        bottom_df = comp_df.tail(10).copy()
        bottom_chart = (
            alt.Chart(bottom_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=_pretty(metric)),
                y=alt.Y("country:N", sort="x", title=""),
                color=alt.value(COLORS["danger"]),
                tooltip=["country:N", alt.Tooltip(f"{metric}:Q", format=",.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(bottom_chart, width="stretch")

    # ASEAN group trend
    st.markdown("#### ASEAN-Group Trend (Mean Across Countries)")
    trend_metric = st.selectbox(
        "Trend Metric",
        options=numeric_columns,
        key="desc_trend_metric",
        index=numeric_columns.index("gini_index") if "gini_index" in numeric_columns else 0,
    )
    trend_df = (
        df.groupby("year", as_index=False)[trend_metric]
        .mean()
        .dropna(subset=[trend_metric])
        .sort_values("year")
    )
    trend_chart = (
        alt.Chart(trend_df)
        .mark_area(
            line={"color": COLORS["secondary"]},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color=COLORS["secondary"], offset=0),
                    alt.GradientStop(color="rgba(46,134,171,0.08)", offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(f"{trend_metric}:Q", title=_pretty(trend_metric)),
            tooltip=["year:O", alt.Tooltip(f"{trend_metric}:Q", format=",.3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(trend_chart, width="stretch")

    # Correlation heatmap (Altair — interactive)
    st.markdown("#### Correlation Matrix")
    st.caption("Hover over cells to see exact correlation values. Strong blue = positive, strong red = negative.")
    corr_csv = load_optional_csv(CORRELATION_PATH)
    if corr_csv is not None and not corr_csv.empty:
        corr_plot = corr_csv.set_index(corr_csv.columns[0])
    else:
        corr_plot = df[numeric_columns].corr()

    # Melt for Altair
    corr_reset = corr_plot.reset_index()
    id_col = corr_reset.columns[0]
    melted = corr_reset.melt(id_vars=id_col, var_name="Variable B", value_name="Correlation")
    melted.rename(columns={id_col: "Variable A"}, inplace=True)

    heatmap = (
        alt.Chart(melted)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("Variable B:N", title=None, axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("Variable A:N", title=None, axis=alt.Axis(labelFontSize=9)),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                legend=alt.Legend(title="r"),
            ),
            tooltip=[
                alt.Tooltip("Variable A:N"),
                alt.Tooltip("Variable B:N"),
                alt.Tooltip("Correlation:Q", format=".3f"),
            ],
        )
        .properties(width=580, height=580)
    )
    heatmap_text_color = "#E2E8F0" if str(st.get_option("theme.base") or "light").lower() == "dark" else "#2D3748"
    # Add text labels
    text = (
        alt.Chart(melted)
        .mark_text(fontSize=8, color=heatmap_text_color)
        .encode(
            x=alt.X("Variable B:N"),
            y=alt.Y("Variable A:N"),
            text=alt.Text("Correlation:Q", format=".2f"),
        )
    )
    st.altair_chart(heatmap + text, width="stretch")


# ── Tab: Predictive Analytics ───────────────────────────────────────────────

def _render_predictive_analytics(df: pd.DataFrame) -> None:
    st.header("Predictive Analytics")
    _section_intro(
        "Multiple machine-learning models (Linear, Ridge, Lasso, Gradient Boosting, Random Forest) "
        "were trained on the panel dataset for each target variable. This section lets you compare "
        "model accuracy, inspect predictions vs. actuals, review residual distributions, and "
        "identify which features matter most. This tab primarily supports Story 3 and Story 4."
    )
    _glossary_expander(["RMSE", "MAE", "R\u00b2", "MAPE"])

    model_tables = load_model_result_tables(RESULTS_DIR)
    prediction_tables = load_target_tables(RESULTS_DIR, "predictions_")
    importance_tables = load_target_tables(RESULTS_DIR, "feature_importance_")
    country_avg_tables = load_target_tables(RESULTS_DIR, "country_average_")

    if not model_tables:
        st.warning("No predictive result tables found. Run `python scripts/run_pipeline.py --stage models`.")
        return

    targets = sorted(model_tables.keys())
    target = st.selectbox(
        "Target Variable",
        targets,
        index=0,
        format_func=_pretty,
    )
    results_df = model_tables[target].sort_values("rmse")

    # Best model KPIs — custom HTML grid for readable model names
    best_row = results_df.iloc[0]
    best_model_name = str(best_row["model"]).replace("_", " ").title()
    st.markdown(
        f'<div class="kpi-grid">'
        f'  <div class="kpi-card kpi-card-highlight">'
        f'    <div class="kpi-label">Best Model</div>'
        f'    <div class="kpi-value kpi-value-sm">{best_model_name}</div>'
        f'  </div>'
        f'  <div class="kpi-card">'
        f'    <div class="kpi-label">RMSE</div>'
        f'    <div class="kpi-value">{best_row["rmse"]:,.2f}</div>'
        f'  </div>'
        f'  <div class="kpi-card">'
        f'    <div class="kpi-label">R\u00b2</div>'
        f'    <div class="kpi-value">{best_row["r2"]:.4f}</div>'
        f'  </div>'
        f'  <div class="kpi-card">'
        f'    <div class="kpi-label">MAE</div>'
        f'    <div class="kpi-value">{best_row["mae"]:,.2f}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    best_r2 = float(best_row["r2"])
    model_tone = "positive" if best_r2 >= 0.75 else "neutral" if best_r2 >= 0.40 else "risk"
    _insight_callout(
        "Model Readout",
        f"For {_pretty(target)}, {best_model_name} currently performs best "
        f"(RMSE {float(best_row['rmse']):,.2f}, MAE {float(best_row['mae']):,.2f}, R\u00b2 {best_r2:.3f}).",
        tone=model_tone,
    )

    # Full results table
    with st.expander("Full Model Comparison Table", expanded=False):
        st.dataframe(results_df, width="stretch", hide_index=True)

    # Interactive bar chart
    st.markdown("#### Model Comparison")
    metric_choice = st.selectbox(
        "Comparison Metric",
        options=["rmse", "mae", "r2", "mape_percent"],
        format_func=lambda x: {"rmse": "RMSE", "mae": "MAE", "r2": "R\u00b2", "mape_percent": "MAPE %"}[x],
        index=0,
        key="pred_metric_choice",
    )
    ascending = metric_choice != "r2"
    ranked = results_df.sort_values(metric_choice, ascending=ascending).copy()
    bar_color = COLORS["secondary"] if metric_choice != "r2" else COLORS["success"]
    metric_chart = (
        alt.Chart(ranked)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=ranked["model"].tolist(), title="Model",
                     axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{metric_choice}:Q", title=metric_choice.upper().replace("_PERCENT", " %")),
            color=alt.value(bar_color),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("mae:Q", format=",.4f"),
                alt.Tooltip("rmse:Q", format=",.4f"),
                alt.Tooltip("r2:Q", format=".4f"),
                alt.Tooltip("mape_percent:Q", format=",.2f", title="MAPE %"),
            ],
        )
        .properties(height=350)
    )
    st.altair_chart(metric_chart, width="stretch")

    # Actual vs. Predicted scatter
    pred_df = prediction_tables.get(target)
    if pred_df is not None and not pred_df.empty:
        st.markdown("#### Actual vs. Predicted")
        st.caption("Each dot is a test-set observation. The black line marks perfect prediction (y = x).")
        available_models = sorted(pred_df["model"].dropna().astype(str).unique().tolist())
        default_idx = available_models.index(str(best_row["model"])) if str(best_row["model"]) in available_models else 0

        col_m, col_c = st.columns([1, 2])
        with col_m:
            selected_model = st.selectbox(
                "Prediction Model",
                options=available_models,
                index=default_idx,
                key="pred_scatter_model",
            )
        with col_c:
            countries = sorted(pred_df[pred_df["model"] == selected_model]["country"].dropna().astype(str).unique().tolist())
            selected_countries = st.multiselect(
                "Filter Countries",
                options=countries,
                default=countries,
                key="pred_scatter_countries",
            )

        view_df = pred_df[(pred_df["model"] == selected_model) & (pred_df["country"].isin(selected_countries))].copy()

        if not view_df.empty:
            axis_min = float(min(view_df["actual"].min(), view_df["predicted"].min()))
            axis_max = float(max(view_df["actual"].max(), view_df["predicted"].max()))
            pad = (axis_max - axis_min) * 0.05
            line_df = pd.DataFrame({"actual": [axis_min - pad, axis_max + pad], "predicted": [axis_min - pad, axis_max + pad]})

            scatter = (
                alt.Chart(view_df)
                .mark_circle(size=70, opacity=0.7)
                .encode(
                    x=alt.X("actual:Q", title=f"Actual {_pretty(target)}",
                             scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])),
                    y=alt.Y("predicted:Q", title=f"Predicted {_pretty(target)}",
                             scale=alt.Scale(domain=[axis_min - pad, axis_max + pad])),
                    color=alt.Color("country:N", title="Country"),
                    tooltip=[
                        alt.Tooltip("country:N"),
                        alt.Tooltip("year:Q"),
                        alt.Tooltip("actual:Q", format=",.2f"),
                        alt.Tooltip("predicted:Q", format=",.2f"),
                    ],
                )
            )
            diagonal = alt.Chart(line_df).mark_line(color="black", strokeDash=[6, 4], opacity=0.6).encode(
                x="actual:Q", y="predicted:Q",
            )
            st.altair_chart((scatter + diagonal).interactive().properties(height=420), width="stretch")

            # Residual histogram
            st.markdown("#### Residual Distribution")
            st.caption("Residuals should be roughly centered at zero. Skew or heavy tails indicate systematic bias.")
            view_df["residual"] = view_df["actual"] - view_df["predicted"]
            residual_chart = (
                alt.Chart(view_df)
                .mark_bar(opacity=0.85, color=COLORS["secondary"])
                .encode(
                    x=alt.X("residual:Q", bin=alt.Bin(maxbins=30), title="Residual (Actual \u2212 Predicted)"),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
                .properties(height=280)
            )
            st.altair_chart(residual_chart, width="stretch")

            # Residual-based uncertainty view
            st.markdown("#### Uncertainty View (Residual-Based)")
            st.caption(
                "Approximate 80% uncertainty interval derived from test residual quantiles "
                "(q10 to q90). This is diagnostic uncertainty, not a formal probabilistic forecast."
            )
            residual_non_na = view_df["residual"].dropna()
            if len(residual_non_na) >= 5:
                q10 = float(residual_non_na.quantile(0.10))
                q90 = float(residual_non_na.quantile(0.90))
                sigma = float(residual_non_na.std(ddof=0))
                view_df["lower80"] = view_df["predicted"] + q10
                view_df["upper80"] = view_df["predicted"] + q90
                coverage80 = float(
                    ((view_df["actual"] >= view_df["lower80"]) & (view_df["actual"] <= view_df["upper80"]))
                    .mean()
                    * 100.0
                )

                u1, u2, u3 = st.columns(3)
                u1.metric("Residual \u03c3", f"{sigma:,.3f}")
                u2.metric("80% Interval Width", f"{(q90 - q10):,.3f}")
                u3.metric("Observed 80% Coverage", f"{coverage80:.1f}%")

                yearly_band = (
                    view_df.groupby("year", as_index=False)
                    .agg(
                        actual=("actual", "mean"),
                        predicted=("predicted", "mean"),
                        lower80=("lower80", "mean"),
                        upper80=("upper80", "mean"),
                    )
                    .sort_values("year")
                )

                band = (
                    alt.Chart(yearly_band)
                    .mark_area(opacity=0.2, color=COLORS["secondary"])
                    .encode(
                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("lower80:Q", title=_pretty(target)),
                        y2="upper80:Q",
                        tooltip=[
                            "year:O",
                            alt.Tooltip("lower80:Q", format=",.3f", title="Lower 80%"),
                            alt.Tooltip("upper80:Q", format=",.3f", title="Upper 80%"),
                        ],
                    )
                )
                pred_line = (
                    alt.Chart(yearly_band)
                    .mark_line(color=COLORS["secondary"])
                    .encode(
                        x=alt.X("year:O"),
                        y=alt.Y("predicted:Q"),
                        tooltip=[alt.Tooltip("predicted:Q", format=",.3f", title="Predicted Mean")],
                    )
                )
                act_line = (
                    alt.Chart(yearly_band)
                    .mark_line(color=COLORS["danger"], strokeDash=[5, 3])
                    .encode(
                        x=alt.X("year:O"),
                        y=alt.Y("actual:Q"),
                        tooltip=[alt.Tooltip("actual:Q", format=",.3f", title="Actual Mean")],
                    )
                )
                st.altair_chart((band + pred_line + act_line).interactive().properties(height=320), width="stretch")
            else:
                st.info("Insufficient residual data to compute uncertainty diagnostics for the current filter.")

            # Per-country error diagnostics
            st.markdown("#### Per-Country Error Diagnostics")
            st.caption("Identifies countries where the model struggles most, useful for targeted policy refinement.")
            country_rows = []
            for country, g in view_df.groupby("country"):
                if g.empty:
                    continue
                err = g["actual"] - g["predicted"]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mae = float(np.mean(np.abs(err)))
                ss_res = float(np.sum((g["actual"] - g["predicted"]) ** 2))
                ss_tot = float(np.sum((g["actual"] - g["actual"].mean()) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else np.nan
                country_rows.append({"country": country, "n_test_rows": int(len(g)), "mae": mae, "rmse": rmse, "r2": r2})

            if country_rows:
                country_perf_df = pd.DataFrame(country_rows).sort_values("rmse")
                with st.expander("Detailed Country Error Table", expanded=False):
                    st.dataframe(country_perf_df, width="stretch", hide_index=True)
                perf_chart = (
                    alt.Chart(country_perf_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("rmse:Q", title="RMSE"),
                        y=alt.Y("country:N", sort="-x", title=""),
                        color=alt.Color("rmse:Q", scale=_risk_scale(), legend=None),
                        tooltip=[
                            alt.Tooltip("country:N"),
                            alt.Tooltip("n_test_rows:Q", title="Test Rows"),
                            alt.Tooltip("mae:Q", format=",.2f"),
                            alt.Tooltip("rmse:Q", format=",.2f"),
                            alt.Tooltip("r2:Q", format=".3f"),
                        ],
                    )
                    .properties(height=max(250, len(country_perf_df) * 28))
                )
                st.altair_chart(perf_chart, width="stretch")
    else:
        st.info(
            "Prediction point data not available. Run "
            "`python scripts/run_pipeline.py --stage models --skip-ols --skip-fixed-effects` "
            "to generate `predictions_*.csv` files."
        )

    # Feature importance + predictor relationship
    importance_df = importance_tables.get(target)
    if importance_df is not None and not importance_df.empty:
        st.markdown("#### Feature Importance")
        if "model" in importance_df.columns:
            model_options = sorted(importance_df["model"].dropna().astype(str).unique().tolist())
            selected_importance_model = st.selectbox(
                "Importance Model",
                options=model_options,
                key=f"pred_importance_model_{target}",
            )
            view_importance_df = importance_df[importance_df["model"] == selected_importance_model].copy()
        else:
            selected_importance_model = "Random Forest"
            view_importance_df = importance_df.copy()
            st.caption("Current pipeline exports feature-importance tables from Random Forest.")

        st.caption(
            "Shows which variables contribute most to predictions. Higher importance = stronger influence on the target."
        )
        top_n = st.slider(
            "Top Features",
            min_value=5,
            max_value=min(20, len(view_importance_df)),
            value=min(10, len(view_importance_df)),
            step=1,
            key=f"pred_top_features_{target}",
        )
        top_features = view_importance_df.head(top_n).sort_values("importance", ascending=True)
        imp_chart = (
            alt.Chart(top_features)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title=f"Importance ({selected_importance_model})"),
                y=alt.Y("feature:N", sort="-x", title=""),
                color=alt.Color("importance:Q", scale=_neutral_scale(), legend=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=max(250, top_n * 28))
        )
        st.altair_chart(imp_chart, width="stretch")
    else:
        st.info("Feature importance table not found for the selected target.")

    if target in df.columns:
        predictor_candidates = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in {"year", target}
        ]
        if predictor_candidates:
            st.markdown("#### Predictor vs. Target Relationship")
            st.caption(
                "Use this to inspect how any predictor aligns with the selected target across observations."
            )
            p1, p2 = st.columns([2, 2])
            with p1:
                predictor = st.selectbox(
                    "Predictor",
                    options=sorted(predictor_candidates),
                    key=f"pred_relationship_predictor_{target}",
                )
            with p2:
                relation_view = st.selectbox(
                    "Relationship View",
                    options=["All Observations", "Country Averages"],
                    key=f"pred_relationship_view_{target}",
                )

            rel_df = df[["country", "year", predictor, target]].dropna().copy()
            if relation_view == "Country Averages":
                rel_df = (
                    rel_df.groupby("country", as_index=False)[[predictor, target]]
                    .mean()
                )

            if not rel_df.empty:
                color_encoding = (
                    alt.Color("country:N", title="Country")
                    if "country" in rel_df.columns and rel_df["country"].nunique() <= 20
                    else alt.value(COLORS["secondary"])
                )

                scatter = (
                    alt.Chart(rel_df)
                    .mark_circle(size=65, opacity=0.65)
                    .encode(
                        x=alt.X(f"{predictor}:Q", title=_pretty(predictor)),
                        y=alt.Y(f"{target}:Q", title=_pretty(target)),
                        color=color_encoding,
                        tooltip=[alt.Tooltip(c) for c in rel_df.columns],
                    )
                    .properties(height=360)
                )
                trend_line = scatter.transform_regression(predictor, target).mark_line(
                    color=COLORS["danger"], strokeDash=[6, 4]
                )
                st.altair_chart((scatter + trend_line).interactive(), width="stretch")
            else:
                st.info("No rows available for the selected predictor/target pairing.")

    # Country average
    country_avg_df = country_avg_tables.get(target)
    if country_avg_df is not None and not country_avg_df.empty:
        st.markdown(f"#### Country Average \u2014 {_pretty(target)}")
        country_chart = (
            alt.Chart(country_avg_df)
            .mark_bar()
            .encode(
                x=alt.X("average_value:Q", title=f"Average {_pretty(target)}"),
                y=alt.Y("country:N", sort="-x", title=""),
                color=alt.value(COLORS["secondary"]),
                tooltip=["country:N", alt.Tooltip("average_value:Q", format=",.2f")],
            )
            .properties(height=max(250, len(country_avg_df) * 24))
        )
        st.altair_chart(country_chart, width="stretch")

    # Best model per target summary
    summary_rows = []
    for tgt, table in model_tables.items():
        table_sorted = table.sort_values("rmse")
        if table_sorted.empty:
            continue
        row = table_sorted.iloc[0]
        summary_rows.append({"Target": _pretty(tgt), "Best Model": row["model"], "RMSE": row["rmse"], "R\u00b2": row["r2"]})
    if summary_rows:
        st.markdown("#### Best Model Per Target (All Variables)")
        st.dataframe(pd.DataFrame(summary_rows).sort_values("Target"), width="stretch", hide_index=True)


# ── Tab: Econometric Results ────────────────────────────────────────────────

def _render_econometric_results() -> None:
    st.header("Econometric Results")
    _section_intro(
        "Classical econometric analysis using Panel OLS with country fixed effects. "
        "Fixed effects control for time-invariant differences between countries, "
        "isolating within-country relationships between variables. VIF diagnostics "
        "check for multicollinearity among predictors."
    )
    _glossary_expander(["VIF", "R\u00b2"])

    fe_path = RESULTS_DIR / "fixed_effects_summary.txt"
    vif_path = RESULTS_DIR / "fixed_effects_vif.csv"
    cleaned_path = MODELING_CLEANED_PATH
    fe_available = fe_path.exists()
    vif_available = vif_path.exists()

    if cleaned_path.exists():
        cleaned_df = pd.read_csv(cleaned_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Countries", int(cleaned_df["country"].nunique()))
        c2.metric("Years", int(cleaned_df["year"].nunique()))
        c3.metric("Observations", f"{len(cleaned_df):,}")

    _insight_callout(
        "Econometric Status",
        f"Panel fixed-effects summary: {'available' if fe_available else 'missing'}; "
        f"VIF diagnostics: {'available' if vif_available else 'missing'}.",
        tone="positive" if (fe_available and vif_available) else "neutral",
    )

    if fe_available:
        _render_panel_ols_coefficients(fe_path)

    if fe_available:
        with st.expander("Panel OLS (Fixed Effects) Summary", expanded=True):
            st.caption("Fixed effects absorb time-invariant country characteristics, providing more reliable estimates of within-country relationships.")
            st.code(fe_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Fixed effects summary not found.")

    if vif_available:
        vif_df = pd.read_csv(vif_path).sort_values("vif", ascending=False)
        if not vif_df.empty:
            max_vif = float(vif_df["vif"].max())
            _insight_callout(
                "Collinearity Signal",
                f"Maximum VIF is {max_vif:.2f}. Values above 10 indicate stronger multicollinearity risk.",
                tone="risk" if max_vif > 10 else "neutral",
            )
        st.markdown("#### VIF Diagnostics")
        st.caption(
            "Variance Inflation Factor measures how much a predictor correlates with others. "
            "Values above 5 warrant attention; above 10 suggest serious multicollinearity."
        )
        col_t, col_c = st.columns([1, 2])
        with col_t:
            st.dataframe(vif_df, width="stretch", hide_index=True)
        with col_c:
            vif_plot = vif_df.copy()
            vif_plot["severity"] = pd.cut(
                vif_plot["vif"],
                bins=[-float("inf"), 5, 10, float("inf")],
                labels=["Normal", "Moderate", "High"],
            )
            severity_colors = alt.Scale(
                domain=["Normal", "Moderate", "High"],
                range=[COLORS["secondary"], COLORS["warning"], COLORS["danger"]],
            )
            vif_chart = (
                alt.Chart(vif_plot)
                .mark_bar()
                .encode(
                    x=alt.X("vif:Q", title="VIF"),
                    y=alt.Y("feature:N", sort="-x", title=""),
                    color=alt.Color("severity:N", scale=severity_colors, title="Severity"),
                    tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("vif:Q", format=",.2f"), "severity:N"],
                )
                .properties(height=max(250, len(vif_df) * 28))
            )
            rule = alt.Chart(pd.DataFrame({"x": [10]})).mark_rule(color=COLORS["danger"], strokeDash=[6, 4]).encode(x="x:Q")
            st.altair_chart(vif_chart + rule, width="stretch")
    else:
        st.info("VIF file not found.")


# ── Tab: What-If Simulation ─────────────────────────────────────────────────

def _render_simulation(df: pd.DataFrame, key_prefix: str = "sim", show_header: bool = True) -> None:
    del df  # Prescriptive views are sourced from generated scenario artifacts.
    if show_header:
        st.header("What-If Simulation")
    _section_intro(
        "Policy-impact simulation layer for Story 6 (prescriptive recommendations). "
        "Compare baseline and reform pathways using a normalized composite score index (0-100) "
        "for the latest-year country snapshot to estimate post-policy uplift."
    )

    scenario_df = load_optional_csv(SCENARIO_RESULTS_PATH)
    summary_df = load_optional_csv(SCENARIO_SUMMARY_PATH)
    if scenario_df is None or summary_df is None:
        st.info(
            "Scenario artifacts are missing. Run `python scripts/run_pipeline.py --stage models` "
            "to generate `scenario_results.csv` and `scenario_summary.csv`."
        )
        return
    if scenario_df.empty:
        st.warning("Scenario results are empty.")
        return

    latest_year = int(scenario_df["year"].max()) if "year" in scenario_df.columns else None
    if latest_year is not None:
        st.caption(
            f"Simulation basis: latest-year snapshot ({latest_year}). Scores are comparative index values, "
            "not direct GDP/Gini unit forecasts."
        )

    if "best_scenario" in summary_df.columns and not summary_df.empty:
        scenario_counts = summary_df["best_scenario"].value_counts()
        dominant = str(scenario_counts.index[0])
        dominant_n = int(scenario_counts.iloc[0])
        total_n = int(len(summary_df))
        _insight_callout(
            "Cross-Country Pattern",
            f"{dominant} is currently the top scenario for {dominant_n}/{total_n} countries in this run.",
            tone="neutral",
        )

    countries = sorted(scenario_df["country"].dropna().astype(str).unique().tolist())
    country = st.selectbox("Country Scenario Focus", countries, key=f"{key_prefix}_country")

    country_df = scenario_df[scenario_df["country"] == country].copy()
    country_summary = summary_df[summary_df["country"] == country]
    baseline_row = country_df[country_df["scenario"] == "Baseline (No Change)"]
    if baseline_row.empty:
        st.warning("Baseline scenario row is missing for this country.")
        return

    baseline_score = float(baseline_row["projected_score"].iloc[0])
    best_scenario = (
        country_summary["best_scenario"].iloc[0]
        if not country_summary.empty
        else country_df[country_df["scenario"] != "Baseline (No Change)"]
        .sort_values("projected_score", ascending=False)["scenario"]
        .iloc[0]
    )
    best_row = country_df[country_df["scenario"] == best_scenario].iloc[0]

    m1, m2 = st.columns(2)
    m1.metric("Baseline Index", f"{baseline_score:.2f}")
    m2.metric("Best Scenario", str(best_scenario))
    m3, m4 = st.columns(2)
    m3.metric("Best Index", f"{float(best_row['projected_score']):.2f}")
    m4.metric("Best Uplift", f"{float(best_row['uplift_vs_baseline']):+.2f}")

    _insight_callout(
        "Country Scenario Signal",
        f"For {country}, the highest projected gain comes from {best_scenario} "
        f"with an uplift of {float(best_row['uplift_vs_baseline']):+.2f} points.",
        tone="positive" if float(best_row["uplift_vs_baseline"]) > 0 else "neutral",
    )

    scenario_order = [
        "Baseline (No Change)",
        "Scenario A - Health Push",
        "Scenario B - Inequality Reduction",
        "Scenario C - Economic Acceleration",
        "Scenario D - Comprehensive Reform",
    ]
    scenario_metric = st.selectbox(
        "Scenario Metric",
        options=["projected_score", "uplift_vs_baseline"],
        format_func=lambda x: "Projected Composite Index" if x == "projected_score" else "Uplift vs Baseline",
        key=f"{key_prefix}_scenario_metric",
    )
    country_df["scenario"] = pd.Categorical(country_df["scenario"], categories=scenario_order, ordered=True)
    country_df = country_df.sort_values("scenario")

    color_domain = scenario_order
    color_range = ["#95A5A6", "#3182CE", "#DD6B20", "#38A169", "#805AD5"]
    chart = (
        alt.Chart(country_df)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", sort=scenario_order, title="Scenario"),
            y=alt.Y(
                f"{scenario_metric}:Q",
                title="Projected Composite Index (0-100)" if scenario_metric == "projected_score" else "Uplift vs Baseline",
            ),
            color=alt.Color("scenario:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            tooltip=[
                "country:N",
                "scenario:N",
                alt.Tooltip("projected_score:Q", format=".2f"),
                alt.Tooltip("uplift_vs_baseline:Q", format="+.2f"),
            ],
        )
        .properties(height=360)
    )
    rule_value = baseline_score if scenario_metric == "projected_score" else 0.0
    rule = alt.Chart(pd.DataFrame({"y": [rule_value]})).mark_rule(
        color=COLORS["danger"], strokeDash=[6, 4]
    ).encode(y="y:Q")
    st.altair_chart((chart + rule).interactive(), width="stretch")

    scenario_uplift = scenario_df[scenario_df["scenario"] != "Baseline (No Change)"].copy()
    avg_uplift = (
        scenario_uplift.groupby("scenario", as_index=False)["uplift_vs_baseline"]
        .mean()
        .sort_values("uplift_vs_baseline", ascending=False)
    )
    st.markdown("#### Average Uplift by Scenario")
    avg_chart = (
        alt.Chart(avg_uplift)
        .mark_bar()
        .encode(
            x=alt.X("uplift_vs_baseline:Q", title="Average Uplift vs Baseline"),
            y=alt.Y("scenario:N", sort="-x", title=""),
            color=alt.Color("uplift_vs_baseline:Q", scale=_neutral_scale(), legend=None),
            tooltip=["scenario:N", alt.Tooltip("uplift_vs_baseline:Q", format="+.2f")],
        )
        .properties(height=230)
    )
    st.altair_chart(avg_chart, width="stretch")

    st.markdown("#### Scenario Summary by Country")
    st.dataframe(summary_df.sort_values("best_uplift_vs_baseline", ascending=False), width="stretch", hide_index=True)


# ── Tab: Policy Recommendations ─────────────────────────────────────────────

def _render_policy_recommendations(show_header: bool = True, key_prefix: str = "rec") -> None:
    if show_header:
        st.header("Policy Recommendations")
    _section_intro(
        "Story 6 ranks countries using weighted scores across inequality, health, demographics, "
        "and economics for the latest-year snapshot. Policy actions are stage-template "
        "recommendations based on DTM/ETM pairings."
    )

    ranked_df = load_optional_csv(RECOMMENDATION_RANKED_PATH)
    evidence_df = load_optional_csv(RECOMMENDATION_EVIDENCE_PATH)
    if ranked_df is None or ranked_df.empty:
        st.info(
            "Recommendation artifacts are missing. Run `python scripts/run_pipeline.py --stage models` "
            "to generate `policy_recommendations_ranked.csv`."
        )
        return

    ranked_df = ranked_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    latest_year = int(ranked_df["year"].max()) if "year" in ranked_df.columns else None
    unique_bundles = int(ranked_df["policy_bundle"].nunique()) if "policy_bundle" in ranked_df.columns else 0
    if latest_year is not None:
        st.caption(
            f"Recommendation basis: latest-year snapshot ({latest_year}). "
            f"This run contains {unique_bundles} unique policy bundle templates across {len(ranked_df)} countries."
        )
    countries = ranked_df["country"].dropna().astype(str).unique().tolist()
    selected_country = st.selectbox("Country Recommendation Focus", countries, key=f"{key_prefix}_country")
    selected = ranked_df[ranked_df["country"] == selected_country].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Composite Score", f"{float(selected['composite_score']):.2f}")
    dtm = selected.get("dtm_stage", np.nan)
    etm = selected.get("etm_stage", np.nan)
    if pd.notna(dtm) and pd.notna(etm):
        c2.metric("DTM / ETM", f"{int(float(dtm))} / {int(float(etm))}")
    else:
        c2.metric("DTM / ETM", "N/A")
    c3, c4 = st.columns(2)
    c3.metric("Tier", str(selected["recommendation_tier"]))
    c4.metric("Income Group", str(selected.get("income_group", "N/A")))

    st.markdown("#### Recommended Policy Actions")
    for col in ["policy_1", "policy_2", "policy_3"]:
        text = selected.get(col, "")
        if isinstance(text, str) and text.strip():
            st.markdown(f"- {text}")

    tier_scale = alt.Scale(
        domain=[
            "Tier 1 - Sustain & Lead",
            "Tier 2 - Optimise",
            "Tier 3 - Transition",
            "Tier 4 - Critical Intervention",
        ],
        range=["#2ecc71", "#3498db", "#f39c12", "#e74c3c"],
    )
    rank_metric_candidates = [
        c
        for c in ["composite_score", "dtm_stage", "etm_stage"]
        if c in ranked_df.columns and ranked_df[c].notna().any()
    ]
    rank_metric = st.selectbox(
        "Ranking Metric",
        options=rank_metric_candidates if rank_metric_candidates else ["composite_score"],
        format_func=_pretty,
        key=f"{key_prefix}_rank_metric",
    )
    rank_chart = (
        alt.Chart(ranked_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{rank_metric}:Q", title=_pretty(rank_metric)),
            y=alt.Y("country:N", sort="-x", title=""),
            color=alt.Color("recommendation_tier:N", scale=tier_scale, title="Tier"),
            tooltip=[
                "country:N",
                alt.Tooltip("composite_score:Q", format=".2f"),
                "recommendation_tier:N",
                "income_group:N",
            ],
        )
        .properties(height=max(300, len(ranked_df) * 24))
    )
    st.markdown("#### Country Ranking")
    st.altair_chart(rank_chart, width="stretch")

    if evidence_df is not None and not evidence_df.empty:
        country_evidence = evidence_df[evidence_df["country"] == selected_country].copy()
        if not country_evidence.empty:
            st.markdown("#### Dimension Evidence")
            dim_chart = (
                alt.Chart(country_evidence)
                .mark_bar()
                .encode(
                    x=alt.X("dimension:N", title="Dimension"),
                    y=alt.Y("dimension_score:Q", title="Dimension Score"),
                    color=alt.Color("dimension_score:Q", scale=_neutral_scale(), legend=None),
                    tooltip=[
                        "country:N",
                        "dimension:N",
                        alt.Tooltip("dimension_score:Q", format=".2f"),
                        alt.Tooltip("dimension_rank:Q", format=".0f", title="Rank"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(dim_chart, width="stretch")

            strongest = country_evidence.sort_values("dimension_score", ascending=False).iloc[0]
            weakest = country_evidence.sort_values("dimension_score", ascending=True).iloc[0]
            _insight_callout(
                "Country Priority Signal",
                f"{selected_country} shows strongest performance in {strongest['dimension']} "
                f"({float(strongest['dimension_score']):.1f}) and weakest performance in "
                f"{weakest['dimension']} ({float(weakest['dimension_score']):.1f}).",
                tone="neutral",
            )
            st.caption(
                "Dimension evidence reflects normalized weighted component scores, used for transparent ranking support."
            )

    st.markdown("#### Ranked Table")
    st.dataframe(ranked_df, width="stretch", hide_index=True)


# ── Tab: Story Mode ─────────────────────────────────────────────────────────

def _render_story_mode(df: pd.DataFrame) -> None:
    st.header("Story Mode")
    _section_intro(
        "Each story is framed around a specific policy persona and question, making "
        "the analytics tangible. Pick a story to see tailored visualizations and "
        "narrative context. Stories span descriptive (what happened?), predictive "
        "(what might happen?), and prescriptive (what should we do?) analytics."
    )

    # Story selector cards
    story_options = [f"{s['id']} \u2014 {s['type']} \u2014 {s['title']}" for s in USER_STORIES]
    selected = st.selectbox("Select a User Story", story_options, index=0, key="story_selector")
    story_id = selected.split("\u2014")[0].strip()
    story = next((s for s in USER_STORIES if s["id"] == story_id), USER_STORIES[0])

    # Story card
    type_lower = story["type"].lower()
    st.markdown(
        f'<div class="story-card">'
        f'<div class="story-card-top">'
        f'<span class="story-id">{story["id"]}</span>'
        f'<span class="story-type story-type-{type_lower}">{story["type"]}</span>'
        f"</div>"
        f'<h4>{story["title"]}</h4>'
        f'<div class="persona">Persona: {story["persona"]}</div>'
        f'<div class="goal">{story["description"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _story_section(
        "Context → Evidence → Takeaway",
        "Each story is structured for presentation flow: context first, supporting visuals second, decision takeaway last.",
        label="Flow",
    )

    model_tables = load_model_result_tables(RESULTS_DIR)
    prediction_tables = load_target_tables(RESULTS_DIR, "predictions_")

    # ── Story 5: Country Inequality ──
    if story["id"] == "Story 5":
        countries = sorted(df["country"].dropna().astype(str).unique().tolist())
        country = st.selectbox("Country Focus", countries, index=0, key="story5_country")
        country_df = df[df["country"] == country].sort_values("year")
        if country_df.empty:
            st.warning("No records found for selected country.")
            return

        _story_section(
            "Country Snapshot",
            f"Focus on {country} to compare inequality and growth indicators against the current ASEAN average.",
            label="Context",
        )
        latest = country_df.iloc[-1]
        asean_latest = df[df["year"] == df["year"].max()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Gini", f"{latest.get('gini_index', np.nan):.2f}")
        c2.metric("GDP per Capita", f"${latest.get('gdp_per_capita', np.nan):,.0f}")
        gini_diff = np.nan
        if "gini_index" in asean_latest.columns:
            gini_diff = latest["gini_index"] - asean_latest["gini_index"].mean()
            c3.metric("Gini vs ASEAN Avg", f"{gini_diff:+.2f}", delta=f"{gini_diff:+.2f}", delta_color="inverse")

        if not pd.isna(gini_diff):
            tone = "risk" if gini_diff > 0 else "positive"
            direction = "above" if gini_diff > 0 else "below"
            _story_takeaway(
                f"{country}'s latest inequality level is {abs(float(gini_diff)):.2f} points {direction} the ASEAN average.",
                tone=tone,
            )

        _story_section(
            "Indicator Trends",
            "Read all three charts together to assess inequality trajectory, economic capacity, and trade posture.",
            label="Evidence",
        )
        for col_name, label in [("gini_index", "Inequality (Gini Index)"), ("gdp_per_capita", "GDP per Capita"), ("trade_percent_gdp", "Trade Openness (% GDP)")]:
            if col_name not in country_df.columns:
                continue
            st.markdown(f"##### {label}")
            st.caption("Trend by year for the selected country.")
            chart_df = country_df[["year", col_name]].dropna()
            chart = (
                alt.Chart(chart_df)
                .mark_area(
                    line={"color": COLORS["secondary"]},
                    color=alt.Gradient(gradient="linear", stops=[
                        alt.GradientStop(color=COLORS["secondary"], offset=0),
                        alt.GradientStop(color="rgba(46,134,171,0.05)", offset=1),
                    ], x1=1, x2=1, y1=1, y2=0),
                )
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y(f"{col_name}:Q", title=label),
                    tooltip=["year:O", alt.Tooltip(f"{col_name}:Q", format=",.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width="stretch")

        _story_section(
            "Decision Layer",
            "Action prioritization is now available through Story 6 and the Policy Recommendations tab.",
            label="Takeaway",
        )
        st.markdown(
            '<div class="story-placeholder"><strong>Next step:</strong> Open Story 6 for ranked policy bundles and supporting dimension evidence.</div>',
            unsafe_allow_html=True,
        )

    # ── Story 1: ASEAN-Wide Inequality Monitoring ──
    elif story["id"] == "Story 1":
        if "gini_index" not in df.columns:
            st.warning("`gini_index` is required for this story.")
            return
        _story_section(
            "Regional Inequality Monitoring",
            "Track both central tendency and spread to understand whether inequality is converging or diverging across countries.",
            label="Context",
        )
        trend = (
            df.groupby("year", as_index=False)["gini_index"]
            .agg(mean_gini="mean", median_gini="median")
            .sort_values("year")
        )
        gap = (
            df.groupby("year")["gini_index"]
            .quantile([0.9, 0.1])
            .unstack()
            .reset_index()
            .rename(columns={0.9: "p90", 0.1: "p10"})
        )
        gap["gap_p90_p10"] = gap["p90"] - gap["p10"]
        merged = trend.merge(gap[["year", "gap_p90_p10"]], on="year", how="left")

        if not merged.empty and merged["gap_p90_p10"].notna().any():
            gap_change = float(merged["gap_p90_p10"].iloc[-1] - merged["gap_p90_p10"].iloc[0])
            trend_tone = "risk" if gap_change > 0 else "positive"
            trend_dir = "widened" if gap_change > 0 else "narrowed"
            _story_takeaway(
                f"The p90-p10 inequality spread has {trend_dir} by {abs(gap_change):.3f} points over the observed period.",
                tone=trend_tone,
            )

        chart_data = merged.melt("year", var_name="Series", value_name="Value")
        _story_section(
            "ASEAN Trend View",
            "Mean, median, and spread (p90-p10 gap) provide a compact regional trajectory view.",
            label="Evidence",
        )
        trend_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("Value:Q", title="Gini Metric"),
                color=alt.Color("Series:N", title=""),
                strokeDash=alt.StrokeDash("Series:N"),
                tooltip=["year:O", "Series:N", alt.Tooltip("Value:Q", format=".3f")],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(trend_chart, width="stretch")

        latest_year = int(df["year"].max())
        _story_section(
            "Latest-Year Country Ranking",
            f"Cross-sectional ordering for {latest_year} highlights countries with comparatively higher inequality levels.",
            label="Evidence",
        )
        latest_df = df[df["year"] == latest_year][["country", "gini_index"]].sort_values("gini_index", ascending=False)
        bar = (
            alt.Chart(latest_df)
            .mark_bar()
            .encode(
                x=alt.X("gini_index:Q", title=f"Gini Index ({latest_year})"),
                y=alt.Y("country:N", sort="-x", title=""),
                color=alt.Color("gini_index:Q", scale=_risk_scale(), legend=None),
                tooltip=["country:N", alt.Tooltip("gini_index:Q", format=".3f")],
            )
            .properties(height=max(250, len(latest_df) * 24))
        )
        st.altair_chart(bar, width="stretch")

    # ── Story 2: Gini + Trade Correlation ──
    elif story["id"] == "Story 2":
        _story_section(
            "Trade and Inequality Signal",
            "Assess whether trade openness co-moves with inequality and where model limits remain.",
            label="Context",
        )
        gini_results = model_tables.get("gini_index")
        if gini_results is not None:
            st.markdown("#### Gini Model Performance Snapshot")
            gini_sorted = gini_results.sort_values("rmse")
            st.dataframe(gini_sorted, width="stretch", hide_index=True)
            if not gini_sorted.empty:
                gini_best = gini_sorted.iloc[0]
                gini_r2 = float(gini_best.get("r2", np.nan))
                tone = "risk" if pd.notna(gini_r2) and gini_r2 < 0.20 else "neutral"
                _story_takeaway(
                    f"Best current Gini model: {str(gini_best['model']).replace('_', ' ').title()} "
                    f"(R\u00b2 {gini_r2:.3f}, RMSE {float(gini_best['rmse']):.3f}).",
                    tone=tone,
                )
        else:
            st.warning("Run predictive models to populate Gini model results.")

        if {"trade_percent_gdp", "gini_index"}.issubset(df.columns):
            st.markdown("#### Trade Openness vs. Inequality")
            st.caption("Each dot is a country-year. The red dashed line is a linear regression fit.")
            sampled = df[["country", "year", "trade_percent_gdp", "gini_index"]].dropna()
            scatter = (
                alt.Chart(sampled)
                .mark_circle(size=62, opacity=0.5, color=COLORS["secondary"])
                .encode(
                    x=alt.X("trade_percent_gdp:Q", title="Trade (% GDP)"),
                    y=alt.Y("gini_index:Q", title="Gini Index"),
                    tooltip=["country:N", "year:Q",
                             alt.Tooltip("trade_percent_gdp:Q", format=",.1f"),
                             alt.Tooltip("gini_index:Q", format=".3f")],
                )
            )
            trend_line = scatter.transform_regression("trade_percent_gdp", "gini_index").mark_line(
                color=COLORS["danger"], strokeDash=[6, 4]
            )
            st.altair_chart((scatter + trend_line).interactive().properties(height=400), width="stretch")

        preds = prediction_tables.get("gini_index")
        if preds is not None and not preds.empty:
            _story_section(
                "Model Behavior Over Time",
                "Compare predicted and actual inequality trajectories to spot bias and drift across years.",
                label="Evidence",
            )
            st.markdown("#### Predicted vs. Actual Gini Over Time")
            best_model = (
                gini_results.sort_values("rmse").iloc[0]["model"]
                if gini_results is not None and not gini_results.empty
                else preds["model"].iloc[0]
            )
            p = preds[preds["model"] == best_model]
            chart_data = p.melt(id_vars=["country", "year", "model"], value_vars=["actual", "predicted"], var_name="Series", value_name="Gini")
            line = (
                alt.Chart(chart_data)
                .mark_line()
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("Gini:Q", title="Gini Index"),
                    color=alt.Color("Series:N"),
                    detail="country:N",
                    tooltip=["country:N", "year:O", "Series:N", alt.Tooltip("Gini:Q", format=".3f")],
                )
                .properties(height=380)
            )
            st.altair_chart(line.interactive(), width="stretch")

    # ── Story 3: GDP Competitiveness ──
    elif story["id"] == "Story 3":
        _story_section(
            "GDP Growth Competitiveness",
            "Evaluate which GDP target formulation is modeled more reliably, then inspect where country-level errors concentrate.",
            label="Context",
        )
        target_choice = st.radio(
            "GDP Target View",
            options=["gdp_per_capita", "log_gdp_per_capita"],
            format_func=_pretty,
            horizontal=True,
            key="story3_target",
        )
        target_results = model_tables.get(target_choice)
        if target_results is not None:
            st.markdown(f"#### {_pretty(target_choice)} Model Performance")
            target_sorted = target_results.sort_values("rmse")
            st.dataframe(target_sorted, width="stretch", hide_index=True)
            if not target_sorted.empty:
                best = target_sorted.iloc[0]
                best_r2 = float(best.get("r2", np.nan))
                tone = "positive" if pd.notna(best_r2) and best_r2 >= 0.75 else "neutral"
                _story_takeaway(
                    f"Best {_pretty(target_choice)} model: {str(best['model']).replace('_', ' ').title()} "
                    f"(R\u00b2 {best_r2:.3f}, RMSE {float(best['rmse']):,.2f}).",
                    tone=tone,
                )
        else:
            st.warning(f"No predictive table found for {_pretty(target_choice)}.")
            return

        preds = prediction_tables.get(target_choice)
        if preds is None or preds.empty:
            st.info("Prediction point outputs are missing for this target.")
            return
        best_model = target_results.sort_values("rmse").iloc[0]["model"]
        p = preds[preds["model"] == best_model].copy()

        _story_section(
            "ASEAN Average Trajectory",
            "Check whether predicted macro trend follows the observed direction and turning points.",
            label="Evidence",
        )
        st.markdown("#### ASEAN-Average Actual vs. Predicted Over Time")
        years_view = p.groupby("year", as_index=False)[["actual", "predicted"]].mean().sort_values("year")
        chart_data = years_view.melt("year", var_name="Series", value_name="Value")
        chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("Value:Q", title=_pretty(target_choice)),
                color=alt.Color("Series:N"),
                tooltip=["year:O", "Series:N", alt.Tooltip("Value:Q", format=",.2f")],
            )
            .properties(height=360)
            .interactive()
        )
        st.altair_chart(chart, width="stretch")

        _story_section(
            "Country-Level Error Concentration",
            "Higher bars indicate where the selected model is less stable and needs country-specific diagnostics.",
            label="Evidence",
        )
        st.markdown("#### Country-Level Prediction Error")
        st.caption("Countries with higher mean absolute error may have unusual economic dynamics worth investigating.")
        country_err = (
            p.assign(abs_error=lambda x: (x["actual"] - x["predicted"]).abs())
            .groupby("country", as_index=False)["abs_error"]
            .mean()
            .sort_values("abs_error", ascending=False)
        )
        err_chart = (
            alt.Chart(country_err)
            .mark_bar()
            .encode(
                x=alt.X("abs_error:Q", title="Mean Absolute Error"),
                y=alt.Y("country:N", sort="-x", title=""),
                color=alt.Color("abs_error:Q", scale=_risk_scale(), legend=None),
                tooltip=["country:N", alt.Tooltip("abs_error:Q", format=",.2f")],
            )
            .properties(height=max(250, len(country_err) * 24))
        )
        st.altair_chart(err_chart, width="stretch")

    # ── Story 6: Ranked Policy Priorities ──
    elif story["id"] == "Story 6":
        _story_section(
            "Ranked Policy Priorities",
            "Country tiers and policy bundles are now generated by the recommendation engine.",
            label="Evidence",
        )
        _render_policy_recommendations(show_header=False, key_prefix="story6")

    # ── Story 4: Predictive Reform Impact ──
    elif story["id"] == "Story 4":
        _story_section(
            "Predictive Reform Impact",
            "Use model evidence to test how changes in key predictors are associated with GDP and inequality outcomes.",
            label="Context",
        )
        candidate_targets = [t for t in ["gdp_per_capita", "log_gdp_per_capita", "gini_index"] if t in model_tables]
        if not candidate_targets:
            st.info("No predictive target tables available for Story 4 yet.")
            return

        story4_target = st.selectbox(
            "Outcome to Analyze",
            options=candidate_targets,
            format_func=_pretty,
            key="story4_target",
        )
        story4_results = model_tables.get(story4_target)
        if story4_results is None or story4_results.empty:
            st.info("Model performance table missing for the selected Story 4 target.")
            return

        story4_ranked = story4_results.sort_values("rmse")
        st.markdown(f"#### {_pretty(story4_target)} Model Performance")
        st.dataframe(story4_ranked, width="stretch", hide_index=True)

        best_story4 = story4_ranked.iloc[0]
        _story_takeaway(
            f"Best current {_pretty(story4_target)} model is {str(best_story4['model']).replace('_', ' ').title()} "
            f"(R\u00b2 {float(best_story4['r2']):.3f}, RMSE {float(best_story4['rmse']):,.2f}).",
            tone="neutral",
        )

        importance_df = load_target_tables(RESULTS_DIR, "feature_importance_").get(story4_target)
        if importance_df is not None and not importance_df.empty and "feature" in importance_df.columns:
            predictor_options = importance_df["feature"].dropna().astype(str).tolist()
        else:
            predictor_options = [
                c for c in df.select_dtypes(include="number").columns
                if c not in {"year", story4_target}
            ]

        predictor_options = [p for p in predictor_options if p in df.columns]
        predictor_options = sorted(set(predictor_options))
        if not predictor_options:
            st.info("No predictor columns available for Story 4 relationship view.")
            return

        predictor = st.selectbox(
            "Predictor to Stress-Test",
            options=predictor_options,
            format_func=_pretty,
            key="story4_predictor",
        )

        rel_df = df[["country", "year", predictor, story4_target]].dropna().copy()
        if rel_df.empty:
            st.info("No non-null rows available for this predictor/outcome pair.")
            return

        _story_section(
            "Predictor-Outcome Relationship",
            "This chart uses observed data and a fitted trend line to show directional association.",
            label="Evidence",
        )
        scatter = (
            alt.Chart(rel_df)
            .mark_circle(size=60, opacity=0.6, color=COLORS["secondary"])
            .encode(
                x=alt.X(f"{predictor}:Q", title=_pretty(predictor)),
                y=alt.Y(f"{story4_target}:Q", title=_pretty(story4_target)),
                tooltip=[
                    "country:N",
                    "year:Q",
                    alt.Tooltip(f"{predictor}:Q", format=",.3f"),
                    alt.Tooltip(f"{story4_target}:Q", format=",.3f"),
                ],
            )
            .properties(height=360)
        )
        trend_line = scatter.transform_regression(predictor, story4_target).mark_line(
            color=COLORS["danger"], strokeDash=[6, 4]
        )
        st.altair_chart((scatter + trend_line).interactive(), width="stretch")

        _story_section(
            "Sensitivity Preview (Directional)",
            "Applies a simple linear sensitivity lens to estimate how predictor shifts may move the selected outcome.",
            label="Takeaway",
        )
        shock_pct = st.slider(
            "Predictor Change (%)",
            min_value=-20,
            max_value=20,
            value=10,
            step=1,
            key="story4_shock",
        )
        x_vals = rel_df[predictor].to_numpy()
        y_vals = rel_df[story4_target].to_numpy()
        if len(rel_df) >= 3 and np.nanstd(x_vals) > 0:
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            baseline_x = float(np.nanmean(x_vals))
            baseline_y = float(slope * baseline_x + intercept)
            shocked_x = baseline_x * (1.0 + shock_pct / 100.0)
            shocked_y = float(slope * shocked_x + intercept)
            delta_y = shocked_y - baseline_y

            m1, m2, m3 = st.columns(3)
            m1.metric("Baseline Predicted Outcome", f"{baseline_y:,.3f}")
            m2.metric("Shocked Predicted Outcome", f"{shocked_y:,.3f}")
            m3.metric("Directional Change", f"{delta_y:+,.3f}")
            st.caption(
                "Sensitivity preview is a directional linear approximation from observed data, not a causal estimate."
            )
        else:
            st.info("Insufficient variation to compute directional sensitivity for this predictor.")


# ── Footer ──────────────────────────────────────────────────────────────────

def _render_footer() -> None:
    st.markdown(
        '<div class="footer">'
        "ASEAN Policy Dashboard &middot; Built with Streamlit, Altair, scikit-learn, and linearmodels &middot; "
        "Data sourced from World Bank, WHO, and UN indicators"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="ASEAN Policy Dashboard",
        page_icon="\U0001F30F",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _register_altair_theme()

    if not DEFAULT_PANEL_PATH.exists():
        st.error("Final panel dataset not found. Run `python scripts/run_pipeline.py --stage data` first.")
        return

    df = load_panel(DEFAULT_PANEL_PATH)

    if "country" not in df.columns or "year" not in df.columns:
        st.error("Dataset is missing required columns: country, year.")
        return

    _render_sidebar(df)
    _render_header()

    tabs = st.tabs([
        "Intro",
        "Executive Summary",
        "Story Mode",
        "Data Explorer",
        "Descriptive Analytics",
        "Econometric Results",
        "Predictive Analytics",
        "Policy Recommendations",
        "What-If Simulation",
    ])

    with tabs[0]:
        _render_intro(df)
    with tabs[1]:
        _render_executive_summary(df)
    with tabs[2]:
        _render_story_mode(df)
    with tabs[3]:
        _render_data_explorer(df)
    with tabs[4]:
        _render_descriptive_analytics(df)
    with tabs[5]:
        _render_econometric_results()
    with tabs[6]:
        _render_predictive_analytics(df)
    with tabs[7]:
        _render_policy_recommendations()
    with tabs[8]:
        _render_simulation(df)

    _render_footer()


if __name__ == "__main__":
    main()
