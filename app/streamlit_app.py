"""
ASEAN Policy Dashboard — Streamlit Application

Clean, professional, interactive dashboard for ASEAN socioeconomic analysis.
Provides descriptive, predictive, and prescriptive analytics with contextual
explanations accessible to both technical and non-technical audiences.
"""

from __future__ import annotations

import os
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
        "title": "Country Inequality Reduction for Sustainable Growth",
        "persona": "Government member of an ASEAN country",
        "goal": "Identify policies to reduce inequality while sustaining economic development.",
        "description": "As a member of the government for one of ASEAN\u2019s member countries, I want to identify policies to reduce inequality so that we can continue to develop economically.",
    },
    {
        "id": "Story 2",
        "type": "Descriptive",
        "title": "Country-Level Socioeconomic Comparison for ASEAN Progress",
        "persona": "ASEAN Economic Community Department",
        "goal": "Summarize inequality and key socioeconomic indicators by country to compare trends in support of ASEAN’s goal of accelerating economic growth, social progress, and cultural development in the region.",
        "description": "As a member of ASEAN’s Economic Community Department, I want to summarize inequality and key socioeconomic indicators by country to compare trends in an effort to support ASEAN’s goal of accelerating “the economic growth, social progress and cultural development in the region”.",
    },
    {
        "id": "Story 3",
        "type": "Predictive",
        "title": "Predict Gini with Trade-Relevant Signals",
        "persona": "ASEAN Trade Facilitation Division",
        "goal": "Understand predictive relationship between inequality and trade for policy tuning.",
        "description": "As a member of ASEAN’s Trade Facilitation Division, I want to predict how the Gini index is correlated with international trade so that we can identify what policies need to be modified or implemented.",
    },
    {
        "id": "Story 4",
        "type": "Predictive",
        "title": "Predict GDP Growth Competitiveness Signals",
        "persona": "ASEAN country government",
        "goal": "Predict GDP growth pathways to improve competitiveness.",
        "description": "As a member of the government of one of ASEAN’s member countries, I want to predict how to grow GDP across so that our country can grow economically and compete alongside larger countries.",
    },
    {
        "id": "Story 5",
        "type": "Prescriptive",
        "title": "Ranked Policy Recommendations",
        "persona": "ASEAN Economic Research policy advisor",
        "goal": "Prioritize high-impact reforms using modeled relationships and current risk conditions.",
        "description": "As a policy advisor within ASEAN’s Economic Research Institute, I want to receive ranked policy recommendations based on modeled relationships between inequality, trade, governance, and demographic stage so that we can prioritize the most impactful policy reforms for member countries.",
    },
    {
        "id": "Story 6",
        "type": "Prescriptive",
        "title": "What-If Reform Simulation",
        "persona": "ASEAN country economic planner",
        "goal": "Simulate reforms in trade, governance proxies, and demographics before implementation.",
        "description": "As an economic planner in an ASEAN member country, I want to simulate how changes in trade openness, governance, or demographic indicators would affect inequality and GDP outcomes so that we can implement data-driven reforms with economic impact.",
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
}}
[data-testid="stMetric"] label {{
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    font-weight: 700;
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
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid {COLORS["accent"]};
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
}}
.story-card .story-type {{
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}}
.story-type-descriptive {{ background: rgba(56,178,172,0.15); color: {COLORS["secondary"]}; }}
.story-type-predictive {{ background: rgba(49,130,206,0.15); color: #3182CE; }}
.story-type-prescriptive {{ background: rgba(229,62,62,0.15); color: {COLORS["danger"]}; }}

.story-card h4 {{ margin: 0.3rem 0 0.2rem; color: var(--text-color); }}
.story-card .persona {{ font-size: 0.82rem; color: var(--text-color); opacity: 0.6; margin-bottom: 0.3rem; }}
.story-card .goal {{ font-size: 0.88rem; color: var(--text-color); opacity: 0.85; }}

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
            "**Story Mode** \u2014 Narrative walkthroughs tied to real policy questions.\n\n"
            "**Data Explorer** \u2014 Filter, browse, and download the panel dataset.\n\n"
            "**Descriptive** \u2014 Cross-country comparisons and trend analysis.\n\n"
            "**Predictive** \u2014 ML model performance and feature importance.\n\n"
            "**Econometric** \u2014 OLS and Panel OLS regression results.\n\n"
            "**Simulation** \u2014 Placeholder tab (implementation in progress).\n\n"
            "**Recommendations** \u2014 Placeholder tab (implementation in progress)."
        )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.75rem; color:#CBD5E0; opacity:0.85;">'
            "Built for academic evaluation and portfolio demonstration. "
            "Data sourced from World Bank, WHO, and UN indicators."
            "</div>",
            unsafe_allow_html=True,
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

    display_columns = st.multiselect(
        "Columns to Display",
        options=list(filtered.columns),
        default=["country", "year", "gini_index", "life_expectancy", "gdp_per_capita"],
    )
    if display_columns:
        st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)
    else:
        st.dataframe(filtered, use_container_width=True, hide_index=True)

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
        st.altair_chart(line, use_container_width=True)


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

    st.markdown("")  # spacer

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
        st.altair_chart(top_chart, use_container_width=True)
    with c2:
        st.markdown(f"**Bottom 10 \u2014 {_pretty(metric)}**")
        bottom_df = comp_df.tail(10).copy()
        bottom_chart = (
            alt.Chart(bottom_df)
            .mark_bar()
            .encode(
                x=alt.X(f"{metric}:Q", title=_pretty(metric)),
                y=alt.Y("country:N", sort="x", title=""),
                color=alt.value(COLORS["accent"]),
                tooltip=["country:N", alt.Tooltip(f"{metric}:Q", format=",.2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(bottom_chart, use_container_width=True)

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
    st.altair_chart(trend_chart, use_container_width=True)

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
    st.altair_chart(heatmap + text, use_container_width=True)


# ── Tab: Predictive Analytics ───────────────────────────────────────────────

def _render_predictive_analytics() -> None:
    st.header("Predictive Analytics")
    _section_intro(
        "Multiple machine-learning models (Linear, Ridge, Lasso, Gradient Boosting, Random Forest) "
        "were trained on the panel dataset for each target variable. This section lets you compare "
        "model accuracy, inspect predictions vs. actuals, review residual distributions, and "
        "identify which features matter most."
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

    # Full results table
    with st.expander("Full Model Comparison Table", expanded=False):
        st.dataframe(results_df, use_container_width=True, hide_index=True)

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
    st.altair_chart(metric_chart, use_container_width=True)

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
            st.altair_chart((scatter + diagonal).interactive().properties(height=420), use_container_width=True)

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
            st.altair_chart(residual_chart, use_container_width=True)

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
                    st.dataframe(country_perf_df, use_container_width=True, hide_index=True)
                perf_chart = (
                    alt.Chart(country_perf_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("rmse:Q", title="RMSE"),
                        y=alt.Y("country:N", sort="-x", title=""),
                        color=alt.Color("rmse:Q", scale=alt.Scale(scheme="orangered"), legend=None),
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
                st.altair_chart(perf_chart, use_container_width=True)
    else:
        st.info(
            "Prediction point data not available. Run "
            "`python scripts/run_pipeline.py --stage models --skip-ols --skip-fixed-effects` "
            "to generate `predictions_*.csv` files."
        )

    # Feature importance
    importance_df = importance_tables.get(target)
    if importance_df is not None and not importance_df.empty:
        st.markdown("#### Feature Importance (Random Forest)")
        st.caption("Shows which variables contribute most to predictions. Higher importance = stronger influence on the target.")
        top_n = st.slider("Top Features", min_value=5, max_value=min(20, len(importance_df)), value=min(10, len(importance_df)), step=1, key="pred_top_features")
        top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
        imp_chart = (
            alt.Chart(top_features)
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=""),
                color=alt.Color("importance:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
                tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f")],
            )
            .properties(height=max(250, top_n * 28))
        )
        st.altair_chart(imp_chart, use_container_width=True)

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
        st.altair_chart(country_chart, use_container_width=True)

    # Legacy static plots
    with st.expander("Static PNG Plots (Legacy)", expanded=False):
        plot_specs = [
            ("actual_vs_pred", "Actual vs Predicted"),
            ("model_comparison", "Model Comparison"),
            ("feature_importance", "Feature Importance"),
            ("country", "Country Comparison"),
            ("vs_mean", "Model vs Mean Baseline"),
        ]
        for i in range(0, len(plot_specs), 2):
            cols = st.columns(2)
            for j, (suffix, title) in enumerate(plot_specs[i: i + 2]):
                img_path = PLOTS_DIR / f"{target}_{suffix}.png"
                with cols[j]:
                    if img_path.exists():
                        st.image(str(img_path), caption=title, use_container_width=True)
                    else:
                        st.info(f"Missing: {img_path.name}")

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
        st.dataframe(pd.DataFrame(summary_rows).sort_values("Target"), use_container_width=True, hide_index=True)


# ── Tab: Econometric Results ────────────────────────────────────────────────

def _render_econometric_results() -> None:
    st.header("Econometric Results")
    _section_intro(
        "Classical econometric analysis using Ordinary Least Squares (OLS) and Panel OLS "
        "with country fixed effects. Fixed effects control for time-invariant differences "
        "between countries, isolating the within-country relationships between variables. "
        "VIF diagnostics check for multicollinearity among predictors."
    )
    _glossary_expander(["VIF", "R\u00b2"])

    ols_path = RESULTS_DIR / "ols_summary.txt"
    fe_path = RESULTS_DIR / "fixed_effects_summary.txt"
    vif_path = RESULTS_DIR / "fixed_effects_vif.csv"
    cleaned_path = MODELING_CLEANED_PATH

    if cleaned_path.exists():
        cleaned_df = pd.read_csv(cleaned_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Countries", int(cleaned_df["country"].nunique()))
        c2.metric("Years", int(cleaned_df["year"].nunique()))
        c3.metric("Observations", f"{len(cleaned_df):,}")

    if ols_path.exists():
        with st.expander("OLS Regression Summary", expanded=False):
            st.caption("Pooled OLS treats all observations as independent\u200a\u2014\u200aa useful baseline but does not account for country-level heterogeneity.")
            st.code(ols_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("OLS summary not found.")

    if fe_path.exists():
        with st.expander("Panel OLS (Fixed Effects) Summary", expanded=True):
            st.caption("Fixed effects absorb time-invariant country characteristics, providing more reliable estimates of within-country relationships.")
            st.code(fe_path.read_text(encoding="utf-8"), language="text")
    else:
        st.info("Fixed effects summary not found.")

    if vif_path.exists():
        vif_df = pd.read_csv(vif_path).sort_values("vif", ascending=False)
        st.markdown("#### VIF Diagnostics")
        st.caption(
            "Variance Inflation Factor measures how much a predictor correlates with others. "
            "Values above 5 warrant attention; above 10 suggest serious multicollinearity."
        )
        col_t, col_c = st.columns([1, 2])
        with col_t:
            st.dataframe(vif_df, use_container_width=True, hide_index=True)
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
            st.altair_chart(vif_chart + rule, use_container_width=True)
    else:
        st.info("VIF file not found.")


# ── Tab: What-If Simulation ─────────────────────────────────────────────────

def _render_simulation(df: pd.DataFrame, key_prefix: str = "sim") -> None:
    st.header("What-If Simulation")
    _section_intro(
        "This section is pending implementation. "
        "TODO: complete Story 6 scenario engine with baseline-vs-scenario outcome deltas."
    )
    st.info(
        "Planned: scenario controls, predicted metric deltas, and exportable scenario summaries."
    )


# ── Tab: Policy Recommendations ─────────────────────────────────────────────

def _render_policy_recommendations() -> None:
    st.header("Policy Recommendations")
    _section_intro(
        "This section is pending implementation while the recommendation engine is being built."
    )
    st.info(
        "TODO: implement ranked recommendation logic, country-level policy actions, and theme summaries."
    )


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
        f'<span class="story-type story-type-{type_lower}">{story["type"]}</span>'
        f'<h4>{story["title"]}</h4>'
        f'<div class="persona">Persona: {story["persona"]}</div>'
        f'<div class="goal">{story["description"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    model_tables = load_model_result_tables(RESULTS_DIR)
    prediction_tables = load_target_tables(RESULTS_DIR, "predictions_")

    # ── Story 1: Country Inequality ──
    if story["id"] == "Story 1":
        countries = sorted(df["country"].dropna().astype(str).unique().tolist())
        country = st.selectbox("Country Focus", countries, index=0, key="story1_country")
        country_df = df[df["country"] == country].sort_values("year")
        if country_df.empty:
            st.warning("No records found for selected country.")
            return

        latest = country_df.iloc[-1]
        asean_latest = df[df["year"] == df["year"].max()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Gini", f"{latest.get('gini_index', np.nan):.2f}")
        c2.metric("GDP per Capita", f"${latest.get('gdp_per_capita', np.nan):,.0f}")
        if "gini_index" in asean_latest.columns:
            diff = latest["gini_index"] - asean_latest["gini_index"].mean()
            c3.metric("Gini vs ASEAN Avg", f"{diff:+.2f}", delta=f"{diff:+.2f}", delta_color="inverse")

        for col_name, label in [("gini_index", "Inequality (Gini Index)"), ("gdp_per_capita", "GDP per Capita"), ("trade_percent_gdp", "Trade Openness (% GDP)")]:
            if col_name not in country_df.columns:
                continue
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
                .properties(height=260, title=label)
            )
            st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Current Policy Recommendations")
        st.info(
            "This section is pending implementation. TODO: complete recommendation engine outputs for Story 5."
        )

    # ── Story 2: ASEAN-Wide Inequality Monitoring ──
    elif story["id"] == "Story 2":
        if "gini_index" not in df.columns:
            st.warning("`gini_index` is required for this story.")
            return
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

        chart_data = merged.melt("year", var_name="Series", value_name="Value")
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
            .properties(height=380, title="ASEAN Inequality Trends")
            .interactive()
        )
        st.altair_chart(trend_chart, use_container_width=True)

        latest_year = int(df["year"].max())
        latest_df = df[df["year"] == latest_year][["country", "gini_index"]].sort_values("gini_index", ascending=False)
        bar = (
            alt.Chart(latest_df)
            .mark_bar()
            .encode(
                x=alt.X("gini_index:Q", title=f"Gini Index ({latest_year})"),
                y=alt.Y("country:N", sort="-x", title=""),
                color=alt.Color("gini_index:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                tooltip=["country:N", alt.Tooltip("gini_index:Q", format=".3f")],
            )
            .properties(height=max(250, len(latest_df) * 24))
        )
        st.altair_chart(bar, use_container_width=True)

    # ── Story 3: Gini + Trade Prediction ──
    elif story["id"] == "Story 3":
        gini_results = model_tables.get("gini_index")
        if gini_results is not None:
            st.markdown("#### Gini Prediction Model Performance")
            st.dataframe(gini_results.sort_values("rmse"), use_container_width=True, hide_index=True)
        else:
            st.warning("Run predictive models to populate Gini model results.")

        if {"trade_percent_gdp", "gini_index"}.issubset(df.columns):
            st.markdown("#### Trade Openness vs. Inequality")
            st.caption("Each dot is a country-year. The black line is a linear regression fit.")
            sampled = df[["country", "year", "trade_percent_gdp", "gini_index"]].dropna()
            scatter = (
                alt.Chart(sampled)
                .mark_circle(size=60, opacity=0.5)
                .encode(
                    x=alt.X("trade_percent_gdp:Q", title="Trade (% GDP)"),
                    y=alt.Y("gini_index:Q", title="Gini Index"),
                    color=alt.Color("country:N", title="Country"),
                    tooltip=["country:N", "year:Q",
                             alt.Tooltip("trade_percent_gdp:Q", format=",.1f"),
                             alt.Tooltip("gini_index:Q", format=".3f")],
                )
            )
            trend_line = scatter.transform_regression("trade_percent_gdp", "gini_index").mark_line(color="black", strokeDash=[6, 4])
            st.altair_chart((scatter + trend_line).interactive().properties(height=400), use_container_width=True)

        preds = prediction_tables.get("gini_index")
        if preds is not None and not preds.empty:
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
            st.altair_chart(line.interactive(), use_container_width=True)

    # ── Story 4: GDP Competitiveness ──
    elif story["id"] == "Story 4":
        target_choice = st.radio(
            "GDP Target View",
            options=["gdp_per_capita", "log_gdp_per_capita"],
            format_func=_pretty,
            horizontal=True,
            key="story4_target",
        )
        target_results = model_tables.get(target_choice)
        if target_results is not None:
            st.markdown(f"#### {_pretty(target_choice)} Model Performance")
            st.dataframe(target_results.sort_values("rmse"), use_container_width=True, hide_index=True)
        else:
            st.warning(f"No predictive table found for {_pretty(target_choice)}.")
            return

        preds = prediction_tables.get(target_choice)
        if preds is None or preds.empty:
            st.info("Prediction point outputs are missing for this target.")
            return
        best_model = target_results.sort_values("rmse").iloc[0]["model"]
        p = preds[preds["model"] == best_model].copy()

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
        st.altair_chart(chart, use_container_width=True)

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
                color=alt.Color("abs_error:Q", scale=alt.Scale(scheme="orangered"), legend=None),
                tooltip=["country:N", alt.Tooltip("abs_error:Q", format=",.2f")],
            )
            .properties(height=max(250, len(country_err) * 24))
        )
        st.altair_chart(err_chart, use_container_width=True)

    # ── Story 5: Ranked Policy Priorities ──
    elif story["id"] == "Story 5":
        st.markdown("#### Ranked Policy Priorities")
        st.info(
            "TODO: Story 5 implementation pending. Future work will include scoring logic, "
            "country ranking, and grouped policy themes."
        )

    # ── Story 6: What-If Simulation ──
    elif story["id"] == "Story 6":
        _render_simulation(df, key_prefix="story6")


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
        "Story Mode",
        "Data Explorer",
        "Descriptive Analytics",
        "Predictive Analytics",
        "Econometric Results",
        "What-If Simulation",
        "Policy Recommendations",
    ])

    with tabs[0]:
        _render_story_mode(df)
    with tabs[1]:
        _render_data_explorer(df)
    with tabs[2]:
        _render_descriptive_analytics(df)
    with tabs[3]:
        _render_predictive_analytics()
    with tabs[4]:
        _render_econometric_results()
    with tabs[5]:
        _render_simulation(df)
    with tabs[6]:
        _render_policy_recommendations()

    _render_footer()


if __name__ == "__main__":
    main()
