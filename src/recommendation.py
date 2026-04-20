from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import matplotlib
import numpy as np
import pandas as pd

from src.preprocessing import FINAL_PANEL_PATH, normalize_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

PANEL_CANDIDATES = (
    PROJECT_ROOT / "data" / "panel" / "Final_Panel_Dataset_with_DTM_ETM.csv",
    FINAL_PANEL_PATH,
)

# Keep legacy names used by the original prescriptive script.
COLUMN_ALIASES = {
    "top_10percent_income_share_percent": "top_10pct_income_share_pct",
    "bottom_10percent_income_share_percent": "bottom_10pct_income_share_pct",
}

DIMENSION_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "inequality": {
        "gini_index": -0.40,
        "bottom_10pct_income_share_pct": 0.35,
        "top_10pct_income_share_pct": -0.25,
    },
    "health": {
        "life_expectancy": 0.40,
        "infant_mortality": -0.35,
        "infectious_disease_rate": -0.15,
        "noncommunicable_disease_rate": -0.10,
    },
    "demographics": {
        "crude_birth_rate": -0.20,
        "crude_death_rate": -0.30,
        "natural_increase_rate": 0.25,
        "pop_growth": 0.25,
    },
    "economic": {
        "gdp_per_capita": 0.45,
        "average_income_usd": 0.35,
        "trade_percent_gdp": 0.20,
    },
}

RECOMMENDATIONS = {
    (2, 1): [
        "Accelerate public health investment to reduce infant mortality and infectious disease.",
        "Expand access to primary education to drive demographic transition.",
        "Introduce conditional cash-transfer programmes to address extreme inequality.",
    ],
    (2, 2): [
        "Strengthen maternal health services to reduce birth rates sustainably.",
        "Invest in vocational training to build a productive workforce.",
        "Diversify trade partnerships to reduce commodity dependence.",
    ],
    (3, 2): [
        "Scale preventive healthcare infrastructure to shift disease burden.",
        "Introduce progressive taxation to narrow income inequality (Gini >0.5).",
        "Target FDI in manufacturing to accelerate GDP per-capita growth.",
    ],
    (3, 5): [
        "Redirect fiscal space from population growth management to ageing preparedness.",
        "Strengthen non-communicable disease programmes (NCDs are the primary burden).",
        "Invest in innovation and R&D for sustained economic competitiveness.",
    ],
    (4, 5): [
        "Prioritise pension and elderly-care systems given advanced demographic stage.",
        "Monitor and manage inequality as economic maturity can widen income gaps.",
        "Lead on climate-aligned trade and sustainable development policies.",
    ],
}

TIER_COLORS = {
    "Tier 1 - Sustain & Lead": "#2ecc71",
    "Tier 2 - Optimise": "#3498db",
    "Tier 3 - Transition": "#f39c12",
    "Tier 4 - Critical Intervention": "#e74c3c",
}


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_panel_path(panel_path: Optional[Path] = None) -> Path:
    if panel_path is not None:
        path = Path(panel_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Panel dataset not found: {path}")

    for candidate in PANEL_CANDIDATES:
        if candidate.exists():
            return candidate

    joined = ", ".join(str(p) for p in PANEL_CANDIDATES)
    raise FileNotFoundError(f"Panel dataset not found in expected locations: {joined}")


def load_prescriptive_panel(panel_path: Optional[Path] = None) -> pd.DataFrame:
    resolved = resolve_panel_path(panel_path)
    df = normalize_columns(pd.read_csv(resolved))

    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "trade_percent_gdp" in df.columns:
        df["trade_percent_gdp"] = df.groupby("country")["trade_percent_gdp"].transform(
            lambda s: s.fillna(s.median())
        )
        df["trade_percent_gdp"] = df["trade_percent_gdp"].fillna(df["trade_percent_gdp"].median())

    return df


def assign_tier(score: float) -> str:
    if score >= 65:
        return "Tier 1 - Sustain & Lead"
    if score >= 45:
        return "Tier 2 - Optimise"
    if score >= 25:
        return "Tier 3 - Transition"
    return "Tier 4 - Critical Intervention"


def get_recommendations(dtm_stage: float, etm_stage: float) -> List[str]:
    if pd.isna(dtm_stage) or pd.isna(etm_stage):
        return [
            "Monitor key indicators and benchmark against peer countries.",
            "Conduct detailed structural diagnostics.",
            "Engage international partners for tailored advisory support.",
        ]

    key = (int(round(float(dtm_stage))), int(round(float(etm_stage))))
    return RECOMMENDATIONS.get(
        key,
        [
            "Monitor key indicators and benchmark against peer countries.",
            "Conduct detailed structural diagnostics.",
            "Engage international partners for tailored advisory support.",
        ],
    )


def score_panel(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    numeric_cols = scored.select_dtypes(include=[np.number]).columns.difference(
        ["year", "dtm_stage", "etm_stage"]
    )

    df_norm = scored.copy()
    for col in numeric_cols:
        mn, mx = scored[col].min(), scored[col].max()
        df_norm[col] = (scored[col] - mn) / (mx - mn + 1e-9)

    for dim, indicators in DIMENSION_WEIGHTS.items():
        available = [(col, weight) for col, weight in indicators.items() if col in df_norm.columns]
        if not available:
            scored[f"score_{dim}"] = np.nan
            continue
        score = sum(df_norm[col] * weight for col, weight in available)
        scored[f"score_{dim}"] = (score - score.min()) / (score.max() - score.min() + 1e-9) * 100

    score_cols = [f"score_{dim}" for dim in DIMENSION_WEIGHTS if f"score_{dim}" in scored.columns]
    if not score_cols:
        raise ValueError("No score columns were computed. Check panel schema.")

    scored["composite_score"] = scored[score_cols].mean(axis=1)
    scored["recommendation_tier"] = scored["composite_score"].apply(assign_tier)
    return scored


def _latest_snapshot(scored: pd.DataFrame) -> pd.DataFrame:
    latest_year = int(scored["year"].max())
    latest = scored[scored["year"] == latest_year].copy()
    latest = latest.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return latest


def _build_ranked_table(latest: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in latest.iterrows():
        policy_actions = get_recommendations(row.get("dtm_stage", np.nan), row.get("etm_stage", np.nan))
        rows.append(
            {
                "country": row.get("country"),
                "year": int(row.get("year", np.nan)),
                "income_group": row.get("income_group"),
                "dtm_stage": row.get("dtm_stage"),
                "etm_stage": row.get("etm_stage"),
                "score_inequality": round(float(row.get("score_inequality", np.nan)), 2),
                "score_health": round(float(row.get("score_health", np.nan)), 2),
                "score_demographics": round(float(row.get("score_demographics", np.nan)), 2),
                "score_economic": round(float(row.get("score_economic", np.nan)), 2),
                "composite_score": round(float(row.get("composite_score", np.nan)), 2),
                "recommendation_tier": row.get("recommendation_tier"),
                "policy_1": policy_actions[0],
                "policy_2": policy_actions[1],
                "policy_3": policy_actions[2],
                "policy_bundle": " | ".join(policy_actions),
            }
        )
    return pd.DataFrame(rows)


def _build_evidence_table(latest: pd.DataFrame) -> pd.DataFrame:
    dimensions = list(DIMENSION_WEIGHTS.keys())
    long_rows: List[Dict[str, object]] = []
    for _, row in latest.iterrows():
        for dim in dimensions:
            score_col = f"score_{dim}"
            value = float(row.get(score_col, np.nan))
            long_rows.append(
                {
                    "country": row.get("country"),
                    "year": int(row.get("year", np.nan)),
                    "dimension": dim,
                    "dimension_score": round(value, 2),
                    "recommendation_tier": row.get("recommendation_tier"),
                    "composite_score": round(float(row.get("composite_score", np.nan)), 2),
                }
            )

    evidence = pd.DataFrame(long_rows)
    if evidence.empty:
        return evidence
    evidence["dimension_rank"] = evidence.groupby("dimension")["dimension_score"].rank(
        ascending=False, method="min"
    )
    return evidence.sort_values(["dimension", "dimension_rank", "country"]).reset_index(drop=True)


def _save_recommendation_plot(latest: pd.DataFrame, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = [TIER_COLORS.get(tier, "#95a5a6") for tier in latest["recommendation_tier"]]
    bars = ax.barh(latest["country"], latest["composite_score"], color=palette, edgecolor="none")
    ax.set_title("Story 5 - Composite Recommendation Score by Country")
    ax.set_xlabel("Composite Score (0-100)")
    ax.set_ylabel("Country")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, 100)
    for bar, value in zip(bars, latest["composite_score"]):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_recommendation_engine(
    panel_path: Optional[Path] = None,
    ranked_output: Path = RESULTS_DIR / "policy_recommendations_ranked.csv",
    evidence_output: Path = RESULTS_DIR / "policy_recommendation_evidence.csv",
    plot_output: Path = PLOTS_DIR / "policy_recommendation_scores.png",
) -> Dict[str, object]:
    ensure_output_dirs()
    panel_df = load_prescriptive_panel(panel_path=panel_path)
    scored = score_panel(panel_df)
    latest = _latest_snapshot(scored)
    ranked = _build_ranked_table(latest)
    evidence = _build_evidence_table(latest)

    ranked_output.parent.mkdir(parents=True, exist_ok=True)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(ranked_output, index=False)
    evidence.to_csv(evidence_output, index=False)
    _save_recommendation_plot(latest, plot_output)

    return {
        "ran": True,
        "latest_year": int(latest["year"].max()) if not latest.empty else None,
        "panel_path": resolve_panel_path(panel_path),
        "ranked_recommendations": ranked_output,
        "recommendation_evidence": evidence_output,
        "recommendation_plot": plot_output,
    }


__all__ = [
    "DIMENSION_WEIGHTS",
    "PLOTS_DIR",
    "RESULTS_DIR",
    "assign_tier",
    "ensure_output_dirs",
    "get_recommendations",
    "load_prescriptive_panel",
    "resolve_panel_path",
    "run_recommendation_engine",
    "score_panel",
]
