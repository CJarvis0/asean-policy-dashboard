from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import matplotlib
import numpy as np
import pandas as pd

from src.recommendation import (
    DIMENSION_WEIGHTS,
    PLOTS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
    load_prescriptive_panel,
    score_panel,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_SCENARIO = "Baseline (No Change)"

SCENARIOS: Mapping[str, Mapping[str, float]] = {
    BASELINE_SCENARIO: {},
    "Scenario A - Health Push": {
        "life_expectancy": 5.0,
        "infant_mortality": -20.0,
        "infectious_disease_rate": -15.0,
    },
    "Scenario B - Inequality Reduction": {
        "gini_index": -0.08,
        "bottom_10pct_income_share_pct": 3.0,
        "average_income_usd": 15.0,
    },
    "Scenario C - Economic Acceleration": {
        "gdp_per_capita": 25.0,
        "trade_percent_gdp": 10.0,
        "average_income_usd": 20.0,
    },
    "Scenario D - Comprehensive Reform": {
        "life_expectancy": 3.0,
        "infant_mortality": -15.0,
        "gini_index": -0.05,
        "gdp_per_capita": 15.0,
        "average_income_usd": 10.0,
        "infectious_disease_rate": -10.0,
    },
}


def _apply_scenario(row: pd.Series, lever_dict: Mapping[str, float]) -> pd.Series:
    modified = row.copy()
    for col, delta in lever_dict.items():
        if col in modified.index and pd.notna(modified[col]):
            modified[col] = modified[col] + delta
    return modified


def _recompute_raw_score(row: pd.Series, baseline_df: pd.DataFrame) -> float:
    dim_scores: list[float] = []
    for _, indicators in DIMENSION_WEIGHTS.items():
        subtotal = 0.0
        used = 0
        for col, weight in indicators.items():
            if col not in baseline_df.columns or col not in row.index:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            mn, mx = baseline_df[col].min(), baseline_df[col].max()
            normalized = (value - mn) / (mx - mn + 1e-9)
            normalized = float(np.clip(normalized, 0, 1))
            subtotal += normalized * weight
            used += 1
        if used > 0:
            dim_scores.append(subtotal)

    if not dim_scores:
        return float("nan")
    return float(np.mean(dim_scores))


def _scale_raw_results(raw_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    all_raw = [score for country_scores in raw_results.values() for score in country_scores.values()]
    finite_raw = [score for score in all_raw if pd.notna(score)]
    if not finite_raw:
        return {country: {scenario: float("nan") for scenario in SCENARIOS} for country in raw_results}

    r_min = min(finite_raw)
    r_max = max(finite_raw)
    if abs(r_max - r_min) < 1e-12:
        return {country: {scenario: 50.0 for scenario in SCENARIOS} for country in raw_results}

    scaled: Dict[str, Dict[str, float]] = {}
    for country, scores in raw_results.items():
        scaled[country] = {}
        for scenario, value in scores.items():
            if pd.isna(value):
                scaled[country][scenario] = float("nan")
            else:
                scaled[country][scenario] = float(np.clip((value - r_min) / (r_max - r_min + 1e-9) * 100, 0, 100))
    return scaled


def _save_uplift_heatmap(scenario_df: pd.DataFrame, output_path: Path) -> Path:
    uplift_df = scenario_df[scenario_df["scenario"] != BASELINE_SCENARIO].copy()
    pivot = uplift_df.pivot(index="country", columns="scenario", values="uplift_vs_baseline").sort_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_title("Story 6 - Scenario Uplift vs Baseline")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Country")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(im, ax=ax, label="Score Uplift")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_average_uplift_plot(scenario_df: pd.DataFrame, output_path: Path) -> Path:
    uplift_df = scenario_df[scenario_df["scenario"] != BASELINE_SCENARIO].copy()
    order = [name for name in SCENARIOS.keys() if name != BASELINE_SCENARIO]
    averages = uplift_df.groupby("scenario", as_index=False)["uplift_vs_baseline"].mean()
    averages["scenario"] = pd.Categorical(averages["scenario"], categories=order, ordered=True)
    averages = averages.sort_values("scenario")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(averages["scenario"], averages["uplift_vs_baseline"], color="#2E86AB")
    ax.set_title("Average Scenario Uplift Across Countries")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average Uplift vs Baseline")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, averages["uplift_vs_baseline"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{val:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_simulation_engine(
    panel_path: Optional[Path] = None,
    results_output: Path = RESULTS_DIR / "scenario_results.csv",
    summary_output: Path = RESULTS_DIR / "scenario_summary.csv",
    heatmap_plot_output: Path = PLOTS_DIR / "scenario_uplift_heatmap.png",
    average_plot_output: Path = PLOTS_DIR / "scenario_average_uplift.png",
) -> Dict[str, object]:
    ensure_output_dirs()
    panel_df = load_prescriptive_panel(panel_path=panel_path)
    scored = score_panel(panel_df)

    latest_year = int(scored["year"].max())
    latest = scored[scored["year"] == latest_year].copy().sort_values("country")
    latest_by_country = latest.set_index("country")

    raw_results: Dict[str, Dict[str, float]] = {}
    for _, row in latest.iterrows():
        country = str(row["country"])
        raw_results[country] = {}
        for scenario_name, levers in SCENARIOS.items():
            modified_row = _apply_scenario(row, levers)
            raw_results[country][scenario_name] = _recompute_raw_score(modified_row, panel_df)

    scaled_results = _scale_raw_results(raw_results)

    scenario_rows = []
    for country, scores in scaled_results.items():
        baseline = scores[BASELINE_SCENARIO]
        baseline_meta = latest_by_country.loc[country]
        for scenario_name in SCENARIOS.keys():
            projected = scores[scenario_name]
            scenario_rows.append(
                {
                    "country": country,
                    "year": latest_year,
                    "income_group": baseline_meta.get("income_group"),
                    "dtm_stage": baseline_meta.get("dtm_stage"),
                    "etm_stage": baseline_meta.get("etm_stage"),
                    "recommendation_tier": baseline_meta.get("recommendation_tier"),
                    "scenario": scenario_name,
                    "projected_score": round(float(projected), 2),
                    "uplift_vs_baseline": round(float(projected - baseline), 2),
                }
            )

    scenario_df = pd.DataFrame(scenario_rows).sort_values(["country", "scenario"]).reset_index(drop=True)

    summary_rows = []
    for country, country_df in scenario_df.groupby("country", sort=True):
        baseline = float(
            country_df.loc[country_df["scenario"] == BASELINE_SCENARIO, "projected_score"].iloc[0]
        )
        non_baseline = country_df[country_df["scenario"] != BASELINE_SCENARIO].copy()
        best_idx = non_baseline["projected_score"].idxmax()
        best_row = non_baseline.loc[best_idx]
        summary_rows.append(
            {
                "country": country,
                "year": latest_year,
                "baseline_score": round(baseline, 2),
                "best_scenario": best_row["scenario"],
                "best_projected_score": round(float(best_row["projected_score"]), 2),
                "best_uplift_vs_baseline": round(float(best_row["uplift_vs_baseline"]), 2),
                "income_group": best_row["income_group"],
                "dtm_stage": best_row["dtm_stage"],
                "etm_stage": best_row["etm_stage"],
                "recommendation_tier": best_row["recommendation_tier"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("best_uplift_vs_baseline", ascending=False)

    results_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(results_output, index=False)
    summary_df.to_csv(summary_output, index=False)
    _save_uplift_heatmap(scenario_df, heatmap_plot_output)
    _save_average_uplift_plot(scenario_df, average_plot_output)

    return {
        "ran": True,
        "latest_year": latest_year,
        "scenario_results": results_output,
        "scenario_summary": summary_output,
        "scenario_uplift_heatmap": heatmap_plot_output,
        "scenario_average_uplift_plot": average_plot_output,
    }


__all__ = ["BASELINE_SCENARIO", "SCENARIOS", "run_simulation_engine"]
