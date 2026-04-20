"""
Prescriptive analysis runner for Story 5 (recommendations) and Story 6 (simulation).

This module preserves the existing scoring/scenario logic by delegating to:
- src.recommendation.run_recommendation_engine
- src.simulation.run_simulation_engine
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation import run_recommendation_engine
from src.simulation import run_simulation_engine

PANEL_PATH_CANDIDATES = (
    PROJECT_ROOT / "data" / "panel" / "Final_Panel_Dataset_with_DTM_ETM.csv",
    PROJECT_ROOT / "data" / "processed" / "panel" / "Final_Panel_Dataset_with_DTM_ETM.csv",
)


def resolve_panel_path() -> Path:
    for candidate in PANEL_PATH_CANDIDATES:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(p) for p in PANEL_PATH_CANDIDATES)
    raise FileNotFoundError(f"Could not find panel dataset. Checked: {joined}")


def run_prescriptive_analysis(panel_path: Path | None = None) -> Dict[str, object]:
    chosen_panel = panel_path if panel_path is not None else resolve_panel_path()
    recommendation_outputs = run_recommendation_engine(panel_path=chosen_panel)
    simulation_outputs = run_simulation_engine(panel_path=chosen_panel)
    return {
        "panel_path": chosen_panel,
        "recommendation": recommendation_outputs,
        "simulation": simulation_outputs,
    }


def main() -> None:
    outputs = run_prescriptive_analysis()
    print("=" * 72)
    print("PRESCRIPTIVE ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Panel used: {outputs['panel_path']}")

    rec = outputs["recommendation"]
    sim = outputs["simulation"]

    print("\nStory 5 artifacts:")
    print(f"  Ranked recommendations: {rec['ranked_recommendations']}")
    print(f"  Recommendation evidence: {rec['recommendation_evidence']}")
    print(f"  Recommendation plot: {rec['recommendation_plot']}")

    print("\nStory 6 artifacts:")
    print(f"  Scenario results: {sim['scenario_results']}")
    print(f"  Scenario summary: {sim['scenario_summary']}")
    print(f"  Scenario uplift heatmap: {sim['scenario_uplift_heatmap']}")
    print(f"  Scenario average uplift plot: {sim['scenario_average_uplift_plot']}")


if __name__ == "__main__":
    main()
