from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from src.data_loader import prepare_indicator_csvs
from src.preprocessing import (
    FINAL_PANEL_PATH,
    MASTER_PANEL_PATH,
    build_final_panel_with_stages,
    build_master_panel_dataset,
    build_modeling_datasets,
)


def run_data_pipeline(
    build_from_raw_json: bool = False,
    start_year: int = 2000,
    end_year: int = 2023,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}

    if build_from_raw_json:
        prepared = prepare_indicator_csvs(start_year=start_year, end_year=end_year, strict=False)
        outputs.update({f"indicator_{name}": path for name, path in prepared.items()})

    master_path = build_master_panel_dataset(start_year=start_year, end_year=end_year)
    final_path = build_final_panel_with_stages(master_panel_path=master_path)
    modeling_paths = build_modeling_datasets(panel_path=final_path)

    outputs["master_panel"] = master_path
    outputs["final_panel"] = final_path
    outputs.update({f"modeling_{name}": path for name, path in modeling_paths.items()})
    return outputs


def run_model_pipeline(
    run_ols: bool = True,
    run_fixed_effects: bool = True,
    run_predictive: bool = True,
) -> Dict[str, object]:
    from src.models import run_fixed_effects_model, run_ols_regression, run_predictive_models

    outputs: Dict[str, object] = {}

    if run_ols:
        ols_model = run_ols_regression()
        outputs["ols_ran"] = ols_model is not None

    if run_fixed_effects:
        fe_result = run_fixed_effects_model()
        outputs["fixed_effects"] = fe_result

    if run_predictive:
        outputs["predictive_outputs"] = run_predictive_models(panel_path=FINAL_PANEL_PATH)

    # TODO: Re-enable recommendation artifact generation once engine is implemented.
    outputs["recommendations_path"] = None
    return outputs


def run_full_pipeline(
    build_from_raw_json: bool = False,
    start_year: int = 2000,
    end_year: int = 2023,
    run_ols: bool = True,
    run_fixed_effects: bool = True,
    run_predictive: bool = True,
) -> Dict[str, object]:
    data_outputs = run_data_pipeline(
        build_from_raw_json=build_from_raw_json,
        start_year=start_year,
        end_year=end_year,
    )
    model_outputs = run_model_pipeline(
        run_ols=run_ols,
        run_fixed_effects=run_fixed_effects,
        run_predictive=run_predictive,
    )
    return {"data": data_outputs, "models": model_outputs}


def ensure_minimum_datasets() -> Dict[str, Optional[Path]]:
    return {
        "master_panel": MASTER_PANEL_PATH if MASTER_PANEL_PATH.exists() else None,
        "final_panel": FINAL_PANEL_PATH if FINAL_PANEL_PATH.exists() else None,
    }
