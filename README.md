# ASEAN Policy Dashboard

End-to-end analytics and decision-support system for ASEAN-focused socioeconomic panel analysis.

## What Works Now

- Structured raw and processed data folders
- Data pipeline for panel assembly and stage generation (DTM/ETM)
- Econometric outputs (OLS and Panel OLS)
- Multi-target predictive modeling outputs with plots and metrics
- Prescriptive module scaffolding (Recommendation + What-If placeholders)
- Streamlit dashboard entry point

## Project Structure

- `data/raw/worldbank`: raw World Bank JSON files
- `data/raw/who`: raw WHO JSON files
- `data/raw/external`: external source CSV files
- `data/processed/indicators`: cleaned indicator CSV files
- `data/processed/panel`: panel datasets (`Master`, `Final`, `Trade_Analysis`)
- `data/processed/modeling`: `panel_cleaned.csv`, `panel_scaled.csv`, `correlation_matrix.csv`
- `outputs/results`: OLS/Panel OLS summaries, VIF, predictive model outputs, and future prescriptive artifacts
- `outputs/plots`: predictive model visualizations
- `src`: reusable modules (`data_loader`, `preprocessing`, `models`, `simulation`, `recommendation`, `pipeline`)
- `scripts/run_pipeline.py`: main CLI entry point
- `app/streamlit_app.py`: dashboard app
- `scripts/legacy`: archived one-off scripts kept for traceability

## Setup

### Platform Support

- Works on **macOS**, **Linux**, and **Windows**.
- `make` workflow is easiest on macOS/Linux.
- On Windows, you can use:
  - PowerShell (manual commands), or
  - WSL / Git Bash with `make`.

### Quickstart (macOS / Linux)

```bash
git clone <your-repo-url>
cd asean-policy-dashboard
make help
make setup PYTHON=python3.12
source venv/bin/activate
make all
make dashboard
```

If your machine already maps `python3` correctly, this also works:

```bash
make setup
source venv/bin/activate
```

### Quickstart (Windows PowerShell)

```powershell
git clone <your-repo-url>
cd asean-policy-dashboard
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --stage all
python -m streamlit run app/streamlit_app.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Optional: `make` on Windows (WSL / Git Bash)

```bash
make setup PYTHON=python3.12
source venv/bin/activate
make all
make dashboard
```

### Requirements

- `git`
- Python 3.12+ recommended
- `make` (optional, required only for make-based workflow)

### Manual Setup (Without Make)

From project root (macOS/Linux):

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

From project root (Windows PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick team setup with `make`:

```bash
make setup
source venv/bin/activate
```

### First-Run Checklist

1. Clone and enter repo.
2. Create/setup env (`make setup` or manual setup).
3. Activate env:
   - macOS/Linux: `source venv/bin/activate`
   - Windows PowerShell: `.\venv\Scripts\Activate.ps1`
4. Build artifacts: `make all`.
5. Launch app:
   - make workflow: `make dashboard`
   - manual workflow: `python -m streamlit run app/streamlit_app.py`

## Pipeline Commands

```bash
# Full run (data + models)
python scripts/run_pipeline.py --stage all

# Data only
python scripts/run_pipeline.py --stage data

# Models only (OLS + Panel OLS + predictive)
python scripts/run_pipeline.py --stage models

# OLS + Panel OLS only
python scripts/run_pipeline.py --stage models --skip-predictive

# Predictive only
python scripts/run_pipeline.py --stage models --skip-ols --skip-fixed-effects

# Rebuild indicators from raw JSON before running
python scripts/run_pipeline.py --stage all --build-from-raw-json
```

Equivalent `make` commands:

```bash
make all
make data
make models
make models-econ
make models-predictive
make all-from-raw
```

## Dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

### Presentation Mode

The dashboard includes a `Story Mode` tab that aligns analysis to six user stories:

- 2 descriptive stories (country-level inequality trends and ASEAN-wide inequality monitoring)
- 2 predictive stories (Gini prediction with trade context, GDP prediction performance)
- 2 prescriptive story placeholders (ranked policy priorities, what-if simulation for reforms) pending implementation

## Data/Modeling Notes

- Econometric inference uses `data/processed/modeling/panel_cleaned.csv`.
- Predictive modeling uses `data/processed/panel/Final_Panel_Dataset_with_DTM_ETM.csv`.
- `dtm_stage` and `etm_stage` are retained in final panel for predictive use, but excluded from econometric cleaned features to reduce multicollinearity pressure.

## Generated Outputs

- OLS summary: `outputs/results/ols_summary.txt`
- Panel OLS summary: `outputs/results/fixed_effects_summary.txt`
- Panel OLS VIF table: `outputs/results/fixed_effects_vif.csv`
- Predictive metrics: `outputs/results/model_results_*.csv`
- Predictive point-level outputs for interactive dashboard charts: `outputs/results/predictions_*.csv`
- Feature-importance and country-average outputs for interactive dashboard charts:
  - `outputs/results/feature_importance_*.csv`
  - `outputs/results/country_average_*.csv`
- Predictive plots: `outputs/plots/*.png`

## Prescriptive Modules Status (Story 5 + Story 6)

The Recommendation Engine and What-If Simulation are not complete yet.

- `src/recommendation.py` is currently a placeholder and does not produce ranked recommendations.
- `src/simulation.py` is currently a placeholder and does not run scenario inference.
- `src/pipeline.py` intentionally sets `recommendations_path` to `None` until prescriptive features are implemented.
- Dashboard tabs and Story Mode sections for Story 5/Story 6 are scaffolded and marked as pending implementation.

## Small Roadmap: Recommendation Engine + What-If Analysis

### Step 1 — Lock Inputs from Existing Analytics

- Use validated indicators from descriptive/econometric work as policy levers:
  - `gini_index`
  - `trade_percent_gdp`
  - `gdp_per_capita`
  - `life_expectancy`
  - `infant_mortality`
  - demographic stage variables (`dtm_stage`, `etm_stage`)
- Reuse existing predictive results to define target outcomes and baseline feature schema.

### Step 2 — Implement Recommendation Engine (Story 5)

- Build a transparent country-level scoring framework:
  - risk score
  - expected impact score
  - feasibility score
  - composite priority score
- Generate ranked artifacts:
  - `outputs/results/policy_recommendations_ranked.csv`
  - `outputs/results/policy_recommendation_evidence.csv`
- Integrate outputs into `src/pipeline.py` and the dashboard Recommendation tab.

### Step 3 — Implement What-If Simulation (Story 6)

- Add scenario controls that adjust policy levers for a selected country-year baseline.
- Use trained predictive models to estimate outcome deltas:
  - `Δgini_index`
  - `Δgdp_per_capita` / `Δlog_gdp_per_capita`
  - `Δlife_expectancy`
  - `Δinfant_mortality`
- Generate scenario artifacts:
  - `outputs/results/scenario_results.csv`
  - `outputs/results/scenario_summary.csv`
- Integrate baseline vs scenario visualizations into Story 6 and the Simulation tab.

### Step 4 — QA and Presentation Readiness

- Add validation checks (input bounds, missing values, and reproducibility metadata).
- Add clear assumptions/limitations text for prescriptive outputs in the dashboard.
- Finalize Story 5 and Story 6 presentation flow with evidence + scenario comparisons.

## Troubleshooting

- If `python scripts/run_pipeline.py ...` says dependencies are missing, your `python` and `pip` are likely from different installations.
- Fix by activating `venv` and installing requirements with `python -m pip install -r requirements.txt`.
