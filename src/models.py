from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.preprocessing import FINAL_PANEL_PATH, MODELING_DIR, normalize_columns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"


def _ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_ols_regression(
    cleaned_path: Path = MODELING_DIR / "panel_cleaned.csv",
    summary_output: Path = RESULTS_DIR / "ols_summary.txt",
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    if not cleaned_path.exists():
        return None

    df = pd.read_csv(cleaned_path)
    df = normalize_columns(df)
    required = {"gini_index", "country", "year"}
    if not required.issubset(df.columns):
        return None

    y = pd.to_numeric(df["gini_index"], errors="coerce")
    X = df.drop(columns=["gini_index", "country", "year"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    model_df = pd.concat([y.rename("gini_index"), X], axis=1).dropna()
    if model_df.empty:
        return None

    y_model = model_df["gini_index"].astype(float)
    X_model = sm.add_constant(model_df.drop(columns=["gini_index"]).astype(float))

    model = sm.OLS(y_model, X_model).fit()
    _ensure_output_dirs()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(str(model.summary()), encoding="utf-8")
    return model


def run_fixed_effects_model(
    cleaned_path: Path = MODELING_DIR / "panel_cleaned.csv",
    summary_output: Path = RESULTS_DIR / "fixed_effects_summary.txt",
    vif_output: Path = RESULTS_DIR / "fixed_effects_vif.csv",
) -> Dict[str, object]:
    try:
        from linearmodels.panel import PanelOLS
    except ImportError:
        return {"ran": False, "reason": "linearmodels is not installed"}

    if not cleaned_path.exists():
        return {"ran": False, "reason": f"Missing file: {cleaned_path}"}

    df = pd.read_csv(cleaned_path)
    df = normalize_columns(df)
    if "gini_index" not in df.columns or "country" not in df.columns or "year" not in df.columns:
        return {"ran": False, "reason": "Required columns missing for panel FE model"}

    df_panel = df.set_index(["country", "year"])
    y = pd.to_numeric(df_panel["gini_index"], errors="coerce")
    X = df_panel.drop(columns=["gini_index"], errors="ignore").apply(pd.to_numeric, errors="coerce")

    X = X.drop(
        columns=[
            "income_group_low_income",
            "income_group_lower_middle_income",
            "income_group_upper_middle_income",
        ],
        errors="ignore",
    )

    keep_mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    if X.empty:
        return {"ran": False, "reason": "No complete rows after cleaning"}

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    if X.empty:
        return {"ran": False, "reason": "No informative features after dropping constants"}

    vif_df = pd.DataFrame(
        {
            "feature": X.columns,
            "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    ).sort_values("vif", ascending=False)
    cond_number = float(np.linalg.cond(X))

    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="robust")

    _ensure_output_dirs()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        f"{results}\n\nApproximate Condition Number: {cond_number}\n",
        encoding="utf-8",
    )
    vif_output.parent.mkdir(parents=True, exist_ok=True)
    vif_df.to_csv(vif_output, index=False)

    return {
        "ran": True,
        "results": results,
        "vif": vif_df,
        "condition_number": cond_number,
        "summary_path": summary_output,
        "vif_path": vif_output,
    }


def _safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = y_true.replace(0, np.nan)
    mape = (np.abs((y_true - y_pred) / denom)).mean() * 100
    return float(mape) if pd.notna(mape) else float("nan")


def run_predictive_models(
    panel_path: Path = FINAL_PANEL_PATH,
    train_cutoff_year: int = 2018,
) -> Dict[str, Path]:
    if not panel_path.exists():
        return {}

    _ensure_output_dirs()

    df_full = pd.read_csv(panel_path)
    df_full = normalize_columns(df_full)

    if "gdp_per_capita" in df_full.columns:
        df_full["log_gdp_per_capita"] = np.log(df_full["gdp_per_capita"].replace(0, np.nan))

    targets = [
        "gini_index",
        "life_expectancy",
        "infant_mortality",
        "gdp_per_capita",
        "log_gdp_per_capita",
    ]
    base_features = [
        "gdp_per_capita",
        "population",
        "pop_growth",
        "crude_birth_rate",
        "crude_death_rate",
        "life_expectancy",
        "infant_mortality",
        "infectious_disease_rate",
        "noncommunicable_disease_rate",
        "trade_percent_gdp",
        "dtm_stage",
        "etm_stage",
    ]

    exclude_map = {
        "life_expectancy": {"infant_mortality"},
        "infant_mortality": {"life_expectancy"},
        "gdp_per_capita": {"log_gdp_per_capita"},
        "log_gdp_per_capita": {"gdp_per_capita"},
    }

    models = {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "lasso_regression": Lasso(alpha=0.01, max_iter=10000),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    saved: Dict[str, Path] = {}

    for target in targets:
        if target not in df_full.columns:
            continue

        df_source = df_full.copy().reset_index(drop=True)
        df_source["_row_id"] = df_source.index
        row_country_map = df_source.set_index("_row_id")["country"] if "country" in df_source.columns else None

        df = pd.get_dummies(df_source.copy(), columns=["country"], drop_first=True)
        country_dummies = [c for c in df.columns if c.startswith("country_")]
        current_features = [
            feature
            for feature in base_features
            if feature in df.columns and feature != target and feature not in exclude_map.get(target, set())
        ]
        features = current_features + country_dummies
        if not features:
            continue

        keep_cols = features + [target, "year", "_row_id"]
        if not set(keep_cols).issubset(df.columns):
            continue

        df = df.dropna(subset=keep_cols)
        if df.empty:
            continue

        train = df[df["year"] < train_cutoff_year]
        test = df[df["year"] >= train_cutoff_year]
        if train.empty or test.empty:
            continue

        X_train = train[features].copy()
        y_train = pd.to_numeric(train[target], errors="coerce")
        X_test = test[features].copy()
        y_test = pd.to_numeric(test[target], errors="coerce")

        keep_mask_train = y_train.notna() & X_train.notna().all(axis=1)
        keep_mask_test = y_test.notna() & X_test.notna().all(axis=1)
        X_train = X_train.loc[keep_mask_train]
        y_train = y_train.loc[keep_mask_train]
        X_test = X_test.loc[keep_mask_test]
        y_test = y_test.loc[keep_mask_test]
        test_meta = test.loc[keep_mask_test, ["_row_id", "year"]].copy()
        test_meta = test_meta.loc[y_test.index]
        if row_country_map is not None:
            test_meta["country"] = test_meta["_row_id"].map(row_country_map)
        else:
            test_meta["country"] = "UNKNOWN"

        if X_train.empty or X_test.empty:
            continue

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if current_features:
            X_train_scaled[current_features] = scaler.fit_transform(X_train[current_features])
            X_test_scaled[current_features] = scaler.transform(X_test[current_features])

        rows: List[Dict[str, float]] = []
        prediction_frames: List[pd.DataFrame] = []
        prediction_store: Dict[str, np.ndarray] = {}
        trained_rf: Optional[RandomForestRegressor] = None

        for model_name, model in models.items():
            if model_name in {"linear_regression", "ridge_regression", "lasso_regression"}:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            prediction_store[model_name] = preds
            if model_name == "random_forest":
                trained_rf = model

            rows.append(
                {
                    "model": model_name,
                    "mae": mean_absolute_error(y_test, preds),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                    "r2": r2_score(y_test, preds),
                    "mape_percent": _safe_mape(y_test, preds),
                }
            )

            pred_df = test_meta.copy()
            pred_df["model"] = model_name
            pred_df["actual"] = y_test.to_numpy()
            pred_df["predicted"] = preds
            prediction_frames.append(pred_df)

        results_df = pd.DataFrame(rows).sort_values("rmse")
        safe_target = target.replace(" ", "_")
        results_path = RESULTS_DIR / f"model_results_{safe_target}.csv"
        results_df.to_csv(results_path, index=False)
        saved[f"results_{safe_target}"] = results_path

        if prediction_frames:
            predictions_df = pd.concat(prediction_frames, ignore_index=True)
            predictions_df = predictions_df[["country", "year", "model", "actual", "predicted"]]
            predictions_path = RESULTS_DIR / f"predictions_{safe_target}.csv"
            predictions_df.to_csv(predictions_path, index=False)
            saved[f"predictions_{safe_target}"] = predictions_path

        best_model = str(results_df.iloc[0]["model"])
        best_preds = prediction_store[best_model]

        plt.figure()
        plt.scatter(y_test, best_preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
        plt.title(f"{target}: Actual vs Predicted ({best_model})")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plot_path = PLOTS_DIR / f"{safe_target}_actual_vs_pred.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        saved[f"plot_actual_vs_pred_{safe_target}"] = plot_path

        mean_val = float(np.mean(y_test))
        mean_preds = np.full_like(best_preds, mean_val, dtype=float)
        plt.figure()
        plt.scatter(y_test, best_preds, alpha=0.5, label="Model")
        plt.scatter(y_test, mean_preds, alpha=0.5, label="Mean")
        plt.legend()
        plt.title(f"{target}: Model vs Mean")
        plt.grid(True)
        plot_path = PLOTS_DIR / f"{safe_target}_vs_mean.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        saved[f"plot_vs_mean_{safe_target}"] = plot_path

        ax = results_df.set_index("model")[["mae", "rmse"]].plot(kind="bar")
        ax.set_title(f"{target}: Model Comparison")
        ax.grid(True)
        plot_path = PLOTS_DIR / f"{safe_target}_model_comparison.png"
        ax.get_figure().savefig(plot_path, bbox_inches="tight")
        plt.close(ax.get_figure())
        saved[f"plot_model_comparison_{safe_target}"] = plot_path

        if "country" in df_full.columns:
            country_avg_df = (
                df_full.groupby("country", as_index=False)[target]
                .mean()
                .rename(columns={target: "average_value"})
                .sort_values("average_value", ascending=False)
            )
            country_avg_path = RESULTS_DIR / f"country_average_{safe_target}.csv"
            country_avg_df.to_csv(country_avg_path, index=False)
            saved[f"country_average_{safe_target}"] = country_avg_path

            country_avg = country_avg_df.set_index("country")["average_value"].sort_values()
            plt.figure()
            country_avg.plot(kind="barh")
            plt.title(f"{target}: Country Comparison")
            plt.grid(True)
            plot_path = PLOTS_DIR / f"{safe_target}_country.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            saved[f"plot_country_{safe_target}"] = plot_path

        if trained_rf is not None:
            importance = pd.Series(trained_rf.feature_importances_, index=features)
            importance_df = (
                importance.sort_values(ascending=False)
                .rename_axis("feature")
                .reset_index(name="importance")
            )
            importance_path = RESULTS_DIR / f"feature_importance_{safe_target}.csv"
            importance_df.to_csv(importance_path, index=False)
            saved[f"feature_importance_{safe_target}"] = importance_path

            plt.figure()
            importance.sort_values().tail(8).plot(kind="barh")
            plt.title(f"{target}: Feature Importance")
            plt.grid(True)
            plot_path = PLOTS_DIR / f"{safe_target}_feature_importance.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            saved[f"plot_feature_importance_{safe_target}"] = plot_path

    return saved
