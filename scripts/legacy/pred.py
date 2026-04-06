# =========================================================
# MULTI-TARGET PANEL PREDICTIVE PIPELINE (FINAL FINAL VERSION)
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Scaling
from sklearn.preprocessing import StandardScaler

# =========================================================
# CREATE OUTPUT FOLDER
# =========================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df_full = pd.read_csv("Final_Panel_Dataset_with_DTM_ETM.csv")
df_full.columns = df_full.columns.str.strip().str.lower()

# =========================================================
# ADD LOG GDP (IMPORTANT)
# =========================================================

df_full["log_gdp_per_capita"] = np.log(df_full["gdp_per_capita"])

# =========================================================
# TARGET VARIABLES
# =========================================================

targets = [
    "gini index",
    "life_expectancy",
    "infant_mortality",
    "gdp_per_capita",
    "log_gdp_per_capita"
]

# =========================================================
# BASE FEATURES
# =========================================================

base_features = [
    "gdp_per_capita",
    "population",
    "pop_growth",
    "crude_birth_rate",
    "crude_death_rate",
    "life_expectancy",
    "infant_mortality",
    "infectious_disease_rate",
    "noncommunicable_disease_rate"
]

# =========================================================
# EXCLUDE HIGHLY RELATED VARIABLES
# =========================================================

exclude_map = {
    "life_expectancy": ["infant_mortality"],
    "infant_mortality": ["life_expectancy"],
    "gdp_per_capita": ["log_gdp_per_capita"],
    "log_gdp_per_capita": ["gdp_per_capita"]
}

# =========================================================
# MODELS
# =========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# =========================================================
# LOOP THROUGH TARGETS
# =========================================================

for TARGET in targets:

    print(f"\n==============================")
    print(f"RUNNING MODELS FOR: {TARGET}")
    print(f"==============================")

    df = df_full.copy()
    df_original = df.copy()

    # Add country dummies
    df = pd.get_dummies(df, columns=["country"], drop_first=True)
    country_dummies = [col for col in df.columns if col.startswith("country_")]

    # Dynamic feature selection
    current_features = [
        f for f in base_features
        if f != TARGET and f not in exclude_map.get(TARGET, [])
    ]

    features = current_features + country_dummies

    df = df.dropna(subset=features + [TARGET])

    # Train/test split
    train = df[df["year"] < 2018]
    test = df[df["year"] >= 2018]

    X_train = train[features]
    y_train = train[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[current_features] = scaler.fit_transform(X_train[current_features])
    X_test_scaled[current_features] = scaler.transform(X_test[current_features])

    predictions_dict = {}
    results = []

    # =========================================================
    # TRAIN MODELS
    # =========================================================

    for name, model in models.items():

        if name in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        predictions_dict[name] = preds

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # 🔥 NEW: MAPE
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE (%)": mape
        })

    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by="RMSE"))

    # Save results
    safe_target = TARGET.replace(" ", "_")
    results_df.to_csv(f"{OUTPUT_DIR}/model_results_{safe_target}.csv", index=False)

    # =========================================================
    # VISUALS
    # =========================================================

    best_model_name = results_df.sort_values(by="RMSE").iloc[0]["Model"]
    best_preds = predictions_dict[best_model_name]

    # Actual vs Predicted
    plt.figure()
    plt.scatter(y_test, best_preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    plt.title(f"{TARGET}: Actual vs Predicted ({best_model_name})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{safe_target}_actual_vs_pred.png")
    plt.close()

    # Model vs Mean
    mean_val = np.mean(y_test)
    mean_preds = np.full_like(y_test, mean_val)

    plt.figure()
    plt.scatter(y_test, best_preds, alpha=0.5, label="Model")
    plt.scatter(y_test, mean_preds, alpha=0.5, label="Mean")
    plt.legend()
    plt.title(f"{TARGET}: Model vs Mean")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{safe_target}_vs_mean.png")
    plt.close()

    # Model comparison
    results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar")
    plt.title(f"{TARGET}: Model Comparison")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{safe_target}_model_comparison.png")
    plt.close()

    # Country comparison
    country_avg = df_original.groupby("country")[TARGET].mean().sort_values()
    plt.figure()
    country_avg.plot(kind="barh")
    plt.title(f"{TARGET}: Country Comparison")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{safe_target}_country.png")
    plt.close()

    # Feature importance
    rf_model = models["Random Forest"]
    importance = pd.Series(rf_model.feature_importances_, index=features)

    plt.figure()
    importance.sort_values().tail(8).plot(kind="barh")
    plt.title(f"{TARGET}: Feature Importance")
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{safe_target}_feature_importance.png")
    plt.close()

print("\nALL TARGETS COMPLETE.")