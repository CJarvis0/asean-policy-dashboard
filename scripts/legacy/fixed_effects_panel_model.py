from linearmodels.panel import PanelOLS
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------
# 1. LOAD CLEANED DATA
# -----------------------------
df = pd.read_csv("panel_cleaned.csv")

# -----------------------------
# 2. SET PANEL INDEX
# -----------------------------
df_panel = df.set_index(["country", "year"])

# -----------------------------
# 3. DEFINE VARIABLES
# -----------------------------
y = df_panel["gini_index"]

X = df_panel.drop(columns=["gini_index"])

# Convert to numeric
X = X.astype(float)
y = y.astype(float)

# REMOVE INCOME GROUP (absorbed by FE)
X = X.drop(columns=[
    "income_group_Low Income",
    "income_group_Lower Middle Income",
    "income_group_Upper Middle Income"
], errors="ignore")

# -----------------------------
# 4. VIF CALCULATION
# -----------------------------
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\nVariance Inflation Factors (VIF):")
print(vif_data.sort_values(by="VIF", ascending=False))

# -----------------------------
# 5. CONDITION NUMBER
# -----------------------------
cond_number = np.linalg.cond(X)
print("\nApproximate Condition Number:", cond_number)

# -----------------------------
# 6. FIXED EFFECTS MODEL
# -----------------------------
model = PanelOLS(
    y,
    X,
    entity_effects=True,
    time_effects=True
)

results = model.fit(cov_type="robust")

# -----------------------------
# 7. OUTPUT RESULTS
# -----------------------------
print("\nFixed Effects Model Results:")
print(results)