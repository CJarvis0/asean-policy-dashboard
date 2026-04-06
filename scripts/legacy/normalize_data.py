import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("Master_Panel_Dataset.csv")

# -----------------------------
# 0. CLEAN COLUMN NAMES
# -----------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("%", "percent")
)

# -----------------------------
# NEW FIX: STABILIZE INCOME GROUP
# -----------------------------
if "income_group" in df.columns:
    df["income_group"] = df.groupby("country")["income_group"] \
        .transform(lambda x: x.mode()[0])

# -----------------------------
# 1. HANDLE MONETARY VARIABLES
# -----------------------------
df["average_income_real"] = df["average_income_usd"]

# -----------------------------
# 2. HANDLE POPULATION
# -----------------------------
df["log_population"] = np.log(df["population"].replace(0, np.nan))

# -----------------------------
# 3. STANDARDIZE DEMOGRAPHIC RATES
# -----------------------------
rate_columns = [
    "natural_increase_rate",
    "infant_mortality",
    "infectious_disease_rate",
    "noncommunicable_disease_rate"
]

for col in rate_columns:
    if col in df.columns:
        df[col] = df[col] / 1000

# -----------------------------
# 4. CLEAN PERCENT VARIABLES
# -----------------------------
percent_columns = [
    "top_10percent_income_share_percent",
    "bottom_10percent_income_share_percent",
    "trade_percent_gdp"
]

for col in percent_columns:
    if col in df.columns:
        df[col] = df[col] / 100

# -----------------------------
# 5. DROP VARIABLES (UPDATED FIX)
# -----------------------------
df = df.drop(columns=[
    # Redundant / derived
    "average_income_usd",
    "average_income_real",
    "natural_increase_rate",
    "population",
    "log_population",

    # HIGH multicollinearity (health cluster)
    "infant_mortality",
    "infectious_disease_rate",
    "hiv_prevalence",
    "noncommunicable_disease_rate",

    # NEW FIX (demographic overlap)
    "crude_birth_rate",
    "crude_death_rate"

], errors="ignore")

# -----------------------------
# 6. HANDLE CATEGORICAL VARIABLES (AFTER FIX)
# -----------------------------
if "income_group" in df.columns:
    df = pd.get_dummies(df, columns=["income_group"], drop_first=True)

# -----------------------------
# 7. HANDLE MISSING VALUES
# -----------------------------
df = df.sort_values(["country", "year"])
df = df.groupby("country", group_keys=False).apply(lambda x: x.ffill().bfill())
df = df.dropna()

# -----------------------------
# 8. SAVE CLEANED DATASET
# -----------------------------
df.to_csv("panel_cleaned.csv", index=False)
print("Saved: panel_cleaned.csv")

# -----------------------------
# 9. STANDARDIZE (Z-SCORE)
# -----------------------------
exclude_cols = ["country", "year"]

numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------
# 10. SAVE SCALED DATASET
# -----------------------------
df_scaled.to_csv("panel_scaled.csv", index=False)
print("Saved: panel_scaled.csv")

# -----------------------------
# 11. VALIDATION CHECKS
# -----------------------------
print("\nColumns after cleaning:")
print(df.columns.tolist())

print("\nCheck scaling (mean ~0, std ~1):")
print(df_scaled[numeric_cols].agg(['mean','std']))

# -----------------------------
# 12. CORRELATION MATRIX
# -----------------------------
corr = df_scaled[numeric_cols].corr()

print("\nCorrelation Matrix:")
print(corr)

corr.to_csv("correlation_matrix.csv")
print("Saved: correlation_matrix.csv")

# -----------------------------
# 13. OLS REGRESSION
# -----------------------------
y = df["gini_index"]

X = df.drop(columns=["gini_index", "country", "year"])

X = X.astype(float)
y = y.astype(float)

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print("\nOLS Regression Results:")
print(model.summary())

print("\nCondition Number:", model.condition_number)