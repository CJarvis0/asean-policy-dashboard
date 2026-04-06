import pandas as pd
import numpy as np

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("Master_Panel_Dataset.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Remove HIV prevalence column (not used in DTM or ETM)
if "hiv_prevalence" in df.columns:
    df = df.drop(columns=["hiv_prevalence"])

# Verify required columns exist
required_cols = ["country", "year"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Ensure proper data types and sorting
df["year"] = df["year"].astype(int)
df = df.sort_values(["country", "year"]).reset_index(drop=True)

# --------------------------------------------------
# Step 1: Handle Missing Values
# --------------------------------------------------

# Forward fill within each country
df = df.groupby("country", as_index=False).apply(lambda group: group.ffill())
df = df.reset_index(drop=True)

# Backward fill within each country
df = df.groupby("country", as_index=False).apply(lambda group: group.bfill())
df = df.reset_index(drop=True)

# Fill remaining numeric NA values with country-level mean
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
    lambda x: x.fillna(x.mean())
)

# Check remaining missing values
print("\nRemaining missing values after imputation:")
print(df.isna().sum())

# --------------------------------------------------
# Step 2: DTM Classification
# --------------------------------------------------
def classify_dtm(row):
    cbr = row["crude_birth_rate"]
    cdr = row["crude_death_rate"]
    life_exp = row["life_expectancy"]

    if pd.isna(cbr) or pd.isna(cdr) or pd.isna(life_exp):
        return np.nan
    elif cbr > 30 and cdr > 20:
        return 1  # Stage 1: High Stationary
    elif cbr > 25 and cdr <= 20:
        return 2  # Stage 2: Early Expanding
    elif 15 < cbr <= 25 and cdr <= 12:
        return 3  # Stage 3: Late Expanding
    elif cbr <= 15 and cdr <= 12 and life_exp >= 70:
        return 4  # Stage 4: Low Stationary
    else:
        return 3  # Transitional Stage

df["dtm_stage"] = df.apply(classify_dtm, axis=1)

# --------------------------------------------------
# Step 3: ETM Classification
# --------------------------------------------------
def classify_etm(row):
    infectious = row["infectious_disease_rate"]
    noncommunicable = row["noncommunicable_disease_rate"]
    life_exp = row["life_expectancy"]

    if pd.isna(infectious) or pd.isna(noncommunicable) or pd.isna(life_exp):
        return np.nan
    elif infectious >= 500:
        return 1  # Pestilence & Famine
    elif 200 <= infectious < 500:
        return 2  # Receding Pandemics
    elif infectious < 200 and noncommunicable >= 300:
        return 3  # Degenerative Diseases
    elif infectious < 150 and noncommunicable >= 400 and life_exp >= 70:
        return 4  # Delayed Degenerative Diseases
    else:
        return 5  # Re-emergence of Infectious Diseases

df["etm_stage"] = df.apply(classify_etm, axis=1)

# --------------------------------------------------
# Step 4: Save Final Dataset
# --------------------------------------------------
df = df.sort_values(["country", "year"]).reset_index(drop=True)
df.to_csv("Final_Panel_Dataset_with_DTM_ETM.csv", index=False)

print("\nFinal dataset created successfully.")
print("DTM Stage Distribution:\n", df["dtm_stage"].value_counts().sort_index())
print("\nETM Stage Distribution:\n", df["etm_stage"].value_counts().sort_index())

# --------------------------------------------------
# Optional: Create Trade-Analysis Dataset (Exclude Nigeria)
# --------------------------------------------------
if "trade_percent_gdp" in df.columns:
    trade_analysis_df = df[df["country"] != "NGA"].copy()
    trade_analysis_df.to_csv("Trade_Analysis_Dataset.csv", index=False)
    print("\nTrade_Analysis_Dataset.csv created (Nigeria excluded for trade analysis).")