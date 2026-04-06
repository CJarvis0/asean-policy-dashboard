import pandas as pd
import pycountry

START_YEAR = 2000
END_YEAR = 2023

# --- Function to convert country names to ISO3 ---
def country_to_iso3(name):
    custom_mapping = {
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "United States": "USA",
        "United Kingdom": "GBR",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Vietnam": "VNM",
        "Iran": "IRN",
        "Egypt": "EGY"
    }

    if pd.isna(name):
        return None

    name = name.strip()

    if name in custom_mapping:
        return custom_mapping[name]

    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None


# -----------------------------
# Load Inequality Dataset (Base)
# -----------------------------
inequality_df = pd.read_csv("global_income_inequality.csv")
inequality_df.columns = inequality_df.columns.str.lower()

# Convert country names to ISO3
inequality_df["country"] = inequality_df["country"].apply(country_to_iso3)

# Drop rows where conversion failed
inequality_df = inequality_df.dropna(subset=["country"])

# Ensure correct year format and filter range
inequality_df["year"] = inequality_df["year"].astype(int)
inequality_df = inequality_df[
    (inequality_df["year"] >= START_YEAR) &
    (inequality_df["year"] <= END_YEAR)
]

# Remove duplicates from inequality dataset
inequality_df = inequality_df.drop_duplicates(subset=["country", "year"])

valid_countries = inequality_df["country"].unique()

print("Countries in inequality dataset:", valid_countries)


# -----------------------------
# Files to Merge
# -----------------------------
files = {
    "CBR.csv": "crude_birth_rate",
    "CDR.csv": "crude_death_rate",
    "NIR.csv": "natural_increase_rate",
    "LifeExpectancyAtBirth.csv": "life_expectancy",
    "InfantMortalityRate.csv": "infant_mortality",
    "PopGrowthRate.csv": "pop_growth",
    "TotalPop.csv": "population",
    "HIVPrevalence_filtered.csv": "hiv_prevalence",
    "InfectiousAndParasiticDiseases.csv": "infectious_disease_rate",
    "NoncommunicableDiseases.csv": "noncommunicable_disease_rate",
    "GDP.csv": "gdp_per_capita",
    "TRADE.csv": "trade_percent_gdp"
}

dfs = []

# -----------------------------
# Load and Clean Each Dataset
# -----------------------------
for file, col_name in files.items():
    print(f"Processing {file}...")

    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()

    # Rename generic "value" column if present
    if "value" in df.columns:
        df = df.rename(columns={"value": col_name})

    # Ensure correct column naming
    value_cols = [c for c in df.columns if c not in ["country", "year"]]
    if col_name not in df.columns and len(value_cols) == 1:
        df = df.rename(columns={value_cols[0]: col_name})

    # Standardize formats
    df["year"] = df["year"].astype(int)
    df["country"] = df["country"].astype(str).str.strip()

    # Filter by year and country
    df = df[(df["year"] >= START_YEAR) & (df["year"] <= END_YEAR)]
    df = df[df["country"].isin(valid_countries)]

    # Remove duplicate country-year rows
    df = df.drop_duplicates(subset=["country", "year"])

    dfs.append(df)


# -----------------------------
# Merge All Data (Left Join)
# -----------------------------
panel_df = inequality_df.copy()

for df in dfs:
    panel_df = pd.merge(panel_df, df, on=["country", "year"], how="left")


# -----------------------------
# Clean Up Duplicate Columns
# -----------------------------
if "population_x" in panel_df.columns and "population_y" in panel_df.columns:
    panel_df["population"] = panel_df["population_x"].combine_first(panel_df["population_y"])
    panel_df = panel_df.drop(columns=["population_x", "population_y"])


# Final cleanup
panel_df = panel_df.drop_duplicates(subset=["country", "year"])
panel_df = panel_df.sort_values(["country", "year"]).reset_index(drop=True)

# Save final dataset
panel_df.to_csv("Master_Panel_Dataset.csv", index=False)


# -----------------------------
# Validation Output
# -----------------------------
print("\nMaster_Panel_Dataset.csv created successfully.")
print("Countries:", panel_df["country"].unique())
print("Number of countries:", panel_df["country"].nunique())
print("Year range:", panel_df["year"].min(), "to", panel_df["year"].max())
print("Total rows:", len(panel_df))