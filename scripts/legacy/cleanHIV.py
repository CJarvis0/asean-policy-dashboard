import json
import pandas as pd

INPUT_FILE = "HIVPrevalence.json"
OUTPUT_FILE = "HIVPrevalence_filtered.csv"

COUNTRIES = {
    "USA", "IND", "BRA", "DEU", "NGA",
    "CHN", "GBR", "RUS", "JPN", "ZAF",
    "CAN", "MEX", "FRA", "AUS", "SAU"
}

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for row in data["value"]:
    if row["SpatialDimType"] != "COUNTRY":
        continue

    if row["SpatialDim"] not in COUNTRIES:
        continue

    year = row["TimeDim"]
    if year < 2000 or year > 2023:
        continue

    if row["NumericValue"] is None:
        continue

    records.append({
        "country": row["SpatialDim"],
        "year": year,
        "hiv_prevalence": row["NumericValue"]
    })

df = pd.DataFrame(records)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
