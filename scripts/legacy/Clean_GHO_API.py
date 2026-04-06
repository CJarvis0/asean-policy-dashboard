import json
import pandas as pd
import sys

"""
Usage:
    python who_json_to_csv.py input.json output.csv
"""

if len(sys.argv) != 3:
    print("Usage: python who_json_to_csv.py <input.json> <output.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

records = []

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data["value"]

for row in rows:
    # Keep only country-level data
    if row.get("SpatialDimType") != "COUNTRY":
        continue

    numeric_value = row.get("NumericValue")
    if numeric_value is None:
        continue

    records.append({
        "country": row["SpatialDim"],   # ISO3
        "year": int(row["TimeDim"]),
        "value": float(numeric_value)
    })

df = pd.DataFrame(records)

if df.empty:
    print("No valid rows found.")
    sys.exit(0)

df.sort_values(["country", "year"], inplace=True)
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows → {output_file}")
