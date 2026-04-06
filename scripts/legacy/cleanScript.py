import json
import pandas as pd
import sys
from pathlib import Path

"""
Usage:
    python worldbank_json_to_csv.py CBR.json CBR.csv
"""

if len(sys.argv) != 3:
    print("Usage: python worldbank_json_to_csv.py <input.json> <output.csv>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

records = []

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# World Bank data is always in index 1
rows = data[1]

for row in rows:
    if row["value"] is None:
        continue

    records.append({
        "country": row["countryiso3code"],
        "year": int(row["date"]),
        "value": row["value"]
    })

df = pd.DataFrame(records)

if df.empty:
    print("No valid rows found.")
    sys.exit(0)

df.sort_values(["country", "year"], inplace=True)
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows → {output_file}")
