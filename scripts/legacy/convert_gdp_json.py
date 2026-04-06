import json
import pandas as pd

# Load JSON file
with open("GDP.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# World Bank JSON structure: [metadata, actual_data]
records = data[1]

# Extract relevant fields
rows = []
for entry in records:
    if entry["value"] is not None:
        rows.append({
            "country": entry["countryiso3code"],
            "year": int(entry["date"]),
            "gdp_per_capita": round(float(entry["value"]), 2)
        })

# Create DataFrame
df = pd.DataFrame(rows)

# Save CSV
df = df.sort_values(["country", "year"])
df.to_csv("GDP.csv", index=False)

print("GDP.csv created successfully.")