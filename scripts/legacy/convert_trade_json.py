import json
import pandas as pd

# Load JSON file
with open("TRADE.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = data[1]

rows = []
for entry in records:
    if entry["value"] is not None:
        rows.append({
            "country": entry["countryiso3code"],
            "year": int(entry["date"]),
            "trade_percent_gdp": round(float(entry["value"]), 2)
        })

df = pd.DataFrame(rows)

df = df.sort_values(["country", "year"])
df.to_csv("TRADE.csv", index=False)

print("TRADE.csv created successfully.")