import pandas as pd

# Load CBR and CDR datasets
cbr = pd.read_csv("CBR.csv")
cdr = pd.read_csv("CDR.csv")

# Standardize column names (safety)
cbr.columns = [c.lower() for c in cbr.columns]
cdr.columns = [c.lower() for c in cdr.columns]

# Rename value columns if needed
cbr = cbr.rename(columns={"value": "cbr"})
cdr = cdr.rename(columns={"value": "cdr"})

# Merge on country + year
df = pd.merge(
    cbr[["country", "year", "cbr"]],
    cdr[["country", "year", "cdr"]],
    on=["country", "year"],
    how="inner"
)

# Compute Natural Increase Rate (per 1,000 people)
df["nir"] = (df["cbr"] - df["cdr"]).round(3)

# Keep final schema
nir_df = df[["country", "year", "nir"]].sort_values(["country", "year"])

# Save output
nir_df.to_csv("NIR.csv", index=False)

print(f"NIR.csv created with {len(nir_df)} rows")
