import pandas as pd

df = pd.read_csv("Master_Panel_Dataset.csv")

def classify_dtm(row):
    if pd.isna(row["cbr"]) or pd.isna(row["cdr"]):
        return None
    if row["cbr"] > 35 and row["cdr"] > 35:
        return 1
    elif row["cbr"] > 30 and row["cdr"] < 20:
        return 2
    elif 15 <= row["cbr"] <= 30 and row["cdr"] < 15:
        return 3
    elif row["cbr"] < 15 and row["cdr"] < 10:
        return 4
    else:
        return None

df["DTM_stage"] = df.apply(classify_dtm, axis=1)

df.to_csv("Master_Panel_Dataset_with_DTM.csv", index=False)
print("DTM stages added.")