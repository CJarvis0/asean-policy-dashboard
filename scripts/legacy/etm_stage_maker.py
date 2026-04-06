import pandas as pd

df = pd.read_csv("Master_Panel_Dataset_with_DTM.csv")


def classify_etm(row):
    if pd.isna(row["life_expectancy"]):
        return None

    infectious = row.get("infectious_disease_rate", None)

    if row["life_expectancy"] < 40:
        return 1
    elif 40 <= row["life_expectancy"] < 60:
        return 2
    elif 60 <= row["life_expectancy"] < 75:
        return 3
    elif row["life_expectancy"] >= 75:
        return 4
    else:
        return None


df["ETM_stage"] = df.apply(classify_etm, axis=1)

df.to_csv("Master_Panel_Dataset_with_DTM_ETM.csv", index=False)
print("ETM stages added.")