from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import INDICATOR_FILE_TO_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_EXTERNAL_DIR = DATA_DIR / "raw" / "external"
INDICATORS_DIR = DATA_DIR / "processed" / "indicators"
PANEL_DIR = DATA_DIR / "processed" / "panel"
MODELING_DIR = DATA_DIR / "processed" / "modeling"

MASTER_PANEL_PATH = PANEL_DIR / "Master_Panel_Dataset.csv"
FINAL_PANEL_PATH = PANEL_DIR / "Final_Panel_Dataset_with_DTM_ETM.csv"
TRADE_PANEL_PATH = PANEL_DIR / "Trade_Analysis_Dataset.csv"


def normalize_column_name(column: str) -> str:
    cleaned = column.strip().lower()
    cleaned = cleaned.replace("%", "percent")
    cleaned = re.sub(r"[^\w]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def country_to_iso3(name: object) -> Optional[str]:
    custom_mapping = {
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "United States": "USA",
        "United Kingdom": "GBR",
        "South Korea": "KOR",
        "North Korea": "PRK",
        "Vietnam": "VNM",
        "Iran": "IRN",
        "Egypt": "EGY",
        "Czech Republic": "CZE",
    }

    if pd.isna(name):
        return None

    raw = str(name).strip()
    if len(raw) == 3 and raw.isalpha():
        return raw.upper()
    if raw in custom_mapping:
        return custom_mapping[raw]

    try:
        import pycountry

        return pycountry.countries.lookup(raw).alpha_3
    except Exception:
        return None


def _resolve_value_column(df: pd.DataFrame, expected: str) -> str:
    if expected in df.columns:
        return expected

    aliases = {
        "natural_increase_rate": ["nir"],
        "crude_birth_rate": ["value", "cbr"],
        "crude_death_rate": ["value", "cdr"],
    }
    for alias in aliases.get(expected, []):
        if alias in df.columns:
            return alias

    fallback = [c for c in df.columns if c not in {"country", "year"}]
    if len(fallback) == 1:
        return fallback[0]
    raise ValueError(f"Unable to resolve value column for {expected}. Columns: {list(df.columns)}")


def load_inequality_dataset(
    path: Path,
    start_year: int = 2000,
    end_year: int = 2023,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)

    if "country" not in df.columns or "year" not in df.columns:
        raise ValueError("Inequality dataset must include 'country' and 'year' columns.")

    df["country"] = df["country"].apply(country_to_iso3)
    df = df.dropna(subset=["country"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    return df.drop_duplicates(subset=["country", "year"]).sort_values(["country", "year"])


def build_master_panel_dataset(
    inequality_path: Path = RAW_EXTERNAL_DIR / "global_income_inequality.csv",
    output_path: Path = MASTER_PANEL_PATH,
    start_year: int = 2000,
    end_year: int = 2023,
) -> Path:
    inequality_df = load_inequality_dataset(
        inequality_path,
        start_year=start_year,
        end_year=end_year,
    )
    valid_countries = set(inequality_df["country"].unique())

    panel_df = inequality_df.copy()

    for filename, expected_column in INDICATOR_FILE_TO_COLUMN.items():
        file_path = INDICATORS_DIR / filename
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        df = normalize_columns(df)

        if "country" not in df.columns or "year" not in df.columns:
            continue

        resolved = _resolve_value_column(df, expected_column)
        if resolved != expected_column:
            df = df.rename(columns={resolved: expected_column})

        keep_cols = ["country", "year", expected_column]
        df = df[keep_cols].copy()
        df["country"] = df["country"].astype(str).str.strip().str.upper()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        df = df[df["country"].isin(valid_countries)]
        df = df.drop_duplicates(subset=["country", "year"])

        panel_df = pd.merge(panel_df, df, on=["country", "year"], how="left")

    # Avoid duplicated population columns when both inequality source and indicator source include population.
    if "population_x" in panel_df.columns and "population_y" in panel_df.columns:
        panel_df["population"] = panel_df["population_x"].combine_first(panel_df["population_y"])
        panel_df = panel_df.drop(columns=["population_x", "population_y"])

    panel_df = panel_df.drop_duplicates(subset=["country", "year"])
    panel_df = panel_df.sort_values(["country", "year"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False)
    return output_path


def classify_dtm(row: pd.Series) -> float:
    cbr = row.get("crude_birth_rate")
    cdr = row.get("crude_death_rate")
    life_exp = row.get("life_expectancy")

    if pd.isna(cbr) or pd.isna(cdr) or pd.isna(life_exp):
        return np.nan
    if cbr > 30 and cdr > 20:
        return 1
    if cbr > 25 and cdr <= 20:
        return 2
    if 15 < cbr <= 25 and cdr <= 12:
        return 3
    if cbr <= 15 and cdr <= 12 and life_exp >= 70:
        return 4
    return 3


def classify_etm(row: pd.Series) -> float:
    infectious = row.get("infectious_disease_rate")
    noncommunicable = row.get("noncommunicable_disease_rate")
    life_exp = row.get("life_expectancy")

    if pd.isna(infectious) or pd.isna(noncommunicable) or pd.isna(life_exp):
        return np.nan
    if infectious >= 500:
        return 1
    if 200 <= infectious < 500:
        return 2
    if infectious < 200 and noncommunicable >= 300:
        return 3
    if infectious < 150 and noncommunicable >= 400 and life_exp >= 70:
        return 4
    return 5


def build_final_panel_with_stages(
    master_panel_path: Path = MASTER_PANEL_PATH,
    output_path: Path = FINAL_PANEL_PATH,
    trade_output_path: Path = TRADE_PANEL_PATH,
) -> Path:
    df = pd.read_csv(master_panel_path)
    df = normalize_columns(df)

    if "country" not in df.columns or "year" not in df.columns:
        raise ValueError("Master panel must include 'country' and 'year'.")

    df["country"] = df["country"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Country-level forward/backward fill keeps temporal dynamics realistic for panel data.
    fill_cols = [c for c in df.columns if c != "country"]
    df[fill_cols] = df.groupby("country")[fill_cols].ffill()
    df[fill_cols] = df.groupby("country")[fill_cols].bfill()

    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in {"year"}]
    if numeric_cols:
        df[numeric_cols] = df.groupby("country")[numeric_cols].transform(lambda x: x.fillna(x.mean()))
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df["dtm_stage"] = df.apply(classify_dtm, axis=1)
    df["etm_stage"] = df.apply(classify_etm, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if "trade_percent_gdp" in df.columns:
        trade_df = df[df["country"] != "NGA"].copy()
        trade_output_path.parent.mkdir(parents=True, exist_ok=True)
        trade_df.to_csv(trade_output_path, index=False)

    return output_path


def build_modeling_datasets(
    panel_path: Path = FINAL_PANEL_PATH,
    cleaned_output: Path = MODELING_DIR / "panel_cleaned.csv",
    scaled_output: Path = MODELING_DIR / "panel_scaled.csv",
    corr_output: Path = MODELING_DIR / "correlation_matrix.csv",
) -> Dict[str, Path]:
    df = pd.read_csv(panel_path)
    df = normalize_columns(df)

    if "income_group" in df.columns:
        mode_series = df.groupby("country")["income_group"].transform(
            lambda s: s.mode().iat[0] if not s.mode().empty else np.nan
        )
        df["income_group"] = mode_series

    if "average_income_usd" in df.columns:
        df["average_income_real"] = df["average_income_usd"]

    if "population" in df.columns:
        df["log_population"] = np.log(df["population"].replace(0, np.nan))

    rate_columns = [
        "natural_increase_rate",
        "infant_mortality",
        "infectious_disease_rate",
        "noncommunicable_disease_rate",
    ]
    for col in rate_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1000

    percent_columns = [
        "top_10percent_income_share_percent",
        "bottom_10percent_income_share_percent",
        "trade_percent_gdp",
    ]
    for col in percent_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100

    drop_columns = {
        "average_income_usd",
        "average_income_real",
        "natural_increase_rate",
        "population",
        "log_population",
        "infant_mortality",
        "infectious_disease_rate",
        "hiv_prevalence",
        "noncommunicable_disease_rate",
        "crude_birth_rate",
        "crude_death_rate",
        # Stage variables can inflate VIF in econometric inference models.
        "dtm_stage",
        "etm_stage",
    }
    keep_drop = [c for c in drop_columns if c in df.columns]
    if keep_drop:
        df = df.drop(columns=keep_drop)

    if "income_group" in df.columns:
        df = pd.get_dummies(df, columns=["income_group"], drop_first=True)

    df = df.sort_values(["country", "year"])
    fill_cols = [c for c in df.columns if c != "country"]
    df[fill_cols] = df.groupby("country")[fill_cols].ffill()
    df[fill_cols] = df.groupby("country")[fill_cols].bfill()
    df = df.dropna().reset_index(drop=True)

    cleaned_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cleaned_output, index=False)

    exclude_cols = {"country", "year"}
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    scaler = StandardScaler()
    df_scaled = df.copy()
    if numeric_cols:
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    scaled_output.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(scaled_output, index=False)

    corr = df_scaled[numeric_cols].corr() if numeric_cols else pd.DataFrame()
    corr_output.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(corr_output, index=True)

    return {
        "cleaned": cleaned_output,
        "scaled": scaled_output,
        "correlation": corr_output,
    }
