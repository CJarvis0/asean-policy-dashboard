from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_WORLD_BANK_DIR = DATA_DIR / "raw" / "worldbank"
RAW_WHO_DIR = DATA_DIR / "raw" / "who"
PROCESSED_INDICATORS_DIR = DATA_DIR / "processed" / "indicators"


@dataclass(frozen=True)
class IndicatorSpec:
    source: str
    json_name: str
    csv_name: str
    value_column: str


# Files used by the panel dataset build.
INDICATOR_FILE_TO_COLUMN: Dict[str, str] = {
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
    "TRADE.csv": "trade_percent_gdp",
}


INDICATOR_SPECS: Sequence[IndicatorSpec] = (
    IndicatorSpec("worldbank", "CBR.json", "CBR.csv", "crude_birth_rate"),
    IndicatorSpec("worldbank", "CDR.json", "CDR.csv", "crude_death_rate"),
    IndicatorSpec(
        "worldbank",
        "InfantMortalityRate.json",
        "InfantMortalityRate.csv",
        "infant_mortality",
    ),
    IndicatorSpec(
        "worldbank",
        "LifeExpectancyAtBirth.json",
        "LifeExpectancyAtBirth.csv",
        "life_expectancy",
    ),
    IndicatorSpec("worldbank", "PopGrowthRate.json", "PopGrowthRate.csv", "pop_growth"),
    IndicatorSpec("worldbank", "TotalPop.json", "TotalPop.csv", "population"),
    IndicatorSpec("worldbank", "GDP.json", "GDP.csv", "gdp_per_capita"),
    IndicatorSpec("worldbank", "TRADE.json", "TRADE.csv", "trade_percent_gdp"),
    IndicatorSpec("who", "HIVPrevalence.json", "HIVPrevalence_filtered.csv", "hiv_prevalence"),
    IndicatorSpec(
        "who",
        "InfectiousAndParasiticDiseases.json",
        "InfectiousAndParasiticDiseases.csv",
        "infectious_disease_rate",
    ),
    IndicatorSpec(
        "who",
        "NoncommunicableDiseases.json",
        "NoncommunicableDiseases.csv",
        "noncommunicable_disease_rate",
    ),
    # Optional dataset retained for validation analysis.
    IndicatorSpec(
        "who",
        "LifeExpectancyCrossValidation.json",
        "LifeExpectancyCrossValidation.csv",
        "life_expectancy_cross_validation",
    ),
)


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _country_code(value: object) -> Optional[str]:
    if value is None:
        return None
    country = str(value).strip().upper()
    if len(country) != 3:
        return None
    return country


def _build_dataframe(records: List[Tuple[str, int, float]], value_column: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["country", "year", value_column])
    df = pd.DataFrame(records, columns=["country", "year", value_column])
    df = df.dropna(subset=["country", "year", value_column])
    df = df.drop_duplicates(subset=["country", "year"])
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    return df


def _parse_world_bank_rows(payload: object, value_column: str) -> pd.DataFrame:
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return pd.DataFrame(columns=["country", "year", value_column])

    records: List[Tuple[str, int, float]] = []
    for row in payload[1]:
        if not isinstance(row, dict):
            continue
        country = _country_code(row.get("countryiso3code"))
        year = _safe_int(row.get("date"))
        value = _safe_float(row.get("value"))
        if country is None or year is None or value is None:
            continue
        records.append((country, year, value))
    return _build_dataframe(records, value_column)


def _parse_who_rows(payload: object, value_column: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["country", "year", value_column])

    rows = payload.get("value", [])
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["country", "year", value_column])

    records: List[Tuple[str, int, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("SpatialDimType") != "COUNTRY":
            continue
        country = _country_code(row.get("SpatialDim"))
        year = _safe_int(row.get("TimeDim"))
        value = _safe_float(row.get("NumericValue"))
        if country is None or year is None or value is None:
            continue
        records.append((country, year, value))
    return _build_dataframe(records, value_column)


def _resolve_source_dir(source: str) -> Path:
    if source == "worldbank":
        return RAW_WORLD_BANK_DIR
    if source == "who":
        return RAW_WHO_DIR
    raise ValueError(f"Unsupported source: {source}")


def _filter_years(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()


def convert_indicator_json(
    spec: IndicatorSpec,
    start_year: int = 2000,
    end_year: int = 2023,
) -> Path:
    source_dir = _resolve_source_dir(spec.source)
    input_path = source_dir / spec.json_name
    output_path = PROCESSED_INDICATORS_DIR / spec.csv_name
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if spec.source == "worldbank":
        df = _parse_world_bank_rows(payload, spec.value_column)
    else:
        df = _parse_who_rows(payload, spec.value_column)

    df = _filter_years(df, start_year=start_year, end_year=end_year)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _resolve_value_column(df: pd.DataFrame, preferred: Sequence[str]) -> str:
    df_cols = {col.lower(): col for col in df.columns}
    for candidate in preferred:
        if candidate.lower() in df_cols:
            return df_cols[candidate.lower()]

    fallback = [c for c in df.columns if c.lower() not in {"country", "year"}]
    if len(fallback) == 1:
        return fallback[0]
    raise ValueError(f"Could not resolve value column from columns: {list(df.columns)}")


def compute_natural_increase_rate(
    cbr_path: Optional[Path] = None,
    cdr_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    cbr_input = cbr_path or (PROCESSED_INDICATORS_DIR / "CBR.csv")
    cdr_input = cdr_path or (PROCESSED_INDICATORS_DIR / "CDR.csv")
    nir_output = output_path or (PROCESSED_INDICATORS_DIR / "NIR.csv")

    cbr = pd.read_csv(cbr_input)
    cdr = pd.read_csv(cdr_input)

    cbr_col = _resolve_value_column(cbr, ["crude_birth_rate", "value", "cbr"])
    cdr_col = _resolve_value_column(cdr, ["crude_death_rate", "value", "cdr"])

    cbr = cbr.rename(columns={cbr_col: "crude_birth_rate"})
    cdr = cdr.rename(columns={cdr_col: "crude_death_rate"})

    merged = pd.merge(
        cbr[["country", "year", "crude_birth_rate"]],
        cdr[["country", "year", "crude_death_rate"]],
        on=["country", "year"],
        how="inner",
    )
    merged["natural_increase_rate"] = (
        merged["crude_birth_rate"] - merged["crude_death_rate"]
    ).round(3)

    nir = merged[["country", "year", "natural_increase_rate"]].sort_values(
        ["country", "year"]
    )
    nir_output.parent.mkdir(parents=True, exist_ok=True)
    nir.to_csv(nir_output, index=False)
    return nir_output


def prepare_indicator_csvs(
    start_year: int = 2000,
    end_year: int = 2023,
    strict: bool = False,
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for spec in INDICATOR_SPECS:
        try:
            out_path = convert_indicator_json(spec, start_year=start_year, end_year=end_year)
            outputs[spec.csv_name] = out_path
        except FileNotFoundError:
            if strict:
                raise
    nir_output = compute_natural_increase_rate()
    outputs[nir_output.name] = nir_output
    return outputs
