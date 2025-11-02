#!/usr/bin/env python3

"""
Preprocesses the test dataset for inference.

This script reads a CSV file containing the test data, computes
additional features to match those used during model training, and writes
out a processed CSV ready for scoring. It is intended to be run inside
the Docker container and expects the input file at `/app/input/test.csv`.
The processed file is written to `/app/work/processed.csv`.

Steps performed:

1. Convert the `transaction_time` column into separate time‑derived
   features: `hour` (hour of day), `dow` (day of week, Monday=0), and
   `is_weekend` (1 if Saturday or Sunday, else 0). Missing values in
   these columns are filled with sensible defaults.
2. Compute a geographic distance feature `dist_km` using the latitude
   and longitude of the transaction and the merchant. A simple
   Haversine formula is used to measure the great‑circle distance in
   kilometers.
3. Fill numeric columns with the median of the column to handle
   missing values. Categorical features are coerced to strings and
   missing values are replaced with "na".
4. Drop unused coordinate columns and retain only those features
   specified in `features.json` so the model sees exactly what it
   expects.

The processed file will include the original `id` column so that the
predictions can be matched back to the correct rows when creating
`sample_submission.csv`.
"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def haversine(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Return the great‑circle distance between two points on Earth in km."""
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def preprocess(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    # Create time features
    dt = pd.to_datetime(df.get("transaction_time"), errors="coerce")
    df["hour"] = dt.dt.hour.fillna(-1).astype(int)
    df["dow"] = dt.dt.dayofweek.fillna(-1).astype(int)
    df["is_weekend"] = ((df["dow"] >= 5).astype(int))

    # Compute geographic distance (in km)
    df["dist_km"] = haversine(
        df.get("lat"), df.get("lon"), df.get("merchant_lat"), df.get("merchant_lon")
    )

    # Fill numeric columns with their median values
    numeric_cols = [
        "amount",
        "population_city",
        "hour",
        "dow",
        "is_weekend",
        "dist_km",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        series = pd.to_numeric(df[col], errors="coerce")
        median_val = float(series.median()) if series.count() > 0 else 0.0
        df[col] = series.fillna(median_val)

    # Fill categorical columns
    cat_cols = ["cat_id", "gender"]
    for col in cat_cols:
        if col not in df.columns:
            df[col] = "na"
        else:
            df[col] = df[col].astype(str).fillna("na")

    # Drop latitude and longitude columns since they are not model features
    df = df.drop(columns=[c for c in ["lat", "lon", "merchant_lat", "merchant_lon", "transaction_time"] if c in df.columns], errors="ignore")

    # Ensure all required features are present
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0

    # Return dataframe restricted to feature_list plus id
    cols = [c for c in ["id"] + feature_list if c in df.columns]
    return df[cols]


def main() -> None:
    input_path = Path("./input/test.csv")
    output_dir = Path("./work")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load feature list
    with open("features.json", "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    df_raw = pd.read_csv(input_path)
    df_processed = preprocess(df_raw.copy(), feature_list)

    df_processed.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    main()