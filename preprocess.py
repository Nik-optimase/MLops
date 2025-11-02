import pandas as pd
import numpy as np
import json
from pathlib import Path

def clean_text_series(s):
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"[^\w\s\-\.\,\/\&\(\)]", "", regex=True)
    s = s.str.lower()
    return s.fillna("na")

def strip_fraud_prefix(s):
    s = s.astype("string").fillna("na")
    s = s.str.replace(r"^\s*fraud[_\-\s]+", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.str.replace(r"[^\w\s\-\.\,\/\&\(\)]", "", regex=True)
    s = s.str.lower()
    return s.replace("", "na")

def pad_zip(s):
    s = s.astype("string").fillna("na")
    s = s.str.replace(r"[^\d]", "", regex=True).str.slice(0, 9)
    return s.replace("", "na")

def _hav_km(a_lat, a_lon, b_lat, b_lon):
    lat1 = np.radians(a_lat.astype(float))
    lon1 = np.radians(a_lon.astype(float))
    lat2 = np.radians(b_lat.astype(float))
    lon2 = np.radians(b_lon.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * 6371.0 * np.arcsin(np.sqrt(h))

def main():
    input_path = Path("/app/input/test.csv")
    output_path = Path("/app/work/prepared.csv")

    # === загружаем сохранённые артефакты с обучения ===
    with open("medians.json", "r") as f:
        medians = json.load(f)
    with open("rare_maps.json", "r") as f:
        rare_maps = json.load(f)

    df = pd.read_csv(input_path)

    # очистка текстов
    for col in ["cat_id", "name_1", "name_2", "gender", "street", "one_city", "us_state"]:
        if col in df.columns:
            df[col] = clean_text_series(df[col])
    if "merch" in df.columns:
        df["merch"] = strip_fraud_prefix(df["merch"])
    if "post_code" in df.columns:
        df["post_code"] = pad_zip(df["post_code"])

    # временные признаки
    if "transaction_time" in df.columns:
        dt = pd.to_datetime(df["transaction_time"], errors="coerce", infer_datetime_format=True)
        df["hour"] = dt.dt.hour.fillna(-1).astype(int)
        df["dow"] = dt.dt.dayofweek.fillna(-1).astype(int)
        df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # гео-расстояние
    if all(x in df.columns for x in ["lat", "lon", "merchant_lat", "merchant_lon"]):
        df["dist_km"] = _hav_km(df["lat"], df["lon"], df["merchant_lat"], df["merchant_lon"])

    # заполняем пропуски медианами
    for c, med in medians.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)

    # rare mapping категориальных
    for c, mapping in rare_maps.items():
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("na").map(lambda x: mapping.get(x, "rare")).astype("string")

    # выбираем только нужные фичи
    with open("features.json", "r") as f:
        features = json.load(f)
    df = df[[c for c in features if c in df.columns]]

    df.to_csv(output_path, index=False)
    print(f"[1/3] Preprocessing done — saved {output_path}")

if __name__ == "__main__":
    main()
