"""
etl_pipeline.py
NASA POWER CSV extraction, feature engineering, SQLite loading.
"""

import time
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np
import requests

from utils import (
    get_db_connection, ensure_data_dir, LOCATIONS,
    NASA_BASE_URL, NASA_PARAMETERS, compute_green_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5


def fetch_nasa_data(location_name, latitude, longitude, start_date, end_date):
    """Fetch from NASA POWER API (CSV). Retries on failure."""
    params = {
        "parameters": NASA_PARAMETERS,
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "CSV",
    }

    logger.info(f"Fetching: {location_name} ({start_date} to {end_date})")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(NASA_BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            if attempt == MAX_RETRIES:
                logger.error(f"All retries exhausted for {location_name}")
                return pd.DataFrame()
            time.sleep(RETRY_DELAY * attempt)

    # Parse CSV (skip header block ending with -END HEADER-)
    try:
        lines = response.text.split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            if "-END HEADER-" in line:
                data_start = i + 1
                break
        csv_text = "\n".join(lines[data_start:])
        df = pd.read_csv(StringIO(csv_text), skipinitialspace=True)
    except Exception as exc:
        logger.error(f"CSV parse failed for {location_name}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Build date column
    try:
        df["date"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" +
            df["MO"].astype(str).str.zfill(2) + "-" +
            df["DY"].astype(str).str.zfill(2),
            format="%Y-%m-%d", errors="coerce",
        )
    except KeyError:
        logger.error(f"Missing date columns for {location_name}")
        return pd.DataFrame()

    df.drop(columns=["YEAR", "MO", "DY", "DOY"], errors="ignore", inplace=True)
    df["location"] = location_name
    df["latitude"] = latitude
    df["longitude"] = longitude

    for col in ["ALLSKY_SFC_SW_DWN", "WS10M", "T2M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([-999.0, -999], np.nan)

    df.dropna(subset=["date"], inplace=True)
    logger.info(f"Got {len(df)} rows for {location_name}")
    time.sleep(1.5)
    return df


def engineer_features(df):
    """
    Feature engineering:
    - Green score (solar OR wind, not both required)
    - Lag features of raw solar/wind (1d, 3d, 7d)
    - Rolling 7d average of raw solar/wind
    - Calendar features
    - Location one-hot encoding
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values(["location", "date"], inplace=True)

    # Calendar
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Green score: hours where solar OR wind can generate power
    scores = df.apply(
        lambda row: compute_green_score(
            row.get("ALLSKY_SFC_SW_DWN", 0) or 0,
            row.get("WS10M", 0) or 0,
            row.get("month", 1),
        ),
        axis=1, result_type="expand",
    )
    df["green_compute_hours"] = scores[0]
    df["solar_viable_hours"] = scores[1]
    df["wind_viable_hours"] = scores[2]

    # Lag features: past solar and wind values (no target leakage)
    for lag in [1, 3, 7]:
        df[f"solar_lag_{lag}d"] = df.groupby("location")["ALLSKY_SFC_SW_DWN"].shift(lag)
        df[f"wind_lag_{lag}d"] = df.groupby("location")["WS10M"].shift(lag)

    # Rolling 7d mean of raw inputs (not target)
    df["solar_7d_avg"] = (
        df.groupby("location")["ALLSKY_SFC_SW_DWN"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["wind_7d_avg"] = (
        df.groupby("location")["WS10M"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Location one-hot (always create all 5 columns)
    location_dummies = pd.get_dummies(df["location"], prefix="loc")
    for loc_name in LOCATIONS:
        col = f"loc_{loc_name}"
        if col not in location_dummies.columns:
            location_dummies[col] = 0
    df = pd.concat([df, location_dummies], axis=1)

    return df


def load_to_sqlite(df):
    """Safe load: preserve data for locations not in current batch."""
    if df.empty:
        logger.warning("Empty dataframe, skipping load.")
        return

    ensure_data_dir()
    conn = get_db_connection()
    df_save = df.copy()
    df_save["date"] = df_save["date"].astype(str)

    # Preserve existing data for locations not in this batch
    existing = pd.DataFrame()
    try:
        existing = pd.read_sql_query("SELECT * FROM energy_data", conn)
        new_locations = df_save["location"].unique().tolist()
        existing = existing[~existing["location"].isin(new_locations)]
    except Exception:
        pass

    # Align columns between old and new
    if not existing.empty:
        all_cols = list(set(existing.columns) | set(df_save.columns))
        for col in all_cols:
            if col not in existing.columns:
                existing[col] = 0
            if col not in df_save.columns:
                df_save[col] = 0
        combined = pd.concat([existing, df_save], ignore_index=True)
    else:
        combined = df_save

    conn.execute("DROP TABLE IF EXISTS energy_data")
    combined.to_sql("energy_data", conn, if_exists="replace", index=False)

    conn.execute(
        "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
        ("last_etl_run", datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    logger.info(f"Loaded {len(combined)} rows.")


def run_full_etl():
    logger.info("=" * 50)
    logger.info("STARTING ETL")
    logger.info("=" * 50)

    end_date = datetime.now() - timedelta(days=2)
    start_date = end_date - timedelta(days=3 * 365)

    all_frames = []
    for loc_name, loc_info in LOCATIONS.items():
        df = fetch_nasa_data(
            loc_name, loc_info["lat"], loc_info["lon"],
            start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"),
        )
        if not df.empty:
            all_frames.append(df)

    if not all_frames:
        logger.error("No data fetched.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.drop_duplicates(subset=["date", "location"], keep="last", inplace=True)

    logger.info(f"Combined: {len(combined)} rows, {combined['location'].nunique()} locations")

    featured = engineer_features(combined)

    logger.info(
        f"Solar: {featured['ALLSKY_SFC_SW_DWN'].min():.1f} - "
        f"{featured['ALLSKY_SFC_SW_DWN'].max():.1f} kWh/m2/day"
    )
    logger.info(
        f"Wind: {featured['WS10M'].min():.1f} - {featured['WS10M'].max():.1f} m/s"
    )
    logger.info(
        f"Green hours: {featured['green_compute_hours'].min():.1f} - "
        f"{featured['green_compute_hours'].max():.1f} hrs/day"
    )
    logger.info(
        f"Green hours avg: {featured['green_compute_hours'].mean():.1f} hrs/day"
    )

    load_to_sqlite(featured)

    logger.info("ETL COMPLETE")
    return featured


if __name__ == "__main__":
    run_full_etl()