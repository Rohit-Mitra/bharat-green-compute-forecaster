"""
etl_pipeline.py
NASA POWER API extraction (CSV), feature engineering, SQLite loading.
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
RETRY_DELAY = 5  # seconds


def fetch_nasa_data(location_name, latitude, longitude, start_date, end_date):
    """
    Fetch daily data from NASA POWER API using CSV format.
    Retries up to MAX_RETRIES on failure with exponential backoff.
    """
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

    # Retry loop with exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(NASA_BASE_URL, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {location_name}: {exc}")
            if attempt == MAX_RETRIES:
                logger.error(f"All retries exhausted for {location_name}")
                return pd.DataFrame()
            time.sleep(RETRY_DELAY * attempt)

    # Parse CSV: skip everything before -END HEADER-
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
        logger.warning(f"Empty response for {location_name}")
        return pd.DataFrame()

    # Build date column
    try:
        df["date"] = pd.to_datetime(
            df["YEAR"].astype(str) + "-" +
            df["MO"].astype(str).str.zfill(2) + "-" +
            df["DY"].astype(str).str.zfill(2),
            format="%Y-%m-%d",
            errors="coerce",
        )
    except KeyError:
        logger.error(f"Missing date columns for {location_name}")
        return pd.DataFrame()

    df.drop(columns=["YEAR", "MO", "DY", "DOY"], errors="ignore", inplace=True)

    df["location"] = location_name
    df["latitude"] = latitude
    df["longitude"] = longitude

    # Replace NASA fill values
    for col in ["ALLSKY_SFC_SW_DWN", "WS10M", "T2M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([-999.0, -999], np.nan)

    df.dropna(subset=["date"], inplace=True)
    logger.info(f"Fetched {len(df)} rows for {location_name}")
    time.sleep(1.5)
    return df


def engineer_features(df):
    """
    Feature engineering. Computes green energy score and time-series lag features.
    No target-derived features are used as model inputs.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.sort_values(["location", "date"], inplace=True)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Green score using proper overlap calculation
    scores = df.apply(
        lambda row: compute_green_score(
            row.get("ALLSKY_SFC_SW_DWN", 0) or 0,
            row.get("WS10M", 0) or 0,
            row.get("month", 1),
        ),
        axis=1,
        result_type="expand",
    )
    df["green_compute_hours"] = scores[0]
    df["solar_viable_hours"] = scores[1]
    df["wind_viable_hours"] = scores[2]

    # Lag features (per location) for time-series modeling
    # These use PAST values only, so no target leakage
    for lag in [1, 3, 7]:
        df[f"solar_lag_{lag}d"] = (
            df.groupby("location")["ALLSKY_SFC_SW_DWN"]
            .shift(lag)
        )
        df[f"wind_lag_{lag}d"] = (
            df.groupby("location")["WS10M"]
            .shift(lag)
        )

    # 7-day rolling mean of RAW inputs (not target)
    df["solar_7d_avg"] = (
        df.groupby("location")["ALLSKY_SFC_SW_DWN"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    df["wind_7d_avg"] = (
        df.groupby("location")["WS10M"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # Location one-hot encoding
    location_dummies = pd.get_dummies(df["location"], prefix="loc")
    df = pd.concat([df, location_dummies], axis=1)

    return df


def load_to_sqlite(df):
    """
    Upsert data into energy_data table.
    Uses INSERT OR REPLACE to avoid losing data on partial ETL runs.
    """
    if df.empty:
        logger.warning("Empty dataframe, nothing to load.")
        return

    ensure_data_dir()
    conn = get_db_connection()

    df_save = df.copy()
    df_save["date"] = df_save["date"].astype(str)

    # Create table if not exists (first run)
    df_save.head(0).to_sql("energy_data", conn, if_exists="append", index=False)

    # Delete only rows for locations we have new data for, then insert
    locations_updated = df_save["location"].unique().tolist()
    for loc in locations_updated:
        conn.execute("DELETE FROM energy_data WHERE location = ?", (loc,))

    df_save.to_sql("energy_data", conn, if_exists="append", index=False)

    # Update metadata
    conn.execute(
        "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
        ("last_etl_run", datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    logger.info(f"Loaded {len(df_save)} rows for {len(locations_updated)} locations.")


def run_full_etl():
    """Fetch all locations, engineer features, load to SQLite."""
    logger.info("=" * 50)
    logger.info("STARTING ETL PIPELINE")
    logger.info("=" * 50)

    end_date = datetime.now() - timedelta(days=2)
    start_date = end_date - timedelta(days=3 * 365)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    all_frames = []
    for loc_name, loc_info in LOCATIONS.items():
        df = fetch_nasa_data(
            loc_name, loc_info["lat"], loc_info["lon"], start_str, end_str
        )
        if not df.empty:
            all_frames.append(df)
        else:
            logger.warning(f"No data for {loc_name}, skipping.")

    if not all_frames:
        logger.error("No data fetched for any location.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.drop_duplicates(subset=["date", "location"], keep="last", inplace=True)

    logger.info(f"Combined: {len(combined)} rows, {combined['location'].nunique()} locations")

    featured = engineer_features(combined)

    # Log sanity check
    logger.info(
        f"Solar range: {featured['ALLSKY_SFC_SW_DWN'].min():.2f} - "
        f"{featured['ALLSKY_SFC_SW_DWN'].max():.2f} kWh/m2/day"
    )
    logger.info(
        f"Wind range: {featured['WS10M'].min():.2f} - "
        f"{featured['WS10M'].max():.2f} m/s"
    )
    logger.info(
        f"Green hours range: {featured['green_compute_hours'].min():.2f} - "
        f"{featured['green_compute_hours'].max():.2f} hrs/day"
    )

    load_to_sqlite(featured)

    logger.info("ETL PIPELINE COMPLETE")
    logger.info("=" * 50)
    return featured


if __name__ == "__main__":
    df = run_full_etl()
    if not df.empty:
        print(f"Rows: {len(df)}")
        print(f"Locations: {df['location'].unique().tolist()}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")