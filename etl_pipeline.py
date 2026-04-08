"""
=============================================================================
etl_pipeline.py — Bharat Green Compute Forecaster
=============================================================================
Handles all NASA POWER API extraction, data transformation, feature
engineering, and SQLite loading for 5 key Indian cities.

NASA POWER API docs: https://power.larc.nasa.gov/docs/
=============================================================================
"""

import os
import time
import sqlite3
import logging
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np
import requests

from utils import (
    get_db_connection,
    get_db_path,
    ensure_data_dir,
    LOCATIONS,
    NASA_BASE_URL,
    NASA_PARAMETERS,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. EXTRACT — Fetch data from NASA POWER API
# ===========================================================================

def fetch_nasa_data(
    location_name: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily solar, wind, and temperature data from NASA POWER API.

    Parameters
    ----------
    location_name : str
        Human-readable city name (e.g., "Delhi NCR").
    latitude : float
        Latitude of the location.
    longitude : float
        Longitude of the location.
    start_date : str
        Start date in 'YYYYMMDD' format.
    end_date : str
        End date in 'YYYYMMDD' format.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        [date, location, latitude, longitude,
         ALLSKY_SFC_SW_DWN, WS10M, T2M]
    """
    # --- Check SQLite cache first -------------------------------------------
    cached = _load_from_cache(location_name, start_date, end_date)
    if cached is not None and len(cached) > 0:
        logger.info(
            f"✅ Cache hit for {location_name} ({start_date}–{end_date}): "
            f"{len(cached)} rows"
        )
        return cached

    # --- Build API request --------------------------------------------------
    params = {
        "parameters": NASA_PARAMETERS,
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date,
        "end": end_date,
        "format": "JSON",
    }

    logger.info(
        f"🌐 Fetching NASA POWER data for {location_name} "
        f"({start_date}–{end_date})..."
    )

    try:
        response = requests.get(NASA_BASE_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error(f"❌ API request failed for {location_name}: {exc}")
        return pd.DataFrame()

    # --- Parse JSON response ------------------------------------------------
    try:
        data = response.json()
        properties = data.get("properties", {})
        parameter_data = properties.get("parameter", {})
    except (ValueError, KeyError) as exc:
        logger.error(f"❌ Failed to parse API response for {location_name}: {exc}")
        return pd.DataFrame()

    if not parameter_data:
        logger.warning(f"⚠️ No parameter data returned for {location_name}")
        return pd.DataFrame()

    # --- Build DataFrame from JSON parameters --------------------------------
    df = pd.DataFrame(parameter_data)
    df.index.name = "date_str"
    df = df.reset_index()
    df.rename(columns={"index": "date_str"}, inplace=True)

    # Convert date string to datetime
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%d", errors="coerce")
    df.drop(columns=["date_str"], inplace=True)

    # Add location metadata
    df["location"] = location_name
    df["latitude"] = latitude
    df["longitude"] = longitude

    # --- Clean: replace NASA fill values (-999) with NaN --------------------
    for col in ["ALLSKY_SFC_SW_DWN", "WS10M", "T2M"]:
        if col in df.columns:
            df[col] = df[col].replace(-999.0, np.nan)
            df[col] = df[col].replace(-999, np.nan)

    # Drop rows with missing dates
    df.dropna(subset=["date"], inplace=True)

    logger.info(
        f"✅ Fetched {len(df)} rows for {location_name}"
    )

    # --- Cache to SQLite ----------------------------------------------------
    _save_to_cache(df)

    # Small delay to respect rate limits
    time.sleep(1.5)

    return df


def _load_from_cache(
    location_name: str, start_date: str, end_date: str
) -> pd.DataFrame | None:
    """Load previously cached data from SQLite raw_cache table."""
    ensure_data_dir()
    try:
        conn = get_db_connection()
        # Check if table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_cache'"
        )
        if cursor.fetchone() is None:
            conn.close()
            return None

        query = """
            SELECT * FROM raw_cache
            WHERE location = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        df = pd.read_sql_query(query, conn, params=(location_name, start_dt, end_dt))
        conn.close()

        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            return df
        return None
    except Exception as exc:
        logger.debug(f"Cache read error: {exc}")
        return None


def _save_to_cache(df: pd.DataFrame) -> None:
    """Save fetched data to SQLite raw_cache table."""
    if df.empty:
        return
    ensure_data_dir()
    try:
        conn = get_db_connection()
        df_to_save = df.copy()
        df_to_save["date"] = df_to_save["date"].astype(str)
        df_to_save.to_sql("raw_cache", conn, if_exists="append", index=False)
        conn.close()
        logger.info(f"💾 Cached {len(df_to_save)} rows to raw_cache")
    except Exception as exc:
        logger.warning(f"⚠️ Cache write error: {exc}")


# ===========================================================================
# 2. TRANSFORM — Feature Engineering
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required engineered features to the raw data.

    New columns:
        - solar_yield_estimate   : normalized solar fraction (0–1)
        - wind_yield_estimate    : normalized wind fraction (0–1)
        - green_compute_hours    : 24 * solar_fraction * wind_fraction
        - rolling_7d_avg         : 7-day rolling mean of green_compute_hours
        - is_weekend             : 1 if Saturday/Sunday, else 0
        - month                  : month number (1–12)
        - day_of_year            : day of year (1–366)
    """
    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    # --- Solar yield estimate -----------------------------------------------
    # ALLSKY_SFC_SW_DWN is in kWh/m²/day (NASA POWER units)
    # We treat 300+ as "good solar" → fraction = min(value/300, 1.0)
    # Some days can exceed 300 significantly → cap at ~8 kWh/m²/day for norm
    max_solar = 8.0  # approximate peak solar irradiance in kWh/m²/day
    df["solar_yield_estimate"] = (
        df["ALLSKY_SFC_SW_DWN"].fillna(0).clip(lower=0) / max_solar
    ).clip(upper=1.0)

    # --- Wind yield estimate ------------------------------------------------
    # Optimal wind: 5–15 m/s → fraction peaks at 1.0 in that band
    ws = df["WS10M"].fillna(0)
    wind_fraction = pd.Series(0.0, index=df.index)

    # Inside optimal band [5, 15] → fraction based on how centred
    in_band = (ws >= 5) & (ws <= 15)
    wind_fraction[in_band] = 1.0

    # Below 5 → ramp up linearly from 0 at 0 m/s to 1.0 at 5 m/s
    below = ws < 5
    wind_fraction[below] = (ws[below] / 5.0).clip(lower=0.0)

    # Above 15 → ramp down linearly from 1.0 at 15 m/s to 0 at 25 m/s
    above = ws > 15
    wind_fraction[above] = ((25.0 - ws[above]) / 10.0).clip(lower=0.0)

    df["wind_yield_estimate"] = wind_fraction.clip(lower=0.0, upper=1.0)

    # --- Green Compute Hours ------------------------------------------------
    # Hours/day where BOTH solar AND wind are good
    df["green_compute_hours"] = (
        24.0 * df["solar_yield_estimate"] * df["wind_yield_estimate"]
    )

    # --- Rolling 7-day average (per location) --------------------------------
    df.sort_values(["location", "date"], inplace=True)
    df["rolling_7d_avg"] = (
        df.groupby("location")["green_compute_hours"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    # --- Calendar features --------------------------------------------------
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    return df


# ===========================================================================
# 3. LOAD — Write processed data to SQLite
# ===========================================================================

def load_to_sqlite(df: pd.DataFrame) -> None:
    """
    Load the fully engineered DataFrame into the `energy_data` table
    in SQLite (data/energy.db). Replaces existing data.
    """
    if df.empty:
        logger.warning("⚠️ Empty DataFrame — nothing to load.")
        return

    ensure_data_dir()
    conn = get_db_connection()

    df_save = df.copy()
    df_save["date"] = df_save["date"].astype(str)

    # Drop and recreate for clean load
    conn.execute("DROP TABLE IF EXISTS energy_data")
    df_save.to_sql("energy_data", conn, if_exists="replace", index=False)

    # Record ETL timestamp
    conn.execute("DROP TABLE IF EXISTS etl_metadata")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
        ("last_etl_run", datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()

    logger.info(f"✅ Loaded {len(df_save)} rows into energy_data table.")


# ===========================================================================
# 4. ORCHESTRATOR — Run full ETL pipeline
# ===========================================================================

def run_full_etl() -> pd.DataFrame:
    """
    Master ETL function:
      1. Fetch NASA POWER data for all 5 Indian locations (last 3 years)
      2. Clean & engineer features
      3. Load to SQLite
      4. Return the final DataFrame

    Returns
    -------
    pd.DataFrame
        Fully processed and feature-engineered data.
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING FULL ETL PIPELINE")
    logger.info("=" * 60)

    # Date range: ~3 years back to today (or near-today)
    end_date = datetime.now() - timedelta(days=2)  # NASA data has ~2-day lag
    start_date = end_date - timedelta(days=3 * 365)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    all_frames = []

    for loc_name, loc_info in LOCATIONS.items():
        logger.info(f"📍 Processing: {loc_name}")
        df = fetch_nasa_data(
            location_name=loc_name,
            latitude=loc_info["lat"],
            longitude=loc_info["lon"],
            start_date=start_str,
            end_date=end_str,
        )
        if not df.empty:
            all_frames.append(df)
        else:
            logger.warning(f"⚠️ No data for {loc_name} — skipping.")

    if not all_frames:
        logger.error("❌ No data fetched for any location!")
        return pd.DataFrame()

    # Combine all locations
    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates (from overlapping cache + fresh fetch)
    combined.drop_duplicates(subset=["date", "location"], keep="last", inplace=True)

    logger.info(f"📊 Combined dataset: {len(combined)} rows across {combined['location'].nunique()} locations")

    # Engineer features
    logger.info("⚙️ Engineering features...")
    featured = engineer_features(combined)

    # Load to SQLite
    logger.info("💾 Loading to SQLite...")
    load_to_sqlite(featured)

    logger.info("=" * 60)
    logger.info("✅ ETL PIPELINE COMPLETE")
    logger.info("=" * 60)

    return featured


# ===========================================================================
# CLI entry point — run ETL directly
# ===========================================================================

if __name__ == "__main__":
    df = run_full_etl()
    if not df.empty:
        print(f"\n{'='*40}")
        print(f"Total rows: {len(df)}")
        print(f"Locations:  {df['location'].unique().tolist()}")
        print(f"Date range: {df['date'].min()} → {df['date'].max()}")
        print(f"Columns:    {df.columns.tolist()}")
        print(f"{'='*40}")