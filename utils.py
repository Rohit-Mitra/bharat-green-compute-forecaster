"""
utils.py
Shared constants, DB helpers, and domain calculations.
"""

import os
import sqlite3
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
DB_PATH = os.path.join(DATA_DIR, "energy.db")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = "ALLSKY_SFC_SW_DWN,WS10M,T2M"

# --- Solar thresholds (ALLSKY_SFC_SW_DWN is kWh/m2/day from NASA) ---
# Below this value, solar generation is negligible
SOLAR_MIN_THRESHOLD = 3.0  # kWh/m2/day
# Above this value, solar fraction saturates at 1.0
SOLAR_NORMALIZE_CAP = 6.5  # kWh/m2/day

# --- Wind thresholds (WS10M is m/s from NASA) ---
# Real turbine power curve zones
WIND_CUT_IN = 3.5       # below this, turbines don't spin
WIND_OPTIMAL_LOW = 5.0   # start of peak efficiency band
WIND_OPTIMAL_HIGH = 15.0  # end of peak efficiency band
WIND_CUT_OUT = 25.0      # above this, turbines shut down for safety

# --- Typical solar daylight hours by month in India (hours of usable sun) ---
# Source: India Meteorological Department averages
INDIA_SOLAR_HOURS = {
    1: 9.5, 2: 10.0, 3: 11.0, 4: 12.0, 5: 12.5, 6: 12.5,
    7: 11.5, 8: 11.0, 9: 11.0, 10: 10.5, 11: 9.5, 12: 9.0,
}

# --- India grid emission factor (tCO2 per MWh) ---
# Source: Central Electricity Authority (CEA), CO2 Baseline Database v19, 2024
INDIA_GRID_CO2_FACTOR = 0.82

LOCATIONS = {
    "Delhi NCR": {
        "lat": 28.6139, "lon": 77.2090,
        "description": "National Capital Region",
    },
    "Bangalore, Karnataka": {
        "lat": 12.9716, "lon": 77.5946,
        "description": "Silicon Valley of India",
    },
    "Mumbai, Maharashtra": {
        "lat": 19.0760, "lon": 72.8777,
        "description": "Financial capital",
    },
    "Hyderabad, Telangana": {
        "lat": 17.3850, "lon": 78.4867,
        "description": "HITEC City data center hub",
    },
    "Jodhpur, Rajasthan": {
        "lat": 26.2389, "lon": 73.0242,
        "description": "Thar Desert solar zone",
    },
}


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def get_db_path():
    ensure_data_dir()
    return DB_PATH


def get_db_connection():
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_model_path():
    ensure_data_dir()
    return MODEL_PATH


def format_date_display(dt):
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%B %d, %Y at %I:%M %p")


def get_last_etl_time():
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT value FROM etl_metadata WHERE key = 'last_etl_run'"
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def get_model_metrics():
    metrics = {}
    try:
        conn = get_db_connection()
        cursor = conn.execute("SELECT key, value FROM etl_metadata")
        for row in cursor.fetchall():
            metrics[row[0]] = row[1]
        conn.close()
    except Exception:
        pass
    return metrics


def estimate_co2_saved(green_energy_mwh):
    """
    CO2 saved in tonnes.
    green_energy_mwh: total green energy in MWh that displaced grid power.
    """
    return round(green_energy_mwh * INDIA_GRID_CO2_FACTOR, 2)


def compute_green_score(solar_kwh, wind_ms, month):
    """
    Compute a green energy score (0-24 scale) representing
    the estimated overlap hours where both solar AND wind
    are generating usable power.

    This is NOT simply 24 * fraction_a * fraction_b.
    Instead we compute separate solar and wind viable hours,
    then estimate their overlap probabilistically.

    Returns (green_score, solar_hours, wind_hours)
    """
    # Solar: compute fraction of solar resource available
    if solar_kwh <= SOLAR_MIN_THRESHOLD:
        solar_fraction = 0.0
    else:
        solar_fraction = min(
            (solar_kwh - SOLAR_MIN_THRESHOLD) /
            (SOLAR_NORMALIZE_CAP - SOLAR_MIN_THRESHOLD),
            1.0,
        )

    # Solar is only available during daylight
    daylight = INDIA_SOLAR_HOURS.get(month, 11.0)
    solar_viable_hours = daylight * solar_fraction

    # Wind: compute fraction from turbine power curve
    if wind_ms < WIND_CUT_IN:
        wind_fraction = 0.0
    elif wind_ms < WIND_OPTIMAL_LOW:
        wind_fraction = (wind_ms - WIND_CUT_IN) / (WIND_OPTIMAL_LOW - WIND_CUT_IN)
    elif wind_ms <= WIND_OPTIMAL_HIGH:
        wind_fraction = 1.0
    elif wind_ms < WIND_CUT_OUT:
        wind_fraction = (WIND_CUT_OUT - wind_ms) / (WIND_CUT_OUT - WIND_OPTIMAL_HIGH)
    else:
        wind_fraction = 0.0

    # Wind can blow 24 hours
    wind_viable_hours = 24.0 * wind_fraction

    # Overlap estimation:
    # P(both available in a given hour) = P(solar_on) * P(wind_on)
    # assuming independence, which is a reasonable first-order assumption
    p_solar = solar_viable_hours / 24.0
    p_wind = wind_viable_hours / 24.0
    overlap_hours = 24.0 * p_solar * p_wind

    # But overlap cannot exceed the smaller of the two
    overlap_hours = min(overlap_hours, solar_viable_hours, wind_viable_hours)

    return round(overlap_hours, 4), round(solar_viable_hours, 4), round(wind_viable_hours, 4)