"""
utils.py
Constants, DB helpers, green energy scoring.
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

# Solar thresholds (ALLSKY_SFC_SW_DWN is kWh/m2/day)
# India range: 1-7 kWh/m2/day
# Below 2.0: too weak for meaningful PV generation
# Above 6.0: near-maximum, fraction saturates at 1.0
SOLAR_MIN = 2.0
SOLAR_MAX = 6.0

# Wind thresholds (WS10M is m/s)
# Lowered for modern small/medium turbines common in India
# Source: MNRE small wind turbine specifications
WIND_CUT_IN = 2.5
WIND_OPTIMAL_LOW = 4.0
WIND_OPTIMAL_HIGH = 15.0
WIND_CUT_OUT = 25.0

# Average daylight hours per month across India (hours of sun above horizon)
# Source: India Meteorological Department
DAYLIGHT_HOURS = {
    1: 9.5, 2: 10.0, 3: 11.0, 4: 12.0, 5: 12.5, 6: 12.5,
    7: 11.5, 8: 11.0, 9: 11.0, 10: 10.5, 11: 9.5, 12: 9.0,
}

# India grid CO2 factor: 0.82 tCO2/MWh
# Source: Central Electricity Authority, CO2 Baseline Database v19, 2024
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


def compute_green_score(solar_kwh, wind_ms, month):
    """
    Green Compute Hours: hours/day where renewable energy (solar OR wind OR both)
    can sustain compute loads.

    NOT "both must be active simultaneously." Solar alone during daytime counts.
    Wind alone at night counts. Having both is a bonus.

    Method:
        During daylight: P(green) = P(solar) + P(wind) - P(solar)*P(wind)
        During night: P(green) = P(wind)
        Total = daylight * P(day_green) + night * P(night_green)

    Returns (green_hours, solar_hours, wind_hours)
    """
    # Solar fraction: linear ramp from SOLAR_MIN to SOLAR_MAX
    if solar_kwh <= SOLAR_MIN:
        solar_frac = 0.0
    elif solar_kwh >= SOLAR_MAX:
        solar_frac = 1.0
    else:
        solar_frac = (solar_kwh - SOLAR_MIN) / (SOLAR_MAX - SOLAR_MIN)

    # Wind fraction: turbine power curve
    if wind_ms < WIND_CUT_IN:
        wind_frac = 0.0
    elif wind_ms < WIND_OPTIMAL_LOW:
        wind_frac = (wind_ms - WIND_CUT_IN) / (WIND_OPTIMAL_LOW - WIND_CUT_IN)
    elif wind_ms <= WIND_OPTIMAL_HIGH:
        wind_frac = 1.0
    elif wind_ms < WIND_CUT_OUT:
        wind_frac = (WIND_CUT_OUT - wind_ms) / (WIND_CUT_OUT - WIND_OPTIMAL_HIGH)
    else:
        wind_frac = 0.0

    daylight = DAYLIGHT_HOURS.get(month, 11.0)
    night = 24.0 - daylight

    # Daytime: either solar or wind or both generates power
    # Probability union: P(A or B) = P(A) + P(B) - P(A)*P(B)
    day_green = min(solar_frac + wind_frac - solar_frac * wind_frac, 1.0)

    # Nighttime: only wind
    night_green = wind_frac

    green_hours = daylight * day_green + night * night_green
    solar_hours = daylight * solar_frac
    wind_hours = 24.0 * wind_frac

    return round(green_hours, 4), round(solar_hours, 4), round(wind_hours, 4)


# --- DB helpers ---

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


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
    """CO2 saved in tonnes = green MWh * grid emission factor."""
    return round(green_energy_mwh * INDIA_GRID_CO2_FACTOR, 2)