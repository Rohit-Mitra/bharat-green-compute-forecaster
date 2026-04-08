"""
=============================================================================
utils.py — Bharat Green Compute Forecaster
=============================================================================
Shared helper functions: DB connections, paths, constants, date utilities.
=============================================================================
"""

import os
import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")
DB_PATH = os.path.join(DATA_DIR, "energy.db")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")

# ---------------------------------------------------------------------------
# NASA POWER API constants
# ---------------------------------------------------------------------------
NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = "ALLSKY_SFC_SW_DWN,WS10M,T2M"

# ---------------------------------------------------------------------------
# Hardcoded Indian locations (5 key AI data center + renewable hubs)
# ---------------------------------------------------------------------------
LOCATIONS = {
    "Delhi NCR": {
        "lat": 28.6139,
        "lon": 77.2090,
        "description": "National Capital Region — India's political & tech hub",
        "icon_color": "blue",
    },
    "Bangalore, Karnataka": {
        "lat": 12.9716,
        "lon": 77.5946,
        "description": "Silicon Valley of India — largest IT/AI cluster",
        "icon_color": "green",
    },
    "Mumbai, Maharashtra": {
        "lat": 19.0760,
        "lon": 72.8777,
        "description": "Financial capital — major data center corridor",
        "icon_color": "orange",
    },
    "Hyderabad, Telangana": {
        "lat": 17.3850,
        "lon": 78.4867,
        "description": "HITEC City — fastest-growing data center market",
        "icon_color": "purple",
    },
    "Jodhpur, Rajasthan": {
        "lat": 26.2389,
        "lon": 73.0242,
        "description": "Thar Desert solar hub — India's highest solar irradiance",
        "icon_color": "red",
    },
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def ensure_data_dir() -> None:
    """Create the data/ directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)


def get_db_path() -> str:
    """Return the full path to energy.db."""
    ensure_data_dir()
    return DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection to energy.db."""
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
    return conn


def get_model_path() -> str:
    """Return the full path to model.pkl."""
    ensure_data_dir()
    return MODEL_PATH


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def format_date_display(dt) -> str:
    """Format a datetime for display in the dashboard."""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    return dt.strftime("%B %d, %Y at %I:%M %p")


def get_last_etl_time() -> str | None:
    """Read last ETL run timestamp from SQLite metadata."""
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT value FROM etl_metadata WHERE key = 'last_etl_run'"
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def get_model_metrics() -> dict:
    """Read saved model metrics from SQLite metadata."""
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


# ---------------------------------------------------------------------------
# Green Compute calculation helpers
# ---------------------------------------------------------------------------

def estimate_co2_saved(green_hours: float, power_mw: float = 50.0) -> float:
    """
    Estimate CO₂ saved (in tonnes) by using green energy instead of grid.

    Parameters
    ----------
    green_hours : float
        Total green compute hours.
    power_mw : float
        Power consumption of the data center in MW (default 50 MW).

    Returns
    -------
    float
        Estimated CO₂ savings in metric tonnes.
        India grid emission factor ≈ 0.82 tCO₂/MWh (CEA 2024).
    """
    india_grid_emission_factor = 0.82  # tCO₂ per MWh
    energy_mwh = green_hours * power_mw  # MWh
    return round(energy_mwh * india_grid_emission_factor, 2)


def estimate_grok_models(green_hours_per_day: float, demand_mw: float) -> float:
    """
    Estimate how many 'Grok-scale' AI model training runs can be powered
    sustainably per month.

    Assumptions (public estimates):
        - Grok-3 training ≈ 100 MW for ~90 days ≈ 216,000 MWh
        - 1 Grok-unit = 216,000 MWh

    Parameters
    ----------
    green_hours_per_day : float
        Average daily green compute hours available.
    demand_mw : float
        Available green power capacity in MW.

    Returns
    -------
    float
        Number of Grok-scale training runs achievable per month.
    """
    grok_energy_mwh = 216_000  # total MWh for one Grok-3 training
    monthly_green_mwh = green_hours_per_day * 30 * demand_mw
    if grok_energy_mwh == 0:
        return 0.0
    return round(monthly_green_mwh / grok_energy_mwh, 4)