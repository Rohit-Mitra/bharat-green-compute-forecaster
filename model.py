"""
=============================================================================
model.py — Bharat Green Compute Forecaster
=============================================================================
XGBoost-based ML model for predicting future Green Compute Hours.

- train_model()             : train on historical data, save model.pkl
- predict_next_30_days()    : generate 30-day forecast for a location
=============================================================================
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import get_db_connection, get_model_path, LOCATIONS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns used for training (must match engineer_features output)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "ALLSKY_SFC_SW_DWN",
    "WS10M",
    "T2M",
    "solar_yield_estimate",
    "wind_yield_estimate",
    "rolling_7d_avg",
    "is_weekend",
    "month",
    "day_of_year",
    "lat_encoded",
    "lon_encoded",
]

TARGET_COL = "green_compute_hours"


# ===========================================================================
# 1. LOAD DATA FROM SQLITE
# ===========================================================================

def load_training_data() -> pd.DataFrame:
    """Load energy_data table from SQLite and prepare for training."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM energy_data", conn)
        conn.close()
    except Exception as exc:
        logger.error(f"❌ Could not load training data: {exc}")
        return pd.DataFrame()

    if df.empty:
        logger.error("❌ energy_data table is empty. Run ETL first.")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", TARGET_COL], inplace=True)

    # Encode location as lat/lon numerical features
    df["lat_encoded"] = df["latitude"].astype(float)
    df["lon_encoded"] = df["longitude"].astype(float)

    # Fill any remaining NaNs in features
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"📊 Training data loaded: {len(df)} rows")
    return df


# ===========================================================================
# 2. TRAIN MODEL
# ===========================================================================

def train_model() -> dict:
    """
    Train an XGBoost Regressor on historical Green Compute Hours data.

    Returns
    -------
    dict
        Training metrics: {mae, rmse, r2, n_samples, timestamp}
    """
    logger.info("🏋️ Starting model training...")

    df = load_training_data()
    if df.empty:
        return {"error": "No training data available."}

    # Filter to rows where all feature columns exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_features) < 5:
        logger.error(f"❌ Not enough features: {available_features}")
        return {"error": "Insufficient features."}

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    # Train/test split (80/20, time-aware: last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- XGBoost Regressor --------------------------------------------------
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate -----------------------------------------------------------
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"📈 Model Metrics — MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

    # --- Save model ---------------------------------------------------------
    model_path = get_model_path()
    joblib.dump(
        {"model": model, "features": available_features},
        model_path,
    )
    logger.info(f"💾 Model saved to {model_path}")

    # --- Save metrics to DB -------------------------------------------------
    try:
        conn = get_db_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
            ("model_mae", f"{mae:.4f}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
            ("model_rmse", f"{rmse:.4f}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
            ("model_r2", f"{r2:.4f}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
            ("model_trained_at", datetime.now().isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
            ("model_n_samples", str(len(X))),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"⚠️ Could not save metrics to DB: {exc}")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": len(X),
        "timestamp": datetime.now().isoformat(),
    }


# ===========================================================================
# 3. PREDICT NEXT 30 DAYS
# ===========================================================================

def predict_next_30_days(location_name: str) -> pd.DataFrame:
    """
    Generate a 30-day Green Compute Hours forecast for a given location.

    Uses the trained XGBoost model + synthetic future features
    derived from historical seasonal patterns.

    Parameters
    ----------
    location_name : str
        One of the 5 hardcoded Indian city names.

    Returns
    -------
    pd.DataFrame
        Columns: [date, predicted_green_hours, confidence_lower, confidence_upper]
    """
    model_path = get_model_path()

    # --- Load model ---------------------------------------------------------
    if not os.path.exists(model_path):
        logger.warning("⚠️ Model not found. Training now...")
        train_model()

    if not os.path.exists(model_path):
        logger.error("❌ Model still not found after training attempt.")
        return pd.DataFrame()

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]

    # --- Load historical data for this location to derive patterns ----------
    try:
        conn = get_db_connection()
        df_hist = pd.read_sql_query(
            "SELECT * FROM energy_data WHERE location = ? ORDER BY date",
            conn,
            params=(location_name,),
        )
        conn.close()
    except Exception as exc:
        logger.error(f"❌ Could not load historical data: {exc}")
        return pd.DataFrame()

    if df_hist.empty:
        logger.error(f"❌ No historical data for {location_name}")
        return pd.DataFrame()

    df_hist["date"] = pd.to_datetime(df_hist["date"])

    # --- Build future feature matrix ----------------------------------------
    loc_info = LOCATIONS.get(location_name, {})
    lat = loc_info.get("lat", df_hist["latitude"].iloc[0])
    lon = loc_info.get("lon", df_hist["longitude"].iloc[0])

    last_date = df_hist["date"].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq="D",
    )

    future_rows = []
    for d in future_dates:
        # Use same day-of-year from historical data (seasonal average)
        doy = d.dayofyear
        month = d.month
        is_weekend = 1 if d.dayofweek in [5, 6] else 0

        # Seasonal average from historical data for this day-of-year (±7 days)
        mask = (df_hist["day_of_year"] >= doy - 7) & (
            df_hist["day_of_year"] <= doy + 7
        )
        seasonal_slice = df_hist[mask]

        if seasonal_slice.empty:
            seasonal_slice = df_hist  # fallback to all data

        row = {
            "date": d,
            "ALLSKY_SFC_SW_DWN": seasonal_slice["ALLSKY_SFC_SW_DWN"].median(),
            "WS10M": seasonal_slice["WS10M"].median(),
            "T2M": seasonal_slice["T2M"].median(),
            "solar_yield_estimate": seasonal_slice[
                "solar_yield_estimate"
            ].median(),
            "wind_yield_estimate": seasonal_slice["wind_yield_estimate"].median(),
            "rolling_7d_avg": seasonal_slice["rolling_7d_avg"].median(),
            "is_weekend": is_weekend,
            "month": month,
            "day_of_year": doy,
            "lat_encoded": lat,
            "lon_encoded": lon,
        }
        future_rows.append(row)

    df_future = pd.DataFrame(future_rows)

    # --- Predict ------------------------------------------------------------
    X_future = df_future[[c for c in feature_cols if c in df_future.columns]]

    # Add any missing columns as 0
    for col in feature_cols:
        if col not in X_future.columns:
            X_future[col] = 0

    X_future = X_future[feature_cols]

    predictions = model.predict(X_future)

    # Clip predictions to valid range [0, 24]
    predictions = np.clip(predictions, 0, 24)

    # --- Confidence interval (simple heuristic: ±15% or ±std) ---------------
    hist_std = df_hist["green_compute_hours"].std()
    confidence_margin = max(hist_std * 0.5, 0.5)  # at least 0.5 hours

    result = pd.DataFrame(
        {
            "date": future_dates,
            "predicted_green_hours": np.round(predictions, 2),
            "confidence_lower": np.round(
                np.clip(predictions - confidence_margin, 0, 24), 2
            ),
            "confidence_upper": np.round(
                np.clip(predictions + confidence_margin, 0, 24), 2
            ),
        }
    )

    logger.info(
        f"🔮 Generated 30-day forecast for {location_name}: "
        f"avg={result['predicted_green_hours'].mean():.2f} hrs/day"
    )

    return result


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    # Train model
    metrics = train_model()
    print(f"\n📊 Training Metrics: {metrics}")

    # Predict for each location
    for loc in LOCATIONS:
        preds = predict_next_30_days(loc)
        if not preds.empty:
            print(f"\n🔮 {loc}: avg predicted = {preds['predicted_green_hours'].mean():.2f} hrs/day")