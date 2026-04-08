"""
model.py
XGBoost model that predicts green_compute_hours from raw weather inputs,
lag features, and calendar/location features. No target-derived features.
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import get_db_connection, get_model_path, LOCATIONS, compute_green_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Only raw inputs, lags, calendar, and location dummies.
# No solar_viable_hours, wind_viable_hours, or rolling avg of target.
BASE_FEATURES = [
    "ALLSKY_SFC_SW_DWN",
    "WS10M",
    "T2M",
    "is_weekend",
    "month",
    "day_of_year",
    "solar_lag_1d",
    "solar_lag_3d",
    "solar_lag_7d",
    "wind_lag_1d",
    "wind_lag_3d",
    "wind_lag_7d",
    "solar_7d_avg",
    "wind_7d_avg",
]

TARGET_COL = "green_compute_hours"


def _get_feature_cols(df):
    """Return BASE_FEATURES + any location one-hot columns present in df."""
    loc_cols = [c for c in df.columns if c.startswith("loc_")]
    return [c for c in BASE_FEATURES + loc_cols if c in df.columns]


def load_training_data():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM energy_data", conn)
        conn.close()
    except Exception as exc:
        logger.error(f"Could not load training data: {exc}")
        return pd.DataFrame()

    if df.empty:
        logger.error("energy_data is empty. Run ETL first.")
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", TARGET_COL], inplace=True)

    # Drop rows where lag features are NaN (first 7 days per location)
    lag_cols = [c for c in df.columns if "lag_" in c]
    df.dropna(subset=lag_cols, inplace=True)

    logger.info(f"Training data: {len(df)} rows after dropping NaN lags")
    return df


def train_model():
    logger.info("Starting model training...")

    df = load_training_data()
    if df.empty:
        return {"error": "No training data available."}

    feature_cols = _get_feature_cols(df)
    if len(feature_cols) < 5:
        return {"error": f"Not enough features: {feature_cols}"}

    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]

    # Time-based split: first 80% train, last 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Metrics -- MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    model_path = get_model_path()
    joblib.dump({"model": model, "features": feature_cols}, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    try:
        conn = get_db_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        for key, val in [
            ("model_mae", f"{mae:.4f}"),
            ("model_rmse", f"{rmse:.4f}"),
            ("model_r2", f"{r2:.4f}"),
            ("model_trained_at", datetime.now().isoformat()),
            ("model_n_samples", str(len(X))),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)",
                (key, val),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Could not save metrics: {exc}")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "n_samples": len(X),
    }


def predict_next_30_days(location_name):
    """
    30-day forecast using autoregressive prediction.
    Day 1 uses real lag values from the last historical row.
    Day 2+ uses predicted values fed back as lags.
    """
    model_path = get_model_path()

    # Check data exists
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='energy_data'"
        )
        if cursor.fetchone() is None:
            conn.close()
            return pd.DataFrame()
        conn.close()
    except Exception:
        return pd.DataFrame()

    # Load or train model
    if not os.path.exists(model_path):
        result = train_model()
        if "error" in result:
            return pd.DataFrame()

    if not os.path.exists(model_path):
        return pd.DataFrame()

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]

    # Load historical data for this location
    try:
        conn = get_db_connection()
        df_hist = pd.read_sql_query(
            "SELECT * FROM energy_data WHERE location = ? ORDER BY date",
            conn, params=(location_name,),
        )
        conn.close()
    except Exception:
        return pd.DataFrame()

    if df_hist.empty:
        return pd.DataFrame()

    df_hist["date"] = pd.to_datetime(df_hist["date"])

    # Get the last 7 days of actual data for initial lag values
    last_7 = df_hist.tail(7).copy()
    solar_history = list(last_7["ALLSKY_SFC_SW_DWN"].values)
    wind_history = list(last_7["WS10M"].values)

    last_date = df_hist["date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")

    # Location one-hot: figure out which loc_ columns exist in feature_cols
    loc_cols = {c: 0 for c in feature_cols if c.startswith("loc_")}
    loc_col_name = f"loc_{location_name}"
    if loc_col_name in loc_cols:
        loc_cols[loc_col_name] = 1

    predictions = []

    for d in future_dates:
        doy = d.dayofyear
        month = d.month
        is_weekend = 1 if d.dayofweek in [5, 6] else 0

        # Seasonal estimate for today's raw values (median of same +-7 DOY)
        mask = (df_hist["day_of_year"] >= doy - 7) & (df_hist["day_of_year"] <= doy + 7)
        seasonal = df_hist[mask]
        if seasonal.empty:
            seasonal = df_hist

        solar_today = seasonal["ALLSKY_SFC_SW_DWN"].median()
        wind_today = seasonal["WS10M"].median()
        temp_today = seasonal["T2M"].median()

        # Lag features from actual history (first few days) or predicted history
        solar_lag_1 = solar_history[-1] if len(solar_history) >= 1 else solar_today
        solar_lag_3 = solar_history[-3] if len(solar_history) >= 3 else solar_today
        solar_lag_7 = solar_history[-7] if len(solar_history) >= 7 else solar_today

        wind_lag_1 = wind_history[-1] if len(wind_history) >= 1 else wind_today
        wind_lag_3 = wind_history[-3] if len(wind_history) >= 3 else wind_today
        wind_lag_7 = wind_history[-7] if len(wind_history) >= 7 else wind_today

        solar_7d = np.mean(solar_history[-7:]) if len(solar_history) >= 7 else solar_today
        wind_7d = np.mean(wind_history[-7:]) if len(wind_history) >= 7 else wind_today

        row = {
            "ALLSKY_SFC_SW_DWN": solar_today,
            "WS10M": wind_today,
            "T2M": temp_today,
            "is_weekend": is_weekend,
            "month": month,
            "day_of_year": doy,
            "solar_lag_1d": solar_lag_1,
            "solar_lag_3d": solar_lag_3,
            "solar_lag_7d": solar_lag_7,
            "wind_lag_1d": wind_lag_1,
            "wind_lag_3d": wind_lag_3,
            "wind_lag_7d": wind_lag_7,
            "solar_7d_avg": solar_7d,
            "wind_7d_avg": wind_7d,
        }
        row.update(loc_cols)

        X_row = pd.DataFrame([row])
        # Ensure column order matches training
        for col in feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[feature_cols]

        pred = model.predict(X_row)[0]
        pred = float(np.clip(pred, 0, 24))
        predictions.append(pred)

        # Update history for next iteration (autoregressive)
        solar_history.append(solar_today)
        wind_history.append(wind_today)

    # Confidence: use per-month std from historical data
    df_hist["_month"] = df_hist["date"].dt.month
    month_std = df_hist.groupby("_month")["green_compute_hours"].std().to_dict()

    conf_lower = []
    conf_upper = []
    for i, d in enumerate(future_dates):
        std = month_std.get(d.month, 1.0)
        margin = std * 0.6
        conf_lower.append(round(max(predictions[i] - margin, 0), 2))
        conf_upper.append(round(min(predictions[i] + margin, 24), 2))

    result = pd.DataFrame({
        "date": future_dates,
        "predicted_green_hours": [round(p, 2) for p in predictions],
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
    })

    logger.info(
        f"Forecast for {location_name}: avg={result['predicted_green_hours'].mean():.2f} hrs/day"
    )
    return result


if __name__ == "__main__":
    metrics = train_model()
    print(f"Training: {metrics}")
    for loc in LOCATIONS:
        preds = predict_next_30_days(loc)
        if not preds.empty:
            print(f"{loc}: avg={preds['predicted_green_hours'].mean():.2f} hrs/day")