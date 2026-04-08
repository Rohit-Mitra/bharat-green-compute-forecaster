"""
model.py
XGBoost model for forecasting green compute hours.

Key design decision: today's solar/wind (ALLSKY_SFC_SW_DWN, WS10M) are NOT
used as features because they directly determine the target via the scoring
formula. The model must predict from PAST observations only (lag features,
rolling averages, calendar, location). This forces it to learn actual
weather patterns instead of arithmetic.
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import get_db_connection, get_model_path, LOCATIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Features the model can see: only PAST values and calendar/location info.
# Explicitly excluded: ALLSKY_SFC_SW_DWN (today), WS10M (today),
# green_compute_hours (target), solar_viable_hours, wind_viable_hours
BASE_FEATURES = [
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
    loc_cols = sorted([c for c in df.columns if c.startswith("loc_")])
    return [c for c in BASE_FEATURES + loc_cols if c in df.columns]


def load_training_data():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM energy_data", conn)
        conn.close()
    except Exception as exc:
        logger.error(f"Could not load data: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date", TARGET_COL], inplace=True)

    # Drop rows where lags are NaN (first 7 days per location)
    lag_cols = [c for c in df.columns if "lag_" in c]
    df.dropna(subset=lag_cols, inplace=True)

    logger.info(f"Training data: {len(df)} rows")
    return df


def train_model():
    logger.info("Training model...")

    df = load_training_data()
    if df.empty:
        return {"error": "No data. Run ETL first."}

    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]

    # Time-based 80/20 split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    joblib.dump({"model": model, "features": feature_cols}, get_model_path())

    try:
        conn = get_db_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS etl_metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        for k, v in [
            ("model_mae", f"{mae:.4f}"), ("model_rmse", f"{rmse:.4f}"),
            ("model_r2", f"{r2:.4f}"),
            ("model_trained_at", datetime.now().isoformat()),
            ("model_n_samples", str(len(X))),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO etl_metadata (key, value) VALUES (?, ?)", (k, v)
            )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def predict_next_30_days(location_name):
    """
    30-day forecast using lag features.

    For future days, solar/wind lag values come from:
    - Days 1-7: real historical observations
    - Days 8+: seasonal medians filling the buffer

    This means accuracy naturally degrades for days further out,
    which is realistic for weather-dependent forecasting.
    """
    model_path = get_model_path()

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

    if not os.path.exists(model_path):
        result = train_model()
        if "error" in result:
            return pd.DataFrame()

    if not os.path.exists(model_path):
        return pd.DataFrame()

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_cols = artifact["features"]

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

    # Last 7 days of real observations for initial lag buffer
    last_rows = df_hist.tail(7)
    solar_buffer = list(last_rows["ALLSKY_SFC_SW_DWN"].fillna(0).values)
    wind_buffer = list(last_rows["WS10M"].fillna(0).values)

    last_date = df_hist["date"].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq="D")

    # Location one-hot columns
    loc_cols = {c: 0 for c in feature_cols if c.startswith("loc_")}
    loc_key = f"loc_{location_name}"
    if loc_key in loc_cols:
        loc_cols[loc_key] = 1

    predictions = []

    for d in future_dates:
        doy = d.dayofyear
        month = d.month

        # Seasonal estimate for today (used to extend the buffer)
        mask = (df_hist["day_of_year"] >= doy - 7) & (df_hist["day_of_year"] <= doy + 7)
        seasonal = df_hist[mask] if mask.sum() > 0 else df_hist
        solar_est = seasonal["ALLSKY_SFC_SW_DWN"].median()
        wind_est = seasonal["WS10M"].median()
        temp_est = seasonal["T2M"].median()

        # Lag features from buffer (real data for first days, seasonal after)
        n = len(solar_buffer)
        row = {
            "T2M": temp_est,
            "is_weekend": 1 if d.dayofweek in [5, 6] else 0,
            "month": month,
            "day_of_year": doy,
            "solar_lag_1d": solar_buffer[-1] if n >= 1 else solar_est,
            "solar_lag_3d": solar_buffer[-3] if n >= 3 else solar_est,
            "solar_lag_7d": solar_buffer[-7] if n >= 7 else solar_est,
            "wind_lag_1d": wind_buffer[-1] if n >= 1 else wind_est,
            "wind_lag_3d": wind_buffer[-3] if n >= 3 else wind_est,
            "wind_lag_7d": wind_buffer[-7] if n >= 7 else wind_est,
            "solar_7d_avg": np.mean(solar_buffer[-7:]) if n >= 7 else solar_est,
            "wind_7d_avg": np.mean(wind_buffer[-7:]) if n >= 7 else wind_est,
        }
        row.update(loc_cols)

        X_row = pd.DataFrame([row])
        for col in feature_cols:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[feature_cols]

        pred = float(np.clip(model.predict(X_row)[0], 0, 24))
        predictions.append(pred)

        # Extend buffer with seasonal estimate (not predicted green hours)
        solar_buffer.append(solar_est)
        wind_buffer.append(wind_est)

    # Confidence: per-month historical std
    df_hist["_m"] = df_hist["date"].dt.month
    month_std = df_hist.groupby("_m")["green_compute_hours"].std().to_dict()

    result = pd.DataFrame({
        "date": future_dates,
        "predicted_green_hours": [round(p, 2) for p in predictions],
        "confidence_lower": [
            round(max(predictions[i] - month_std.get(future_dates[i].month, 1.0) * 0.6, 0), 2)
            for i in range(30)
        ],
        "confidence_upper": [
            round(min(predictions[i] + month_std.get(future_dates[i].month, 1.0) * 0.6, 24), 2)
            for i in range(30)
        ],
    })

    logger.info(f"Forecast {location_name}: avg={result['predicted_green_hours'].mean():.1f} hrs/day")
    return result


if __name__ == "__main__":
    print(train_model())
    for loc in LOCATIONS:
        p = predict_next_30_days(loc)
        if not p.empty:
            print(f"{loc}: {p['predicted_green_hours'].mean():.1f} hrs/day")