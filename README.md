# Bharat Green Compute Forecaster

End-to-End ETL + ML Pipeline for Predicting Sustainable "Green Compute Hours" for India's AI Data Centers.

## Live Demo

[Launch App](https://bharat-green-compute-forecaster.streamlit.app)

## What It Does

Forecasts daily hours where both solar irradiance and wind speed are sufficient to power AI data centers sustainably, across 5 Indian cities.

Uses 3 years of NASA POWER satellite data, proper turbine power curves, India-specific solar daylight hours, and an XGBoost model with autoregressive lag features.

## Features

- NASA POWER API ETL with retry logic and safe upserts
- Scientifically grounded green energy scoring (not arbitrary fractions)
- XGBoost regression with lag features, calendar features, and location encoding
- Autoregressive 30-day forecasting (predictions feed back as inputs)
- Per-month confidence intervals based on historical variance
- Interactive Folium map of India with forecast popups
- Plotly time-series and bar charts
- What-If simulator with sourced CO2 emission factors

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ETL | requests, pandas, sqlite3 |
| ML | XGBoost, scikit-learn, joblib |
| Dashboard | Streamlit, Plotly, Folium |
| Data Source | NASA POWER API (free, no key) |

## Run Locally

```
git clone https://github.com/YOUR_USERNAME/bharat-green-compute-forecaster.git
cd bharat-green-compute-forecaster
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data assets .streamlit
streamlit run app.py
```