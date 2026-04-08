# IN Bharat Green Compute Forecaster

> **End-to-End ETL + ML Pipeline for Predicting Sustainable "Green Compute Hours" for India's AI Data Centers**
> *Inspired by Elon Musk's xAI / Tesla / SpaceX Terafab vision – April 2026

## 🔗 Live Demo

> **[▶ Launch App on Streamlit Cloud](https://bharat-green-compute-forecaster.streamlit.app)**

## 💡 Inspiration

India is on track to become the **world's 3rd-largest data center market** by 2026,
with the government's **500 GW non-fossil fuel target** and the **India AI Mission**
driving a massive convergence of renewable energy and AI compute.

This project forecasts how many **"Green Compute Hours"** — hours per day where
both solar irradiance AND wind speeds are sufficient to power AI chip fabs and
data centers sustainably — are available across 5 key Indian cities.

The concept is inspired by Elon Musk's **Terafab** initiative, which envisions
massive AI compute factories powered entirely by renewable energy.

---

## ✨ Features

|
 Feature 
|
 Description 
|
|
---
|
---
|
|
**
NASA POWER ETL
**
|
 Automated extraction of 3 years of solar, wind & temperature data 
|
|
**
SQLite Caching
**
|
 All API responses cached locally — zero redundant calls 
|
|
**
Feature Engineering
**
|
 Solar yield, wind yield, rolling averages, calendar features 
|
|
**
XGBoost ML Model
**
|
 Predicts next 30 days of Green Compute Hours per city 
|
|
**
Interactive India Map
**
|
 Folium map with color-coded markers & forecast popups 
|
|
**
Plotly Dashboards
**
|
 Historical + forecast line charts, CO₂ savings metrics 
|
|
**
What-If Simulator
**
|
 Slide AI demand (MW) → see how many Grok-scale models run green 
|
|
**
Pipeline Monitor
**
|
 Real-time ETL status, data freshness indicators 
|

---

## 🛠 Tech Stack

|
 Layer 
|
 Technology 
|
|
---
|
---
|
|
 Language 
|
 Python 3.10+ 
|
|
 ETL 
|
 requests, pandas, sqlite3 
|
|
 ML 
|
 XGBoost, scikit-learn, joblib 
|
|
 Dashboard 
|
 Streamlit, Plotly, Folium, streamlit-folium 
|
|
 Data Source 
|
[
NASA POWER API
](
https://power.larc.nasa.gov/
)
 