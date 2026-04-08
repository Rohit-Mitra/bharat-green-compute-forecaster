"""
app.py
Streamlit dashboard with 5 tabs: Home, Map, Forecast, Simulator, Pipeline.
"""

import os
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from utils import (
    LOCATIONS, get_db_connection, get_last_etl_time, get_model_metrics,
    format_date_display, estimate_co2_saved, ensure_data_dir,
    INDIA_GRID_CO2_FACTOR,
)
from etl_pipeline import run_full_etl
from model import train_model, predict_next_30_days

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Bharat Green Compute Forecaster",
    page_icon="https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Data loaders (cached) -------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_energy_data():
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY date", conn)
        conn.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def get_all_forecasts():
    """Compute forecasts for all locations once, not per re-render."""
    forecasts = {}
    for loc in LOCATIONS:
        forecasts[loc] = predict_next_30_days(loc)
    return forecasts


# -- Sidebar ----------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("Bharat Green Compute")
        st.caption("Forecasting sustainable AI compute across India")
        st.divider()

        if st.button("Refresh Data (Run ETL + Train)", width="stretch"):
            with st.spinner("Running pipeline... this takes a few minutes."):
                try:
                    run_full_etl()
                    train_model()
                    st.cache_data.clear()
                    st.success("Pipeline complete. Data refreshed.")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

        st.divider()
        last_etl = get_last_etl_time()
        if last_etl:
            st.metric("Last ETL Run", format_date_display(last_etl))
        else:
            st.warning("No ETL run yet. Click Refresh above.")

        st.divider()
        st.markdown("### About")
        st.markdown(
            "Predicts daily green compute windows using 3 years of "
            "[NASA POWER](https://power.larc.nasa.gov/) solar and wind data "
            "for 5 Indian cities. Built for India's AI data center growth, "
            "April 2026."
        )


# -- Tab 1: Home -----------------------------------------------------------

def render_home():
    st.markdown("# Bharat Green Compute Forecaster")
    st.markdown(
        "Predicting sustainable energy windows for AI data centers across India."
    )
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Why Green Compute?")
        st.markdown(
            "India's data center capacity will triple by 2027. AI training "
            "runs consume 10-100+ MW for months. Scheduling these during "
            "renewable energy peaks reduces cost and emissions."
        )
    with c2:
        st.markdown("### The Approach")
        st.markdown(
            "We compute daily overlap hours where both solar irradiance "
            "and wind speed are sufficient for power generation, using "
            "real turbine curves and India-specific solar daylight data."
        )
    with c3:
        st.markdown("### Pipeline")
        st.markdown(
            "1. Extract 3 years of NASA solar/wind data\n"
            "2. Engineer lag and calendar features\n"
            "3. Train XGBoost with autoregressive prediction\n"
            "4. Forecast next 30 days per city"
        )

    st.divider()

    df = load_energy_data()
    if not df.empty:
        st.markdown("### Overview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Locations", f"{df['location'].nunique()}")
        m2.metric("Days of Data", f"{df['date'].nunique():,}")
        m3.metric("Avg Green Hrs/Day", f"{df['green_compute_hours'].mean():.1f}")
        m4.metric("Peak Green Hrs", f"{df['green_compute_hours'].max():.1f}")

        st.markdown("### City Summary")
        summary = (
            df.groupby("location")["green_compute_hours"]
            .agg(["mean", "max", "std", "count"])
            .round(2)
            .rename(columns={
                "mean": "Avg Hrs/Day", "max": "Peak Hrs",
                "std": "Std Dev", "count": "Data Points",
            })
            .sort_values("Avg Hrs/Day", ascending=False)
        )
        st.dataframe(summary, width="stretch")
    else:
        st.info("Click Refresh Data in the sidebar to get started.")

    st.divider()
    st.caption(f"Loaded: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")


# -- Tab 2: India Map ------------------------------------------------------

def render_india_map():
    st.markdown("## India Green Compute Map")
    st.markdown(
        "Markers colored by predicted 30-day average green hours. "
        "Click for details."
    )

    df = load_energy_data()
    forecasts = get_all_forecasts()

    india_map = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="CartoDB positron")

    for loc_name, loc_info in LOCATIONS.items():
        lat, lon = loc_info["lat"], loc_info["lon"]
        desc = loc_info["description"]

        fc = forecasts.get(loc_name, pd.DataFrame())
        avg_pred = fc["predicted_green_hours"].mean() if not fc.empty else 0
        max_pred = fc["predicted_green_hours"].max() if not fc.empty else 0

        hist_avg = 0
        if not df.empty:
            loc_data = df[df["location"] == loc_name]
            if not loc_data.empty:
                hist_avg = loc_data["green_compute_hours"].mean()

        if avg_pred >= 8:
            color, status = "green", "Excellent"
        elif avg_pred >= 4:
            color, status = "orange", "Moderate"
        elif avg_pred > 0:
            color, status = "red", "Low"
        else:
            color, status = "gray", "No Forecast"

        popup_html = f"""
        <div style="font-family:Arial; width:250px; padding:8px;">
            <h4 style="margin:0;">{loc_name}</h4>
            <p style="color:#666; margin:2px 0; font-size:12px;">{desc}</p>
            <hr style="margin:6px 0;">
            <table style="width:100%; font-size:13px;">
                <tr><td>Status</td><td><b>{status}</b></td></tr>
                <tr><td>30-Day Avg</td><td><b>{avg_pred:.1f}</b> hrs/day</td></tr>
                <tr><td>30-Day Peak</td><td>{max_pred:.1f} hrs/day</td></tr>
                <tr><td>Historical Avg</td><td>{hist_avg:.1f} hrs/day</td></tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{loc_name}: {avg_pred:.1f} hrs/day",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(india_map)

    st_folium(india_map, width=1100, height=600, returned_objects=[])

    st.markdown(
        "**Legend:** Green >= 8 hrs | Orange 4-8 hrs | Red < 4 hrs | Gray = no data"
    )


# -- Tab 3: Forecast Dashboard ---------------------------------------------

def render_forecast_dashboard():
    st.markdown("## Forecast Dashboard")

    selected = st.selectbox("Select City", options=list(LOCATIONS.keys()))

    df = load_energy_data()
    forecasts = get_all_forecasts()
    forecast = forecasts.get(selected, pd.DataFrame())

    if df.empty:
        st.warning("No data available. Run ETL first.")
        return

    loc_data = df[df["location"] == selected].copy().sort_values("date")
    if loc_data.empty:
        st.warning(f"No data for {selected}.")
        return

    # Metrics
    st.markdown(f"### {selected}")
    m1, m2, m3 = st.columns(3)

    hist_avg = loc_data["green_compute_hours"].mean()
    hist_max = loc_data["green_compute_hours"].max()
    forecast_avg = forecast["predicted_green_hours"].mean() if not forecast.empty else 0

    m1.metric("Historical Avg", f"{hist_avg:.1f} hrs/day")
    m2.metric("Historical Peak", f"{hist_max:.1f} hrs/day")
    m3.metric(
        "30-Day Forecast Avg", f"{forecast_avg:.1f} hrs/day",
        delta=f"{forecast_avg - hist_avg:+.1f}" if forecast_avg else None,
    )

    st.divider()

    # Historical chart (last 365 days)
    st.markdown("### Historical (Last 12 Months)")
    recent = loc_data.tail(365)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=recent["date"], y=recent["green_compute_hours"],
        mode="lines", name="Actual",
        line=dict(color="#2ecc71", width=1.5),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.12)",
    ))
    # Rolling avg of raw solar as a context line
    if "solar_7d_avg" in recent.columns:
        fig_hist.add_trace(go.Scatter(
            x=recent["date"], y=recent["solar_7d_avg"],
            mode="lines", name="Solar 7d Avg (kWh/m2/day)",
            line=dict(color="#f39c12", width=1, dash="dot"),
            yaxis="y2",
        ))

    fig_hist.update_layout(
        xaxis_title="Date", yaxis_title="Green Compute Hours",
        yaxis2=dict(title="Solar (kWh/m2/day)", overlaying="y", side="right"),
        hovermode="x unified", template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_hist, width="stretch")

    # Forecast chart
    if not forecast.empty:
        st.markdown("### 30-Day Forecast")
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["confidence_upper"], forecast["confidence_lower"][::-1]]),
            fill="toself", fillcolor="rgba(52,152,219,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="Confidence",
        ))
        fig_fc.add_trace(go.Scatter(
            x=forecast["date"], y=forecast["predicted_green_hours"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#3498db", width=2.5), marker=dict(size=4),
        ))
        fig_fc.update_layout(
            xaxis_title="Date", yaxis_title="Predicted Green Hours",
            hovermode="x unified", template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_fc, width="stretch")

        with st.expander("View Forecast Table"):
            st.dataframe(
                forecast.style.format({
                    "predicted_green_hours": "{:.2f}",
                    "confidence_lower": "{:.2f}",
                    "confidence_upper": "{:.2f}",
                }),
                width="stretch",
            )

    # Monthly bar chart
    st.markdown("### Monthly Comparison")
    loc_data["month_name"] = loc_data["date"].dt.month_name()
    loc_data["month_num"] = loc_data["date"].dt.month
    monthly = (
        loc_data.groupby(["month_num", "month_name"])["green_compute_hours"]
        .mean().reset_index().sort_values("month_num")
    )
    fig_mo = px.bar(
        monthly, x="month_name", y="green_compute_hours",
        color="green_compute_hours", color_continuous_scale="Greens",
        labels={"month_name": "Month", "green_compute_hours": "Avg Green Hrs/Day"},
    )
    fig_mo.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(fig_mo, width="stretch")


# -- Tab 4: What-If Simulator ----------------------------------------------

def render_what_if():
    st.markdown("## What-If Simulator")
    st.markdown(
        "Estimate sustainable compute capacity at different power levels and efficiencies."
    )
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        selected = st.selectbox("City", options=list(LOCATIONS.keys()), key="sim_loc")
        demand_mw = st.slider("Green Power Capacity (MW)", 10, 500, 100, 10)
        dc_pue = st.slider(
            "Data Center PUE",
            min_value=1.1, max_value=2.5, value=1.4, step=0.1,
            help="Power Usage Effectiveness. 1.0 = perfect. Typical India DC: 1.4-1.8",
        )

    forecasts = get_all_forecasts()
    forecast = forecasts.get(selected, pd.DataFrame())

    if forecast.empty:
        st.warning("No forecast available. Run the pipeline first.")
        return

    avg_green_hours = forecast["predicted_green_hours"].mean()
    effective_mw = demand_mw / dc_pue
    daily_mwh = avg_green_hours * effective_mw
    monthly_mwh = daily_mwh * 30
    yearly_mwh = daily_mwh * 365
    co2_yearly = estimate_co2_saved(yearly_mwh)

    with col2:
        st.markdown("### Results")
        st.metric("Avg Green Hrs/Day", f"{avg_green_hours:.1f}")
        st.metric("Effective Compute Power", f"{effective_mw:.0f} MW")
        st.metric("Monthly Green Energy", f"{monthly_mwh:,.0f} MWh")
        st.metric(
            "CO2 Saved / Year", f"{co2_yearly:,.0f} tonnes",
            help=f"India grid factor: {INDIA_GRID_CO2_FACTOR} tCO2/MWh (CEA 2024)",
        )

    st.divider()

    # City comparison
    st.markdown("### All Cities at Current Settings")
    rows = []
    for loc in LOCATIONS:
        fc = forecasts.get(loc, pd.DataFrame())
        if not fc.empty:
            avg = fc["predicted_green_hours"].mean()
            eff = demand_mw / dc_pue
            rows.append({
                "City": loc,
                "Avg Green Hrs/Day": round(avg, 1),
                "Monthly MWh": round(avg * eff * 30, 0),
                "CO2 Saved/Year (t)": round(estimate_co2_saved(avg * eff * 365), 0),
            })

    if rows:
        comp_df = pd.DataFrame(rows).sort_values("Avg Green Hrs/Day", ascending=False)
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["Avg Green Hrs/Day", "Monthly MWh"], color="#d4edda"
            ),
            width="stretch", hide_index=True,
        )

        fig = px.bar(
            comp_df, x="City", y="Avg Green Hrs/Day",
            color="Avg Green Hrs/Day", color_continuous_scale="Greens",
            title=f"Comparison at {demand_mw} MW, PUE {dc_pue}",
        )
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### Methodology")
    st.markdown(
        f"- Effective MW = Capacity / PUE = {demand_mw} / {dc_pue} = {effective_mw:.0f} MW\n"
        f"- Green Energy = Green Hours x Effective MW\n"
        f"- CO2 Saved = Green Energy x {INDIA_GRID_CO2_FACTOR} tCO2/MWh "
        f"([CEA Baseline Database v19, 2024]"
        f"(https://cea.nic.in/dashboard/?lang=en))"
    )


# -- Tab 5: Pipeline Status ------------------------------------------------

def render_pipeline_status():
    st.markdown("## Pipeline Status")

    last_etl = get_last_etl_time()
    metrics = get_model_metrics()

    c1, c2, c3 = st.columns(3)
    with c1:
        if last_etl:
            st.success(f"Last ETL: {format_date_display(last_etl)}")
        else:
            st.error("ETL not run yet.")
    with c2:
        trained = metrics.get("model_trained_at")
        if trained:
            st.success(f"Model trained: {format_date_display(trained)}")
        else:
            st.error("Model not trained.")
    with c3:
        st.info(f"Training samples: {metrics.get('model_n_samples', 'N/A')}")

    st.divider()

    # Model metrics
    st.markdown("### Model Performance")
    if metrics.get("model_mae"):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"{metrics['model_mae']} hrs")
        mc2.metric("RMSE", f"{metrics['model_rmse']} hrs")
        mc3.metric("R2", metrics["model_r2"])
    else:
        st.warning("No metrics yet. Train the model first.")

    st.divider()

    # Data freshness
    st.markdown("### Data Freshness")
    df = load_energy_data()
    if not df.empty:
        rows = []
        for loc in LOCATIONS:
            loc_df = df[df["location"] == loc]
            if not loc_df.empty:
                latest = loc_df["date"].max()
                days_old = (pd.Timestamp.now() - latest).days
                rows.append({
                    "Location": loc,
                    "Earliest": loc_df["date"].min().strftime("%Y-%m-%d"),
                    "Latest": latest.strftime("%Y-%m-%d"),
                    "Rows": len(loc_df),
                    "Days Old": days_old,
                    "Status": "Fresh" if days_old <= 7 else "Stale",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()

    # Database info
    st.markdown("### Database")
    db_path = os.path.join("data", "energy.db")
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        st.metric("Size", f"{size_mb:.2f} MB")
    else:
        st.warning("Database not found.")

    st.divider()

    # Manual controls
    st.markdown("### Manual Controls")
    ca, cb, cc = st.columns(3)
    with ca:
        if st.button("Run ETL Only", width="stretch"):
            with st.spinner("Running ETL..."):
                try:
                    run_full_etl()
                    st.cache_data.clear()
                    st.success("ETL complete.")
                except Exception as e:
                    st.error(f"ETL error: {e}")
    with cb:
        if st.button("Train Model Only", width="stretch"):
            with st.spinner("Training..."):
                try:
                    result = train_model()
                    st.cache_data.clear()
                    st.success(f"Trained. R2={result.get('r2', 'N/A')}")
                except Exception as e:
                    st.error(f"Error: {e}")
    with cc:
        if st.button("Clear Cache", width="stretch"):
            st.cache_data.clear()
            st.success("Cache cleared.")


# -- Main ------------------------------------------------------------------

def main():
    ensure_data_dir()
    render_sidebar()

    df_check = load_energy_data()
    if df_check.empty:
        st.warning(
            "No data found. Click Refresh Data in the sidebar to run the "
            "ETL pipeline. First run takes 3-8 minutes."
        )

    tabs = st.tabs(["Home", "India Map", "Forecast", "Simulator", "Pipeline"])

    with tabs[0]:
        render_home()
    with tabs[1]:
        render_india_map()
    with tabs[2]:
        render_forecast_dashboard()
    with tabs[3]:
        render_what_if()
    with tabs[4]:
        render_pipeline_status()

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#888; padding:1rem 0;'>"
        "Bharat Green Compute Forecaster | "
        "Data: <a href='https://power.larc.nasa.gov/'>NASA POWER</a> | "
        "ML: XGBoost | Dashboard: Streamlit | 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()