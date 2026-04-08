"""
app.py
Streamlit dashboard: Home, Map, Forecast, Simulator, Pipeline.
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

st.set_page_config(
    page_title="Bharat Green Compute Forecaster",
    page_icon="https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    return {loc: predict_next_30_days(loc) for loc in LOCATIONS}


# -- Sidebar ---------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("Bharat Green Compute")
        st.caption("Sustainable AI compute forecasting for India")
        st.divider()

        if st.button("Refresh Data (Run ETL + Train)", width="stretch"):
            with st.spinner("Running pipeline... takes a few minutes."):
                try:
                    run_full_etl()
                    train_model()
                    st.cache_data.clear()
                    st.success("Pipeline complete.")
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

        st.divider()
        last_etl = get_last_etl_time()
        if last_etl:
            st.metric("Last ETL Run", format_date_display(last_etl))
        else:
            st.warning("No ETL run yet.")

        st.divider()
        st.markdown("### About")
        st.markdown(
            "Forecasts hours/day where renewable energy (solar and/or wind) "
            "can power AI data centers across 5 Indian cities. Uses 3 years of "
            "[NASA POWER](https://power.larc.nasa.gov/) data and XGBoost."
        )


# -- Tab 1: Home -----------------------------------------------------------

def render_home():
    st.markdown("# Bharat Green Compute Forecaster")
    st.markdown("Predicting renewable energy windows for India's AI data centers.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Why This Matters")
        st.markdown(
            "India's data center capacity will triple by 2027. AI training "
            "runs consume 10-100+ MW. Knowing when renewable energy peaks "
            "helps schedule workloads to cut costs and emissions."
        )
    with c2:
        st.markdown("### How It Works")
        st.markdown(
            "Green compute hours = hours/day where **solar or wind or both** "
            "can generate power. Solar counts during daylight. Wind counts "
            "24 hours. Having both is a bonus, not a requirement."
        )
    with c3:
        st.markdown("### The Model")
        st.markdown(
            "XGBoost trained on lag features (past solar/wind), calendar, "
            "and location. Today's weather is NOT given as input -- the "
            "model must learn temporal patterns to predict tomorrow."
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
            .agg(["mean", "max", "std", "count"]).round(2)
            .rename(columns={
                "mean": "Avg Hrs/Day", "max": "Peak Hrs",
                "std": "Std Dev", "count": "Data Points",
            })
            .sort_values("Avg Hrs/Day", ascending=False)
        )
        st.dataframe(summary, width="stretch")
    else:
        st.info("Click Refresh Data in the sidebar to start.")

    st.caption(f"Loaded: {datetime.now().strftime('%B %d, %Y %I:%M %p')}")


# -- Tab 2: Map ------------------------------------------------------------

def render_india_map():
    st.markdown("## India Green Compute Map")
    st.markdown("Colored by predicted 30-day average green hours. Click markers for details.")

    df = load_energy_data()
    forecasts = get_all_forecasts()

    m = folium.Map(location=[22.5, 78.9], zoom_start=5, tiles="CartoDB positron")

    for loc_name, info in LOCATIONS.items():
        lat, lon = info["lat"], info["lon"]
        fc = forecasts.get(loc_name, pd.DataFrame())
        avg = fc["predicted_green_hours"].mean() if not fc.empty else 0
        peak = fc["predicted_green_hours"].max() if not fc.empty else 0

        hist = 0
        if not df.empty:
            ld = df[df["location"] == loc_name]
            if not ld.empty:
                hist = ld["green_compute_hours"].mean()

        if avg >= 8:
            color, status = "green", "Excellent"
        elif avg >= 5:
            color, status = "orange", "Moderate"
        elif avg > 0:
            color, status = "red", "Low"
        else:
            color, status = "gray", "No Data"

        popup = f"""
        <div style="font-family:Arial; width:240px;">
            <h4 style="margin:0;">{loc_name}</h4>
            <p style="color:#666; font-size:12px;">{info['description']}</p>
            <hr>
            <table style="font-size:13px; width:100%;">
                <tr><td>Status</td><td><b>{status}</b></td></tr>
                <tr><td>30d Forecast Avg</td><td><b>{avg:.1f}</b> hrs/day</td></tr>
                <tr><td>30d Forecast Peak</td><td>{peak:.1f} hrs/day</td></tr>
                <tr><td>Historical Avg</td><td>{hist:.1f} hrs/day</td></tr>
            </table>
        </div>
        """

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup, max_width=270),
            tooltip=f"{loc_name}: {avg:.1f} hrs/day",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(m)

    st_folium(m, width=1100, height=600, returned_objects=[])
    st.markdown("Green >= 8 hrs | Orange 5-8 hrs | Red < 5 hrs | Gray = no data")


# -- Tab 3: Forecast -------------------------------------------------------

def render_forecast():
    st.markdown("## Forecast Dashboard")

    selected = st.selectbox("Select City", list(LOCATIONS.keys()))
    df = load_energy_data()
    forecasts = get_all_forecasts()
    fc = forecasts.get(selected, pd.DataFrame())

    if df.empty:
        st.warning("No data. Run ETL first.")
        return

    loc = df[df["location"] == selected].copy().sort_values("date")
    if loc.empty:
        st.warning(f"No data for {selected}.")
        return

    hist_avg = loc["green_compute_hours"].mean()
    hist_max = loc["green_compute_hours"].max()
    fc_avg = fc["predicted_green_hours"].mean() if not fc.empty else 0

    st.markdown(f"### {selected}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical Avg", f"{hist_avg:.1f} hrs/day")
    m2.metric("Historical Peak", f"{hist_max:.1f} hrs/day")
    m3.metric("30-Day Forecast Avg", f"{fc_avg:.1f} hrs/day",
              delta=f"{fc_avg - hist_avg:+.1f}" if fc_avg else None)

    # CO2: compute for a user-specified capacity shown in simulator
    yearly_green_mwh = hist_avg * 365 * 50  # 50 MW default for context
    m4.metric("Est CO2 Saved/yr (50 MW DC)", f"{estimate_co2_saved(yearly_green_mwh):,.0f} t")

    st.divider()

    # Historical chart
    st.markdown("### Historical (Last 12 Months)")
    recent = loc.tail(365)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["green_compute_hours"],
        mode="lines", name="Green Hours",
        line=dict(color="#27ae60", width=1.5),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.1)",
    ))
    if "solar_viable_hours" in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["solar_viable_hours"],
            mode="lines", name="Solar Hours",
            line=dict(color="#f39c12", width=1, dash="dot"),
        ))
    if "wind_viable_hours" in recent.columns:
        fig.add_trace(go.Scatter(
            x=recent["date"], y=recent["wind_viable_hours"],
            mode="lines",             name="Wind Hours",
            line=dict(color="#3498db", width=1, dash="dot"),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Hours/Day",
        hovermode="x unified", template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    # Forecast chart
    if not fc.empty:
        st.markdown("### 30-Day Forecast")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pd.concat([fc["date"], fc["date"][::-1]]),
            y=pd.concat([fc["confidence_upper"], fc["confidence_lower"][::-1]]),
            fill="toself", fillcolor="rgba(52,152,219,0.12)",
            line=dict(color="rgba(255,255,255,0)"), name="Confidence",
        ))
        fig2.add_trace(go.Scatter(
            x=fc["date"], y=fc["predicted_green_hours"],
            mode="lines+markers", name="Predicted",
            line=dict(color="#2980b9", width=2.5), marker=dict(size=4),
        ))
        fig2.update_layout(
            xaxis_title="Date", yaxis_title="Predicted Green Hours",
            hovermode="x unified", template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2, width="stretch")

        with st.expander("View Forecast Table"):
            st.dataframe(
                fc.style.format({
                    "predicted_green_hours": "{:.2f}",
                    "confidence_lower": "{:.2f}",
                    "confidence_upper": "{:.2f}",
                }),
                width="stretch",
            )

    # Monthly bar chart
    st.markdown("### Monthly Pattern")
    loc["month_name"] = loc["date"].dt.month_name()
    loc["month_num"] = loc["date"].dt.month
    monthly = (
        loc.groupby(["month_num", "month_name"])["green_compute_hours"]
        .mean().reset_index().sort_values("month_num")
    )
    fig3 = px.bar(
        monthly, x="month_name", y="green_compute_hours",
        color="green_compute_hours", color_continuous_scale="Greens",
        labels={"month_name": "Month", "green_compute_hours": "Avg Green Hrs/Day"},
    )
    fig3.update_layout(template="plotly_white", height=400, showlegend=False)
    st.plotly_chart(fig3, width="stretch")


# -- Tab 4: Simulator ------------------------------------------------------

def render_simulator():
    st.markdown("## What-If Simulator")
    st.markdown("Estimate sustainable compute capacity under different scenarios.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        selected = st.selectbox("City", list(LOCATIONS.keys()), key="sim_loc")
        demand_mw = st.slider("Green Power Capacity (MW)", 10, 500, 100, 10)
        pue = st.slider(
            "Data Center PUE", 1.1, 2.5, 1.4, 0.1,
            help="Power Usage Effectiveness. 1.0 = perfect. India typical: 1.4-1.8",
        )

    forecasts = get_all_forecasts()
    fc = forecasts.get(selected, pd.DataFrame())

    if fc.empty:
        st.warning("No forecast. Run pipeline first.")
        return

    avg_hrs = fc["predicted_green_hours"].mean()
    eff_mw = demand_mw / pue
    daily_mwh = avg_hrs * eff_mw
    monthly_mwh = daily_mwh * 30
    yearly_mwh = daily_mwh * 365
    co2 = estimate_co2_saved(yearly_mwh)

    with col2:
        st.markdown("### Results")
        st.metric("Avg Green Hrs/Day", f"{avg_hrs:.1f}")
        st.metric("Effective Compute Power", f"{eff_mw:.0f} MW")
        st.metric("Monthly Green Energy", f"{monthly_mwh:,.0f} MWh")
        st.metric(
            "CO2 Saved / Year", f"{co2:,.0f} tonnes",
            help=f"Grid factor: {INDIA_GRID_CO2_FACTOR} tCO2/MWh (CEA 2024)",
        )

    st.divider()

    # City comparison
    st.markdown("### All Cities Comparison")
    rows = []
    for loc_name in LOCATIONS:
        f = forecasts.get(loc_name, pd.DataFrame())
        if not f.empty:
            a = f["predicted_green_hours"].mean()
            e = demand_mw / pue
            rows.append({
                "City": loc_name,
                "Avg Green Hrs/Day": round(a, 1),
                "Monthly MWh": round(a * e * 30, 0),
                "CO2 Saved/Year (t)": round(estimate_co2_saved(a * e * 365), 0),
            })

    if rows:
        comp = pd.DataFrame(rows).sort_values("Avg Green Hrs/Day", ascending=False)
        st.dataframe(
            comp.style.highlight_max(
                subset=["Avg Green Hrs/Day", "Monthly MWh"], color="#d4edda",
            ),
            width="stretch", hide_index=True,
        )

        fig = px.bar(
            comp, x="City", y="Avg Green Hrs/Day",
            color="Avg Green Hrs/Day", color_continuous_scale="Greens",
            title=f"Comparison at {demand_mw} MW, PUE {pue}",
        )
        fig.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.markdown("### Methodology")
    st.markdown(
        f"- Effective MW = Capacity / PUE = {demand_mw} / {pue} = {eff_mw:.0f} MW\n"
        f"- Daily Green MWh = Green Hours x Effective MW\n"
        f"- CO2 Saved = Green MWh x {INDIA_GRID_CO2_FACTOR} tCO2/MWh "
        f"[CEA 2024](https://cea.nic.in/dashboard/?lang=en)"
    )


# -- Tab 5: Pipeline -------------------------------------------------------

def render_pipeline():
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
        t = metrics.get("model_trained_at")
        if t:
            st.success(f"Model trained: {format_date_display(t)}")
        else:
            st.error("Model not trained.")
    with c3:
        st.info(f"Training samples: {metrics.get('model_n_samples', 'N/A')}")

    st.divider()

    st.markdown("### Model Performance")
    if metrics.get("model_mae"):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("MAE", f"{metrics['model_mae']} hrs")
        mc2.metric("RMSE", f"{metrics['model_rmse']} hrs")
        mc3.metric("R2", metrics["model_r2"])
    else:
        st.warning("No metrics. Train model first.")

    st.divider()

    st.markdown("### Data Freshness")
    df = load_energy_data()
    if not df.empty:
        rows = []
        for loc_name in LOCATIONS:
            ld = df[df["location"] == loc_name]
            if not ld.empty:
                latest = ld["date"].max()
                rows.append({
                    "Location": loc_name,
                    "Earliest": ld["date"].min().strftime("%Y-%m-%d"),
                    "Latest": latest.strftime("%Y-%m-%d"),
                    "Rows": len(ld),
                    "Days Old": (pd.Timestamp.now() - latest).days,
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()

    st.markdown("### Database")
    db_path = os.path.join("data", "energy.db")
    if os.path.exists(db_path):
        st.metric("Size", f"{os.path.getsize(db_path) / (1024*1024):.2f} MB")

    st.divider()

    st.markdown("### Manual Controls")
    ca, cb, cc = st.columns(3)
    with ca:
        if st.button("Run ETL Only", width="stretch"):
            with st.spinner("Running ETL..."):
                try:
                    run_full_etl()
                    st.cache_data.clear()
                    st.success("ETL done.")
                except Exception as e:
                    st.error(str(e))
    with cb:
        if st.button("Train Model Only", width="stretch"):
            with st.spinner("Training..."):
                try:
                    r = train_model()
                    st.cache_data.clear()
                    st.success(f"Done. R2={r.get('r2', 'N/A')}")
                except Exception as e:
                    st.error(str(e))
    with cc:
        if st.button("Clear Cache", width="stretch"):
            st.cache_data.clear()
            st.success("Cleared.")


# -- Main ------------------------------------------------------------------

def main():
    ensure_data_dir()
    render_sidebar()

    df_check = load_energy_data()
    if df_check.empty:
        st.warning(
            "No data found. Click Refresh Data in the sidebar. "
            "First run takes 3-8 minutes."
        )

    tabs = st.tabs(["Home", "India Map", "Forecast", "Simulator", "Pipeline"])

    with tabs[0]:
        render_home()
    with tabs[1]:
        render_india_map()
    with tabs[2]:
        render_forecast()
    with tabs[3]:
        render_simulator()
    with tabs[4]:
        render_pipeline()

    st.divider()
    st.markdown(
        "<div style='text-align:center; color:#888; padding:1rem;'>"
        "Bharat Green Compute Forecaster | "
        "Data: <a href='https://power.larc.nasa.gov/'>NASA POWER</a> | "
        "ML: XGBoost | 2026</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()