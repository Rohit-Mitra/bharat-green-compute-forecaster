"""
=============================================================================
app.py — Bharat Green Compute Forecaster
=============================================================================
Main Streamlit dashboard with 5 tabs:
  1. Home — overview & live timestamp
  2. India Map — Folium map with forecast popups
  3. Forecast Dashboard — Plotly charts + metrics
  4. What-If Simulator — interactive sliders
  5. Pipeline Status — ETL & model health
=============================================================================
"""

import os
import sys
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
    LOCATIONS,
    get_db_connection,
    get_last_etl_time,
    get_model_metrics,
    format_date_display,
    estimate_co2_saved,
    estimate_grok_models,
    ensure_data_dir,
)
from etl_pipeline import run_full_etl
from model import train_model, predict_next_30_days

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🇮🇳 Bharat Green Compute Forecaster",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# HELPER: Load data from DB
# ===========================================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_energy_data() -> pd.DataFrame:
    """Load the energy_data table from SQLite."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM energy_data ORDER BY date", conn)
        conn.close()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_forecast(location: str) -> pd.DataFrame:
    """Get 30-day forecast for a location (cached)."""
    return predict_next_30_days(location)


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar():
    """Render the sidebar with controls and info."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/"
            "Flag_of_India.svg/1200px-Flag_of_India.svg.png",
            width=60,
        )
        st.title("🇮🇳 Bharat Green Compute")
        st.caption("Forecasting sustainable AI compute across India")

        st.divider()

        # --- Refresh data button -------------------------------------------
        if st.button("🔄 Refresh Data (Run ETL + Train)", width="stretch"):
            with st.spinner("Running ETL pipeline... This may take a few minutes."):
                try:
                    run_full_etl()
                    train_model()
                    st.cache_data.clear()
                    st.success("✅ Pipeline complete! Data refreshed.")
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")

        st.divider()

        # --- Last updated ---------------------------------------------------
        last_etl = get_last_etl_time()
        if last_etl:
            st.metric("Last ETL Run", format_date_display(last_etl))
        else:
            st.warning("No ETL run yet. Click Refresh above.")

        st.divider()

        # --- About section --------------------------------------------------
        st.markdown("### ℹ️ About")
        st.markdown(
            """
            **Built for India's AI Mission & data center growth – April 2026.**

            Inspired by Elon Musk's Terafab initiative and India's ambitious
            500 GW non-fossil fuel energy target.

            Data source: [NASA POWER API](https://power.larc.nasa.gov/)

            ---
            *Made with ❤️ using Python, XGBoost & Streamlit*
            """
        )


# ===========================================================================
# TAB 1: HOME
# ===========================================================================

def render_home():
    """Render the Home tab."""
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 2.8rem; margin-bottom: 0.2rem;'>
                🇮🇳 Bharat Green Compute Forecaster
            </h1>
            <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>
                End-to-End ETL + ML Pipeline for Predicting Sustainable
                "Green Compute Hours" for India's AI Data Centers
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Context columns ---------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌞 Why Green Compute?")
        st.markdown(
            """
            India's data center capacity is set to **triple by 2027**. As AI
            workloads (like Grok, Gemini, GPT) demand massive energy, powering
            them with solar & wind becomes critical for sustainability and cost.
            """
        )

    with col2:
        st.markdown("### 🏭 The Terafab Vision")
        st.markdown(
            """
            Elon Musk's **Terafab** concept envisions AI chip factories powered
            entirely by renewable energy. India — with the world's **5th-largest
            solar capacity** — is perfectly positioned to lead this revolution.
            """
        )

    with col3:
        st.markdown("### 📊 What This App Does")
        st.markdown(
            """
            1. **Extracts** 3 years of NASA solar/wind data for 5 Indian cities
            2. **Engineers** features & trains an XGBoost model
            3. **Predicts** next 30 days of Green Compute Hours
            4. **Visualizes** everything on interactive maps & charts
            """
        )

    st.divider()

    # --- Quick metrics from data -------------------------------------------
    df = load_energy_data()
    if not df.empty:
        st.markdown("### 📈 Quick Overview")
        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "📍 Locations Tracked",
            f"{df['location'].nunique()}",
        )
        m2.metric(
            "📅 Days of Data",
            f"{df['date'].nunique():,}",
        )
        m3.metric(
            "⚡ Avg Green Hours/Day",
            f"{df['green_compute_hours'].mean():.1f} hrs",
        )
        m4.metric(
            "🌟 Max Green Hours (Single Day)",
            f"{df['green_compute_hours'].max():.1f} hrs",
        )

        # Per-city summary
        st.markdown("### 🏙️ City-wise Green Compute Summary")
        summary = (
            df.groupby("location")["green_compute_hours"]
            .agg(["mean", "max", "std", "count"])
            .round(2)
            .rename(
                columns={
                    "mean": "Avg Hours/Day",
                    "max": "Peak Hours",
                    "std": "Std Dev",
                    "count": "Data Points",
                }
            )
            .sort_values("Avg Hours/Day", ascending=False)
        )
        st.dataframe(summary, width="stretch")
    else:
        st.info(
            "👈 Click **Refresh Data** in the sidebar to run the ETL pipeline "
            "and start exploring!"
        )

    # --- Timestamp ---------------------------------------------------------
    st.divider()
    st.caption(f"🕐 Dashboard loaded at: {datetime.now().strftime('%B %d, %Y %I:%M %p IST')}")


# ===========================================================================
# TAB 2: INDIA MAP
# ===========================================================================

def render_india_map():
    """Render the Folium India map with color-coded location markers."""
    st.markdown("## 🗺️ India Green Compute Map")
    st.markdown(
        "Markers are color-coded by **predicted average Green Compute Hours** "
        "for the next 30 days. Click any marker for details."
    )

    # Center map on India
    india_map = folium.Map(
        location=[22.5, 78.9],
        zoom_start=5,
        tiles="CartoDB positron",
    )

    df = load_energy_data()

    for loc_name, loc_info in LOCATIONS.items():
        lat = loc_info["lat"]
        lon = loc_info["lon"]
        desc = loc_info["description"]

        # Get forecast
        forecast = get_forecast(loc_name)
        if not forecast.empty:
            avg_pred = forecast["predicted_green_hours"].mean()
            max_pred = forecast["predicted_green_hours"].max()
        else:
            avg_pred = 0
            max_pred = 0

        # Historical average
        hist_avg = 0
        if not df.empty:
            loc_data = df[df["location"] == loc_name]
            if not loc_data.empty:
                hist_avg = loc_data["green_compute_hours"].mean()

        # Color based on predicted green hours
        if avg_pred >= 8:
            color = "green"
            status = "🟢 Excellent"
        elif avg_pred >= 4:
            color = "orange"
            status = "🟡 Moderate"
        else:
            color = "red"
            status = "🔴 Low"

        # Build popup HTML
        popup_html = f"""
        <div style="font-family: Arial; width: 280px; padding: 10px;">
            <h3 style="margin: 0; color: #1a1a2e;">{loc_name}</h3>
            <p style="color: #666; margin: 4px 0;">{desc}</p>
            <hr style="margin: 8px 0;">
            <table style="width: 100%; font-size: 13px;">
                <tr>
                    <td><b>Status</b></td>
                    <td>{status}</td>
                </tr>
                <tr>
                    <td><b>30-Day Forecast Avg</b></td>
                    <td><b>{avg_pred:.1f}</b> hrs/day</td>
                </tr>
                <tr>
                    <td><b>30-Day Forecast Peak</b></td>
                    <td>{max_pred:.1f} hrs/day</td>
                </tr>
                <tr>
                    <td><b>Historical Avg</b></td>
                    <td>{hist_avg:.1f} hrs/day</td>
                </tr>
                <tr>
                    <td><b>Coordinates</b></td>
                    <td>{lat:.4f}, {lon:.4f}</td>
                </tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{loc_name}: {avg_pred:.1f} hrs/day",
            icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
        ).add_to(india_map)

    # Render map
    st_folium(india_map, width=1100, height=600, returned_objects=[])

    # Legend
    st.markdown(
        """
        **Legend:** 🟢 Green (≥8 hrs) = Excellent &nbsp;|&nbsp;
        🟡 Orange (4–8 hrs) = Moderate &nbsp;|&nbsp;
        🔴 Red (<4 hrs) = Low green compute potential
        """
    )


# ===========================================================================
# TAB 3: FORECAST DASHBOARD
# ===========================================================================

def render_forecast_dashboard():
    """Render the forecast charts and metrics for a selected location."""
    st.markdown("## 📊 Forecast Dashboard")

    # Location selector
    selected_location = st.selectbox(
        "Select City",
        options=list(LOCATIONS.keys()),
        index=0,
    )

    df = load_energy_data()
    forecast = get_forecast(selected_location)

    if df.empty:
        st.warning("No historical data available. Run the ETL pipeline first.")
        return

    loc_data = df[df["location"] == selected_location].copy()
    loc_data.sort_values("date", inplace=True)

    if loc_data.empty:
        st.warning(f"No data for {selected_location}.")
        return

    # --- Key Metrics --------------------------------------------------------
    st.markdown(f"### 📍 {selected_location}")
    m1, m2, m3, m4 = st.columns(4)

    hist_avg = loc_data["green_compute_hours"].mean()
    hist_max = loc_data["green_compute_hours"].max()

    forecast_avg = (
        forecast["predicted_green_hours"].mean() if not forecast.empty else 0
    )

    co2_saved = estimate_co2_saved(hist_avg * 365, power_mw=50)

    m1.metric("📊 Historical Avg", f"{hist_avg:.1f} hrs/day")
    m2.metric("🌟 Historical Peak", f"{hist_max:.1f} hrs/day")
    m3.metric(
        "🔮 30-Day Forecast Avg",
        f"{forecast_avg:.1f} hrs/day",
        delta=f"{forecast_avg - hist_avg:+.1f}" if forecast_avg else None,
    )
    m4.metric("🌿 Est. CO₂ Saved/Year", f"{co2_saved:,.0f} tonnes")

    st.divider()

    # --- Historical line chart with Plotly ----------------------------------
    st.markdown("### 📈 Historical Green Compute Hours")

    # Use last 365 days for cleaner chart
    recent = loc_data.tail(365)

    fig_hist = go.Figure()

    fig_hist.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["green_compute_hours"],
            mode="lines",
            name="Actual Green Hours",
            line=dict(color="#2ecc71", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.15)",
        )
    )

    fig_hist.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["rolling_7d_avg"],
            mode="lines",
            name="7-Day Rolling Avg",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        )
    )

    fig_hist.update_layout(
        title=f"Daily Green Compute Hours — {selected_location} (Last 12 Months)",
        xaxis_title="Date",
        yaxis_title="Green Compute Hours",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig_hist, width="stretch")

    # --- Forecast chart ----------------------------------------------------
    if not forecast.empty:
        st.markdown("### 🔮 30-Day Forecast")

        fig_forecast = go.Figure()

        # Confidence band
        fig_forecast.add_trace(
            go.Scatter(
                x=pd.concat([forecast["date"], forecast["date"][::-1]]),
                y=pd.concat(
                    [forecast["confidence_upper"], forecast["confidence_lower"][::-1]]
                ),
                fill="toself",
                fillcolor="rgba(52, 152, 219, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence Band",
                showlegend=True,
            )
        )

        # Predicted line
        fig_forecast.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["predicted_green_hours"],
                mode="lines+markers",
                name="Predicted Green Hours",
                line=dict(color="#3498db", width=2.5),
                marker=dict(size=5),
            )
        )

        fig_forecast.update_layout(
            title=f"30-Day Green Compute Forecast — {selected_location}",
            xaxis_title="Date",
            yaxis_title="Predicted Green Compute Hours",
            hovermode="x unified",
            template="plotly_white",
            height=400,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig_forecast, width="stretch")

        # Forecast data table
        with st.expander("📋 View Forecast Data Table"):
            st.dataframe(
                forecast.style.format(
                    {
                        "predicted_green_hours": "{:.2f}",
                        "confidence_lower": "{:.2f}",
                        "confidence_upper": "{:.2f}",
                    }
                ),
                width="stretch",
            )

    # --- Monthly comparison chart ------------------------------------------
    st.markdown("### 📅 Monthly Comparison")

    loc_data["month_name"] = loc_data["date"].dt.month_name()
    loc_data["month_num"] = loc_data["date"].dt.month
    monthly = (
        loc_data.groupby(["month_num", "month_name"])["green_compute_hours"]
        .mean()
        .reset_index()
        .sort_values("month_num")
    )

    fig_monthly = px.bar(
        monthly,
        x="month_name",
        y="green_compute_hours",
        color="green_compute_hours",
        color_continuous_scale="Greens",
        labels={
            "month_name": "Month",
            "green_compute_hours": "Avg Green Hours/Day",
        },
        title=f"Average Monthly Green Compute Hours — {selected_location}",
    )
    fig_monthly.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    st.plotly_chart(fig_monthly, width="stretch")


# ===========================================================================
# TAB 4: WHAT-IF SIMULATOR
# ===========================================================================

def render_what_if():
    """Render the What-If simulator for AI training demand scenarios."""
    st.markdown("## 🧪 What-If Simulator")
    st.markdown(
        "Explore how many **Grok-scale AI model training runs** can be powered "
        "sustainably at different locations and power capacities."
    )

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        selected_location = st.selectbox(
            "🏙️ Select City",
            options=list(LOCATIONS.keys()),
            key="whatif_location",
        )

        demand_mw = st.slider(
            "⚡ Available Green Power Capacity (MW)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="How much renewable power capacity (in MW) is available at this location?",
        )

        dc_pue = st.slider(
            "🏭 Data Center PUE (Power Usage Effectiveness)",
            min_value=1.1,
            max_value=2.5,
            value=1.4,
            step=0.1,
            help="PUE of 1.0 = all power goes to compute. Typical Indian DC: 1.4–1.8",
        )

    # Get forecast for selected location
    forecast = get_forecast(selected_location)

    if forecast.empty:
        st.warning("No forecast available. Run the pipeline first.")
        return

    avg_green_hours = forecast["predicted_green_hours"].mean()

    # Effective compute power after PUE
    effective_mw = demand_mw / dc_pue

    # Calculations
    daily_green_mwh = avg_green_hours * effective_mw
    monthly_green_mwh = daily_green_mwh * 30
    yearly_green_mwh = daily_green_mwh * 365

    grok_models = estimate_grok_models(avg_green_hours, effective_mw)
    co2_saved_yearly = estimate_co2_saved(avg_green_hours * 365, effective_mw)

    # H100 GPU equivalents (each H100 ≈ 0.7 kW = 0.0007 MW)
    h100_gpus = int(effective_mw / 0.0007)
    h100_hours_daily = avg_green_hours * h100_gpus

    with col2:
        st.markdown("### 📊 Results")

        st.metric(
            "🔮 Avg Green Hours/Day (Forecast)",
            f"{avg_green_hours:.1f} hrs",
        )
        st.metric(
            "⚡ Effective Compute Power",
            f"{effective_mw:.0f} MW",
        )
        st.metric(
            "🔋 Monthly Green Energy",
            f"{monthly_green_mwh:,.0f} MWh",
        )
        st.metric(
            "🤖 Grok-Scale Trainings / Month",
            f"{grok_models:.4f}",
            help="Based on Grok-3 estimate: ~216,000 MWh per training run",
        )

    st.divider()

    # Detailed breakdown
    st.markdown("### 📋 Detailed Breakdown")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🖥️ H100 GPU Equivalent", f"{h100_gpus:,}")
    r2.metric("⏱️ GPU-Hours/Day (Green)", f"{h100_hours_daily:,.0f}")
    r3.metric("🌿 CO₂ Saved/Year", f"{co2_saved_yearly:,.0f} tonnes")
    r4.metric("📅 Yearly Green Energy", f"{yearly_green_mwh:,.0f} MWh")

    st.divider()

    # --- Comparison across all cities --------------------------------------
    st.markdown("### 🏙️ City Comparison at Current Settings")

    comparison_rows = []
    for loc in LOCATIONS:
        fc = get_forecast(loc)
        if not fc.empty:
            avg_gh = fc["predicted_green_hours"].mean()
            eff_mw = demand_mw / dc_pue
            monthly_mwh = avg_gh * eff_mw * 30
            grok_eq = estimate_grok_models(avg_gh, eff_mw)
            co2 = estimate_co2_saved(avg_gh * 365, eff_mw)
            comparison_rows.append(
                {
                    "City": loc,
                    "Avg Green Hrs/Day": round(avg_gh, 1),
                    "Monthly Green MWh": round(monthly_mwh, 0),
                    "Grok Trainings/Month": round(grok_eq, 4),
                    "CO₂ Saved/Year (tonnes)": round(co2, 0),
                }
            )

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows).sort_values(
            "Avg Green Hrs/Day", ascending=False
        )

        # Color the best city
        st.dataframe(
            comp_df.style.highlight_max(
                subset=["Avg Green Hrs/Day", "Monthly Green MWh", "Grok Trainings/Month"],
                color="#d4edda",
            ),
            width="stretch",
            hide_index=True,
        )

        # Bar chart comparison
        fig_comp = px.bar(
            comp_df,
            x="City",
            y="Avg Green Hrs/Day",
            color="Avg Green Hrs/Day",
            color_continuous_scale="Greens",
            title=f"Green Compute Potential Across Cities ({demand_mw} MW capacity, PUE {dc_pue})",
        )
        fig_comp.update_layout(template="plotly_white", height=400, showlegend=False)
        st.plotly_chart(fig_comp, width="stretch")

    st.divider()

    # --- Scenario explanation -----------------------------------------------
    st.markdown("### 💡 How is this calculated?")
    st.markdown(
        f"""
        | Parameter | Value |
        |---|---|
        | Green Power Capacity | **{demand_mw} MW** |
        | PUE | **{dc_pue}** |
        | Effective Compute Power | **{demand_mw / dc_pue:.0f} MW** |
        | Grok-3 Training Energy | **~216,000 MWh** (estimated) |
        | India Grid CO₂ Factor | **0.82 tCO₂/MWh** (CEA 2024) |
        | H100 GPU Power Draw | **~0.7 kW each** |

        **Formula:**
        - Green Energy (MWh/day) = Green Hours × Effective MW
        - Grok Trainings/Month = (Green MWh/month) ÷ 216,000
        - CO₂ Saved = Green Energy × 0.82 tCO₂/MWh
        """
    )


# ===========================================================================
# TAB 5: PIPELINE STATUS
# ===========================================================================

def render_pipeline_status():
    """Render the Pipeline Status monitoring tab."""
    st.markdown("## ⚙️ Pipeline Status & Data Health")

    # --- ETL Status ---------------------------------------------------------
    st.markdown("### 🔄 ETL Pipeline")

    last_etl = get_last_etl_time()
    metrics = get_model_metrics()

    col1, col2, col3 = st.columns(3)

    with col1:
        if last_etl:
            st.success(f"✅ Last ETL Run: {format_date_display(last_etl)}")
        else:
            st.error("❌ ETL has not been run yet.")

    with col2:
        trained_at = metrics.get("model_trained_at")
        if trained_at:
            st.success(f"✅ Model Trained: {format_date_display(trained_at)}")
        else:
            st.error("❌ Model not trained yet.")

    with col3:
        n_samples = metrics.get("model_n_samples", "N/A")
        st.info(f"📊 Training Samples: {n_samples}")

    st.divider()

    # --- Model Metrics ------------------------------------------------------
    st.markdown("### 🤖 Model Performance")

    if metrics:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "MAE (Mean Absolute Error)",
            f"{metrics.get('model_mae', 'N/A')} hrs",
        )
        mc2.metric(
            "RMSE (Root Mean Squared Error)",
            f"{metrics.get('model_rmse', 'N/A')} hrs",
        )
        mc3.metric(
            "R² Score",
            metrics.get("model_r2", "N/A"),
        )
    else:
        st.warning("No model metrics available. Train the model first.")

    st.divider()

    # --- Data Freshness per location ----------------------------------------
    st.markdown("### 📅 Data Freshness by Location")

    df = load_energy_data()
    if not df.empty:
        freshness_rows = []
        for loc_name in LOCATIONS:
            loc_df = df[df["location"] == loc_name]
            if not loc_df.empty:
                latest = loc_df["date"].max()
                earliest = loc_df["date"].min()
                count = len(loc_df)
                days_old = (pd.Timestamp.now() - latest).days
                freshness_rows.append(
                    {
                        "Location": loc_name,
                        "Earliest Date": earliest.strftime("%Y-%m-%d"),
                        "Latest Date": latest.strftime("%Y-%m-%d"),
                        "Data Points": count,
                        "Days Since Last Update": days_old,
                        "Status": "✅ Fresh" if days_old <= 7 else "⚠️ Stale",
                    }
                )

        if freshness_rows:
            fresh_df = pd.DataFrame(freshness_rows)
            st.dataframe(fresh_df, width="stretch", hide_index=True)
    else:
        st.warning("No data in database.")

    st.divider()

    # --- Database info ------------------------------------------------------
    st.markdown("### 💾 Database Info")

    db_path = os.path.join("data", "energy.db")
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path)
        db_size_mb = db_size / (1024 * 1024)

        d1, d2 = st.columns(2)
        d1.metric("Database Size", f"{db_size_mb:.2f} MB")
        d2.metric("Database Path", db_path)

        # Show tables
        try:
            conn = get_db_connection()
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
            conn.close()
            st.markdown("**Tables in database:**")
            for _, row in tables.iterrows():
                st.code(row["name"], language=None)
        except Exception as e:
            st.warning(f"Could not read tables: {e}")
    else:
        st.warning("Database file not found. Run ETL pipeline first.")

    st.divider()

    # --- Manual pipeline controls -------------------------------------------
    st.markdown("### 🛠️ Manual Controls")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("🔄 Run ETL Only", width="stretch"):
            with st.spinner("Running ETL..."):
                try:
                    run_full_etl()
                    st.cache_data.clear()
                    st.success("✅ ETL complete!")
                except Exception as e:
                    st.error(f"❌ ETL error: {e}")

    with col_b:
        if st.button("🤖 Train Model Only", width="stretch"):
            with st.spinner("Training model..."):
                try:
                    result = train_model()
                    st.cache_data.clear()
                    st.success(f"✅ Model trained! Metrics: {result}")
                except Exception as e:
                    st.error(f"❌ Training error: {e}")

    with col_c:
        if st.button("🗑️ Clear Cache", width="stretch"):
            st.cache_data.clear()
            st.success("✅ Streamlit cache cleared!")


# ===========================================================================
# MAIN APP — Tab Layout
# ===========================================================================

def main():
    """Main application entry point."""
    # Ensure directories exist
    ensure_data_dir()

    # Render sidebar
    render_sidebar()

    # --- Auto-initialize: run ETL + train on first launch if no data -------
    df_check = load_energy_data()
    if df_check.empty:
        st.warning(
            "⚠️ **First Launch Detected!** No data found in the database. "
            "Click the **🔄 Refresh Data** button in the sidebar to run the "
            "ETL pipeline and train the model. This will take a few minutes."
        )

    # --- Main tab layout ---------------------------------------------------
    tab_home, tab_map, tab_forecast, tab_whatif, tab_status = st.tabs(
        [
            "🏠 Home",
            "🗺️ India Map",
            "📊 Forecast Dashboard",
            "🧪 What-If Simulator",
            "⚙️ Pipeline Status",
        ]
    )

    with tab_home:
        render_home()

    with tab_map:
        render_india_map()

    with tab_forecast:
        render_forecast_dashboard()

    with tab_whatif:
        render_what_if()

    with tab_status:
        render_pipeline_status()

    # --- Footer ------------------------------------------------------------
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 1rem 0;'>
            <p>
                🇮🇳 <b>Bharat Green Compute Forecaster</b> — 
                Built for India's AI Mission & Data Center Growth | April 2026
                <br>
                Inspired by Elon Musk's Terafab Vision |
                Data: <a href="https://power.larc.nasa.gov/" target="_blank">NASA POWER API</a> |
                ML: XGBoost | Dashboard: Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()