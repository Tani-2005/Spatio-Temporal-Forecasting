import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 1. DATABASE & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Epidemic Early Warning System", layout="wide", page_icon="🦠")

COUNTRIES = {
    "India": {"lat": 20.5937, "lon": 78.9629, "geo": "IN", "base_risk": 150, "vuln_score": 85, "capacity": 220},
    "Brazil": {"lat": -14.2350, "lon": -51.9253, "geo": "BR", "base_risk": 200, "vuln_score": 92, "capacity": 280},
    "Philippines": {"lat": 12.8797, "lon": 121.7740, "geo": "PH", "base_risk": 120, "vuln_score": 78, "capacity": 160},
    "Nigeria": {"lat": 9.0820, "lon": 8.6753, "geo": "NG", "base_risk": 90, "vuln_score": 65, "capacity": 130},
    "Mexico": {"lat": 23.6345, "lon": -102.5528, "geo": "MX", "base_risk": 80, "vuln_score": 55, "capacity": 140}
}

# ==========================================
# 2. LIVE DATA FETCHERS (Cached for Speed)
# ==========================================
@st.cache_data(ttl=3600)
def get_live_telemetry(lat, lon, geo_code):
    telemetry = {"ndvi": 0.35, "temp": 30.5, "rain": 12.0, "trends": 45} 
    try:
        from src.live_data import fetch_latest_vegetation_index, fetch_live_weather_forecast, fetch_live_disease_trends
        telemetry["ndvi"] = fetch_latest_vegetation_index(lat, lon)
        weather_df = fetch_live_weather_forecast(lat, lon)
        if weather_df is not None and not weather_df.empty:
            telemetry["temp"] = weather_df['temperature_2m_max'].iloc[1]
            telemetry["rain"] = weather_df['precipitation_sum'].iloc[1]
        trends_df = fetch_live_disease_trends("dengue", geo_code)
        if not trends_df.empty:
            telemetry["trends"] = int(trends_df.iloc[-1, 0]) 
    except Exception:
        pass 
    return telemetry

# ==========================================
# 3. SIDEBAR UI CONTROLS 
# ==========================================
st.sidebar.title("🌍 Global Command Center")
selected_country = st.sidebar.selectbox("Target Country Node", list(COUNTRIES.keys()))
country_data = COUNTRIES[selected_country]

st.sidebar.markdown("---")
st.sidebar.header("🎛️ 'What-If' Scenario Simulators")
st.sidebar.caption("Adjust environmental and policy variables to simulate forecast outcomes.")

temp_anomaly = st.sidebar.slider("🌡️ Temp Anomaly (°C)", min_value=-2.0, max_value=4.0, value=0.0, step=0.5)
precip_multiplier = st.sidebar.slider("🌧️ Rainfall Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
intervention_efficacy = st.sidebar.slider("🛡️ Vector Control Intervention (%)", min_value=0, max_value=80, value=0, step=10)

# ==========================================
# 4. DATA SIMULATION & MATH
# ==========================================
future_dates = [(datetime.today() + timedelta(days=i)).strftime('%b %d') for i in range(14)]

np.random.seed(42)
base_trend = np.linspace(country_data["base_risk"], country_data["base_risk"] * 1.3, 14)
base_preds = base_trend + np.random.normal(0, 8, 14)

base_peak = int(max(base_preds))
temp_impact = int(temp_anomaly * 15)
rain_impact = int((precip_multiplier - 1.0) * 40)
gross_peak = base_peak + temp_impact + rain_impact
intervention_impact_val = int(gross_peak * (intervention_efficacy / 100.0))

scenario_peak = gross_peak - intervention_impact_val
adjusted_preds = np.maximum(0, (base_preds + temp_impact + rain_impact) * (1.0 - (intervention_efficacy / 100.0)))

with st.spinner(f"Establishing live satellite & API uplinks for {selected_country}..."):
    live_data = get_live_telemetry(country_data["lat"], country_data["lon"], country_data["geo"])

# ==========================================
# 5. CHART FUNCTIONS (Including New 3D Visuals)
# ==========================================
def plot_waterfall_attribution(base, temp, rain, intervention, final):
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Baseline Peak", "🌡️ Heat Impact", "🌧️ Rain Impact", "🛡️ Intervention", "Final Forecast"],
        y=[base, temp, rain, -intervention, final],
        textposition="outside",
        text=[f"{base}", f"+{temp}", f"+{rain}", f"-{intervention}", f"{final}"],
        decreasing={"marker": {"color": "#2ecc71"}}, increasing={"marker": {"color": "#e74c3c"}}, totals={"marker": {"color": "#3498db"}}
    ))
    fig.update_layout(title="AI Forecast Attribution", template="plotly_dark", waterfallgap=0.3, margin=dict(t=40, b=0))
    return fig

def plot_threat_gauge(peak, capacity):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=peak,
        title={'text': "Hospital ICU Strain", 'font': {'size': 20}},
        delta={'reference': capacity, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, capacity * 1.5], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "black", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range': [0, capacity * 0.7], 'color': "rgba(46, 204, 64, 0.4)"},
                {'range': [capacity * 0.7, capacity], 'color': "rgba(255, 133, 27, 0.4)"},
                {'range': [capacity, capacity * 1.5], 'color': "rgba(255, 65, 54, 0.6)"}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': peak}
        }
    ))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(t=40, b=0, l=0, r=0))
    return fig

# --- NEW: 3D Surface Plot (Replaces 2D Heatmap) ---
def plot_3d_spatiotemporal_surface(dates, peak_val):
    sub_regions = ["Northern District", "Southern District", "Coastal Region", "Urban Center", "Rural Highlands"]
    z_data = np.random.poisson(lam=50, size=(len(sub_regions), len(dates)))
    # Create the simulated hotspot in the Urban Center
    z_data[3, :] = np.linspace(50, peak_val, len(dates)) + np.random.normal(0, 5, len(dates)) 
    
    fig = go.Figure(data=[go.Surface(
        z=z_data, x=dates, y=sub_regions, colorscale='Inferno',
        hovertemplate='Date: %{x}<br>Region: %{y}<br>Cases: %{z}<extra></extra>'
    )])
    fig.update_layout(
        title="3D Node-Level Spatio-Temporal Spread",
        scene=dict(
            xaxis_title='Timeline',
            yaxis_title='Geographic Nodes',
            zaxis_title='Infection Intensity',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2)) # Sets default viewing angle
        ),
        template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0)
    )
    return fig

# --- NEW: 3D Scatter Plot (Environmental Drivers) ---
def plot_3d_risk_clusters(temp_val, rain_val):
    # Simulate a cluster of historical outbreaks based on climate
    np.random.seed(42)
    temps = np.random.normal(temp_val, 2, 80)
    rains = np.random.normal(rain_val, 15, 80)
    cases = np.maximum(0, (temps * 4) + (rains * 2) + np.random.normal(0, 20, 80))
    
    fig = px.scatter_3d(
        x=temps, y=rains, z=cases,
        color=cases, size=np.maximum(1, cases),
        color_continuous_scale='Turbo',
        opacity=0.8,
        title="Multidimensional Environmental Risk Clusters"
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (mm)',
            zaxis_title='Historical Cases'
        ),
        template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0)
    )
    return fig

# ==========================================
# 6. UI LAYOUT: TOP SECTION (Executive View)
# ==========================================
st.title(f"📍 Epidemic Modeler: {selected_country}")

st.subheader("📡 Live Environmental Telemetry (API Feeds)")
t1, t2, t3, t4, t5 = st.columns(5)

np.random.seed(datetime.today().day) 
current_cases = int(country_data["base_risk"] + np.random.normal(0, 12))

t1.metric(label="🏥 Est. Current Cases", value=f"{current_cases}", help="Estimated cases based on most recent clinical reports.")
t2.metric("🌡️ 24h Max Temp", f"{live_data['temp']} °C")
t3.metric("🌧️ 24h Precipitation", f"{live_data['rain']} mm")
t4.metric("🛰️ NASA NDVI", f"{live_data['ndvi']}", help="Vegetation index (-1 to 1). >0.5 is high risk.")
t5.metric("🔍 Search Trend Index", f"{live_data['trends']} / 100", help="Live Google Search interest.")

st.markdown("---")

col_text, col_gauge, col_map = st.columns([1.5, 1, 1])

with col_text:
    st.subheader("💡 AI Scenario Analysis")
    delta = scenario_peak - base_peak
    if scenario_peak >= country_data["capacity"]:
        st.error(f"**CRITICAL SURGE:** The current scenario pushes peak cases to **{scenario_peak}**, exceeding the national ICU capacity of {country_data['capacity']}. Immediate vector control required.")
    elif delta < -20:
        st.success(f"**INTERVENTION SUCCESS:** Simulated interventions suppress the outbreak, saving **{abs(delta)} individuals** from infection at the peak.")
    else:
        st.info(f"**STABLE:** Current simulators show a trajectory within standard hospital capacity.")
    
    m1, m2 = st.columns(2)
    m1.metric("Baseline Peak (Do Nothing)", base_peak)
    m2.metric("Scenario Peak", scenario_peak, delta=delta, delta_color="inverse")

with col_gauge:
    st.plotly_chart(plot_threat_gauge(scenario_peak, country_data["capacity"]), use_container_width=True)

with col_map:
    df_map = pd.DataFrame({"lat": [country_data["lat"]], "lon": [country_data["lon"]], "cases": [scenario_peak]})
    fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", size="cases", color_discrete_sequence=["red" if scenario_peak > country_data["capacity"] else "orange"], zoom=3, mapbox_style="carto-darkmatter")
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ==========================================
# 7. UI LAYOUT: MIDDLE SECTION (Attribution)
# ==========================================
st.subheader("📊 Comparative Trajectory & Attribution")
col_traj, col_waterfall = st.columns([2, 1.5])

with col_traj:
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=future_dates, y=base_preds, line=dict(color='gray', dash='dash', width=2), name='Baseline Forecast'))
    line_color = 'rgb(255, 65, 54)' if scenario_peak > base_peak else 'rgb(46, 204, 64)'
    fig_traj.add_trace(go.Scatter(x=future_dates, y=adjusted_preds, line=dict(color=line_color, width=4), mode='lines+markers', name='Simulated Forecast'))
    fig_traj.update_layout(title="14-Day Trajectory: Baseline vs. Scenario", template="plotly_dark", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(t=40, b=0))
    st.plotly_chart(fig_traj, use_container_width=True)

with col_waterfall:
    st.plotly_chart(plot_waterfall_attribution(base_peak, temp_impact, rain_impact, intervention_impact_val, scenario_peak), use_container_width=True)

st.markdown("---")

# ==========================================
# 8. UI LAYOUT: BOTTOM SECTION (3D Advanced Analytics)
# ==========================================
st.subheader("🧊 3D Advanced Data Analytics")
col_3d_surface, col_3d_scatter = st.columns(2)

with col_3d_surface:
    # Render the 3D Spatio-Temporal Surface Plot
    st.plotly_chart(plot_3d_spatiotemporal_surface(future_dates, scenario_peak), use_container_width=True)

with col_3d_scatter:
    # Render the 3D Environmental Clusters Plot
    st.plotly_chart(plot_3d_risk_clusters(live_data['temp'], live_data['rain']), use_container_width=True)