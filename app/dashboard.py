import streamlit as st
import requests
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- UI/UX Layout ---
st.set_page_config(page_title="Epidemic Early Warning System", layout="wide")

st.title("🦠 Spatio-Temporal Epidemic Forecasting")
st.markdown("Predicting localized outbreak risks using Deep Learning (STGCN).")

# --- Sidebar for User Interaction ---
st.sidebar.header("Control Panel")
selected_region = st.sidebar.selectbox("Select Focus Region", ["San Juan (Puerto Rico)", "Iquitos (Peru)"])
forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", 1, 4, 4)

# --- Main Dashboard Split ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Spatio-Temporal Risk Map")
    # Build an interactive map centered on Puerto Rico
    m = folium.Map(location=[18.4663, -66.1057], zoom_start=8)
    
    # Add a mock "High Risk" heatmap marker for UI/UX demonstration
    folium.CircleMarker(
        location=[18.4663, -66.1057],
        radius=30,
        color="red",
        fill=True,
        fill_color="red",
        popup="San Juan: High Risk Forecast"
    ).add_to(m)
    
    # Render the map in Streamlit
    st_folium(m, width=700, height=400)

with col2:
    st.subheader("Run Prediction Model")
    if st.button("Generate Forecast", type="primary"):
        with st.spinner('Querying FastAPI Backend...'):
            # In a real app, this hits your local FastAPI server running on port 8000
            # response = requests.post("http://127.0.0.1:8000/predict", json={...})
            
            # Mocking the API response for the UI layout
            mock_data = pd.DataFrame({
                "Week": ["Week +1", "Week +2", "Week +3", "Week +4"],
                "Predicted Cases": [120, 135, 150, 180]
            })
            
            st.success("Forecast generated successfully!")
            st.dataframe(mock_data, hide_index=True)
            
            st.line_chart(mock_data.set_index("Week"))
            
            if mock_data["Predicted Cases"].max() > 140:
                st.error("⚠️ Critical Outbreak Warning triggered for Week +3.")