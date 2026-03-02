import pandas as pd
import requests
from pytrends.request import TrendReq
import ee
import datetime

# ==========================================
# 1. EARTH ENGINE INITIALIZATION
# ==========================================
# This attempts to connect to Google's satellite servers.
# You must run `earthengine authenticate` in your terminal once before this works.
try:
    ee.Initialize()
    EE_INITIALIZED = True
except Exception as e:
    print(f"Earth Engine Warning: Not initialized. ({e})")
    print("Run `earthengine authenticate` in your terminal to enable satellite data.")
    EE_INITIALIZED = False

# ==========================================
# 2. METEOROLOGICAL DATA (Weather)
# ==========================================
def fetch_live_weather_forecast(lat, lon):
    """
    Fetches the 14-day weather forecast for the exact coordinates.
    Source: Open-Meteo (Free Tier)
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto",
        "forecast_days": 14
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() # Check for HTTP errors
        
        data = response.json()
        df = pd.DataFrame(data['daily'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None

# ==========================================
# 3. SYNDROMIC SURVEILLANCE (Google Trends)
# ==========================================
def fetch_live_disease_trends(keyword="dengue symptoms", geo_code="IN"):
    """
    Pulls live Google Search interest to act as an early warning proxy.
    Source: Google Trends via PyTrends
    """
    try:
        # tz=360 is US CST, but timeframe 'today 1-m' gets the last 30 days globally
        pytrend = TrendReq(hl='en-US', tz=360, timeout=(10,25))
        pytrend.build_payload(kw_list=[keyword], timeframe='today 1-m', geo=geo_code)
        
        trends_df = pytrend.interest_over_time()
        
        # Clean up the dataframe if data was successfully returned
        if not trends_df.empty:
            trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')
            
        return trends_df
    except Exception as e:
        print(f"PyTrends Error (Likely Rate Limit): {e}")
        # Return an empty dataframe so the main app doesn't crash
        return pd.DataFrame() 

# ==========================================
# 4. SATELLITE TELEMETRY (Vegetation Index)
# ==========================================
def fetch_latest_vegetation_index(lat, lon):
    """
    Pulls the latest NDVI (Vegetation Density) from the NASA MODIS Satellite.
    Scale: -1.0 (Water/Concrete) to 1.0 (Dense Rainforest).
    Source: Google Earth Engine
    """
    if not EE_INITIALIZED:
        return 0.3 # Fallback moderate value if auth fails
        
    try:
        point = ee.Geometry.Point([lon, lat])
        
        # Get data from the last 30 days
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Access NASA MODIS Vegetation dataset
        dataset = ee.ImageCollection('MODIS/061/MOD13Q1') \
                    .filterBounds(point) \
                    .filterDate(start_date, end_date) \
                    .sort('system:time_start', False)
        
        latest_image = dataset.first()
        
        # Extract the exact pixel value at our coordinate (250m resolution)
        ndvi_data = latest_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=250
        ).getInfo()
        
        # MODIS stores NDVI scaled by 10,000. Revert it to -1 to 1 scale.
        raw_ndvi = ndvi_data.get('NDVI', 0)
        
        if raw_ndvi is None:
            return 0.0 # No data found for this exact pixel
            
        actual_ndvi = raw_ndvi * 0.0001 
        return round(actual_ndvi, 4)

    except Exception as e:
        print(f"Earth Engine API Error: {e}")
        return 0.3 # Fallback value

# ==========================================
# 5. STANDALONE TESTING BLOCK
# ==========================================
# This allows you to run `python src/live_data.py` in your terminal to test APIs
if __name__ == "__main__":
    print("--- Testing Live Data Pipeline ---")
    
    # Test coordinates: Bangalore, India
    test_lat, test_lon, test_geo = 12.9716, 77.5946, "IN"
    
    print("\n1. Pinging Open-Meteo...")
    weather = fetch_live_weather_forecast(test_lat, test_lon)
    if weather is not None:
        print(f"Success! Retrieved 14 days of weather. Max Temp tomorrow: {weather['temperature_2m_max'].iloc[1]}°C")
        
    print("\n2. Pinging Google Trends...")
    trends = fetch_live_disease_trends("dengue symptoms", test_geo)
    if not trends.empty:
        print(f"Success! Retrieved {len(trends)} days of search history.")
    else:
        print("Failed or Rate Limited.")
        
    print("\n3. Pinging NASA Earth Engine...")
    ndvi = fetch_latest_vegetation_index(test_lat, test_lon)
    print(f"Success! Latest NDVI for coordinates: {ndvi}")
    
    print("\n--- Pipeline Check Complete ---")