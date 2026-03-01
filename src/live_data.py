import pandas as pd
import requests
from pytrends.request import TrendReq

def fetch_live_weather_forecast(lat, lon):
    """Fetches the 14-day weather forecast for the selected country's coordinates."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": 14
    }
    try:
        response = requests.get(url, params=params).json()
        df = pd.DataFrame(response['daily'])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None

def fetch_live_disease_trends(keyword="dengue symptoms", geo_code="IN"):
    """
    Pulls live Google Search interest for a specific country.
    """
    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        # timeframe='today 1-m' gets data for the last 30 days
        pytrend.build_payload(kw_list=[keyword], timeframe='today 1-m', geo=geo_code)
        trends_df = pytrend.interest_over_time()
        
        if not trends_df.empty:
            trends_df = trends_df.drop(columns=['isPartial'])
        return trends_df
    except Exception as e:
        print(f"PyTrends Error: {e}")
        return pd.DataFrame() # Return empty dataframe on rate limit