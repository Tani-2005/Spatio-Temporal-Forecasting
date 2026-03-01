import requests
import pandas as pd

def fetch_historical_weather(lat, lon, start_date, end_date):
    """
    Pulls daily weather data from Open-Meteo's Archive API.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Format 'YYYY-MM-DD'
        end_date (str): Format 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: A daily time-series dataframe of weather variables.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "relative_humidity_2m_mean"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['daily'])
        
        # Convert the string dates to actual Pandas Datetime objects
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    else:
        print(f"Error fetching data: HTTP {response.status_code}")
        return None

def aggregate_to_weekly(daily_df):
    """
    Converts daily weather data into weekly summaries to match the DengAI dataset.
    
    Args:
        daily_df (pd.DataFrame): The daily dataframe from fetch_historical_weather.
        
    Returns:
        pd.DataFrame: Weekly aggregated dataframe.
    """
    # Resample by week ('W'). We take the mean for temp/humidity, and sum for rain.
    weekly_df = daily_df.resample('W').agg({
        'temperature_2m_mean': 'mean',
        'relative_humidity_2m_mean': 'mean',
        'precipitation_sum': 'sum'
    })
    
    # Rename columns for clarity
    weekly_df.columns = ['avg_weekly_temp', 'avg_weekly_humidity', 'total_weekly_precip']
    
    return weekly_df