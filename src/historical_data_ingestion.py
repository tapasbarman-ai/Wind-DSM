import os
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry

def fetch_historical_wind_data(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch 10 years of historical weather data using the Open-Meteo Archive API.
    Since we are getting 87,600+ rows, we use the FlatBuffers implementation via openmeteo_requests for speed/efficiency.
    """
    print(f"Initializing Open-Meteo Historical Archive API...")
    print(f"Location -> Lat: {lat}, Lon: {lon}")
    print(f"Date Range -> From {start_date} to {end_date}")
    
    # Setup the Open-Meteo API client with cache and retry on error
    # This prevents redownloading if we run the script multiple times
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # API Endpoint for historical data
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "surface_pressure", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m"],
        "timezone": "auto"
    }
    
    print("Calling API and downloading payload...")
    responses = openmeteo.weather_api(url, params=params)

    # Process first location
    response = responses[0]
    
    print(f"Coordinates processed: {response.Latitude()}°N, {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    
    # Process hourly data. The order of variables needs to be the exact same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(4).ValuesAsNumpy()

    print("Formatting arrays into Pandas DataFrame...")
    hourly_data = {"time": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
    hourly_data["wind_gust"] = hourly_wind_gusts_10m

    df = pd.DataFrame(data = hourly_data)
    
    # Clean up empty rows if any at very end
    df = df.dropna(subset=['wind_speed_100m'])
    
    # Step 1: Convert degrees to U and V vectors for ML engineering
    # Meteorological wind direction is the direction FROM which the wind blows.
    wind_dir_rad = np.radians(df['wind_direction_100m'])
    
    # U and V vectors representing the direction the wind is blowing TO
    df['U'] = -df['wind_speed_100m'] * np.sin(wind_dir_rad)
    df['V'] = -df['wind_speed_100m'] * np.cos(wind_dir_rad)
    
    return df

if __name__ == "__main__":
    import yaml
    
    os.makedirs("data/01_raw", exist_ok=True)
    
    # Load all wind farms from the configuration file
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    for park in config['parks']:
        park_id = park['id']
        lat = park['lat']
        lon = park['lng']
        
        # We will just fetch the 10yr dataset for each park to standardize
        ds_name = "10yr"
        start_date = "2015-01-01"
        end_date = "2024-12-31"
        
        out_file = f"data/01_raw/{park_id}_{ds_name}_historical.csv"
        
        if os.path.exists(out_file):
            print(f"\nSkipping {park['name']} - data already downloaded.")
            continue
            
        print(f"\n=============================================")
        print(f"Downloading {ds_name} Dataset for {park['name']} ({start_date} to {end_date})")
        print(f"=============================================")
        
        try:
            df_park = fetch_historical_wind_data(lat, lon, start_date, end_date)
            print(f"\n--------- DOWNLOAD COMPLETE ---------")
            print(f"Dataset Shape: {df_park.shape}")
            print(f"Saving to -> {out_file}")
            df_park.to_csv(out_file, index=False)
            print(f"Done saving {park['name']} dataset!")
        except Exception as e:
            print(f"Failed to download data for {park['name']}: {e}")
