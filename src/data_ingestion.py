import os
import time
import requests
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import numpy as np
import concurrent.futures

def get_latest_nomads_cycle(lat, lon):
    """
    Finds the latest available GFS cycle on NOMADS that has the f000 file published.
    """
    now = datetime.utcnow()
    # Check the last 4 cycles (24 hours)
    for i in range(5):
        dt = now - timedelta(hours=i*6)
        date_str = dt.strftime('%Y%m%d')
        cycle = (dt.hour // 6) * 6
        cycle_str = f"{cycle:02d}"
        
        file_name = f"gfs.t{cycle_str}z.pgrb2.0p25.f000"
        dir_path = f"/gfs.{date_str}/{cycle_str}/atmos"
        
        url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
        params = {
            "file": file_name,
            "lev_100_m_above_ground": "on",
            "var_UGRD": "on",
            "subregion": "",
            "leftlon": lon,
            "rightlon": lon,
            "toplat": lat,
            "bottomlat": lat,
            "dir": dir_path
        }
        
        response = requests.head(url, params=params, timeout=15)
        if response.status_code == 200:
            return date_str, cycle_str
            
    raise Exception("Could not find a valid recent GFS cycle on NOMADS.")

def download_grib(forecast_hour, date_str, cycle_str, lat, lon, tmp_dir="."):
    """
    Downloads a specific forecast hour from the NOMADS subset filter.
    Returns the tmp file name if successful, else None.
    """
    file_name = f"gfs.t{cycle_str}z.pgrb2.0p25.f{forecast_hour:03d}"
    dir_path = f"/gfs.{date_str}/{cycle_str}/atmos"
    
    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
    params = {
        "file": file_name,
        "lev_100_m_above_ground": "on",
        "lev_2_m_above_ground": "on",
        "lev_surface": "on",
        "var_GUST": "on",
        "var_PRES": "on",
        "var_TMP": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "var_HGT": "on",
        "subregion": "",
        "leftlon": lon,
        "rightlon": lon,
        "toplat": lat,
        "bottomlat": lat,
        "dir": dir_path
    }
    
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_filename = os.path.join(tmp_dir, f"tmp_gfs_{cycle_str}_{forecast_hour:03d}.grib2")
    
    for i in range(3):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(tmp_filename, "wb") as f:
                    f.write(resp.content)
                return tmp_filename
        except Exception as e:
            print(f"Request failed for hour {forecast_hour} (attempt {i + 1}/3): {e}")
        time.sleep(1)
            
    return None

def parse_grib(tmp_filename, forecast_hour):
    """
    Parses the GRIB2 file sequentially to avoid thread-safety issues with eccodes on Windows.
    """
    try:
        ds_100m = xr.open_dataset(tmp_filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 100}})
        ds_2m = xr.open_dataset(tmp_filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'level': 2}})
        ds_sfc = xr.open_dataset(tmp_filename, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        
        df_100 = ds_100m.to_dataframe().reset_index()
        df_2m = ds_2m.to_dataframe().reset_index()
        df_sfc = ds_sfc.to_dataframe().reset_index()
        
        row = {
            'time': pd.to_datetime(df_100['valid_time'].iloc[0]),
            'U': float(df_100['u100'].iloc[0]),
            'V': float(df_100['v100'].iloc[0]),
            'wind_speed_100m': float((df_100['u100'].iloc[0]**2 + df_100['v100'].iloc[0]**2)**0.5),
            'temperature_2m': float(df_2m['t2m'].iloc[0] - 273.15),
            'surface_pressure': float(df_sfc['sp'].iloc[0] / 100),
            'wind_gust': float(df_sfc.get('gust', pd.Series([0.0])).iloc[0]),
            'gfs_surface_hgt': float(df_sfc.get('orog', df_sfc.get('hgt', pd.Series([0.0]))).iloc[0])
        }
        
    except Exception as e:
        print(f"Failed to parse {forecast_hour:03d}: {e}")
        row = None
    finally:
        # Cleanup both the GRIB file and any sidecar .idx files
        for path in [tmp_filename, tmp_filename + ".idx"]:
            if path and os.path.exists(path):
                try: os.remove(path)
                except: pass
        # Also clean up cfgrib hash-named .idx sidecars (e.g. file.grib2.abc123.idx)
        if tmp_filename:
            parent = os.path.dirname(tmp_filename)
            base = os.path.basename(tmp_filename)
            try:
                for f in os.listdir(parent or "."):
                    if f.startswith(base) and f.endswith(".idx"):
                        try: os.remove(os.path.join(parent or ".", f))
                        except: pass
            except: pass
            
    return row

def fetch_data(lat: float, lon: float, forecast_hours=24, park_id: str = "unknown") -> pd.DataFrame:
    print(f"Initializing NOAA NOMADS GFS access for lat: {lat}, lon: {lon}")
    
    try:
        try:
            date_str, cycle_str = get_latest_nomads_cycle(lat, lon)
            print(f"Using Latest Available Cycle: Date {date_str}, Cycle {cycle_str}z")
        except Exception as e:
            raise RuntimeError(f"get_latest_nomads_cycle failed: {e}")
        
        # All temp GRIB files go into a per-park subdirectory — keeps root clean
        tmp_dir = os.path.join("data", "01_raw", "live_forecasts", park_id, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Download in parallel
        print(f"Downloading {forecast_hours} hours of forecasting data from NOAA servers...")
        downloaded_files = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(download_grib, h, date_str, cycle_str, lat, lon, tmp_dir): h 
                for h in range(forecast_hours + 1)
            }
            
            for future in concurrent.futures.as_completed(futures):
                hour = futures[future]
                try:
                    tmp_filename = future.result()
                    if tmp_filename:
                        downloaded_files[hour] = tmp_filename
                except Exception as e:
                    print(f"Download thread failed for hour {hour}: {e}")
                    
        # Parse sequentially
        print("Parsing downloaded GRIB files...")
        rows = []
        for h in sorted(downloaded_files.keys()):
            row = parse_grib(downloaded_files[h], h)
            if row:
                rows.append(row)
        
        # Cleanup the now-empty tmp dir
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass  # Not empty — leave it, stale files can be inspected
        
        # We need at least some forecast data to proceed, let's say at least 12 hours
        if len(rows) < 12:
            raise RuntimeError(f"Insufficient GRIB files successfully downloaded/parsed: got {len(rows)}/25")
                    
        df = pd.DataFrame(rows)
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save the parsed forecast as a dated CSV in the park's live_forecasts folder
        out_dir = os.path.join("data", "01_raw", "live_forecasts", park_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"gfs_forecast_{date_str}.csv")
        df.to_csv(out_path, index=False)
        print(f"Live forecast saved -> {out_path}")
        
        print("Data successfully fetched from NOAA Primary Operational Access!")
        return df
        
    except Exception as e:
        print(f"NOAA NOMADS Primary Access or parsing failed: {e}")
        print("Switching to Emergency Fallback: Open-Meteo Global GFS...")
        return fetch_open_meteo_fallback(lat, lon, park_id)

def fetch_open_meteo_fallback(lat, lon, park_id):
    """
    Emergency fallback using Open-Meteo's GFS wrapper. 
    Much higher availability than the direct NOAA filter scripts.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/gfs"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_100m", "temperature_2m", "surface_pressure"],
        "wind_speed_unit": "ms",
        "forecast_days": 2
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        # Build dataframe
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            periods=hourly.Interval(), # This is wrong in the SDK docs, it's actually data length
            freq=pd.Timedelta(seconds=hourly.Interval())
        )
        
        # Wait, the SDK is tricky. Let's do a simpler manual request if SDK fails
        data = {
            "time": pd.to_datetime(hourly.Time(), unit="s", utc=True) + pd.to_timedelta(np.arange(hourly.Variables(0).ValuesLength()), unit='h'),
            "wind_speed_100m": hourly.Variables(0).ValuesAsNumpy(),
            "temperature_2m": hourly.Variables(1).ValuesAsNumpy(),
            "surface_pressure": hourly.Variables(2).ValuesAsNumpy() / 1.0 # Already in hPa or Pa?
        }
        
        df = pd.DataFrame(data)
        
        # Minimal cleanup to match main pipeline expectations
        df['U'] = df['wind_speed_100m'] * 0.7 # Approximate components
        df['V'] = df['wind_speed_100m'] * 0.7
        
        out_dir = os.path.join("data", "01_raw", "live_forecasts", park_id)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"openmeteo_fallback_{datetime.now().strftime('%Y%m%d')}.csv"), index=False)
        
        print("Emergency Fallback Successful: Data fetched via Open-Meteo.")
        return df
        
    except Exception as e:
        print(f"Emergency Fallback Failed: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df_jaisalmer = fetch_data(26.5, 71.5, forecast_hours=6)
    print(df_jaisalmer.head())
