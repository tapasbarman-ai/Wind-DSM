import os
import time
import requests
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
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
        
        response = requests.head(url, params=params)
        if response.status_code == 200:
            return date_str, cycle_str
            
    raise Exception("Could not find a valid recent GFS cycle on NOMADS.")

def download_grib(forecast_hour, date_str, cycle_str, lat, lon):
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
        "subregion": "",
        "leftlon": lon,
        "rightlon": lon,
        "toplat": lat,
        "bottomlat": lat,
        "dir": dir_path
    }
    
    tmp_filename = f"tmp_gfs_{cycle_str}_{forecast_hour:03d}.grib2"
    
    for _ in range(3):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(tmp_filename, "wb") as f:
                f.write(resp.content)
            return tmp_filename
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
            'wind_gust': float(df_sfc.get('gust', pd.Series([0.0])).iloc[0])
        }
        
    except Exception as e:
        print(f"Failed to parse {forecast_hour:03d}: {e}")
        row = None
    finally:
        # Cleanup
        if os.path.exists(tmp_filename):
            try: os.remove(tmp_filename)
            except: pass
        if os.path.exists(tmp_filename + ".idx"):
            try: os.remove(tmp_filename + ".idx")
            except: pass
            
    return row

def fetch_data(lat: float, lon: float, forecast_hours=24) -> pd.DataFrame:
    print(f"Initializing NOAA NOMADS GFS access for lat: {lat}, lon: {lon}")
    
    date_str, cycle_str = get_latest_nomads_cycle(lat, lon)
    print(f"Using Latest Available Cycle: Date {date_str}, Cycle {cycle_str}z")
    
    # Download in parallel
    print(f"Downloading {forecast_hours} hours of forecasting data from NOAA servers...")
    downloaded_files = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_grib, h, date_str, cycle_str, lat, lon): h 
            for h in range(forecast_hours)
        }
        
        for future in concurrent.futures.as_completed(futures):
            hour = futures[future]
            tmp_filename = future.result()
            if tmp_filename:
                downloaded_files[hour] = tmp_filename
                
    # Parse sequentially
    print("Parsing downloaded GRIB files...")
    rows = []
    for h in sorted(downloaded_files.keys()):
        row = parse_grib(downloaded_files[h], h)
        if row:
            rows.append(row)
                
    df = pd.DataFrame(rows)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Data successfully fetched from NOAA Primary Operational Access!")
    return df

if __name__ == "__main__":
    df_jaisalmer = fetch_data(26.5, 71.5, forecast_hours=6)
    print(df_jaisalmer.head())
