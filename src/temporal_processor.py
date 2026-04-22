import pandas as pd
from scipy.interpolate import pchip_interpolate
import numpy as np

def interpolate_to_15min_pchip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates hourly weather data (Wind Speed, Temperature, Pressure)
    down to 15-minute intervals using PCHIP (Piecewise Cubic Hermite Interpolating Polynomial).
    
    Why PCHIP instead of Linear or Standard Cubic Spline?
    Cubic splines cause 'overshoot' (creating physically impossible high artificial peaks).
    PCHIP guarantees a smooth curvature tracking momentum, whilst refusing to exceed the 
    known bounds of the hourly points.
    
    This matches the exact 96-block 15-min reporting requirement for Indian CERC DSM calculations.
    """
    print("⏳ Applying 15-Minute PCHIP Temporal Interpolation (CERC 96-Block Standards)...")
    
    df = df.sort_values('time').copy()
    
    # We only interpolate the raw atmospheric variables.
    # The Physics power output must be fully recalculated based on these newly smoothed 15-min variables.
    cols_to_interpolate = ['wind_speed_100m', 'temperature_2m', 'surface_pressure']
    
    # Filter only available columns
    cols_to_interpolate = [c for c in cols_to_interpolate if c in df.columns]

    # Create a 15-min time grid traversing the original dataframe
    # We use `.resample('15min').asfreq()` on the index
    df_time_idx = df.set_index('time')
    df_15min = df_time_idx[cols_to_interpolate].resample('15min').asfreq()
    
    # Gather exact numeric coordinates for PCHIP
    # Hour 1 = 0, Hour 2 = 1.0, and 1:15 PM = 0.25
    x_orig = np.arange(len(df))
    x_new = np.linspace(0, len(df) - 1, len(df_15min))
    
    for col in cols_to_interpolate:
        y_orig = df[col].values
        # Perform the Pchip interpolation
        y_interp = pchip_interpolate(x_orig, y_orig, x_new)
        df_15min[col] = y_interp
        
    df_15min = df_15min.reset_index()
    
    return df_15min
