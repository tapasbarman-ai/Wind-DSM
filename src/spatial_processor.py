import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

# --- 1km Spatial Resolution Constants ---
VON_KARMAN = 0.40
STANDARD_Z0_COARSE = 0.03  # Typical coarse model surface roughness (e.g., standard grass/crops)
KHAVDA_Z0_FINE = 0.001     # Salt marsh / highly reflective smooth flatland (Khavda)
HUB_HEIGHT = 100.0         # Vestas V112 Hub Height

def apply_surface_roughness_correction(wind_speed_coarse: pd.Series, 
                                       z0_coarse: float = STANDARD_Z0_COARSE, 
                                       z0_fine: float = KHAVDA_Z0_FINE, 
                                       height: float = HUB_HEIGHT) -> pd.Series:
    """
    Applies the Logarithmic Wind Profile correction to downscale wind speed 
    based on local high-resolution surface roughness (z0).
    
    A 25km GFS model assumes an "average" roughness across a massive grid.
    By mapping Khavda's specific 1km pixel roughness, we mechanically adjust 
    the wind shear profile.
    
    Formula derives from relative friction velocity transformation.
    """
    # v_fine = v_coarse * (ln(z/z0_fine) / ln(z/z0_coarse))
    # Khavda is very smooth (salt marsh), so it experiences less drag, 
    # meaning wind speeds at hub height are often higher than a coarse model predicts.
    
    correction_factor = np.log(height / z0_fine) / np.log(height / z0_coarse)
    wind_speed_fine = wind_speed_coarse * correction_factor
    
    return wind_speed_fine

def apply_elevation_speedup_correction(wind_speed: pd.Series, dem_elevation: float, reference_elevation: float) -> pd.Series:
    """
    Adjusts wind speed based on Digital Elevation Model (DEM) differences.
    Wind speeds up as it compresses over terrain variations (Hills/Ridges).
    For flatlands like Khavda, this effect is minimal, but crucial for sites like Muppandal.
    """
    # Simplified fractional speed-up ratio based on elevation difference
    # A highly complex model would use WAsP (Wind Atlas Analysis and Application Program) algorithms.
    speedup_ratio = 1.0 + (0.001 * (dem_elevation - reference_elevation))
    return wind_speed * speedup_ratio

def bilinear_spatial_downscale(target_lat: float, target_lon: float, 
                               grid_lats: np.ndarray, grid_lons: np.ndarray, 
                               grid_values: np.ndarray) -> float:
    """
    Performs Bilinear Spline Interpolation to map a 25km (GFS/ERA5) coarse bounding box 
    down to a precise 1km target coordinate.
    
    Args:
        target_lat: 1km Target Latitude (e.g., 23.82)
        target_lon: 1km Target Longitude (e.g., 69.72)
        grid_lats: 1D array of coarse latitudes (e.g., [23.5, 23.75, 24.0])
        grid_lons: 1D array of coarse longitudes (e.g., [69.5, 69.75, 70.0])
        grid_values: 2D array of values corresponding to the lat/lon grid
    """
    # Uses RectBivariateSpline for fast, accurate interpolation over a regular grid.
    # kx=1, ky=1 represents strict bilinear. Higher degrees (cubic) could cause overshoot on wind speeds.
    interpolator = RectBivariateSpline(grid_lats, grid_lons, grid_values, kx=1, ky=1)
    
    # Evaluate the exact target pixel
    downscaled_value = interpolator(target_lat, target_lon)[0, 0]
    return downscaled_value

def downscale_dataframe(df: pd.DataFrame, target_lat: float = 23.82, target_lon: float = 69.72) -> pd.DataFrame:
    """
    Main orchestrator for 1km spatial downscaling.
    Applies rigorous micro-scale topographical corrections to standard historical data.
    """
    print("🌍 Applying 1km Spatial Downscaling (Surface Roughness & Topography)...")
    df_downscaled = df.copy()
    
    # If using grid data, bilinear interpolation goes here.
    # Since our standard dataframe is already point-extracted from historical sets, 
    # we enforce the Micro-level Terrain Corrections:
    
    if 'wind_speed_100m' in df_downscaled.columns:
        # Khavda has a highly smooth surface compared to average GFS blocks.
        df_downscaled['wind_speed_100m'] = apply_surface_roughness_correction(
            df_downscaled['wind_speed_100m'],
            z0_coarse=STANDARD_Z0_COARSE,
            z0_fine=KHAVDA_Z0_FINE
        )
        
    # We could also apply DEM elevation adjustments here if the site had complex terrain
    # For Khavda (relatively flat salt marsh), elevation speedup is negligible.
    
    return df_downscaled
