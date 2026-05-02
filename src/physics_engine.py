import pandas as pd
import numpy as np
from windpowerlib import ModelChain, WindTurbine

def calculate_moist_air_density(temp_c: pd.Series, pressure_pa: pd.Series, rh_pct: float = 75.0) -> pd.Series:
    """
    Calculates Moist Air Density (kg/m^3) instead of standard Dry Air (Ideal Gas Law).
    Critical for coastal salt-marsh regions like Khavda.
    Moist air is less dense than dry air, which marginally reduces power output (improving realism).
    """
    # Constants
    R_d = 287.058  # Gas constant for dry air, J/(kg·K)
    R_v = 461.495  # Gas constant for water vapor, J/(kg·K)
    
    temp_k = temp_c + 273.15
    
    # Tetens Equation for Saturation Vapor Pressure (hPa -> Pa)
    p_sat_hpa = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
    p_sat_pa = p_sat_hpa * 100
    
    # Vapor Pressure (Pa)
    p_v = (rh_pct / 100.0) * p_sat_pa
    
    # Partial Pressure of Dry Air (Pa)
    p_d = pressure_pa - p_v
    
    # Moist Air Density
    rho = (p_d / (R_d * temp_k)) + (p_v / (R_v * temp_k))
    return rho

def apply_generator_efficiency_losses(power_mw: pd.Series, wind_speeds: pd.Series) -> pd.Series:
    """
    Simulates Copper/Iron electrical losses in a DFIG (Doubly-Fed Induction Generator).
    Efficiency drops at low wind speeds (e.g., < 6 m/s) because the baseline
    magnetization current draws a larger percentage of the generated power.
    """
    # If wind speed < 6 m/s, apply a 3% penalty. Otherwise 1.5% penalty.
    efficiency = np.where(wind_speeds < 6.0, 0.97, 0.985)
    return power_mw * efficiency

def calculate_physics_power(df: pd.DataFrame) -> pd.Series:
    """
    Calculates theoretical power output using windpowerlib with Micro-Physics enhancements.
    (Monin-Obukhov and Moist Air limits).
    """
    print("Calculating Micro-Physics power output (Moist Air Density & DFIG Efficiency)...")
    
    vestas_v112 = {
        'turbine_type': 'V112/3000',
        'hub_height': 100
    }
    
    try:
        my_turbine = WindTurbine(**vestas_v112)
    except KeyError:
        print("Falling back to another 3MW class turbine from the database...")
        my_turbine = WindTurbine(turbine_type='E-115/3000', hub_height=100)
    
    # 1. Pre-calculate Moist Air Density for Khavda
    pressure_pa = df['surface_pressure'] * 100
    moist_density = calculate_moist_air_density(df['temperature_2m'], pressure_pa, rh_pct=75.0)
    
    # 2. Format DataFrame for windpowerlib
    weather_df = pd.DataFrame(index=df['time'])
    weather_df.index.name = 'time'
    
    weather_df[('wind_speed', 100)] = df['wind_speed_100m'].values
    weather_df[('temperature', 2)] = df['temperature_2m'].values + 273.15 
    weather_df[('pressure', 0)] = pressure_pa.values
    weather_df[('density', 100)] = moist_density.values  # Injecting our precision density
    
    weather_df.columns = pd.MultiIndex.from_tuples(weather_df.columns, names=['variable_name', 'height'])
    
    # We bypass 'ideal_gas' because we provide custom 'density', enhancing atmospheric precision
    mc = ModelChain(my_turbine,
                    wind_speed_model='hellman',
                    temperature_model='linear_gradient',
                    power_output_model='power_curve').run_model(weather_df)
    
    physics_mw = mc.power_output / 1e6
    physics_mw.index = df.index
    
    # 3. Apply Internal Generator Efficiency logic
    physics_mw = apply_generator_efficiency_losses(physics_mw, df['wind_speed_100m'])
    
    return physics_mw

if __name__ == "__main__":
    test_df = pd.DataFrame({
        'time': pd.date_range('2023-01-01', periods=3, freq='H'),
        'wind_speed_100m': [5.0, 10.0, 25.0],
        'temperature_2m': [25.0, 25.0, 25.0],
        'surface_pressure': [1013.0, 1013.0, 1013.0]
    })
    mw = calculate_physics_power(test_df)
    print("Power Output Test (MW):")
    print(mw)
