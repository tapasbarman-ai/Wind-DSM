import pandas as pd
import numpy as np

# A standard 3 MW Vestas V112 turbine
CAPACITY_MW = 3.0  
BASE_PPA_PRICE_INR = 3000  # Rs. 3/kWh -> Rs. 3000/MWh baseline assumption

def get_acp_multiplier(time_series: pd.Series) -> pd.Series:
    """
    Simulates the Area Clearing Price (ACP) from the Indian Energy Exchange (IEX).
    Under 2024 CERC rules, Deviation Settlement is no longer tied strictly to PPA.
    It dynamically tracks live, volatile grid market prices.
    
    Deviating during Peak hours (6 PM to 10 PM) incurs massive financial penalties
    since the grid is heavily stressed and Replacement Electricity is expensive.
    """
    hours = time_series.dt.hour
    
    # Peak Hours: 18:00 to 22:00 -> Highly stressed grid, high market price (Multiplier 2.5x)
    # Solar Dip Hours: 06:00 to 09:00 -> Morning ramp, moderately high price (Multiplier 1.5x)
    # Night/Off-Peak: 00:00 to 05:00 -> Low demand (Multiplier 0.8x)
    
    conditions = [
        (hours >= 18) & (hours <= 22),
        (hours >= 6) & (hours <= 9),
        (hours >= 0) & (hours <= 5)
    ]
    choices = [2.5, 1.5, 0.8]
    
    # Normal daytime hours default to 1.0
    multipliers = np.select(conditions, choices, default=1.0)
    return pd.Series(multipliers, index=time_series.index)

def calculate_penalties(df: pd.DataFrame) -> dict:
    """
    Codes the Indian CERC 2024 DSM logic using ACP-Linked dynamic pricing.
    Optimized for 15-Minute Block Settlements.
    """
    
    def calculate_dsm_cost(forecast: pd.Series, actual: pd.Series, time_series: pd.Series) -> pd.Series:
        abs_error_mw = np.abs(forecast - actual)
        error_pct = (abs_error_mw / CAPACITY_MW) * 100
        
        # Calculate time duration mathematically to support both Hourly and 15-Min data seamlessly
        # e.g., 15 minutes = 0.25 Hours. Therefore, Power (MW) * Time (h) = Energy Deviation (MWh)
        time_diffs = time_series.diff().dt.total_seconds().fillna(3600) / 3600.0
        
        # Fallback to hourly if differences are zero or negative
        block_duration_hrs = np.where(time_diffs > 0, time_diffs, 1.0)
        deviation_mwh = abs_error_mw * block_duration_hrs
        
        # Obtain Dynamic Prices based on IEX ACP 
        acp_multipliers = get_acp_multiplier(time_series)
        dynamic_price_inr = BASE_PPA_PRICE_INR * acp_multipliers
        
        penalties = np.zeros(len(error_pct))
        
        # CERC Band 1: 0-10% Error -> Safety buffer, no penalty assessed
        # CERC Band 2: 10-20% Error -> 10% of Live ACP Price
        mask_10_20 = (error_pct > 10) & (error_pct <= 20)
        penalties[mask_10_20] = deviation_mwh[mask_10_20] * (dynamic_price_inr[mask_10_20] * 0.10)
        
        # CERC Band 3: 20-30% Error -> 20% of Live ACP Price
        mask_20_30 = (error_pct > 20) & (error_pct <= 30)
        penalties[mask_20_30] = deviation_mwh[mask_20_30] * (dynamic_price_inr[mask_20_30] * 0.20)
        
        # CERC Band 4: > 30% Error -> 30% of Live ACP Price penalty severity
        mask_30_plus = error_pct > 30
        penalties[mask_30_plus] = deviation_mwh[mask_30_plus] * (dynamic_price_inr[mask_30_plus] * 0.30)
        
        return pd.Series(penalties, index=actual.index)

    df['physics_penalty_inr'] = calculate_dsm_cost(df['physics_mw'], df['actual_mw'], df['time'])
    
    # We assess the penalty utilizing the standard median P50 forecast
    forecast_col = 'final_forecast_mw' if 'final_forecast_mw' in df.columns else 'final_forecast_mw_q50'
    df['ml_penalty_inr'] = calculate_dsm_cost(df[forecast_col], df['actual_mw'], df['time'])
    
    total_physics_penalty = df['physics_penalty_inr'].sum()
    total_ml_penalty = df['ml_penalty_inr'].sum()
    
    savings = total_physics_penalty - total_ml_penalty
    
    return {
        "physics_penalty": total_physics_penalty,
        "ml_penalty": total_ml_penalty,
        "savings": savings
    }
