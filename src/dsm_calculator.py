import pandas as pd
import numpy as np

def calculate_penalties(df: pd.DataFrame, park_id: str, zone: int, capacity_mw: float, ppa_rate: float = 3.0) -> dict:
    """
    Calculates penalties based on the specific Zone and Regulation provided by the user.
    
    Zones:
    1: Tight Band (Tamil Nadu) - 10% Tolerance, AvC based.
    2: Standard Band (MH, RJ, GJ, etc.) - 15% Tolerance, AvC based.
    3: Market Linked (ISTS/Khavda) - 10% Tolerance, Schedule based.
    """
    
    def calculate_dsm_cost(forecast: pd.Series, actual: pd.Series, time_series: pd.Series) -> pd.Series:
        abs_error_mw = np.abs(forecast - actual)
        
        # Calculate time duration for energy (MWh)
        time_diffs = time_series.diff().dt.total_seconds().fillna(900) / 3600.0
        block_duration_hrs = np.where(time_diffs > 0, time_diffs, 0.25) # Default 15-min
        deviation_mwh = abs_error_mw * block_duration_hrs
        deviation_units = deviation_mwh * 1000 # MWh to kWh (Units)
        
        penalties = np.zeros(len(forecast))
        
        if zone == 1:
            # Zone 1: Tight (Tamil Nadu) - Denominator: AvC (Capacity)
            error_pct = (abs_error_mw / capacity_mw) * 100
            
            # 10-20%: ₹0.50
            mask1 = (error_pct > 10) & (error_pct <= 20)
            penalties[mask1] = deviation_units[mask1] * 0.50
            
            # 20-30%: ₹1.00
            mask2 = (error_pct > 20) & (error_pct <= 30)
            penalties[mask2] = deviation_units[mask2] * 1.00
            
            # >30%: ₹1.50
            mask3 = (error_pct > 30)
            penalties[mask3] = deviation_units[mask3] * 1.50
            
        elif zone == 2:
            # Zone 2: Standard - Denominator: AvC (Capacity)
            error_pct = (abs_error_mw / capacity_mw) * 100
            
            # 15-25%: ₹0.50
            mask1 = (error_pct > 15) & (error_pct <= 25)
            penalties[mask1] = deviation_units[mask1] * 0.50
            
            # 25-35%: ₹1.00
            mask2 = (error_pct > 25) & (error_pct <= 35)
            penalties[mask2] = deviation_units[mask2] * 1.00
            
            # >35%: ₹1.50
            mask3 = (error_pct > 35)
            penalties[mask3] = deviation_units[mask3] * 1.50
            
        elif zone == 3:
            # Zone 3: Market Linked (ISTS) - Denominator: Scheduled Generation (Forecast)
            # Avoid division by zero if forecast is 0
            safe_forecast = np.where(forecast > 0, forecast, 0.0001)
            error_pct = (abs_error_mw / safe_forecast) * 100
            
            # Band: >10%
            # Rate: 10% of PPA per unit
            mask = (error_pct > 10)
            penalty_per_unit = ppa_rate * 0.10
            penalties[mask] = deviation_units[mask] * penalty_per_unit
            
        return pd.Series(penalties, index=actual.index)

    # Physics Penalty
    df['physics_penalty_inr'] = calculate_dsm_cost(df['physics_mw'], df['actual_mw'], df['time'])
    
    # ML Penalty
    forecast_col = 'final_forecast_mw' if 'final_forecast_mw' in df.columns else 'final_forecast_mw_q50'
    df['ml_penalty_inr'] = calculate_dsm_cost(df[forecast_col], df['actual_mw'], df['time'])
    
    total_physics_penalty = df['physics_penalty_inr'].sum()
    total_ml_penalty = df['ml_penalty_inr'].sum()
    savings = total_physics_penalty - total_ml_penalty
    
    return {
        "physics_penalty": total_physics_penalty,
        "ml_penalty": total_ml_penalty,
        "savings": savings,
        "zone": zone
    }
