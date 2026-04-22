import os
import pandas as pd
from src.data_ingestion import fetch_data
from src.spatial_processor import downscale_dataframe
from src.temporal_processor import interpolate_to_15min_pchip
from src.physics_engine import calculate_physics_power
from src.ml_pipeline import generate_synthetic_scada, feature_engineering, train_xgboost, predict_with_xgb
from src.dsm_calculator import calculate_penalties

HISTORICAL_DATA_PATH = "data/01_raw/khavda_10yr_historical.csv"

def prepare_historical_data():
    print(f"Loading historical truth data from {HISTORICAL_DATA_PATH}...")
    if not os.path.exists(HISTORICAL_DATA_PATH):
        raise FileNotFoundError(f"Please run src/historical_data_ingestion.py first to download {HISTORICAL_DATA_PATH}")
        
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. Micro-scale Spatial & Temporal Transformation
    df = downscale_dataframe(df)
    df = interpolate_to_15min_pchip(df)
    
    # 2. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # --- SAVE PHYSICS BASELINE ---
    os.makedirs("data/02_physics_baseline", exist_ok=True)
    df[['time', 'wind_speed_100m', 'temperature_2m', 'surface_pressure', 'physics_mw']].to_csv("data/02_physics_baseline/physics_output_historical_15min.csv", index=False)
    
    # 3. Synthetic SCADA Creation
    df = generate_synthetic_scada(df)
    
    # 4. Feature Engineering
    df = feature_engineering(df)
    
    return df

def train_and_backtest():
    df = prepare_historical_data()
    
    # Clean NaNs mapped by rolling window operations
    df = df.dropna().reset_index(drop=True)
    
    # Split: Train on 2015-2023, Test (Backtest) on 2024
    df_train = df[df['time'].dt.year < 2024].copy()
    df_test = df[df['time'].dt.year == 2024].copy()
    
    print(f"Training data size: {len(df_train)} periods (15-min blocks)")
    print(f"Testing data size:  {len(df_test)} periods (15-min blocks)")
    
    # Train the PI-PML XGBoost models
    train_xgboost(df_train)
    
    # Backtest
    print("\n--- Running Financial Backtest on 2024 Data ---")
    df_test['final_forecast_mw'] = predict_with_xgb(df_test)
    
    # --- SAVE ML CORRECTED OUTPUT ---
    os.makedirs("data/03_ml_corrected", exist_ok=True)
    df_test[['time', 'actual_mw', 'physics_mw', 'final_forecast_mw']].to_csv("data/03_ml_corrected/xgboost_output_2024_15min.csv", index=False)
    
    results = calculate_penalties(df_test)
    
    print(f"Physics Model Penalty: ₹ {results['physics_penalty']:,.2f}")
    print(f"ML Enhanced Penalty:   ₹ {results['ml_penalty']:,.2f}")
    print(f"Total DSM Savings:     ₹ {results['savings']:,.2f} over 1 year (For 1 Turbine!)")
    
    # Generate Charts and Reports
    from src.visualization import plot_financial_savings, plot_forecast_vs_actual, generate_reports
    print("\n--- Generating Visualizations & Reports ---")
    plot_financial_savings(results)
    plot_forecast_vs_actual(df_test, days=7)
    generate_reports(df_test, results)
    
def generate_live_forecast(lat, lon):
    print(f"\n--- Generating Live Forecast for Lat: {lat}, Lon: {lon} ---")
    # 1. Get Live Data from NOAA NOMADS GFS Server
    try:
        df = fetch_data(lat, lon, forecast_hours=12)
        if len(df) == 0:
            print("Failed to fetch live data (empty dataframe).")
            return
    except Exception as e:
        print(f"Failed to fetch live API data: {e}")
        return
        
    # 2. Micro-scale Spatial & Temporal Transformation
    df = downscale_dataframe(df)
    df = interpolate_to_15min_pchip(df)
        
    # 3. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # 4. Features
    df = feature_engineering(df)
    
    # 5. ML Correction
    df['final_forecast_mw'] = predict_with_xgb(df)
    
    print("\nReal-time Operational Forecast (Next 5 Hours / 20 Blocks):")
    cols = ['time', 'wind_speed_100m', 'physics_mw', 'final_forecast_mw']
    if 'final_forecast_mw_q10' in df.columns:
        cols.extend(['final_forecast_mw_q10', 'final_forecast_mw_q90'])
    print(df[cols].head(20))

if __name__ == "__main__":
    print("========================================")
    print("   PI-PML WIND DSM OPTIMIZATION PIPELINE")
    print("========================================\n")
    
    # 1. Train model and evaluate financial impact on historical truth
    train_and_backtest()
    
    # 2. Execute a live forecast hitting NOAA servers for the next 12 hours
    generate_live_forecast(23.82, 69.72)
