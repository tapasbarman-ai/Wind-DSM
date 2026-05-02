import os
import pandas as pd
from src.data_ingestion import fetch_data
from src.spatial_processor import downscale_dataframe
from src.temporal_processor import interpolate_to_15min_pchip
from src.physics_engine import calculate_physics_power
from src.ml_pipeline import generate_synthetic_scada, feature_engineering, train_xgboost, predict_with_xgb
from src.dsm_calculator import calculate_penalties

import yaml

def prepare_historical_data(park_id, custom_scada_path=None):
    historical_path = f"data/01_raw/{park_id}_10yr_historical.csv"
    print(f"Loading historical truth data from {historical_path}...")
    if not os.path.exists(historical_path):
        raise FileNotFoundError(f"Please run src/historical_data_ingestion.py first to download {historical_path}")
        
    df = pd.read_csv(historical_path)
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. Micro-scale Spatial & Temporal Transformation
    df = downscale_dataframe(df)
    df = interpolate_to_15min_pchip(df)
    
    # 2. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # --- SAVE PHYSICS BASELINE ---
    os.makedirs(f"data/02_physics_baseline/{park_id}", exist_ok=True)
    df[['time', 'wind_speed_100m', 'temperature_2m', 'surface_pressure', 'physics_mw']].to_csv(f"data/02_physics_baseline/{park_id}/physics_output_historical_15min.csv", index=False)
    
    # 3. Synthetic SCADA Creation or Real SCADA Merging
    if custom_scada_path and os.path.exists(custom_scada_path):
        from src.ml_pipeline import merge_real_scada
        df = merge_real_scada(df, custom_scada_path)
    else:
        df = generate_synthetic_scada(df)
    
    # 4. Feature Engineering
    df = feature_engineering(df)
    
    return df

def train_and_backtest(park_id, capacity_mw, custom_scada_path=None, retrain=False):
    df = prepare_historical_data(park_id, custom_scada_path)
    
    # Clean NaNs mapped by rolling window operations
    df = df.dropna().reset_index(drop=True)
    
    # Split: Train on 2015-2023, Test (Backtest) on 2024
    df_train = df[df['time'].dt.year < 2024].copy()
    df_test = df[df['time'].dt.year == 2024].copy()
    
    if retrain:
        print(f"Training data size: {len(df_train)} periods (15-min blocks)")
        print(f"Testing data size:  {len(df_test)} periods (15-min blocks)")
        
        # Train the PI-PML XGBoost models
        train_xgboost(df_train, park_id=park_id)
    
    # Backtest
    print("\n--- Running Financial Backtest on 2024 Data ---")
    df_test['final_forecast_mw'] = predict_with_xgb(df_test, park_id=park_id)
    
    # --- SAVE ML CORRECTED OUTPUT ---
    os.makedirs(f"data/03_ml_corrected/{park_id}", exist_ok=True)
    df_test[['time', 'actual_mw', 'physics_mw', 'final_forecast_mw']].to_csv(f"data/03_ml_corrected/{park_id}/xgboost_output_2024_15min.csv", index=False)
    
    # --- LOAD CONFIG FOR PENALTY LOGIC ---
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
    park_conf = next(p for p in config['parks'] if p['id'] == park_id)
    
    results = calculate_penalties(
        df_test, 
        park_id=park_id, 
        zone=park_conf.get('zone', 2), 
        capacity_mw=capacity_mw, 
        ppa_rate=park_conf.get('ppa_rate', 3.0)
    )
    
    print(f"Physics Model Penalty: INR {results['physics_penalty']:,.2f}")
    print(f"ML Enhanced Penalty:   INR {results['ml_penalty']:,.2f}")
    print(f"Total DSM Savings:     INR {results['savings']:,.2f} over 1 year (For 1 Turbine!)")
    print(f"Total Estimated Savings for {capacity_mw}MW Plant: INR {(results['savings'] * capacity_mw):,.2f}")
    
    # Generate Charts and Reports
    from src.visualization import plot_financial_savings, plot_forecast_vs_actual, generate_reports, plot_96_block_forecast
    print("\n--- Generating Visualizations & Reports ---")
    plot_financial_savings(results, park_id=park_id)
    plot_forecast_vs_actual(df_test, days=7, park_id=park_id)
    generate_reports(df_test, results, park_id=park_id)
    
    # --- ADDED: LIVE FORECAST PREVIEW ---
    print("\n--- Generating Live 96-Block Forecast Preview ---")
    try:
        # We need the coordinates from the config for the live fetch
        with open("config/wind_farms.yaml", "r") as file:
            config = yaml.safe_load(file)
        park_conf = next(p for p in config['parks'] if p['id'] == park_id)
        
        cerc_df = generate_live_forecast(park_id, park_conf['lat'], park_conf['lng'])
        if cerc_df is not None:
            plot_96_block_forecast(cerc_df, park_id=park_id)
    except Exception as e:
        print(f"Failed to generate live preview: {e}")
    
    return results
    
def generate_live_forecast(park_id, lat, lon):
    print(f"\n--- Generating Live Forecast for Lat: {lat}, Lon: {lon} ---")
    # 1. Get Live Data from NOAA NOMADS GFS Server
    try:
        df = fetch_data(lat, lon, forecast_hours=24, park_id=park_id)
        if len(df) == 0:
            print("Failed to fetch live data (empty dataframe).")
            return None
    except Exception as e:
        print(f"Failed to fetch live API data: {e}")
        return None
        
    # 2. Micro-scale Spatial & Temporal Transformation
    df = downscale_dataframe(df)
    df = interpolate_to_15min_pchip(df)
        
    # 3. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # 4. Features
    df = feature_engineering(df)
    
    # 5. ML Correction
    df['final_forecast_mw'] = predict_with_xgb(df, park_id=park_id)
    
    # Format exactly 96 blocks for CERC day-ahead
    cerc_forecast = df[['time', 'physics_mw', 'final_forecast_mw']].head(96)
    
    return cerc_forecast

if __name__ == "__main__":
    print("========================================")
    print("   PI-PML WIND DSM OPTIMIZATION PIPELINE")
    print("========================================\n")
    
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    for park in config['parks']:
        print(f"\n************************************************")
        print(f"Processing Wind Farm: {park['name']} ({park['capacity_mw']} MW)")
        print(f"************************************************")
        try:
            train_and_backtest(park_id=park['id'], capacity_mw=park['capacity_mw'], retrain=True)
            
            # The live forecast is now triggered on-demand via the FastAPI endpoint.
            # generate_live_forecast(park['id'], park['lat'], park['lng'])
        except Exception as e:
            print(f"Failed processing {park['name']}: {e}")
