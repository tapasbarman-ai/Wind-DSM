import os
import pandas as pd
from src.data_ingestion import fetch_data
from src.spatial_processor import downscale_dataframe
from src.temporal_processor import interpolate_to_15min_pchip
from src.physics_engine import calculate_physics_power
from src.ml_pipeline import generate_synthetic_scada, feature_engineering, train_xgboost, predict_with_xgb
from src.dsm_calculator import calculate_penalties

import yaml

def apply_terrain_processing(df, park_id):
    from src.spatial_processor import TerrainProcessor
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
    farms_list = config.get('farms', config.get('parks', []))
    park_conf = next((p for p in farms_list if p['id'] == park_id), None)
    
    farm_altitude = park_conf.get('altitude_amsl', 100) if park_conf else 100
    hub_height = park_conf.get('hub_height', 100) if park_conf else 100
    alpha = park_conf.get('alpha', 0.14) if park_conf else 0.14
    
    processor = TerrainProcessor(farm_altitude, hub_height, alpha)
    
    if 'gfs_surface_hgt' not in df.columns:
        df['gfs_surface_hgt'] = farm_altitude
        
    df['raw_gfs_ws'] = df['wind_speed_100m']
    df['adjusted_ws'] = processor.adjust_wind_speed(df['wind_speed_100m'], df['gfs_surface_hgt'])
    df['elevation_diff'] = farm_altitude - df['gfs_surface_hgt']
    
    # Avoid div by zero
    df['height_correction_factor'] = df['adjusted_ws'] / df['raw_gfs_ws'].replace(0, 0.001)
    df['terrain_delta'] = df['adjusted_ws'] - df['raw_gfs_ws']
    
    df['wind_speed_100m'] = df['adjusted_ws']
    return df

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
    
    df = apply_terrain_processing(df, park_id)
    
    # 2. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # --- SAVE PHYSICS BASELINE ---
    os.makedirs(f"data/02_physics_baseline/{park_id}", exist_ok=True)
    df[['time', 'wind_speed_100m', 'temperature_2m', 'surface_pressure', 'physics_mw']].to_csv(f"data/02_physics_baseline/{park_id}/physics_output_historical_15min.csv", index=False)
    
    # 3. Synthetic SCADA Creation or Real SCADA Merging
    if custom_scada_path and os.path.exists(custom_scada_path):
        from src.ml_pipeline import merge_real_scada
        # Load capacity from config to scale SCADA correctly
        with open("config/wind_farms.yaml", "r") as file:
            config = yaml.safe_load(file)
        farms_list = config.get('farms', config.get('parks', []))
        park_conf = next((p for p in farms_list if p['id'] == park_id), None)
        capacity_mw = park_conf.get('capacity_mw', 100.0) if park_conf else 100.0
        
        df = merge_real_scada(df, custom_scada_path, capacity_mw=capacity_mw)
    else:
        df = generate_synthetic_scada(df)
    
    # 4. Feature Engineering
    df = feature_engineering(df)
    
    return df

def train_and_backtest(park_id, capacity_mw, custom_scada_path=None, retrain=False, run_live_forecast=False, model_type="pretrained"):
    import json
    
    cache_dir = f"data/03_ml_corrected/{park_id}"
    cache_path = os.path.join(cache_dir, f"backtest_results_{model_type}.json")
    # Backwards compatibility check
    if model_type == "pretrained" and not os.path.exists(cache_path):
        legacy_path = os.path.join(cache_dir, "backtest_results.json")
        if os.path.exists(legacy_path):
            cache_path = legacy_path
            
    plots_exist = (
        os.path.exists(f"outputs/plots/{park_id}/financial_savings_comparison_{model_type}.png") and
        os.path.exists(f"outputs/plots/{park_id}/forecast_vs_actual_timeseries_{model_type}.png")
    )
    
    if not retrain and os.path.exists(cache_path) and plots_exist:
        print(f"[CACHE] Loading pre-computed backtest results for {park_id} ({model_type}) from {cache_path}...")
        try:
            with open(cache_path, "r") as f:
                cached_results = json.load(f)
            
            # Optionally run live forecast preview if requested
            if run_live_forecast:
                print(f"\n--- Generating Live 96-Block Forecast Preview ({model_type}) ---")
                try:
                    with open("config/wind_farms.yaml", "r") as file:
                        config = yaml.safe_load(file)
                    farms_list = config.get('farms', config.get('parks', []))
                    park_conf = next(p for p in farms_list if p['id'] == park_id)
                    cerc_df = generate_live_forecast(park_id, park_conf['lat'], park_conf['lng'], model_type=model_type)
                    if cerc_df is not None:
                        from src.visualization import plot_96_block_forecast
                        plot_96_block_forecast(cerc_df, park_id=park_id, suffix=f"_{model_type}")
                except Exception as e:
                    print(f"Failed to generate live preview: {e}")
                    
            return cached_results
        except Exception as e:
            print(f"[CACHE] Failed to load cache for {park_id} ({e}). Recomputing...")
 
    print(f"Running full backtest for {park_id} ({model_type}) (retrain={retrain})...")
    df = prepare_historical_data(park_id, custom_scada_path)
    
    # Clean NaNs mapped by rolling window operations
    df = df.dropna().reset_index(drop=True)
    
    # Dynamic 80/20 chronological split (resolves fixed 10-year assumptions)
    n_records = len(df)
    split_idx = int(n_records * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    if retrain:
        print(f"Training data size: {len(df_train)} periods (15-min blocks)")
        print(f"Testing data size:  {len(df_test)} periods (15-min blocks)")
        
        # Train the PI-PML XGBoost models
        train_xgboost(df_train, park_id=park_id, model_type=model_type)
    
    # Backtest
    print("\n--- Running Financial Backtest ---")
    df_test['final_forecast_mw'] = predict_with_xgb(df_test, park_id=park_id, model_type=model_type)
    
    # Scale up turbine power to plant level before saving and calculations
    scale_factor = capacity_mw / 3.0
    df_test['actual_mw'] = df_test['actual_mw'] * scale_factor
    df_test['physics_mw'] = df_test['physics_mw'] * scale_factor
    df_test['final_forecast_mw'] = df_test['final_forecast_mw'] * scale_factor
    
    # --- SAVE ML CORRECTED OUTPUT ---
    os.makedirs(f"data/03_ml_corrected/{park_id}", exist_ok=True)
    df_test[['time', 'actual_mw', 'physics_mw', 'final_forecast_mw']].to_csv(f"data/03_ml_corrected/{park_id}/xgboost_output_2024_15min.csv", index=False)
    
    # --- LOAD CONFIG FOR PENALTY LOGIC ---
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
    farms_list = config.get('farms', config.get('parks', []))
    park_conf = next(p for p in farms_list if p['id'] == park_id)
    
    results = calculate_penalties(
        df_test, 
        park_id=park_id, 
        zone=park_conf.get('zone', 2), 
        capacity_mw=capacity_mw, 
        ppa_rate=park_conf.get('ppa_rate', 3.0)
    )
    
    print(f"Physics Model Penalty: INR {results['physics_penalty']:,.2f}")
    print(f"ML Enhanced Penalty:   INR {results['ml_penalty']:,.2f}")
    print(f"Total DSM Savings for {capacity_mw}MW Plant: INR {results['savings']:,.2f} over backtest")
    
    # Generate Charts and Reports
    from src.visualization import plot_financial_savings, plot_forecast_vs_actual, generate_reports, plot_96_block_forecast
    print("\n--- Generating Visualizations & Reports ---")
    plot_financial_savings(results, park_id=park_id, suffix=f"_{model_type}")
    plot_forecast_vs_actual(df_test, days=7, park_id=park_id, suffix=f"_{model_type}")
    generate_reports(df_test, results, park_id=park_id)
    
    # Save results to cache JSON
    try:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[CACHE] Saved backtest results for {park_id} to {cache_path}")
    except Exception as e:
        print(f"[CACHE ERROR] Failed to save cache for {park_id}: {e}")
        
    # --- ADDED: LIVE FORECAST PREVIEW ---
    if run_live_forecast:
        print("\n--- Generating Live 96-Block Forecast Preview ---")
        try:
            cerc_df = generate_live_forecast(park_id, park_conf['lat'], park_conf['lng'], model_type=model_type)
            if cerc_df is not None:
                plot_96_block_forecast(cerc_df, park_id=park_id, suffix=f"_{model_type}")
        except Exception as e:
            print(f"Failed to generate live preview: {e}")
    
    return results
    
def generate_live_forecast(park_id, lat, lon, model_type="pretrained"):
    print(f"\n--- Generating Live Forecast for Lat: {lat}, Lon: {lon} ({model_type}) ---")
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
    
    df = apply_terrain_processing(df, park_id)
        
    # 3. Physics Baseline
    df['physics_mw'] = calculate_physics_power(df)
    
    # 4. Features
    df = feature_engineering(df)
    
    # 5. ML Correction
    df['final_forecast_mw'] = predict_with_xgb(df, park_id=park_id, model_type=model_type)
    
    # Format exactly 96 blocks for CERC day-ahead
    cerc_forecast = df[['time', 'physics_mw', 'final_forecast_mw']].head(96)
    
    # Load configuration to get plant capacity
    try:
        with open("config/wind_farms.yaml", "r") as file:
            config = yaml.safe_load(file)
        farms_list = config.get('farms', config.get('parks', []))
        park_conf = next(p for p in farms_list if p['id'] == park_id)
        capacity_mw = park_conf.get('capacity_mw', 100.0)
    except Exception as e:
        print(f"Warning: Failed to load capacity for {park_id} from config: {e}. Defaulting to 100 MW.")
        capacity_mw = 100.0
        
    scale_factor = capacity_mw / 3.0
    cerc_forecast = cerc_forecast.copy()
    cerc_forecast['physics_mw'] = cerc_forecast['physics_mw'] * scale_factor
    cerc_forecast['final_forecast_mw'] = cerc_forecast['final_forecast_mw'] * scale_factor
    
    return cerc_forecast

if __name__ == "__main__":
    print("========================================")
    print("   PI-PML WIND DSM OPTIMIZATION PIPELINE")
    print("========================================\n")
    
    with open("config/wind_farms.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    farms_list = config.get('farms', config.get('parks', []))
    for park in farms_list:
        print(f"\n************************************************")
        print(f"Processing Wind Farm: {park['name']} ({park['capacity_mw']} MW)")
        print(f"************************************************")
        try:
            train_and_backtest(park_id=park['id'], capacity_mw=park['capacity_mw'], retrain=True)
            
            # The live forecast is now triggered on-demand via the FastAPI endpoint.
            # generate_live_forecast(park['id'], park['lat'], park['lng'])
        except Exception as e:
            print(f"Failed processing {park['name']}: {e}")
