import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import mlflow
import mlflow.xgboost

MODEL_DIR = "models/quantiles"

def generate_synthetic_scada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 'actual_mw' by adding noise and diurnal bias to 'physics_mw'.
    Incorporating 'Surface Albedo' heat effects for the Khavda Salt Flats.
    """
    np.random.seed(42)
    # Base inefficiency
    base_inefficiency = 0.05
    random_noise = np.random.normal(0, 0.02, len(df))
    
    # Thermal Lift / Surface Albedo Penalty
    # Reflective salt creates upward heat plumes causing models to overpredict power
    # Peak solar hours suffer higher efficiency loss due to unstable turbulence.
    hour_val = df['time'].dt.hour
    albedo_lift = 0.04 * np.sin(np.pi * (hour_val - 6) / 12) 
    albedo_lift = np.where(albedo_lift > 0, albedo_lift, 0)
    
    total_loss_factor = base_inefficiency + random_noise + albedo_lift
    
    df['actual_mw'] = df['physics_mw'] * (1 - total_loss_factor)
    df['actual_mw'] = df['actual_mw'].clip(lower=0)
    
    # AI precisely learns the Residual Error left over by the physics engine
    df['target_residual'] = df['actual_mw'] - df['physics_mw']
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    
    df = df.sort_values('time').reset_index(drop=True)
    df['wind_speed_roll3'] = df['wind_speed_100m'].rolling(window=3, min_periods=1).mean()
    df['wind_speed_roll6'] = df['wind_speed_100m'].rolling(window=6, min_periods=1).mean()
    
    # Add Ramp features (Wind Acceleration: Vt - Vt-1)
    df['wind_acceleration'] = df['wind_speed_100m'] - df['wind_speed_100m'].shift(1)
    df['wind_acceleration'] = df['wind_acceleration'].fillna(0)
    
    return df

def train_xgboost(df: pd.DataFrame):
    """
    Trains a Probabilistic Quantile Regressor.
    Predicts the 10th (Conservative), 50th (Median), and 90th (Aggressive) percentiles of power generation.
    """
    print("🧠 Training Quantile XGBoost Regressors (P10, P50, P90)...")
    features = ['wind_speed_100m', 'temperature_2m', 'surface_pressure', 
                'physics_mw', 'hour', 'month', 'wind_speed_roll3', 'wind_speed_roll6', 'wind_acceleration']
    
    if 'U' in df.columns and 'V' in df.columns:
        features.extend(['U', 'V'])
        
    df_train = df.dropna(subset=features + ['target_residual'])
    X = df_train[features]
    y = df_train['target_residual']
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    quantiles = {'q10': 0.1, 'q50': 0.5, 'q90': 0.9}
    
    # Initialize MLflow experiment
    mlflow.set_experiment("Wind_DSM_Quantile_XGBoost")
    
    with mlflow.start_run(run_name="Train_Quantiles"):
        mlflow.log_param("features", features)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        
        for name, alpha in quantiles.items():
            print(f"   -> Fitting {name} (Alpha={alpha})...")
            try:
                # Modern XGBoost supports reg:quantileerror natively
                model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                                         n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                model.fit(X, y)
                mlflow.log_param(f"{name}_alpha", alpha)
                mlflow.log_param(f"{name}_model_type", "quantile_native")
            except Exception as e:
                print("   -> Fallback applied (Standard Regressor due to library version limitations)")
                model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                model.fit(X, y)
                mlflow.log_param(f"{name}_alpha", alpha)
                mlflow.log_param(f"{name}_model_type", "standard_fallback")
                
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
            
            # Log the trained model to MLflow
            mlflow.xgboost.log_model(model, artifact_path=f"models/{name}")
        
    print("   -> All models logged successfully to MLflow tracking.")

def predict_with_xgb(df: pd.DataFrame) -> pd.Series:
    """
    Infers the probabilistic spread of future wind power generation.
    Returns P50 as standard, but makes P10 and P90 available for DSM financial strategy.
    """
    features = ['wind_speed_100m', 'temperature_2m', 'surface_pressure', 
                'physics_mw', 'hour', 'month', 'wind_speed_roll3', 'wind_speed_roll6', 'wind_acceleration']
    
    if 'U' in df.columns and 'V' in df.columns:
        features.extend(['U', 'V'])
        
    X = df[features]
    
    predictions = {}
    for name in ['q10', 'q50', 'q90']:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"Warning: {name} model missing. Using physics fallback.")
            predictions[name] = df['physics_mw']
            continue
            
        model = joblib.load(path)
        pred_residual = model.predict(X)
        predictions[name] = (df['physics_mw'] + pred_residual).clip(lower=0)
    
    df['final_forecast_mw_q10'] = predictions['q10']
    df['final_forecast_mw_q50'] = predictions['q50']
    df['final_forecast_mw_q90'] = predictions['q90']
    
    # We return the Median (P50) for standard tracking, but DSM can utilize q10 to bid safely.
    df['final_forecast_mw'] = predictions['q50']
    return predictions['q50']
