import sys
sys.stdout = open('train_output.txt', 'w')
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

def feature_engineering(df):
    df = df.copy()
    # Ensure chronological order
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Define Target: The ML model must predict the RESIDUAL ERROR, not the absolute MW
    df['target_residual'] = df['act_total'] - df['sch_total']
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    
    # Lag features for the Residual Error (How wrong was the schedule recently?)
    df['res_lag1'] = df['target_residual'].shift(1)
    df['res_lag2'] = df['target_residual'].shift(2)
    df['res_lag4'] = df['target_residual'].shift(4) # 1 hour ago
    
    # Rolling features of the residual
    df['res_roll_mean_4'] = df['target_residual'].rolling(window=4).mean().shift(1)
    
    # Base schedule variations
    df['sch_lag1'] = df['sch_total'].shift(1)
    df['sch_change'] = df['sch_total'] - df['sch_lag1']
    
    # Drop NaNs caused by lagging
    return df.dropna().reset_index(drop=True)

def get_acp_multiplier(time_series: pd.Series) -> pd.Series:
    hours = time_series.dt.hour
    conditions = [
        (hours >= 18) & (hours <= 22),
        (hours >= 6) & (hours <= 9),
        (hours >= 0) & (hours <= 5)
    ]
    choices = [2.5, 1.5, 0.8]
    multipliers = np.select(conditions, choices, default=1.0)
    return pd.Series(multipliers, index=time_series.index)

def calculate_dsm_cost(forecast, actual, time_series, capacity=75.0):
    abs_error_mw = np.abs(forecast - actual)
    error_pct = (abs_error_mw / capacity) * 100
    
    time_diffs = time_series.diff().dt.total_seconds().fillna(900) / 3600.0
    block_duration_hrs = np.where(time_diffs > 0, time_diffs, 0.25)
    deviation_mwh = abs_error_mw * block_duration_hrs
    
    acp_multipliers = get_acp_multiplier(time_series)
    dynamic_price_inr = 3000 * acp_multipliers  # Assume Rs 3000 PPA
    
    penalties = np.zeros(len(error_pct))
    
    mask_10_20 = (error_pct > 10) & (error_pct <= 20)
    penalties[mask_10_20] = deviation_mwh[mask_10_20] * (dynamic_price_inr[mask_10_20] * 0.10)
    
    mask_20_30 = (error_pct > 20) & (error_pct <= 30)
    penalties[mask_20_30] = deviation_mwh[mask_20_30] * (dynamic_price_inr[mask_20_30] * 0.20)
    
    mask_30_plus = error_pct > 30
    penalties[mask_30_plus] = deviation_mwh[mask_30_plus] * (dynamic_price_inr[mask_30_plus] * 0.30)
    
    return pd.Series(penalties, index=actual.index)

if __name__ == "__main__":
    file_path = r"C:\Users\tb619\Videos\Wind\Wind_DSM_Optimization\data\01_raw\rsopl_koppal_2025_merged.csv"
    print(f"Loading true DSM data from {file_path}...")
    df = pd.read_csv(file_path)
    
    df = feature_engineering(df)
    
    # 8 Month / 2 Month Split
    cutoff_date = df['datetime'].min() + pd.DateOffset(months=8)
    
    df_train = df[df['datetime'] <= cutoff_date].copy()
    df_test = df[df['datetime'] > cutoff_date].copy()
    
    print(f"Training data size (8 Months): {len(df_train)} blocks")
    print(f"Testing data size (2 Months):  {len(df_test)} blocks")
    
    # Predict the RESIDUAL, not the total MW
    features = ['sch_total', 'sch_lag1', 'sch_change', 'hour', 'minute', 'month', 'dayofweek',
                'res_lag1', 'res_lag2', 'res_lag4', 'res_roll_mean_4']
    target = 'target_residual'
    
    X_train = df_train[features]
    y_train = df_train[target]
    
    X_test = df_test[features]
    y_test = df_test[target]
    
    print("\nTraining XGBoost Regressor to predict SCHEDULE ERRORS (Residuals)...")
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # ML corrects the baseline schedule by adding the predicted residual
    df_test['pred_residual'] = model.predict(X_test)
    df_test['ml_forecast'] = df_test['sch_total'] + df_test['pred_residual']
    
    # Physical constraints: Generation cannot be negative or above 75MW
    df_test['ml_forecast'] = df_test['ml_forecast'].clip(lower=0, upper=75.0)
    
    print("\n--- Model Evaluation (Test Set: Last 2 Months) ---")
    base_mae = mean_absolute_error(df_test['act_total'], df_test['sch_total'])
    ml_mae = mean_absolute_error(df_test['act_total'], df_test['ml_forecast'])
    print(f"Baseline Forecast MAE: {base_mae:.2f} MW")
    print(f"ML Corrected Forecast (Base + Residual) MAE: {ml_mae:.2f} MW")
    err_reduction = ((base_mae - ml_mae)/base_mae) * 100
    print(f"Error Reduction: {err_reduction:.1f}%")
    
    print("\n--- Financial Savings Validation (CERC DSM Simulator) ---")
    
    # Calculate synthetic penalties for both utilizing the true dynamic price simulator
    df_test['base_penalty'] = calculate_dsm_cost(df_test['sch_total'], df_test['act_total'], df_test['datetime'], capacity=75.0)
    df_test['ml_penalty'] = calculate_dsm_cost(df_test['ml_forecast'], df_test['act_total'], df_test['datetime'], capacity=75.0)
    
    base_penalty_sum = df_test['base_penalty'].sum()
    ml_penalty_sum = df_test['ml_penalty'].sum()
    savings = base_penalty_sum - ml_penalty_sum
    
    print(f"Original Baseline Penalty (Simulated): Rs. {base_penalty_sum:,.2f}")
    print(f"ML Optimized Penalty (Simulated):    Rs. {ml_penalty_sum:,.2f}")
    print(f"Total Money Saved in 2 Months:       Rs. {savings:,.2f}")
    print(f"Projected Annual Savings:            Rs. {savings * 6:,.2f}")
    
    # Saving outputs to verify
    os.makedirs("data/03_ml_corrected", exist_ok=True)
    out_path = "data/03_ml_corrected/real_data_test_results.csv"
    df_test[['datetime', 'sch_total', 'pred_residual', 'ml_forecast', 'act_total', 
             'base_penalty', 'ml_penalty']].to_csv(out_path, index=False)
    print(f"\nSaved test results with ML residual predictions to {out_path}")
