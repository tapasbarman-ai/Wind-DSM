import os
import pandas as pd
import numpy as np

workspace_dir = "."

# Search for the LaTeX figures
figures = ["iiitl_logo.png", "double_engine_diagram.png", "log_profile.png", "penalty_chart.png"]
found_figs = {}

for root, dirs, files in os.walk(workspace_dir):
    for file in files:
        if file in figures:
            found_figs[file] = os.path.relpath(os.path.join(root, file), workspace_dir)

print("--- Figure Search Results ---")
for fig in figures:
    if fig in found_figs:
        print(f"Found: {fig} at {found_figs[fig]}")
    else:
        print(f"NOT Found: {fig}")

# Load test results to print summary stats for validation
test_results_path = "data/03_ml_corrected/real_data_test_results.csv"
if os.path.exists(test_results_path):
    print("\n--- Summary of real_data_test_results.csv ---")
    df = pd.read_csv(test_results_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    base_mae = np.mean(np.abs(df['act_total'] - df['sch_total']))
    ml_mae = np.mean(np.abs(df['act_total'] - df['ml_forecast']))
    mae_reduction = (base_mae - ml_mae) / base_mae * 100
    
    base_penalty = df['base_penalty'].sum()
    ml_penalty = df['ml_penalty'].sum()
    savings = base_penalty - ml_penalty
    ann_savings = savings * 6
    
    print(f"Baseline MAE: {base_mae:.4f} MW")
    print(f"ML MAE:       {ml_mae:.4f} MW")
    print(f"MAE Reduction: {mae_reduction:.2f}%")
    print(f"Baseline Penalty: ₹ {base_penalty:,.2f}")
    print(f"ML Penalty:       ₹ {ml_penalty:,.2f}")
    print(f"Net Savings (2m): ₹ {savings:,.2f}")
    print(f"Projected Annual: ₹ {ann_savings:,.2f}")
else:
    print(f"\n{test_results_path} does not exist.")
