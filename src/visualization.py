import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend — prevents Tkinter thread crashes on Windows
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a nice theme
sns.set_theme(style="whitegrid")

def plot_financial_savings(results: dict, park_id: str = "default", save_dir: str = "outputs/plots"):
    """
    Plots a bar chart comparing the pure Physics penalties vs Physics + ML Penalties.
    """
    save_dir = os.path.join(save_dir, park_id)
    os.makedirs(save_dir, exist_ok=True)
    
    labels = ['Pure Physics Baseline', 'Machine Learning Enhanced']
    penalties = [results['physics_penalty'], results['ml_penalty']]
    
    plt.figure(figsize=(8, 6))
    
    # Bar plot with distinct colors
    colors = ['#e74c3c', '#2ecc71'] # Red for high penalty, Green for low/zero penalty
    bars = plt.bar(labels, penalties, color=colors)
    
    plt.title(f'Indian CERC DSM Penalties ({park_id.upper()})\n(1-Year Backtest)', fontsize=14, pad=20)
    plt.ylabel('Penalty Paid (INR ₹)', fontsize=12)
    
    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(penalties)*0.02), 
                 f'₹ {yval:,.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'financial_savings_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved financial comparison chart to {save_path}")

def plot_forecast_vs_actual(df: pd.DataFrame, days: int = 7, park_id: str = "default", save_dir: str = "outputs/plots"):
    """
    Plots a time-series line chart comparing Actual, Physics, and ML forecasts 
    for a specific slice of the test data.
    """
    save_dir = os.path.join(save_dir, park_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Take a 1-week slice of the data so the chart isn't overcrowded
    # Let's take a week in the summer (e.g. June 2024 for high thermal dynamics)
    target_month = 6
    df_slice = df[df['time'].dt.month == target_month].head(days * 24).copy()
    
    # If the slice is empty, just take the first `days` days available
    if len(df_slice) == 0:
        df_slice = df.head(days * 24).copy()
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(df_slice['time'], df_slice['actual_mw'], label='Actual (Noisy/SCADA)', color='black', linewidth=2)
    plt.plot(df_slice['time'], df_slice['physics_mw'], label='Pure Physics Forecast', color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.plot(df_slice['time'], df_slice['final_forecast_mw'], label='ML Enhanced Forecast', color='#3498db', linestyle='-', linewidth=2, alpha=0.9)
    
    plt.title(f'Wind Generation Forecast vs Actual ({days}-Day Slice) - {park_id.upper()}', fontsize=16)
    plt.xlabel('Date / Time', fontsize=12)
    plt.ylabel('Power Output (MW)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11)
    
    # Clean up x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'forecast_vs_actual_timeseries.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved timeseries chart to {save_path}")

def generate_reports(df_test: pd.DataFrame, results: dict, park_id: str = "default", save_dir: str = "data/04_financials"):
    """
    Saves the absolute results to CSV and a summary text file.
    """
    save_dir = os.path.join(save_dir, park_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save raw test output comparing predictions to actuals
    csv_path = os.path.join(save_dir, "dsm_backtest_results_2024.csv")
    cols = ['time', 'wind_speed_100m', 'temperature_2m', 'actual_mw', 'physics_mw', 'final_forecast_mw', 'physics_penalty_inr', 'ml_penalty_inr']
    df_test[cols].to_csv(csv_path, index=False)
    print(f"-> Saved raw backtest results to {csv_path}")
    
    # Save a high-level summary text doc
    txt_path = os.path.join(save_dir, "executive_summary.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write("      DSM PENALTY OPTIMIZATION SUMMARY (1 YEAR)     \n")
        f.write("====================================================\n\n")
        f.write("MODEL EVALUATION:\n")
        f.write(f"Total Theoretical Physics Penalty: INR ₹ {results['physics_penalty']:,.2f}\n")
        f.write(f"Total ML Enhanced Penalty:         INR ₹ {results['ml_penalty']:,.2f}\n")
        f.write("----------------------------------------------------\n")
        f.write(f"TOTAL MONEY SAVED BY ML:           INR ₹ {results['savings']:,.2f}\n")
        f.write("====================================================\n")
        
    print(f"-> Saved executive summary to {txt_path}")

def plot_96_block_forecast(df_96: pd.DataFrame, park_id: str = "default", save_dir: str = "outputs/plots"):
    """
    Plots the specific 96-block (24-hour) day-ahead schedule.
    """
    save_dir = os.path.join(save_dir, park_id)
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot both physics and ML forecasts for comparison
    plt.fill_between(df_96['time'], df_96['physics_mw'], label='Physics Base', color='#e74c3c', alpha=0.1)
    plt.plot(df_96['time'], df_96['physics_mw'], color='#e74c3c', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.plot(df_96['time'], df_96['final_forecast_mw'], label='ML Optimized Schedule (CERC)', color='#2ecc71', linewidth=3)
    
    plt.title(f'96-Block Day-Ahead Power Schedule - {park_id.upper()}\n(Operational Forecast for Tomorrow)', fontsize=15)
    plt.xlabel('Time (15-min blocks)', fontsize=12)
    plt.ylabel('Planned Power Output (MW)', fontsize=12)
    plt.legend(loc='upper right')
    
    # Formatting
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'live_96_block_forecast.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved live 96-block forecast chart to {save_path}")
