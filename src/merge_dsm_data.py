import os
import pandas as pd

base_dir = r"C:\Users\tb619\Videos\Wind\Wind_DSM_Optimization\data\01_raw\dsm07apr25-08feb26r\07.04.25 to 08.02.26 -Data files"
output_file = r"C:\Users\tb619\Videos\Wind\Wind_DSM_Optimization\data\01_raw\rsopl_koppal_2025_merged.csv"

all_data = []

print("Scanning directories for RSOPL Koppal data...")

# Iterate through all subdirectories
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "commercial_dev2022_rsopl_koppal.csv":
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

if all_data:
    print(f"Found {len(all_data)} weekly files. Merging now...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Clean up the dataframe by dropping NA dates
    merged_df = merged_df.dropna(subset=['date', 'time'])
    
    # Create a single properly typed datetime column
    merged_df['datetime'] = pd.to_datetime(merged_df['date'] + ' ' + merged_df['time'], errors='coerce')
    
    # Sort chronologically so sequential modeling (like XGBoost) works perfectly
    merged_df = merged_df.sort_values(by='datetime').reset_index(drop=True)
    
    # Drop trailing empty columns often introduced by trailing commas in Indian Gov CSVs
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('^Unnamed')]
    
    # Save the master dataset
    merged_df.to_csv(output_file, index=False)
    
    print("\n=== Merging Complete ===")
    print(f"Successfully merged {len(all_data)} weekly files.")
    print(f"Total 15-minute blocks tracked: {len(merged_df):,}")
    print(f"Continuous Date Range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Master Dataset saved to: {output_file}")
    
    print("\n--- Master Dataset Summary ---")
    print(f"Average Scheduled (MW): {merged_df['sch_total'].mean():.2f}")
    print(f"Average Actual (MW): {merged_df['act_total'].mean():.2f}")
    print(f"Total Under-injection Charges: Rs. {merged_df['Underinjection_Charges'].sum():,.2f}")
    print(f"Total Over-injection Charges: Rs. {merged_df['Overinjection_charges'].sum():,.2f}")
else:
    print("Failed: No relevant CSV files found to merge.")
