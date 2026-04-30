import os
import sys
import logging
from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task, task_group

# --- PROJECT IMPORTS ---
# Ensure Airflow can find the 'src' module
sys.path.insert(0, "/opt/airflow")
from src.data_ingestion import fetch_data
from src.spatial_processor import downscale_dataframe
from src.temporal_processor import interpolate_to_15min_pchip
from src.physics_engine import calculate_physics_power
from src.ml_pipeline import feature_engineering, predict_with_xgb

DATA_BASE = "/opt/airflow/data"

@dag(
    dag_id="wind_dsm_production_pipeline",
    start_date=datetime(2026, 4, 20),
    schedule_interval="0 6 * * *",  # Runs daily at 6 AM
    catchup=False,
    tags=["production", "wind", "dsm"],
    params={
        "lat": 15.34,
        "lon": 76.15,
        "forecast_hours": 24,
        "check_only": False
    }
)
def wind_dsm_production_pipeline():

    # =========================================================================
    #  STAGE 1: DATA INGESTION
    # =========================================================================
    @task_group(group_id="S1_Data_Ingestion")
    def stage_s1():
        
        @task(task_id="fetch_live_weather_data")
        def fetch_weather(**context):
            logger = logging.getLogger("airflow.task")
            if context["params"].get("check_only"):
                logger.info("check_only=True, skipping download.")
                return None
                
            lat = context["params"]["lat"]
            lon = context["params"]["lon"]
            forecast_hours = context["params"]["forecast_hours"]
            target_date = context["logical_date"].strftime("%Y-%m-%d")
            
            output_dir = f"{DATA_BASE}/01_raw/live_forecasts"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"{output_dir}/gfs_forecast_{target_date}.csv"
            
            logger.info(f"Fetching {forecast_hours}h GFS forecast for Lat: {lat}, Lon: {lon}")
            try:
                df = fetch_data(lat, lon, forecast_hours=forecast_hours)
                if len(df) == 0:
                    raise ValueError("Failed to fetch live data (empty dataframe).")
                
                df.to_csv(output_file, index=False)
                logger.info(f"Data fetched successfully -> {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise

        return fetch_weather()

    # =========================================================================
    #  STAGE 2: PHYSICS FOUNDATION
    # =========================================================================
    @task_group(group_id="S2_Physics_Foundation")
    def stage_s2(raw_file):

        @task(task_id="spatial_temporal_transformation")
        def transform(input_file):
            logger = logging.getLogger("airflow.task")
            if not input_file:
                logger.warning("No input file provided, skipping transformation.")
                return None
                
            logger.info(f"Transforming file: {input_file}")
            try:
                df = pd.read_csv(input_file)
                df['time'] = pd.to_datetime(df['time'])
                
                logger.info("Running Spatial Downscaling...")
                df = downscale_dataframe(df)
                logger.info("Running Temporal 15-Min Interpolation (96 blocks)...")
                df = interpolate_to_15min_pchip(df)
                
                output_file = input_file.replace("01_raw", "02_physics_baseline").replace("gfs_forecast", "transformed")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file, index=False)
                logger.info(f"Transformation complete -> {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"Transformation failed: {e}")
                raise

        @task(task_id="run_physics_baseline")
        def run_physics(input_file):
            logger = logging.getLogger("airflow.task")
            if not input_file:
                return None
                
            logger.info(f"Running Physics Baseline on: {input_file}")
            try:
                df = pd.read_csv(input_file)
                df['time'] = pd.to_datetime(df['time'])
                
                logger.info("Calculating Aerodynamic Power...")
                df['physics_mw'] = calculate_physics_power(df)
                
                output_file = input_file.replace("transformed", "physics_baseline")
                df.to_csv(output_file, index=False)
                logger.info(f"Physics Baseline complete -> {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"Physics engine failed: {e}")
                raise

        transformed = transform(raw_file)
        return run_physics(transformed)

    # =========================================================================
    #  STAGE 3: RESIDUAL LEARNING (XGBoost)
    # =========================================================================
    @task_group(group_id="S3_Residual_Learning")
    def stage_s3(physics_file):

        @task(task_id="xgboost_residual_correction")
        def run_ml(input_file):
            logger = logging.getLogger("airflow.task")
            if not input_file:
                return None
                
            logger.info(f"Running XGBoost correction on: {input_file}")
            try:
                df = pd.read_csv(input_file)
                df['time'] = pd.to_datetime(df['time'])
                
                logger.info("Engineering Features...")
                df = feature_engineering(df)
                logger.info("Predicting with XGBoost...")
                df['final_forecast_mw'] = predict_with_xgb(df)
                
                output_file = input_file.replace("02_physics_baseline", "03_ml_corrected").replace("physics_baseline", "final_forecast")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df.to_csv(output_file, index=False)
                
                logger.info(f"Final 96-block forecast saved to: {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"ML correction failed: {e}")
                raise
        @task(task_id="cerc_format_submission")
        def format_cerc(input_file):
            logger = logging.getLogger("airflow.task")
            if not input_file:
                return None
            
            try:
                logger.info("Formatting CERC submission...")
                df = pd.read_csv(input_file)
                df['time'] = pd.to_datetime(df['time'])
                
                if len(df) < 96:
                    raise ValueError(f"Insufficient blocks for CERC submission: Expected at least 96, got {len(df)}")
                
                cerc_df = pd.DataFrame()
                cerc_df['Block_No'] = range(1, 97)
                cerc_df['Date'] = df['time'].dt.date.iloc[:96].values
                cerc_df['Time_Block'] = df['time'].dt.time.iloc[:96].values
                cerc_df['Declared_Capacity_MW'] = df['final_forecast_mw'].round(2).iloc[:96].values
                
                output_file = input_file.replace("final_forecast", "cerc_submission")
                cerc_df.to_csv(output_file, index=False)
                logger.info(f"CERC Submission Ready -> {output_file}")
                return output_file
            except Exception as e:
                logger.error(f"CERC formatting failed: {e}")
                raise
                
        forecast_file = run_ml(physics_file)
        return format_cerc(forecast_file)

    # =========================================================================
    # --- EXECUTION FLOW ---
    # =========================================================================
    raw_data = stage_s1()
    physics_data = stage_s2(raw_data)
    final_forecast = stage_s3(physics_data)

# Instantiate the DAG
wind_dsm_dag = wind_dsm_production_pipeline()
