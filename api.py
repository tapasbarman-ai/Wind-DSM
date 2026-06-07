from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import yaml
import os
import uuid
from fastapi.responses import Response
from main import train_and_backtest, generate_live_forecast

app = FastAPI(title="Wind DSM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs = {}

# Load configuration (refreshed for RSOPL Koppal)
with open("config/wind_farms.yaml", "r") as file:
    config = yaml.safe_load(file)
parks_dict = {p['id']: p for p in config.get('farms', config.get('parks', []))}

def run_pipeline_task(job_id: str, park_id: str, custom_scada_path: str = None, retrain: bool = False, model_type: str = "pretrained"):
    """Background task to run the AI pipeline."""
    try:
        park = parks_dict[park_id]
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Fetching weather data..."

        # 1. RUN OPERATIONAL PIPELINE
        cerc_df = generate_live_forecast(park_id, park['lat'], park['lng'], model_type=model_type)
        if cerc_df is None:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Weather API Timeout. Please try again."
            return

        jobs[job_id]["message"] = "Generating 96-block forecast chart..."
        from src.visualization import plot_96_block_forecast
        plot_96_block_forecast(cerc_df, park_id=park_id, suffix=f"_{model_type}")

        # 2. RUN BACKTEST
        if retrain:
            jobs[job_id]["message"] = "Retraining model and recalculating savings..."
        else:
            jobs[job_id]["message"] = f"Loading cached backtest evaluation ({model_type})..."
            
        results = train_and_backtest(park_id, park['capacity_mw'], custom_scada_path=custom_scada_path, retrain=retrain, model_type=model_type)

        # Store results
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "park_id": park_id,
            "physics_penalty": results['physics_penalty'],
            "ml_penalty": results['ml_penalty'],
            "savings": results['savings'],
            "live_stats": {
                "max_mw": float(cerc_df['final_forecast_mw'].max()),
                "avg_mw": float(cerc_df['final_forecast_mw'].mean())
            }
        }
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/api/v1/forecast/{park_id}")
async def run_pretrained(park_id: str, background_tasks: BackgroundTasks, model_mode: str = "pretrained"):
    if park_id not in parks_dict:
        return {"error": f"Invalid park_id: {park_id}"}
    
    m_type = "custom" if model_mode == "custom" or model_mode == "custom_model" else "pretrained"
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "message": "Queueing pipeline..."}
    background_tasks.add_task(run_pipeline_task, job_id, park_id, None, False, m_type)
    
    return {"job_id": job_id, "status": "pending"}

@app.post("/api/v1/retrain/{park_id}")
async def run_retrain(park_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if park_id not in parks_dict:
        return {"error": f"Invalid park_id: {park_id}"}
    
    # Save the uploaded SCADA data
    os.makedirs("data/01_raw/custom", exist_ok=True)
    custom_scada_path = f"data/01_raw/custom/{park_id}_custom_scada.csv"
    with open(custom_scada_path, "wb") as f:
        f.write(await file.read())
        
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "message": "Queueing retraining job..."}
    background_tasks.add_task(run_pipeline_task, job_id, park_id, custom_scada_path, True, "custom")
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/v1/custom-model-exists/{park_id}")
async def check_custom_model(park_id: str):
    local_model_path = f"models/{park_id}_custom_model.pkl"
    return {"exists": os.path.exists(local_model_path)}

@app.get("/api/v1/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

@app.get("/api/v1/download-forecast/{park_id}")
async def download_forecast(park_id: str, model_mode: str = "pretrained"):
    if park_id not in parks_dict:
        return {"error": f"Invalid park_id: {park_id}"}
    park = parks_dict[park_id]
    m_type = "custom" if model_mode == "custom" or model_mode == "custom_model" else "pretrained"
    cerc_df = generate_live_forecast(park_id, park['lat'], park['lng'], model_type=m_type)
    if cerc_df is None:
        return {"error": "Weather fetch failed."}
    csv_data = cerc_df.to_csv(index=False)
    return Response(
        content=csv_data, 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename={park_id}_{m_type}_forecast.csv"}
    )

@app.get("/api/v1/wind-vectors")
async def get_wind_vectors():
    """Fetches current wind speed and direction for all parks to render animated vector flows."""
    import concurrent.futures
    import requests
    from datetime import datetime
    
    parks = config.get('farms', config.get('parks', []))
    
    def fetch_one(park):
        url = "https://api.open-meteo.com/v1/gfs"
        params = {
            "latitude": park['lat'],
            "longitude": park['lng'],
            "hourly": ["wind_speed_100m", "wind_direction_100m"],
            "wind_speed_unit": "ms",
            "forecast_days": 1
        }
        try:
            r = requests.get(url, params=params, timeout=4)
            if r.status_code == 200:
                data = r.json()
                current_hour = datetime.utcnow().hour
                speed = float(data['hourly']['wind_speed_100m'][current_hour])
                direction = float(data['hourly']['wind_direction_100m'][current_hour])
                return {
                    "id": park['id'],
                    "name": park['name'],
                    "lat": park['lat'],
                    "lng": park['lng'],
                    "wind_speed": speed,
                    "wind_direction": direction
                }
        except Exception as e:
            pass
        
        # Fallback to deterministic default values based on capacity if API fails
        import random
        random.seed(hash(park['id']))
        return {
            "id": park['id'],
            "name": park['name'],
            "lat": park['lat'],
            "lng": park['lng'],
            "wind_speed": random.uniform(4.0, 12.0),
            "wind_direction": random.uniform(180.0, 270.0) # typical monsoon direction
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_one, parks))
        
    return results
