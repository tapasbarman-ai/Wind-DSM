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

# Load configuration
with open("config/wind_farms.yaml", "r") as file:
    config = yaml.safe_load(file)
parks_dict = {p['id']: p for p in config['parks']}

def run_pipeline_task(job_id: str, park_id: str, custom_scada_path: str = None, retrain: bool = False):
    """Background task to run the AI pipeline."""
    try:
        park = parks_dict[park_id]
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Fetching weather data..."

        # 1. RUN OPERATIONAL PIPELINE
        cerc_df = generate_live_forecast(park_id, park['lat'], park['lng'])
        if cerc_df is None:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Weather API Timeout. Please try again."
            return

        jobs[job_id]["message"] = "Generating 96-block forecast chart..."
        from src.visualization import plot_96_block_forecast
        plot_96_block_forecast(cerc_df, park_id=park_id)

        # 2. RUN BACKTEST
        jobs[job_id]["message"] = "Calculating financial savings..."
        results = train_and_backtest(park_id, park['capacity_mw'], custom_scada_path=custom_scada_path, retrain=retrain)

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
async def run_pretrained(park_id: str, background_tasks: BackgroundTasks):
    if park_id not in parks_dict:
        return {"error": f"Invalid park_id: {park_id}"}
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "message": "Queueing pipeline..."}
    background_tasks.add_task(run_pipeline_task, job_id, park_id, None, False)
    
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
    background_tasks.add_task(run_pipeline_task, job_id, park_id, custom_scada_path, True)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/v1/job-status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

@app.get("/api/v1/download-forecast/{park_id}")
async def download_forecast(park_id: str):
    if park_id not in parks_dict:
        return {"error": f"Invalid park_id: {park_id}"}
    park = parks_dict[park_id]
    cerc_df = generate_live_forecast(park_id, park['lat'], park['lng'])
    if cerc_df is None:
        return {"error": "Weather fetch failed."}
    csv_data = cerc_df.to_csv(index=False)
    return Response(
        content=csv_data, 
        media_type="text/csv", 
        headers={"Content-Disposition": f"attachment; filename={park_id}_forecast.csv"}
    )
