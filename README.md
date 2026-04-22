# Wind DSM Optimization

This project implements a pipeline to forecast wind power generation (physics-based + ML) and calculate the financial impact using Indian CERC DSM regulations.

## How to Run the Pipeline

The project is fully self-contained and uses a Python Virtual Environment (`.venv`) to protect your global system.

### 1. Open your Terminal (PowerShell)
Make sure you are in the project folder:
```powershell
cd C:\Users\tb619\Videos\Wind\Wind_DSM_Optimization
```

### 2. Run the Orchestrator
Execute the `main.py` file using the isolated Python executable inside the virtual environment:

```powershell
.venv\Scripts\python.exe main.py
```


### What happens when you run it?
1. **Backtesting:** It loads 10 years of Indian Khavda weather data.
2. **Physics:** It calculates the aerodynamic power output for a 3MW Vestas V112 turbine using `windpowerlib` physics.
3. **ML Training:** It trains an `XGBoost` regressor to learn the internal errors (synthetic SCADA noise and diurnal heat bias).
4. **DSM Rules:** It applies Indian CERC financial penalties to determine how much money the ML model saved the wind operator.
5. **Live API Hit:** It connects to the NOAA NOMADS GFS supercomputers and predicts the absolute latest 12-hour wind generation forecast starting *right now*.

## Training on Real-World Data (2025-2026)
We successfully integrated official block-wise grid dispatch data for the **75 MW RSOPL Koppal** wind plant from the Southern Regional Power Committee (SRPC), covering 29,568 blocks over 10 continuous months.

Instead of predicting the standalone real power output from scratch, the PI-PML model acts dynamically to predict the **structural residual error** (`act_total` - `sch_total`) of the physics schedules.

### Experimental Results (Out-of-Sample Last 2 Months Test)
*   **Original Baseline Penalty (2 Months):** ₹ 24,83,254
*   **ML Optimized Penalty (2 Months):** ₹ 1,29,282 
*   **Total Savings Evaluated:** **₹ 23,53,972 (approx. 23.5 Lakhs)**
*   **Projected Annual Rupee Savings for Plant Owner:** **₹ 1,41,23,833 (~1.41 Crores/Year)**
*   **Accuracy Improvement:** Absolute error reduced by **71.2%**
