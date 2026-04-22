# Wind DSM Optimization - Data Repository

This directory contains the datasets used to train and validate the Physics-Informed Machine Learning (PI-PML) model, specifically focusing on the financial impact of Indian CERC Deviation Settlement Mechanism (DSM) regulations.

## The Benchmark Dataset: RSOPL Koppal 2025-2026

We successfully extracted and processed the official, 15-minute block-wise grid dispatch data for the ReNew Surya Ojas Private Limited (RSOPL) wind plant located in Koppal, Karnataka. This data represents the true "ground truth" for the forecasting pipeline.

The raw weekly ZIP archives were sourced directly from the Southern Regional Power Committee (SRPC) public commercial portal and merged chronologically into a single master dataset.

### Master File Details:
* **File Location:** `data/01_raw/rsopl_koppal_2025_merged.csv`
* **Total Time-Series Blocks:** 29,568 continuous 15-minute intervals.
* **Duration:** April 7, 2025, to February 8, 2026 (approx. 10 months).

### Key Features within the Dataset:
* `available_capacity`: The total installed plant capacity (75 MW).
* `sch_total`: The baseline scheduled generation (the traditional forecast).
* `act_total`: The actual measured wind plant output (real sensor reading).
* `dev`: The literal MW generation deviation (`act_total` - `sch_total`).
* `Underinjection_Charges` / `Overinjection_charges`: The actual Rupee (₹) penalties levied by the grid operator per 15-minute block.

### Financial Summary (The Business Pain-Point)
Over this ~10-month period, relying entirely on traditional forecasting models resulted in massive financial volatility under CERC DSM regulations:

* **Cumulative Under-injection Penaltiy:** ₹ -413,483,388 (approx $5M USD)
* **Cumulative Over-injection Charges/Gains:** ₹ +313,081,483
* **Net Loss Status:** Highly volatile cash flows heavily punishing generation under-performance.

### Project Objective:
The primary goal of the ML pipeline (`src/ml_pipeline.py`) is to ingest historical weather data for this exact period, apply mathematical aerodynamic modeling to establish a better physical baseline, and train an XGBoost algorithm to learn the site-specific noise and errors. 

Ultimately, this pipeline must reduce the `dev` metric in this dataset, directly reducing the **₹41 Crore ($5M)** under-injection penalty.

## PI-PML Optimization Results (Real-World Test)

The project leverages an 8-month/2-month chronological train/test split on this exact dataset. The ML model (`XGBoost`) is functionally restricted from predicting the absolute power itself. Instead, it maintains the physics-based schedules (`sch_total`) as the foundation and uses advanced rolling/lag features to predict only the **residual structural errors**. 

### Backtest Results (Final 2 Months):
1. **Statistical Accuracy:**
   * **Baseline Forecast MAE:** 5.96 MW
   * **ML Corrected Forecast (Base + Residual) MAE:** 1.71 MW
   * **Error Reduction:** **71.2% MAE Improvement**

2. **Direct Financial Validation (CERC DSM Simulator):**
   * **Original Baseline Penalty (2 Months):** ₹ 24,83,254
   * **ML Optimized Penalty (2 Months):** ₹ 1,29,282 
   * **Total Savings (2 Months Test):** **₹ 23,53,972 (approx. 23.5 Lakhs)**
   * **Projected Annual Savings for RSOPL:** **₹ 1,41,23,833 (~1.41 Crores/Year)**
