@echo off
:: Wind DSM Optimization Startup Script
:: This script starts the FastAPI backend and a local static file server for the frontend, then opens the dashboard.

setlocal enabledelayedexpansion

echo ===================================================
echo   Starting Wind DSM Optimization Services
echo ===================================================

:: Ensure we are running from the directory of this script
cd /d "%~dp0"

:: Check if the virtual environment exists
if not exist ".venv\Scripts\python.exe" goto no_venv

:: Start the FastAPI backend in a new command prompt window
echo [INFO] Starting FastAPI Backend on port 8000...
start "Wind DSM Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000"

:: Start the Frontend HTTP server in a new command prompt window
echo [INFO] Starting Frontend Web Server on port 8080...
start "Wind DSM Frontend" cmd /k ".venv\Scripts\python.exe -m http.server 8080"

:: Wait a brief moment (approx 3 seconds) using ping to allow the servers to initialize
ping -n 4 127.0.0.1 >nul

:: Open the frontend dashboard in the default browser
echo [INFO] Opening dashboard in browser...
start http://localhost:8080/frontend/

echo ===================================================
echo   All services have been started!
echo   Please keep the command prompt windows open to
echo   keep the application running.
echo ===================================================
pause
exit /b 0

:no_venv
echo [ERROR] Virtual environment (.venv) not found.
echo Please make sure you have run the environment setup.
pause
exit /b 1
