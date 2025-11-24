@echo off
echo ===================================================
echo PSX PREDICTOR v12 - LAUNCHER
echo ===================================================

REM Set paths
set VENV_PYTHON="E:\Ai classes\projects\Claude PSX Project\.venv\Scripts\python.exe"

echo.
echo 1. Running Daily Update ^& Prediction (v12)...
echo.

%VENV_PYTHON% integrated_system_v12.py

echo.
echo 2. Launching Dashboard...
echo.

%VENV_PYTHON% -m streamlit run dashboard/app.py

pause
