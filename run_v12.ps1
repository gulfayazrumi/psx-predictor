# PSX Predictor v12 Launcher
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "PSX PREDICTOR v12 - LAUNCHER" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Running Daily Update & Prediction (v12)..." -ForegroundColor Yellow
Write-Host ""

python integrated_system_v12.py

Write-Host ""
Write-Host "2. Launching Dashboard in Browser..." -ForegroundColor Yellow
Write-Host ""

python -m streamlit run dashboard/app.py

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
