Write-Host "============================================"
Write-Host "       CerebroNet - Starting Services"
Write-Host "============================================"

Write-Host "[1/2] Starting MLflow server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; mlflow server --host 0.0.0.0 --port 5000"

Start-Sleep -Seconds 3

Write-Host "[2/2] Starting FastAPI server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep -Seconds 3

Write-Host "Opening UIs..."
Start-Process "http://localhost:5000"
Start-Process "http://localhost:8000/docs"

Write-Host "============================================"
Write-Host "  MLflow   ->  http://localhost:5000"
Write-Host "  FastAPI  ->  http://localhost:8000/docs"
Write-Host "  Health   ->  http://localhost:8000/health"
Write-Host "  Metrics  ->  http://localhost:8000/metrics"
Write-Host "============================================"