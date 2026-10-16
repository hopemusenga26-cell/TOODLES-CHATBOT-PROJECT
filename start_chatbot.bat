@echo off
rem Starts TOODLES and opens it in the browser. Close this window to stop it.
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error

powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue) { exit 1 }"
if errorlevel 1 (
    echo TOODLES is already running. Opening it in your browser.
    start "" http://localhost:8501
    timeout /t 5 >nul
    exit /b
)

echo Starting TOODLES. Your browser will open in a few seconds.
echo Keep this window open while you use TOODLES; close it to stop.
start "" cmd /c "timeout /t 10 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run src\app.py --server.port 8501
pause
