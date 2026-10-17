@echo off
rem Final checks and evaluation, in order. Run it after the human review sheet is filled in.
rem Every step stops the script if it fails, so the reports are only written from a working setup.
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_VERBOSITY=error
set PY=.venv\Scripts\python.exe

echo.
echo [1/6] Importing the human review sheet
if exist evaluation\human_review.xlsx (
    "%PY%" -m src.review_sheet import evaluation\human_review.xlsx || goto failed
) else (
    echo No evaluation\human_review.xlsx found; correctness will stay "pending".
)

echo.
echo [2/6] Checking the book list
"%PY%" -m src.manifest check || goto failed

echo.
echo [3/6] Running the tests
"%PY%" -m pytest -q || goto failed

echo.
echo [4/6] Checking that Ollama is running
ollama list >nul 2>&1 || (
    echo Ollama is not running. Start the Ollama app, then run this again.
    goto failed
)

echo.
echo [5/6] Evaluating Toodles Fast on the 33 test questions (about 5 minutes)
"%PY%" -m src.evaluate --split test --backend ollama --name test_v2_ollama || goto failed

echo.
echo [6/6] Official report with Toodles Slow (about 30 minutes; keep the laptop plugged in)
choice /c YN /t 20 /d Y /m "Run it now? Y within 20 seconds by default"
if errorlevel 2 (
    echo Skipped. evaluation\report.json keeps its previous results.
) else (
    "%PY%" -m src.evaluate --split test --backend hf || goto failed
)

echo.
echo Done. Results:
"%PY%" -m src.evaluate --summary evaluation/report_test_v2_ollama.json evaluation/report.json
echo Update the results table in README.md with these numbers.
pause
exit /b 0

:failed
echo.
echo Stopped: the step above failed. Fix it, then run this again.
pause
exit /b 1
