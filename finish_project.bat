@echo off
rem Final checks and evaluation, in order. Run it after the human review sheet is filled in.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m src.finish
pause
