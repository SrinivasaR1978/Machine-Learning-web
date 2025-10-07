@echo off
title Flask App Launcher
cd /d "%~dp0"
echo Activating venv...
call venv\Scripts\activate.bat
echo Starting Flask app...
start "" http://127.0.0.1:5000
python app.py
pause
