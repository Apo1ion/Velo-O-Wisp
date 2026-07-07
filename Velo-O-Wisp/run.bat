@echo off
title Velo-O-Wisp AI
cd /d "%~dp0"

echo ==========================================
echo         Starting Velo-O-Wisp AI
echo ==========================================
echo.

start "Velo-O-Wisp Server" cmd /c python app.py

timeout /t 5 /nobreak >nul

start "" http://127.0.0.1:5000

exit
