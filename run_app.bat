@echo off
echo ========================================================
echo Starting WellSense Burnout Prediction Web App...
echo URL: http://localhost:8502
echo ========================================================
.venv\Scripts\streamlit.exe run app.py --server.port 8502
pause
