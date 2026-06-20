@echo off
title MultiAgent RAG Assistant
echo Starting MultiAgent RAG Assistant...
echo.
echo Server will open at http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
pause
