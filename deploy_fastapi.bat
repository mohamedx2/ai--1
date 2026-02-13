@echo off
REM Assistant Agricole Multilingue - FastAPI Deployment Script for Windows
REM This script sets up and runs the FastAPI deployment

echo 🌱 Assistant Agricole Multilingue - FastAPI Deployment
echo ==================================================

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Create static directory if it doesn't exist
if not exist "static" mkdir static

REM Check if static index.html exists
if not exist "static\index.html" (
    echo ❌ Error: static\index.html not found. Please ensure the web interface is built.
    pause
    exit /b 1
)

echo 🚀 Starting FastAPI server...
echo 📱 Web interface will be available at: http://localhost:8000
echo 📚 API documentation at: http://localhost:8000/docs
echo 🔍 Alternative docs at: http://localhost:8000/redoc
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run FastAPI app
python fastapi_app.py
