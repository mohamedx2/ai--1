#!/bin/bash

# Assistant Agricole Multilingue - FastAPI Deployment Script
# This script sets up and runs the FastAPI deployment

echo "🌱 Assistant Agricole Multilingue - FastAPI Deployment"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create static directory if it doesn't exist
mkdir -p static

# Check if static index.html exists
if [ ! -f "static/index.html" ]; then
    echo "❌ Error: static/index.html not found. Please ensure the web interface is built."
    exit 1
fi

echo "🚀 Starting FastAPI server..."
echo "📱 Web interface will be available at: http://localhost:8000"
echo "📚 API documentation at: http://localhost:8000/docs"
echo "🔍 Alternative docs at: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run FastAPI app
python fastapi_app.py
