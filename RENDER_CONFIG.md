# Render Configuration for Assistant Agricole Multilingue

## Environment Variables
- PORT: Automatically set by Render
- GEMINI_API_KEY: Set in Render dashboard

## Build Process
1. Install dependencies from requirements.txt
2. Build static assets
3. Start web service

## Runtime
- Python 3.14
- FastAPI with Uvicorn
- Port binding via PORT environment variable

## Health Check
- Endpoint: /health
- Method: GET
- Expected: 200 OK response
