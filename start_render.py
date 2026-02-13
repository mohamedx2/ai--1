import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the app after loading environment
from fastapi_app import app

# Get port from Render environment variable
port = int(os.environ.get('PORT', 8000))
host = os.getenv('HOST', '0.0.0.0')

# Run the app with Render-compatible settings
uvicorn.run(
    app,
    host=host,
    port=port,
    reload=False,  # Disable reload in production
    log_level="info"
)
