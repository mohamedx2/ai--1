# 🚀 FastAPI Deployment Guide

## Overview

The Assistant Agricole Multilingue now supports FastAPI deployment for production-ready web services. This provides a RESTful API with a modern web interface.

## 🌟 Features

### FastAPI Deployment
- **🌐 RESTful API**: Full REST API with automatic documentation
- **📱 Modern Web Interface**: Responsive web client with voice/text input
- **🔄 Real-time Processing**: Asynchronous request handling
- **📊 API Documentation**: Auto-generated Swagger/OpenAPI docs
- **🔍 Health Monitoring**: Built-in health checks and system stats
- **💬 Conversation Management**: Persistent conversation sessions
- **🌍 Multilingual Support**: Same language detection and response capabilities

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| POST | `/api/text` | Process text input |
| POST | `/api/audio` | Process audio input |
| GET | `/api/conversation/{id}` | Get conversation history |
| DELETE | `/api/conversation/{id}` | Clear conversation |
| GET | `/api/languages` | Get supported languages |
| GET | `/api/stats` | System statistics |
| GET | `/docs` | API documentation (Swagger) |
| GET | `/redoc` | API documentation (ReDoc) |

## 🚀 Quick Start

### Option 1: Using Deployment Scripts

**Windows:**
```bash
deploy_fastapi.bat
```

**Linux/Mac:**
```bash
chmod +x deploy_fastapi.sh
./deploy_fastapi.sh
```

### Option 2: Manual Deployment

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run FastAPI server:**
```bash
python fastapi_app.py
```

3. **Access the application:**
- Web Interface: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

## 📡 API Usage Examples

### Text Input API

```bash
curl -X POST "http://localhost:8000/api/text" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "How much water does wheat need?",
       "conversation_id": "optional_conversation_id"
     }'
```

### Audio Input API

```bash
curl -X POST "http://localhost:8000/api/audio" \
     -H "Content-Type: application/json" \
     -d '{
       "audio_data": "base64_encoded_audio",
       "conversation_id": "optional_conversation_id"
     }'
```

### Response Format

```json
{
  "response": "Wheat requires moderate irrigation...",
  "transcript": "How much water does wheat need?",
  "language": "en",
  "conversation_id": "conv_123456",
  "timestamp": "2026-02-13T05:00:00",
  "processing_time": 2.34
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/fastapi.log

 CORS Settings (for production)
ALLOWED_ORIGINS=https://yourdomain.com
```

### Production Deployment

#### Using Uvicorn (Recommended)

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using Gunicorn (Linux)

```bash
pip install gunicorn
gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t assistant-agricole .
docker run -p 8000:8000 assistant-agricole
```

## 🔒 Security Considerations

### CORS Configuration
Update CORS settings for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Rate Limiting
Consider adding rate limiting for production:

```bash
pip install slowapi
```

### Authentication
Add API key authentication if needed:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Validate token here
    pass
```

## 📊 Monitoring

### Health Checks
- `GET /health` - Basic health status
- `GET /api/stats` - System statistics

### Logging
Logs are automatically created in the `logs/` directory:
- `fastapi_YYYYMMDD.log` - Application logs
- Request/response logging included

### Metrics (Optional)
Add Prometheus metrics:

```bash
pip install prometheus-fastapi-instrumentator
```

## 🌐 Web Interface Features

The FastAPI deployment includes a modern web interface at `/` with:

- **Responsive Design**: Works on desktop and mobile
- **Voice/Text Input**: Toggle between input methods
- **Real-time Status**: Live processing indicators
- **Conversation History**: Persistent chat sessions
- **Multilingual Support**: Same language capabilities
- **Example Questions**: Quick-start examples
- **Modern UI**: Clean, professional interface

## 🔄 Comparison: Gradio vs FastAPI

| Feature | Gradio | FastAPI |
|---------|--------|---------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Customization** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **API Integration** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production Ready** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Mobile Support** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000
```

### Testing the API

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest test_fastapi.py
```

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## 📈 Performance

### Expected Performance
- **Text Processing**: 1-3 seconds
- **Audio Processing**: 2-5 seconds
- **Concurrent Requests**: Supports multiple simultaneous users
- **Memory Usage**: 4-8GB base + additional for concurrent requests

### Optimization Tips
1. **Use Workers**: Run with multiple workers for concurrency
2. **Enable Caching**: Cache frequent responses
3. **Load Balancing**: Use nginx or similar for load balancing
4. **Database**: Use Redis for conversation storage in production

## 🔗 Integration Examples

### JavaScript Client

```javascript
async function askAssistant(text) {
    const response = await fetch('/api/text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });
    return await response.json();
}
```

### Python Client

```python
import requests

def ask_assistant(text, conversation_id=None):
    response = requests.post('http://localhost:8000/api/text', json={
        'text': text,
        'conversation_id': conversation_id
    })
    return response.json()
```

## 🚀 Deployment Checklist

- [ ] Install all dependencies
- [ ] Configure environment variables
- [ ] Set up proper CORS settings
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Test all endpoints
- [ ] Configure reverse proxy (nginx)
- [ ] Set up SSL certificates
- [ ] Configure backup and recovery

---

**🌱 Your Assistant Agricole Multilingue is now ready for production deployment with FastAPI!**
