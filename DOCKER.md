# Docker Deployment Guide

## 🐳 Docker Setup for Assistant Agricole Multilingue

### Prerequisites
- Docker installed on your system
- Docker Compose (for multi-container setup)
- Google Gemini API Key

### Quick Start with Docker Compose (Recommended)

1. **Set Environment Variable:**
   ```bash
   export GEMINI_API_KEY=AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the Application:**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Commands

#### Build and Run
```bash
# Build the Docker image
docker build -t assistant-agricole .

# Run the container
docker run -d \
  --name assistant-agricole \
  -p 8000:8000 \
  -e GEMINI_API_KEY=AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc \
  -v $(pwd)/logs:/app/logs \
  assistant-agricole
```

#### Development with Live Reload
```bash
# Run with volume mounting for development
docker run -d \
  --name assistant-agricole-dev \
  -p 8000:8000 \
  -e GEMINI_API_KEY=AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc \
  -e DEBUG=True \
  -v $(pwd):/app \
  assistant-agricole
```

#### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# Scale for production
docker-compose up -d --scale assistant-agricole=3
```

### Production Deployment

#### With Redis Session Storage
```bash
# Start with Redis for production
docker-compose --profile production up -d
```

#### Environment Variables for Production
Create a `.env` file:
```env
GEMINI_API_KEY=your-production-api-key
DEBUG=False
ALLOWED_ORIGINS=https://yourdomain.com
SECRET_KEY=your-secret-key
```

#### Monitoring
```bash
# Check container health
docker ps

# View resource usage
docker stats

# Check logs
docker-compose logs -f assistant-agricole
```

### Deployment Options

#### 1. Local Development
```bash
docker-compose up -d
```

#### 2. Production Server
```bash
# With persistent data
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

#### 3. Cloud Deployment (AWS, GCP, Azure)
```bash
# Export for cloud deployment
docker save assistant-agricole | gzip > assistant-agricole.tar.gz

# Import on cloud server
docker load < assistant-agricole.tar.gz
```

### Troubleshooting

#### Common Issues
1. **Port Already in Use:**
   ```bash
   # Kill process using port 8000
   sudo lsof -ti:8000 | xargs kill -9
   ```

2. **Permission Issues:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER logs/
   ```

3. **Memory Issues:**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   ```

#### Health Check
```bash
# Test health endpoint
curl http://localhost:8000/health
```

### Benefits of Docker Deployment

- ✅ **Consistent Environment**: Same config everywhere
- ✅ **Easy Scaling**: Multiple containers with Docker Compose
- ✅ **Isolation**: No dependency conflicts
- ✅ **Portability**: Deploy anywhere Docker runs
- ✅ **Version Control**: Image versioning and rollbacks
- ✅ **Monitoring**: Built-in health checks and logging

### Next Steps

1. Deploy to staging environment
2. Set up CI/CD pipeline
3. Configure load balancer for production
4. Set up monitoring and alerting
