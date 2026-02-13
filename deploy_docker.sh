#!/bin/bash

# Docker Deployment Script for Assistant Agricole Multilingue

echo "🌱 Assistant Agricole Multilingue - Docker Deployment"
echo "======================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY environment variable not set."
    echo "Please set it with: export GEMINI_API_KEY=your-api-key"
    echo "Or create a .env file with the key."
fi

echo "📦 Building Docker image..."
docker build -t assistant-agricole .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    echo "🚀 Starting container..."
    docker run -d \
        --name assistant-agricole \
        -p 8000:8000 \
        -e GEMINI_API_KEY=${GEMINI_API_KEY:-AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc} \
        -v $(pwd)/logs:/app/logs \
        assistant-agricole
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started successfully!"
        echo "🌐 Web interface: http://localhost:8000"
        echo "📚 API docs: http://localhost:8000/docs"
        echo "❤️  Health check: http://localhost:8000/health"
        echo ""
        echo "📋 Useful commands:"
        echo "  View logs: docker logs -f assistant-agricole"
        echo "  Stop: docker stop assistant-agricole"
        echo "  Remove: docker rm assistant-agricole"
    else
        echo "❌ Failed to start container"
        exit 1
    fi
else
    echo "❌ Failed to build Docker image"
    exit 1
fi
