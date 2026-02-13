@echo off
REM Docker Deployment Script for Assistant Agricole Multilingue (Windows)

echo 🌱 Assistant Agricole Multilingue - Docker Deployment
echo ======================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop for Windows.
    echo Visit: https://docs.docker.com/desktop/windows/install/
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo 📦 Building Docker image...
docker build -t assistant-agricole .

if %errorlevel% equ 0 (
    echo ✅ Docker image built successfully!
    
    echo 🚀 Starting container...
    docker run -d ^
        --name assistant-agricole ^
        -p 8000:8000 ^
        -e GEMINI_API_KEY=AIzaSyBBQ-okicASaEm1CP9Q8rzvLfYZJMEHvIc ^
        -v "%cd%\logs:/app/logs" ^
        assistant-agricole
    
    if %errorlevel% equ 0 (
        echo ✅ Container started successfully!
        echo 🌐 Web interface: http://localhost:8000
        echo 📚 API docs: http://localhost:8000/docs
        echo ❤️  Health check: http://localhost:8000/health
        echo.
        echo 📋 Useful commands:
        echo   View logs: docker logs -f assistant-agricole
        echo   Stop: docker stop assistant-agricole
        echo   Remove: docker rm assistant-agricole
        echo.
        pause
    ) else (
        echo ❌ Failed to start container
        pause
        exit /b 1
    )
) else (
    echo ❌ Failed to build Docker image
    pause
    exit /b 1
)
