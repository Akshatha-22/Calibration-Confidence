#!/bin/bash
# Quick start script for local development

set -e

echo "🚀 Starting Calibration Confidence Development Environment..."

# Check for docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed"
    exit 1
fi

# Copy env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo "✓ .env created - update with production values if deploying"
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check health
echo "🏥 Checking service health..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is healthy"
else
    echo "❌ Backend health check failed"
    docker-compose logs backend
    exit 1
fi

echo ""
echo "✅ All services are running!"
echo ""
echo "📱 Access Points:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "🛑 To stop services: docker-compose down"
echo "📊 To view logs: docker-compose logs -f"
echo ""
