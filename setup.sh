#!/bin/bash
set -e

echo "🚀 Setting up AI Crypto Trading Application"
echo "=========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your actual API credentials before running!"
else
    echo "✅ Environment file already exists"
fi

# Create necessary directories
echo "📁 Creating application directories..."
mkdir -p {logs,data,models,keys}

# Set permissions
echo "🔒 Setting directory permissions..."
chmod 755 logs data models keys

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Kraken API credentials (optional for paper trading)"
echo "2. Run: docker-compose up -d"
echo "3. Access the API at: http://localhost:8080"
echo "4. Monitor with Grafana at: http://localhost:3000 (admin/admin123)"
echo ""
echo "⚠️  IMPORTANT: Trading is DISABLED by default for safety"
echo "   Enable only after thorough testing in paper mode"
echo ""
echo "📚 See README.md for complete documentation"