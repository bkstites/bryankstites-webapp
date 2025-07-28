#!/bin/bash

# EMS AI Python Backend Setup Script
# This script sets up and starts the Python FastAPI backend

echo "🚑 Setting up EMS AI Python Backend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Navigate to the Back End directory
cd "../Back End"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements_fastapi.txt

# Install additional dependencies if needed
pip install uvicorn fastapi

# Check if the PA State protocols PDF exists
if [ -f "../2023v1-2 PA BLS Protocols.pdf" ]; then
    echo "✅ Found PA State protocols PDF"
else
    echo "⚠️  PA State protocols PDF not found. Please ensure it's in the parent directory."
fi

# Start the FastAPI server
echo "🚀 Starting EMS AI Python Backend..."
echo "📍 Backend will be available at: http://localhost:8000"
echo "📖 API documentation will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload 