# Python Model Integration for EMS AI

This document explains how to integrate the Python machine learning model with the EMS AI application.

## 🎯 Overview

The integration connects your Python FastAPI backend (with the EMS protocols model) to the Next.js frontend, providing enhanced narrative analysis and protocol matching.

## 🏗️ Architecture

```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Next.js App   │ ──────────────► │  Python FastAPI │
│   (Frontend)    │                │   (Backend)     │
│                 │ ◄────────────── │                 │
└─────────────────┘                └─────────────────┘
```

## 🚀 Quick Start

### 1. Start the Python Backend

```bash
# From the bryankstites-webapp directory
./setup-python-backend.sh
```

This will:
- Create a Python virtual environment
- Install all required dependencies
- Start the FastAPI server on http://localhost:8000

### 2. Start the Next.js Frontend

```bash
# In a new terminal, from bryankstites-webapp directory
npm run dev
```

The frontend will be available at http://localhost:3000

## 📋 Requirements

### Python Backend Requirements
- Python 3.8+
- pip3
- The PA State protocols PDF (should be in parent directory)

### Dependencies (automatically installed)
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- nltk>=3.8.0

## 🔧 Configuration

### Environment Variables

The Next.js app will automatically connect to the Python backend at:
- **Development**: `http://localhost:8000`
- **Production**: Set `PYTHON_API_URL` environment variable

### API Endpoints

The Python backend provides these endpoints:

- `POST /analyze` - Analyze patient narrative
- `GET /protocols` - Get available protocols
- `POST /extract-keywords` - Extract keywords from text

## 📊 Enhanced Features

### 1. Protocol Matching
The Python model matches patient narratives against EMS protocols from the PA State protocols PDF.

### 2. Keyword Analysis
Advanced keyword extraction and association analysis using machine learning.

### 3. Triage Scoring
ML-based triage scoring that considers:
- Protocol matches
- Keyword associations
- Urgency indicators
- Medical condition patterns

### 4. Recommendations
AI-generated recommendations based on:
- Protocol matches
- Patient history
- Current symptoms
- Risk factors

## 🧪 Testing the Integration

### 1. Test the Python Backend
```bash
# Test the analyze endpoint
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"narrative": "Patient complaining of chest pain radiating to left arm"}'
```

### 2. Test the Full Integration
1. Start both servers
2. Go to http://localhost:3000
3. Enter patient narrative and vital signs
4. Check results for enhanced analysis

## 🔍 Troubleshooting

### Python Backend Issues
- **Port 8000 in use**: Change port in setup script
- **Missing dependencies**: Run `pip install -r requirements_fastapi.txt`
- **PDF not found**: Ensure PA State protocols PDF is in parent directory

### Frontend Issues
- **Connection failed**: Check if Python backend is running
- **CORS errors**: Backend is configured to allow all origins in development

### Integration Issues
- **No enhanced analysis**: Check browser console for API errors
- **Fallback mode**: App will work with basic analysis if Python backend is unavailable

## 📈 Production Deployment

### Option 1: Separate Services
- Deploy Python backend to separate service (Heroku, Railway, etc.)
- Set `PYTHON_API_URL` environment variable in Vercel
- Update CORS settings in Python backend

### Option 2: Unified Deployment
- Package Python backend with Next.js (more complex)
- Use serverless functions for Python code
- Requires additional configuration

## 🔄 Development Workflow

1. **Start Python backend**: `./setup-python-backend.sh`
2. **Start Next.js frontend**: `npm run dev`
3. **Make changes**: Edit Python code or frontend code
4. **Test**: Use the application interface
5. **Deploy**: Push changes and redeploy

## 📚 API Documentation

Once the Python backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Next Steps

1. **Train on PA Protocols**: Use the PDF to train the model on actual PA State protocols
2. **Enhance Keywords**: Add more medical keywords and associations
3. **Improve Scoring**: Refine the triage scoring algorithm
4. **Add Protocols**: Integrate more EMS protocols into the system 