#!/usr/bin/env python3
"""
EMS AI Triage System - FastAPI Backend
======================================

This FastAPI application integrates the EMS AI Triage System and provides
a REST API for narrative analysis and triage scoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the EMS AI Triage System
from ems_ai_triage_system_enhanced import EnhancedEMSTriageSystem, EnhancedEMSKeywordAssociation

# Initialize FastAPI app
app = FastAPI(
    title="EMS AI Triage System API",
    description="Emergency Medical Services AI Triage System API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Enhanced EMS AI Triage System with PA State Protocols
triage_system = EnhancedEMSTriageSystem()
keyword_analyzer = EnhancedEMSKeywordAssociation()

# Load PA State protocols
try:
    from extract_pa_protocols import extract_pa_protocols_from_pdf
    
    pa_protocols = extract_pa_protocols_from_pdf("../2023v1-2 PA BLS Protocols.pdf")
    
    # Initialize PA protocols
    for protocol in pa_protocols:
        triage_system.add_protocol_data(
            protocol["name"],
            protocol["text"],
            protocol["keywords"]
        )
    print("✅ PA State protocols loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load PA protocols: {e}")
    print("   The system will still work for basic analysis")

# Pydantic models for request/response
class NarrativeRequest(BaseModel):
    narrative: str
    patient_id: Optional[str] = None

class KeywordItem(BaseModel):
    keyword: str
    score: float

class UrgencyIndicators(BaseModel):
    cardiac: List[str] = []
    respiratory: List[str] = []
    neurological: List[str] = []
    trauma: List[str] = []
    other: List[str] = []

class ProtocolMatch(BaseModel):
    protocol_name: str
    match_score: float
    matched_keywords: List[str]

class AnalysisResponse(BaseModel):
    patient_id: str
    narrative: str
    triage_score: float
    urgency_level: str
    keywords: List[KeywordItem]
    urgency_indicators: UrgencyIndicators
    protocol_matches: List[ProtocolMatch]
    recommendations: List[str]
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "EMS AI Triage System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Analyze EMS narrative and provide triage assessment",
            "GET /health": "Health check endpoint",
            "GET /protocols": "Get available protocols"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "EMS AI Triage System API"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_narrative(request: NarrativeRequest):
    """
    Analyze EMS narrative and provide triage assessment
    
    This endpoint takes a narrative string and returns:
    - Triage score (0-10 scale)
    - Urgency level (CRITICAL, URGENT, MODERATE, LOW, MINIMAL)
    - Extracted keywords with scores
    - Urgency indicators by category
    - Matched protocols
    - Clinical recommendations
    """
    try:
        # Generate patient ID if not provided
        if not request.patient_id:
            request.patient_id = f"PT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze the narrative using the Enhanced EMS AI Triage System
        analysis = triage_system.analyze_ems_narrative(request.narrative, request.patient_id)
        
        # Extract keywords using enhanced system
        keywords = keyword_analyzer.extract_keywords(request.narrative, top_k=15)
        
        # Format keywords for response
        keyword_items = [KeywordItem(keyword=kw, score=score) for kw, score in keywords]
        
        # Format urgency indicators (simplified for enhanced system)
        urgency_indicators = UrgencyIndicators(
            cardiac=[],
            respiratory=[],
            neurological=[],
            trauma=[],
            other=[]
        )
        
        # Format protocol matches
        protocol_matches = [
            ProtocolMatch(
                protocol_name=match['protocol_name'],
                match_score=match['match_score'],
                matched_keywords=match['matched_keywords']
            )
            for match in analysis['protocol_matches']
        ]
        
        # Create response
        response = AnalysisResponse(
            patient_id=request.patient_id,
            narrative=request.narrative,
            triage_score=analysis['triage_score'],
            urgency_level=analysis['urgency_level'],
            keywords=keyword_items,
            urgency_indicators=urgency_indicators,
            protocol_matches=protocol_matches,
            recommendations=analysis['recommendations'],
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/protocols")
async def get_protocols():
    """Get available protocols"""
    try:
        protocols = []
        for protocol_name, protocol_data in triage_system.protocol_database.items():
            protocols.append({
                "name": protocol_name,
                "keywords": protocol_data.get('keywords', []),
                "match_count": len(protocol_data.get('keywords', []))
            })
        
        return {
            "protocols": protocols,
            "total_protocols": len(protocols)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get protocols: {str(e)}"
        )

@app.post("/extract-keywords")
async def extract_keywords(request: NarrativeRequest):
    """Extract keywords from text"""
    try:
        keywords = keyword_analyzer.extract_keywords(request.narrative, top_k=10)
        
        return {
            "keywords": [{"keyword": kw, "score": score} for kw, score in keywords],
            "total_keywords": len(keywords)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Keyword extraction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    print("🚑 Starting EMS AI Triage System FastAPI...")
    print("📡 API endpoints available:")
    print("   - GET  /")
    print("   - GET  /health")
    print("   - POST /analyze")
    print("   - GET  /protocols")
    print("   - POST /extract-keywords")
    print("\n🌐 Server starting on http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000) 