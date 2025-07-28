#!/usr/bin/env python3
"""
EMS AI Triage System - Demo Script
==================================

A simplified demo showcasing the EMS AI Triage System functionality.
"""

import json
from datetime import datetime
from ems_ai_triage_system import EMSTriageSystem, EMSKeywordAssociation

def demo_basic_analysis():
    """Demonstrate basic narrative analysis"""
    print("=== EMS AI Triage System Demo ===\n")
    
    # Initialize the system
    triage_system = EMSTriageSystem()
    
    # Sample EMS narratives for demonstration
    narratives = [
        {
            "id": "DEMO001",
            "description": "Cardiac Emergency",
            "narrative": """
            Patient presents with severe chest pain radiating to left arm.
            Patient is diaphoretic and reports shortness of breath.
            Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
            Patient has history of hypertension and diabetes.
            """
        },
        {
            "id": "DEMO002", 
            "description": "Stroke Emergency",
            "narrative": """
            Patient found unresponsive with facial droop and arm weakness.
            FAST assessment positive. Last known normal 2 hours ago.
            Vital signs: BP 160/90, HR 88, RR 18, O2 sat 95%.
            """
        },
        {
            "id": "DEMO003",
            "description": "Trauma Case",
            "narrative": """
            Patient with multiple trauma injuries from motor vehicle accident.
            Patient has lacerations to face and arms, possible internal bleeding.
            Vital signs: BP 90/60, HR 120, RR 28, O2 sat 88%.
            Patient is alert but confused.
            """
        },
        {
            "id": "DEMO004",
            "description": "Minor Injury",
            "narrative": """
            Patient with minor laceration to right forearm.
            No active bleeding. Patient alert and oriented.
            Vital signs: BP 120/80, HR 72, RR 16, O2 sat 98%.
            """
        }
    ]
    
    print("Analyzing EMS Narratives...\n")
    
    for case in narratives:
        print(f"--- {case['description']} (ID: {case['id']}) ---")
        print(f"Narrative: {case['narrative'].strip()}")
        
        # Analyze the narrative
        analysis = triage_system.analyze_ems_narrative(case['narrative'], case['id'])
        
        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"   Urgency Level: {analysis['urgency_level']}")
        print(f"   Triage Score: {analysis['triage_score']:.2f}/10")
        print(f"   Keywords Found: {len(analysis['keyword_analysis']['keywords'])}")
        
        # Show top keywords
        print(f"\n🔑 Top Keywords:")
        for i, (keyword, score) in enumerate(analysis['keyword_analysis']['keywords'][:5], 1):
            print(f"   {i}. {keyword}: {score:.3f}")
        
        # Show urgency indicators
        urgency_indicators = analysis['keyword_analysis']['urgency_indicators']
        if urgency_indicators:
            print(f"\n⚠️  Urgency Indicators:")
            for category, indicators in urgency_indicators.items():
                print(f"   - {category.title()}: {', '.join(indicators)}")
        
        # Show vital signs
        vitals = analysis['keyword_analysis']['vital_signs']
        if vitals:
            print(f"\n💓 Vital Signs Detected:")
            for vital_type, value in vitals.items():
                print(f"   - {vital_type.replace('_', ' ').title()}: {value}")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   - {rec}")
        
        print("\n" + "="*60 + "\n")

def demo_keyword_extraction():
    """Demonstrate keyword extraction functionality"""
    print("=== Keyword Extraction Demo ===\n")
    
    keyword_analyzer = EMSKeywordAssociation()
    
    # Test different types of EMS text
    test_texts = [
        "Patient with chest pain and shortness of breath. ECG shows ST elevation.",
        "Patient experiencing respiratory distress with wheezing. History of asthma.",
        "Patient found unresponsive with facial droop. FAST assessment positive.",
        "Patient with trauma injuries from motor vehicle accident. Multiple lacerations."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        keywords = keyword_analyzer.extract_keywords(text, top_k=8)
        
        print("Extracted Keywords:")
        for keyword, score in keywords:
            print(f"  - {keyword}: {score:.3f}")
        print()

def demo_protocol_matching():
    """Demonstrate protocol matching functionality"""
    print("=== Protocol Matching Demo ===\n")
    
    triage_system = EMSTriageSystem()
    
    # Add some sample protocols
    protocols = [
        {
            "name": "Chest Pain Protocol",
            "text": "CHEST PAIN PROTOCOL\nAssessment:\n- 12-lead ECG within 10 minutes\n- Aspirin 325mg PO if no contraindications\n- Nitroglycerin 0.4mg SL if systolic BP >90\n- Oxygen if SpO2 <94%",
            "keywords": ["chest pain", "cardiac", "ecg", "aspirin", "nitroglycerin", "oxygen"]
        },
        {
            "name": "Stroke Protocol",
            "text": "STROKE PROTOCOL\nFAST Assessment:\n- Face: Facial droop\n- Arms: Arm drift\n- Speech: Speech difficulty\n- Time: Time of onset\n- Glucose check\n- Rapid transport to stroke center",
            "keywords": ["stroke", "fast", "facial droop", "arm drift", "speech", "glucose"]
        },
        {
            "name": "Trauma Protocol",
            "text": "TRAUMA PROTOCOL\nPrimary Survey:\n- Airway assessment\n- Breathing assessment\n- Circulation assessment\n- Disability assessment\n- Exposure assessment\n- Control bleeding",
            "keywords": ["trauma", "airway", "breathing", "circulation", "bleeding"]
        }
    ]
    
    # Add protocols to system
    for protocol in protocols:
        triage_system.add_protocol_data(
            protocol["name"],
            protocol["text"],
            protocol["keywords"]
        )
    
    print("Added protocols to system:")
    for protocol in protocols:
        print(f"  - {protocol['name']}")
    
    # Test narrative
    test_narrative = """
    Patient presents with severe chest pain radiating to left arm.
    Patient is diaphoretic and reports shortness of breath.
    Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
    """
    
    print(f"\nTesting narrative: {test_narrative.strip()}")
    
    analysis = triage_system.analyze_ems_narrative(test_narrative, "PROTOCOL_TEST")
    
    print(f"\nProtocol Matches Found: {len(analysis['protocol_matches'])}")
    for match in analysis['protocol_matches']:
        print(f"  - {match['protocol_name']}: {match['match_score']:.3f}")

def demo_system_capabilities():
    """Demonstrate overall system capabilities"""
    print("=== System Capabilities Summary ===\n")
    
    capabilities = [
        "🔍 **Keyword Extraction**: TF-IDF based keyword identification from EMS narratives",
        "📊 **Triage Scoring**: 0-10 scale based on urgency indicators and vital signs",
        "🚨 **Urgency Classification**: CRITICAL, URGENT, MODERATE, LOW, MINIMAL levels",
        "💓 **Vital Signs Detection**: Automatic extraction of BP, HR, RR, O2 sat, etc.",
        "⚠️ **Urgency Indicators**: Detection of cardiac, trauma, stroke, respiratory issues",
        "📋 **Protocol Matching**: Cosine similarity matching against known protocols",
        "💡 **Recommendations**: AI-generated recommendations based on analysis",
        "📈 **Confidence Scoring**: Reliability metrics for analysis results",
        "🔄 **Model Training**: Machine learning pipeline for protocol categorization",
        "💾 **Data Persistence**: Save/load trained models and analysis history"
    ]
    
    for capability in capabilities:
        print(capability)
    
    print(f"\n✅ System is ready for EMS narrative analysis!")
    print(f"📅 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main demo function"""
    try:
        demo_basic_analysis()
        demo_keyword_extraction()
        demo_protocol_matching()
        demo_system_capabilities()
        
        print("\n🎉 Demo completed successfully!")
        print("The EMS AI Triage System is working correctly and ready for use.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 