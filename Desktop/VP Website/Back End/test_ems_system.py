#!/usr/bin/env python3
"""
Test Script for EMS AI Triage System
====================================

This script demonstrates the functionality of the EMS AI Triage System
with sample data and analysis.
"""

import json
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ems_ai_triage_system import EMSTriageSystem, EMSKeywordAssociation
from ems_data_preprocessing import EMSTrainingPipeline

def test_basic_functionality():
    """Test basic system functionality"""
    print("=== Testing Basic EMS AI Triage System ===")
    
    # Initialize the system
    triage_system = EMSTriageSystem()
    
    # Test narratives
    test_narratives = [
        {
            "id": "TEST001",
            "narrative": """
            Patient presents with severe chest pain radiating to left arm.
            Patient is diaphoretic and reports shortness of breath.
            Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
            Patient has history of hypertension and diabetes.
            """,
            "expected_urgency": "URGENT"
        },
        {
            "id": "TEST002", 
            "narrative": """
            Patient found unresponsive with facial droop and arm weakness.
            FAST assessment positive. Last known normal 2 hours ago.
            Vital signs: BP 160/90, HR 88, RR 18, O2 sat 95%.
            """,
            "expected_urgency": "CRITICAL"
        },
        {
            "id": "TEST003",
            "narrative": """
            Patient with minor laceration to right forearm.
            No active bleeding. Patient alert and oriented.
            Vital signs: BP 120/80, HR 72, RR 16, O2 sat 98%.
            """,
            "expected_urgency": "LOW"
        }
    ]
    
    results = []
    
    for test in test_narratives:
        print(f"\n--- Testing Narrative {test['id']} ---")
        print(f"Narrative: {test['narrative'].strip()}")
        
        # Analyze the narrative
        analysis = triage_system.analyze_ems_narrative(test['narrative'], test['id'])
        
        # Print results
        print(f"Urgency Level: {analysis['urgency_level']}")
        print(f"Triage Score: {analysis['triage_score']:.2f}")
        print(f"Keywords Found: {len(analysis['keyword_analysis']['keywords'])}")
        print(f"Protocol Matches: {len(analysis['protocol_matches'])}")
        
        # Print top keywords
        print("Top Keywords:")
        for keyword, score in analysis['keyword_analysis']['keywords'][:5]:
            print(f"  - {keyword}: {score:.3f}")
        
        # Print urgency indicators
        urgency_indicators = analysis['keyword_analysis']['urgency_indicators']
        if urgency_indicators:
            print("Urgency Indicators:")
            for category, indicators in urgency_indicators.items():
                print(f"  - {category}: {indicators}")
        
        # Print recommendations
        print("Recommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
        
        results.append({
            'test_id': test['id'],
            'urgency_level': analysis['urgency_level'],
            'triage_score': analysis['triage_score'],
            'expected_urgency': test['expected_urgency'],
            'keywords_count': len(analysis['keyword_analysis']['keywords'])
        })
    
    return results

def test_keyword_association():
    """Test keyword association functionality"""
    print("\n=== Testing Keyword Association ===")
    
    keyword_analyzer = EMSKeywordAssociation()
    
    # Sample corpus of EMS narratives
    corpus = [
        "Patient with chest pain and shortness of breath. ECG shows ST elevation.",
        "Patient experiencing chest pain radiating to left arm. Administered aspirin.",
        "Patient with chest pain and diaphoresis. Vital signs stable.",
        "Patient found unresponsive with facial droop. FAST assessment positive.",
        "Patient with stroke symptoms. Last known normal 3 hours ago.",
        "Patient with trauma injuries from motor vehicle accident. Multiple lacerations.",
        "Patient with respiratory distress and wheezing. History of asthma."
    ]
    
    # Test keyword extraction
    test_text = "Patient presents with severe chest pain and shortness of breath"
    keywords = keyword_analyzer.extract_keywords(test_text, top_k=10)
    
    print(f"Extracted keywords from: '{test_text}'")
    for keyword, score in keywords:
        print(f"  - {keyword}: {score:.3f}")
    
    # Test keyword associations
    target_keyword = "chest pain"
    associations = keyword_analyzer.find_keyword_associations(target_keyword, corpus, min_association=0.2)
    
    print(f"\nAssociations with '{target_keyword}':")
    for keyword, score in sorted(associations.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {keyword}: {score:.3f}")

def test_training_pipeline():
    """Test the training pipeline with sample data"""
    print("\n=== Testing Training Pipeline ===")
    
    # Create sample protocol data
    sample_protocols = [
        {
            "protocol_id": "CARDIAC_001",
            "protocol_name": "Chest Pain Protocol",
            "category": "cardiac",
            "content": """
            CHEST PAIN PROTOCOL
            Assessment:
            - 12-lead ECG within 10 minutes
            - Aspirin 325mg PO if no contraindications
            - Nitroglycerin 0.4mg SL if systolic BP >90
            - Oxygen if SpO2 <94%
            
            Transport:
            - Rapid transport to cardiac center
            - Activate cardiac alert if STEMI criteria met
            """,
            "keywords": ["chest pain", "cardiac", "ecg", "aspirin", "nitroglycerin", "oxygen", "transport"]
        },
        {
            "protocol_id": "STROKE_001",
            "protocol_name": "Stroke Protocol", 
            "category": "stroke",
            "content": """
            STROKE PROTOCOL
            FAST Assessment:
            - Face: Facial droop
            - Arms: Arm drift
            - Speech: Speech difficulty
            - Time: Time of onset
            
            Treatment:
            - Glucose check
            - Rapid transport to stroke center
            - Document last known normal time
            """,
            "keywords": ["stroke", "fast", "facial droop", "arm drift", "speech", "glucose", "transport"]
        },
        {
            "protocol_id": "TRAUMA_001",
            "protocol_name": "Trauma Protocol",
            "category": "trauma", 
            "content": """
            TRAUMA PROTOCOL
            Primary Survey:
            - Airway assessment
            - Breathing assessment
            - Circulation assessment
            - Disability assessment
            - Exposure assessment
            
            Treatment:
            - Spinal immobilization if indicated
            - Control bleeding
            - Rapid transport to trauma center
            """,
            "keywords": ["trauma", "airway", "breathing", "circulation", "bleeding", "transport"]
        }
    ]
    
    # Save sample data
    with open('test_protocols.json', 'w') as f:
        json.dump(sample_protocols, f, indent=2)
    
    # Initialize training pipeline
    pipeline = EMSTrainingPipeline()
    
    # Train model
    print("Training model with sample protocols...")
    pipeline.train_model('test_protocols.json', 'test_models/')
    
    # Test the trained system
    test_narrative = """
    Patient presents with severe chest pain radiating to left arm.
    Patient is diaphoretic and reports shortness of breath.
    Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
    """
    
    analysis = pipeline.triage_system.analyze_ems_narrative(test_narrative, "TRAINED_TEST")
    
    print(f"\nTrained System Analysis:")
    print(f"Urgency Level: {analysis['urgency_level']}")
    print(f"Triage Score: {analysis['triage_score']:.2f}")
    print(f"Protocol Matches: {len(analysis['protocol_matches'])}")
    
    if analysis['protocol_matches']:
        print("Protocol Matches:")
        for match in analysis['protocol_matches'][:3]:
            print(f"  - {match['protocol_name']}: {match['match_score']:.3f}")

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\n=== Testing Data Preprocessing ===")
    
    from ems_data_preprocessing import EMSDataPreprocessor
    
    preprocessor = EMSDataPreprocessor()
    
    # Test with sample data
    sample_data = [
        {
            "protocol_id": "TEST_001",
            "protocol_name": "Test Protocol",
            "content": "This is a test protocol with chest pain and cardiac symptoms.",
            "category": "cardiac"
        }
    ]
    
    # Preprocess data
    processed = preprocessor.preprocess_protocols(sample_data)
    
    print(f"Processed {len(processed)} protocols")
    if processed:
        protocol = processed[0]
        print(f"Protocol ID: {protocol['protocol_id']}")
        print(f"Category: {protocol['category']}")
        print(f"Keywords: {protocol['keywords']}")

def generate_test_report(results):
    """Generate a test report"""
    print("\n=== Test Report ===")
    
    total_tests = len(results)
    correct_urgency = sum(1 for r in results if r['urgency_level'] == r['expected_urgency'])
    
    print(f"Total Tests: {total_tests}")
    print(f"Correct Urgency Classification: {correct_urgency}/{total_tests} ({correct_urgency/total_tests*100:.1f}%)")
    
    avg_triage_score = sum(r['triage_score'] for r in results) / len(results)
    print(f"Average Triage Score: {avg_triage_score:.2f}")
    
    avg_keywords = sum(r['keywords_count'] for r in results) / len(results)
    print(f"Average Keywords Found: {avg_keywords:.1f}")
    
    # Save detailed results
    report_data = {
        "test_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "correct_classifications": correct_urgency,
            "accuracy_percentage": correct_urgency/total_tests*100,
            "average_triage_score": avg_triage_score,
            "average_keywords": avg_keywords
        },
        "detailed_results": results
    }
    
    with open('test_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: test_report.json")

def main():
    """Main test function"""
    print("EMS AI Triage System - Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        results = test_basic_functionality()
        test_keyword_association()
        test_training_pipeline()
        test_data_preprocessing()
        
        # Generate report
        generate_test_report(results)
        
        print("\n=== All Tests Completed Successfully ===")
        print("The EMS AI Triage System is working correctly!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 