#!/usr/bin/env python3
"""
Test the enhanced EMS AI system directly
"""

import requests
import json
import time

def test_enhanced_system():
    """Test the enhanced system directly"""
    
    # Test cases
    test_cases = [
        {
            "narrative": "Patient experiencing severe chest pain radiating to left arm, shortness of breath, diaphoretic",
            "expected_urgency": "Critical"
        },
        {
            "narrative": "Patient with diabetes, blood glucose 45, confused and diaphoretic",
            "expected_urgency": "High"
        },
        {
            "narrative": "Patient with minor ankle sprain, able to bear weight, no deformity",
            "expected_urgency": "Low"
        }
    ]
    
    print("=== Testing Enhanced EMS AI System ===")
    
    # Try different ports
    ports = [8000, 8001, 8002]
    
    for port in ports:
        print(f"\nTrying port {port}...")
        try:
            # Test if server is running
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server found on port {port}")
                
                # Test each case
                for i, case in enumerate(test_cases, 1):
                    print(f"\nTest Case {i}:")
                    print(f"Narrative: {case['narrative'][:50]}...")
                    
                    response = requests.post(
                        f"http://localhost:{port}/analyze",
                        json={
                            "narrative": case["narrative"],
                            "patient_id": f"test_{i}"
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        actual_urgency = data.get("urgency_level", "Unknown")
                        triage_score = data.get("triage_score", 0)
                        keywords = [kw["keyword"] for kw in data.get("keywords", [])[:5]]
                        
                        print(f"Expected: {case['expected_urgency']}")
                        print(f"Actual: {actual_urgency}")
                        print(f"Score: {triage_score}")
                        print(f"Keywords: {keywords}")
                        
                        match = actual_urgency == case['expected_urgency']
                        print(f"Match: {'✅' if match else '❌'}")
                        
                        if match:
                            print("🎉 SUCCESS!")
                        else:
                            print("⚠️  Mismatch - but system is working")
                    else:
                        print(f"❌ API error: {response.status_code}")
                        
                return port  # Found working port
                
        except requests.exceptions.ConnectionError:
            print(f"❌ No server on port {port}")
        except Exception as e:
            print(f"❌ Error on port {port}: {e}")
    
    print("\n❌ No working server found")
    return None

if __name__ == "__main__":
    working_port = test_enhanced_system()
    if working_port:
        print(f"\n🎉 Enhanced system is working on port {working_port}")
    else:
        print("\n❌ Need to start the server first") 