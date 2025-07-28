#!/usr/bin/env python3
"""
Debug script for urgency assessment
"""

import re
from ems_ai_triage_system_enhanced import EnhancedEMSKeywordAssociation, EnhancedEMSTriageSystem

def test_urgency_assessment():
    """Test urgency assessment with known cases"""
    
    analyzer = EnhancedEMSKeywordAssociation()
    
    test_cases = [
        {
            "narrative": "Patient experiencing severe chest pain radiating to left arm, shortness of breath, diaphoretic",
            "expected": "Critical"
        },
        {
            "narrative": "Patient with diabetes, blood glucose 45, confused and diaphoretic",
            "expected": "High"
        },
        {
            "narrative": "Patient with minor ankle sprain, able to bear weight, no deformity",
            "expected": "Low"
        }
    ]
    
    print("=== Urgency Assessment Debug ===")
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Narrative: {case['narrative']}")
        print(f"Expected: {case['expected']}")
        
        urgency_level, urgency_score = analyzer.assess_urgency_level(case['narrative'])
        print(f"Actual: {urgency_level}")
        print(f"Score: {urgency_score}")
        
        # Debug urgency scoring
        text_lower = case['narrative'].lower()
        print("Debug - Checking keywords:")
        for urgency_level_name, categories in analyzer.urgency_keywords.items():
            for category, keywords in categories.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        print(f"  Found '{keyword}' in {urgency_level_name} - {category}")
        
        # Extract keywords
        keywords = analyzer.extract_keywords(case['narrative'], top_k=10)
        print(f"Keywords: {[kw[0] for kw in keywords[:5]]}")
        
        # Check vital signs
        vitals = analyzer._extract_vital_signs(case['narrative'])
        print(f"Vital Signs: {vitals}")
        
        # Check if expected matches actual
        match = urgency_level == case['expected']
        print(f"Match: {'✅' if match else '❌'}")

if __name__ == "__main__":
    test_urgency_assessment() 