#!/usr/bin/env python3
"""
Enhanced Narrative Model Testing Framework
=========================================

This script tests the enhanced EMS AI Triage System directly
without relying on the API server.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import statistics

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ems_ai_triage_system_enhanced import EnhancedEMSTriageSystem, EnhancedEMSKeywordAssociation

@dataclass
class TestCase:
    """Represents a test case for the narrative model"""
    id: str
    narrative: str
    expected_protocols: List[str]
    expected_urgency: str
    expected_keywords: List[str]
    category: str
    description: str

@dataclass
class TestResult:
    """Represents the result of a test case"""
    test_case: TestCase
    actual_protocols: List[str]
    actual_urgency: str
    actual_keywords: List[str]
    protocol_accuracy: float
    urgency_accuracy: bool
    keyword_overlap: float
    response_time: float
    success: bool

class EnhancedNarrativeModelTester:
    """Enhanced testing framework for the narrative analysis model"""
    
    def __init__(self):
        self.triage_system = EnhancedEMSTriageSystem()
        self.keyword_analyzer = EnhancedEMSKeywordAssociation()
        self.test_cases = self._load_test_cases()
        
        # Load PA protocols
        self._load_pa_protocols()
        
    def _load_pa_protocols(self):
        """Load PA State protocols"""
        try:
            from extract_pa_protocols import extract_pa_protocols_from_pdf
            
            pa_protocols = extract_pa_protocols_from_pdf("../2023v1-2 PA BLS Protocols.pdf")
            
            # Initialize PA protocols
            for protocol in pa_protocols:
                self.triage_system.add_protocol_data(
                    protocol["name"],
                    protocol["text"],
                    protocol["keywords"]
                )
            print("✅ PA State protocols loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load PA protocols: {e}")
            print("   The system will still work for basic analysis")
        
    def _load_test_cases(self) -> List[TestCase]:
        """Load comprehensive test cases covering various emergency scenarios"""
        
        test_cases = [
            # Cardiac Emergencies
            TestCase(
                id="cardiac_001",
                narrative="Patient experiencing severe chest pain radiating to left arm, shortness of breath, diaphoretic",
                expected_protocols=["PA Chest Pain Protocol"],
                expected_urgency="Critical",
                expected_keywords=["chest pain", "shortness of breath", "diaphoretic"],
                category="Cardiac",
                description="Classic cardiac symptoms"
            ),
            TestCase(
                id="cardiac_002",
                narrative="Patient with history of heart disease complaining of pressure in chest, feeling lightheaded",
                expected_protocols=["PA Chest Pain Protocol"],
                expected_urgency="High",
                expected_keywords=["chest", "pressure", "lightheaded"],
                category="Cardiac",
                description="Atypical cardiac presentation"
            ),
            
            # Respiratory Emergencies
            TestCase(
                id="respiratory_001",
                narrative="Patient unable to breathe, wheezing, using accessory muscles, SpO2 88%",
                expected_protocols=["PA Respiratory Distress Protocol"],
                expected_urgency="Critical",
                expected_keywords=["unable to breathe", "wheezing", "accessory muscles"],
                category="Respiratory",
                description="Severe respiratory distress"
            ),
            TestCase(
                id="respiratory_002",
                narrative="Patient with COPD experiencing increased shortness of breath, coughing up green sputum",
                expected_protocols=["PA Respiratory Distress Protocol"],
                expected_urgency="High",
                expected_keywords=["copd", "shortness of breath", "coughing"],
                category="Respiratory",
                description="COPD exacerbation"
            ),
            
            # Neurological Emergencies
            TestCase(
                id="neuro_001",
                narrative="Patient with sudden facial droop, slurred speech, right arm weakness",
                expected_protocols=["PA Stroke Protocol"],
                expected_urgency="Critical",
                expected_keywords=["facial droop", "slurred speech", "weakness"],
                category="Neurological",
                description="Stroke symptoms"
            ),
            TestCase(
                id="neuro_002",
                narrative="Patient found unconscious, not responding to verbal commands, GCS 6",
                expected_protocols=["PA Stroke Protocol"],
                expected_urgency="Critical",
                expected_keywords=["unconscious", "not responding", "gcs"],
                category="Neurological",
                description="Altered mental status"
            ),
            
            # Trauma Emergencies
            TestCase(
                id="trauma_001",
                narrative="Patient involved in MVC, bleeding from head wound, complaining of neck pain",
                expected_protocols=["PA Trauma Protocol"],
                expected_urgency="Critical",
                expected_keywords=["mvc", "bleeding", "neck pain"],
                category="Trauma",
                description="Motor vehicle collision with injuries"
            ),
            TestCase(
                id="trauma_002",
                narrative="Patient fell from ladder, complaining of back pain, unable to move legs",
                expected_protocols=["PA Trauma Protocol"],
                expected_urgency="High",
                expected_keywords=["fell", "back pain", "unable to move"],
                category="Trauma",
                description="Fall with potential spinal injury"
            ),
            
            # Medical Emergencies
            TestCase(
                id="medical_001",
                narrative="Patient with diabetes, blood glucose 45, confused and diaphoretic",
                expected_protocols=["PA Diabetic Emergency Protocol"],
                expected_urgency="High",
                expected_keywords=["diabetes", "blood glucose", "confused"],
                category="Medical",
                description="Hypoglycemic episode"
            ),
            TestCase(
                id="medical_002",
                narrative="Patient experiencing seizure, tonic-clonic movements, post-ictal confusion",
                expected_protocols=["PA Seizure Protocol"],
                expected_urgency="High",
                expected_keywords=["seizure", "tonic-clonic", "post-ictal"],
                category="Medical",
                description="Active seizure"
            ),
            
            # Pediatric Emergencies
            TestCase(
                id="pediatric_001",
                narrative="2-year-old child with high fever, lethargic, not eating or drinking",
                expected_protocols=["PA Pediatric Protocol"],
                expected_urgency="High",
                expected_keywords=["child", "fever", "lethargic"],
                category="Pediatric",
                description="Pediatric fever with dehydration"
            ),
            
            # Obstetric Emergencies
            TestCase(
                id="obstetric_001",
                narrative="Pregnant patient at 38 weeks, contractions every 3 minutes, water broke",
                expected_protocols=["PA Obstetric Emergency Protocol"],
                expected_urgency="High",
                expected_keywords=["pregnant", "contractions", "water broke"],
                category="Obstetric",
                description="Active labor"
            ),
            
            # Behavioral Emergencies
            TestCase(
                id="behavioral_001",
                narrative="Patient threatening harm to self and others, agitated, refusing medication",
                expected_protocols=["PA Behavioral Emergency Protocol"],
                expected_urgency="High",
                expected_keywords=["threatening", "agitated", "refusing"],
                category="Behavioral",
                description="Psychiatric emergency"
            ),
            
            # Low Acuity Cases
            TestCase(
                id="low_acuity_001",
                narrative="Patient with minor ankle sprain, able to bear weight, no deformity",
                expected_protocols=[],
                expected_urgency="Low",
                expected_keywords=["ankle sprain", "minor"],
                category="Low Acuity",
                description="Minor injury"
            ),
            
            # Complex Cases
            TestCase(
                id="complex_001",
                narrative="Elderly patient with chest pain, diabetes, COPD, and recent stroke history",
                expected_protocols=["PA Chest Pain Protocol", "PA Diabetic Emergency Protocol"],
                expected_urgency="Critical",
                expected_keywords=["chest pain", "diabetes", "copd", "stroke"],
                category="Complex",
                description="Multi-morbidity case"
            )
        ]
        
        return test_cases
    
    def run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single test case and return results"""
        
        start_time = time.time()
        
        try:
            # Analyze the narrative directly using the enhanced system
            analysis = self.triage_system.analyze_ems_narrative(test_case.narrative, f"test_{test_case.id}")
            
            response_time = time.time() - start_time
            
            # Extract results
            actual_protocols = [p["protocol_name"] for p in analysis.get("protocol_matches", [])]
            actual_urgency = analysis.get("urgency_level", "Unknown")
            actual_keywords = [kw[0] for kw in analysis.get("keywords", [])]
            
            # Calculate accuracy metrics
            protocol_accuracy = self._calculate_protocol_accuracy(
                actual_protocols, test_case.expected_protocols
            )
            urgency_accuracy = actual_urgency == test_case.expected_urgency
            keyword_overlap = self._calculate_keyword_overlap(
                actual_keywords, test_case.expected_keywords
            )
            
            return TestResult(
                test_case=test_case,
                actual_protocols=actual_protocols,
                actual_urgency=actual_urgency,
                actual_keywords=actual_keywords,
                protocol_accuracy=protocol_accuracy,
                urgency_accuracy=urgency_accuracy,
                keyword_overlap=keyword_overlap,
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            print(f"Error testing case {test_case.id}: {e}")
            return TestResult(
                test_case=test_case,
                actual_protocols=[],
                actual_urgency="Unknown",
                actual_keywords=[],
                protocol_accuracy=0.0,
                urgency_accuracy=False,
                keyword_overlap=0.0,
                response_time=time.time() - start_time,
                success=False
            )
    
    def _calculate_protocol_accuracy(self, actual: List[str], expected: List[str]) -> float:
        """Calculate protocol matching accuracy"""
        if not expected and not actual:
            return 1.0
        if not expected:
            return 0.0
        if not actual:
            return 0.0
        
        # Calculate precision and recall
        correct = len(set(actual) & set(expected))
        precision = correct / len(actual) if actual else 0
        recall = correct / len(expected) if expected else 0
        
        # Return F1 score
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_keyword_overlap(self, actual: List[str], expected: List[str]) -> float:
        """Calculate keyword overlap percentage"""
        if not expected and not actual:
            return 1.0
        if not expected:
            return 0.0
        if not actual:
            return 0.0
        
        # Convert to sets for case-insensitive comparison
        actual_set = {kw.lower() for kw in actual}
        expected_set = {kw.lower() for kw in expected}
        
        intersection = actual_set & expected_set
        union = actual_set | expected_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive testing and return detailed results"""
        
        print("🚑 Starting Enhanced Narrative Model Testing...")
        print("=" * 60)
        
        results = []
        category_results = {}
        
        for test_case in self.test_cases:
            print(f"Testing: {test_case.id} - {test_case.description}")
            result = self.run_single_test(test_case)
            results.append(result)
            
            # Group by category
            if result.test_case.category not in category_results:
                category_results[result.test_case.category] = []
            category_results[result.test_case.category].append(result)
        
        # Calculate overall metrics
        successful_tests = [r for r in results if r.success]
        
        if not successful_tests:
            return {"error": "No successful tests completed"}
        
        # Overall metrics
        overall_metrics = {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(results),
            "avg_response_time": statistics.mean([r.response_time for r in successful_tests]),
            "avg_protocol_accuracy": statistics.mean([r.protocol_accuracy for r in successful_tests]),
            "urgency_accuracy_rate": sum(1 for r in successful_tests if r.urgency_accuracy) / len(successful_tests),
            "avg_keyword_overlap": statistics.mean([r.keyword_overlap for r in successful_tests])
        }
        
        # Category-specific metrics
        category_metrics = {}
        for category, category_results_list in category_results.items():
            if not category_results_list:
                continue
                
            successful_category = [r for r in category_results_list if r.success]
            if not successful_category:
                continue
                
            category_metrics[category] = {
                "test_count": len(category_results_list),
                "success_count": len(successful_category),
                "success_rate": len(successful_category) / len(category_results_list),
                "avg_protocol_accuracy": statistics.mean([r.protocol_accuracy for r in successful_category]),
                "urgency_accuracy_rate": sum(1 for r in successful_category if r.urgency_accuracy) / len(successful_category),
                "avg_keyword_overlap": statistics.mean([r.keyword_overlap for r in successful_category]),
                "avg_response_time": statistics.mean([r.response_time for r in successful_category])
            }
        
        # Detailed results
        detailed_results = []
        for result in results:
            detailed_results.append({
                "test_id": result.test_case.id,
                "category": result.test_case.category,
                "description": result.test_case.description,
                "narrative": result.test_case.narrative,
                "expected_protocols": result.test_case.expected_protocols,
                "actual_protocols": result.actual_protocols,
                "expected_urgency": result.test_case.expected_urgency,
                "actual_urgency": result.actual_urgency,
                "expected_keywords": result.test_case.expected_keywords,
                "actual_keywords": result.actual_keywords,
                "protocol_accuracy": result.protocol_accuracy,
                "urgency_accuracy": result.urgency_accuracy,
                "keyword_overlap": result.keyword_overlap,
                "response_time": result.response_time,
                "success": result.success
            })
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": overall_metrics,
            "category_metrics": category_metrics,
            "detailed_results": detailed_results
        }
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print a comprehensive summary of test results"""
        
        print("\n" + "=" * 60)
        print("📊 ENHANCED TEST RESULTS SUMMARY")
        print("=" * 60)
        
        metrics = report["overall_metrics"]
        print(f"\n🎯 Overall Performance:")
        print(f"   • Total Tests: {metrics['total_tests']}")
        print(f"   • Success Rate: {metrics['success_rate']:.1%}")
        print(f"   • Avg Response Time: {metrics['avg_response_time']:.3f}s")
        print(f"   • Protocol Accuracy: {metrics['avg_protocol_accuracy']:.1%}")
        print(f"   • Urgency Accuracy: {metrics['urgency_accuracy_rate']:.1%}")
        print(f"   • Keyword Overlap: {metrics['avg_keyword_overlap']:.1%}")
        
        print(f"\n📈 Category Performance:")
        for category, cat_metrics in report["category_metrics"].items():
            print(f"   • {category}:")
            print(f"     - Success Rate: {cat_metrics['success_rate']:.1%}")
            print(f"     - Protocol Accuracy: {cat_metrics['avg_protocol_accuracy']:.1%}")
            print(f"     - Urgency Accuracy: {cat_metrics['urgency_accuracy_rate']:.1%}")
            print(f"     - Avg Response Time: {cat_metrics['avg_response_time']:.3f}s")
        
        print(f"\n✅ Enhanced Model Performance Assessment:")
        if metrics['avg_protocol_accuracy'] >= 0.85:
            print("   • Protocol Matching: EXCELLENT")
        elif metrics['avg_protocol_accuracy'] >= 0.75:
            print("   • Protocol Matching: GOOD")
        else:
            print("   • Protocol Matching: NEEDS IMPROVEMENT")
            
        if metrics['urgency_accuracy_rate'] >= 0.85:
            print("   • Urgency Assessment: EXCELLENT")
        elif metrics['urgency_accuracy_rate'] >= 0.75:
            print("   • Urgency Assessment: GOOD")
        else:
            print("   • Urgency Assessment: NEEDS IMPROVEMENT")
            
        if metrics['avg_keyword_overlap'] >= 0.70:
            print("   • Keyword Recognition: EXCELLENT")
        elif metrics['avg_keyword_overlap'] >= 0.50:
            print("   • Keyword Recognition: GOOD")
        else:
            print("   • Keyword Recognition: NEEDS IMPROVEMENT")
        
        print("\n" + "=" * 60)

def main():
    """Main testing function"""
    
    print("🚑 Enhanced EMS AI Triage System - Narrative Model Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = EnhancedNarrativeModelTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_narrative_model_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("\n🎉 Enhanced testing completed!")

if __name__ == "__main__":
    main() 