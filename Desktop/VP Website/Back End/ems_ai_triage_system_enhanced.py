#!/usr/bin/env python3
"""
Enhanced EMS AI Triage System - Medical Standards Optimized
==========================================================

This enhanced system provides improved accuracy for medical triage with:
- Expanded medical terminology and synonyms
- Sophisticated urgency assessment algorithms
- Enhanced protocol matching with fuzzy logic
- Medical standards compliance
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import logging
from datetime import datetime
import pickle
import os
from difflib import SequenceMatcher

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEMSKeywordAssociation:
    """
    Enhanced keyword association system with medical standards optimization
    """
    
    def __init__(self):
        self.vectorizer = None
        self.medical_synonyms = self._load_medical_synonyms()
        self.urgency_keywords = self._load_urgency_keywords()
        self.vital_signs_patterns = self._load_vital_signs_patterns()
        self.ems_stop_words = self._load_ems_stop_words()
        
        # Initialize NLP components
        self._initialize_nlp()
        
    def _load_medical_synonyms(self) -> Dict[str, List[str]]:
        """Load comprehensive medical synonyms"""
        return {
            # Cardiac terms
            'chest pain': ['cp', 'chest pressure', 'chest discomfort', 'angina', 'cardiac pain'],
            'heart attack': ['mi', 'myocardial infarction', 'cardiac arrest', 'heart failure'],
            'shortness of breath': ['sob', 'dyspnea', 'breathing difficulty', 'respiratory distress'],
            'diaphoretic': ['sweating', 'sweaty', 'perspiring', 'clammy'],
            
            # Respiratory terms
            'wheezing': ['wheeze', 'whistling', 'bronchospasm'],
            'coughing': ['cough', 'productive cough', 'dry cough'],
            'copd': ['chronic obstructive pulmonary disease', 'emphysema', 'chronic bronchitis'],
            'asthma': ['asthmatic', 'bronchial asthma'],
            
            # Neurological terms
            'stroke': ['cva', 'cerebrovascular accident', 'brain attack'],
            'facial droop': ['facial weakness', 'drooping face', 'bell palsy'],
            'slurred speech': ['dysarthria', 'speech difficulty', 'unclear speech'],
            'unconscious': ['unresponsive', 'coma', 'altered mental status', 'ams'],
            'seizure': ['convulsion', 'epileptic', 'tonic-clonic', 'grand mal'],
            
            # Trauma terms
            'bleeding': ['hemorrhage', 'blood loss', 'laceration', 'wound'],
            'fracture': ['broken bone', 'broken', 'fx'],
            'trauma': ['injury', 'accident', 'mvc', 'motor vehicle collision'],
            'neck pain': ['cervical pain', 'neck stiffness', 'whiplash'],
            
            # Medical terms
            'diabetes': ['diabetic', 'dm', 'diabetes mellitus'],
            'hypertension': ['htn', 'high blood pressure', 'hypertensive'],
            'hypoglycemia': ['low blood sugar', 'insulin reaction', 'diabetic emergency'],
            'dka': ['diabetic ketoacidosis', 'ketoacidosis'],
            
            # Pediatric terms
            'child': ['pediatric', 'kid', 'infant', 'baby', 'toddler'],
            'fever': ['pyrexia', 'elevated temperature', 'hyperthermia'],
            'lethargic': ['lethargy', 'tired', 'weak', 'listless'],
            
            # Obstetric terms
            'pregnant': ['pregnancy', 'gestation', 'obstetric'],
            'labor': ['contractions', 'delivery', 'childbirth', 'parturition'],
            'crowning': ['baby crowning', 'delivery imminent'],
            
            # Behavioral terms
            'agitated': ['agitation', 'restless', 'combative', 'violent'],
            'psychiatric': ['mental health', 'behavioral', 'psych', 'psychiatric emergency'],
            'suicidal': ['suicide', 'self-harm', 'self injury'],
            
            # Vital signs
            'blood pressure': ['bp', 'pressure', 'systolic', 'diastolic'],
            'heart rate': ['hr', 'pulse', 'bpm', 'heartbeat'],
            'respiratory rate': ['rr', 'breathing rate', 'respirations'],
            'oxygen saturation': ['spo2', 'o2 sat', 'oxygen level', 'sat'],
            'temperature': ['temp', 'fever', 'pyrexia'],
            'glucose': ['blood sugar', 'bg', 'glu', 'sugar level']
        }
    
    def _load_urgency_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load urgency keywords with severity levels"""
        return {
            'critical': {
                'cardiac': ['chest pain', 'heart attack', 'cardiac arrest', 'mi', 'stemi', 'unstable angina'],
                'respiratory': ['respiratory arrest', 'unable to breathe', 'respiratory failure', 'severe dyspnea'],
                'neurological': ['stroke', 'seizure', 'unconscious', 'coma', 'altered mental status'],
                'trauma': ['major bleeding', 'severe trauma', 'penetrating injury', 'amputation'],
                'medical': ['diabetic emergency', 'severe allergic reaction', 'anaphylaxis', 'overdose']
            },
            'high': {
                'cardiac': ['chest pressure', 'shortness of breath', 'palpitations', 'irregular heartbeat'],
                'respiratory': ['wheezing', 'coughing', 'copd exacerbation', 'asthma attack'],
                'neurological': ['facial droop', 'slurred speech', 'weakness', 'numbness'],
                'trauma': ['fracture', 'dislocation', 'moderate bleeding', 'head injury'],
                'medical': ['diabetes', 'hypertension', 'infection', 'dehydration']
            },
            'moderate': {
                'cardiac': ['mild chest discomfort', 'fatigue', 'dizziness'],
                'respiratory': ['mild shortness of breath', 'cough', 'congestion'],
                'neurological': ['headache', 'dizziness', 'mild confusion'],
                'trauma': ['minor injury', 'bruising', 'sprain', 'minor laceration'],
                'medical': ['mild fever', 'nausea', 'vomiting', 'abdominal pain']
            },
            'low': {
                'cardiac': ['mild fatigue', 'anxiety', 'mild discomfort'],
                'respiratory': ['mild congestion', 'sore throat', 'mild cough'],
                'neurological': ['mild headache', 'anxiety', 'mild dizziness'],
                'trauma': ['minor abrasion', 'small cut', 'minor injury', 'sprain'],
                'medical': ['mild symptoms', 'routine care', 'minor complaint']
            }
        }
    
    def _load_vital_signs_patterns(self) -> Dict[str, str]:
        """Load comprehensive vital signs patterns"""
        return {
            'blood_pressure': r'(\d{2,3}/\d{2,3})\s*(?:bp|blood pressure|pressure)',
            'heart_rate': r'(\d{2,3})\s*(?:bpm|hr|heart rate|pulse)',
            'temperature': r'(\d{2,3}\.?\d*)\s*(?:f|fahrenheit|temp|temperature)',
            'oxygen_saturation': r'(\d{2,3})\s*(?:spo2|o2 sat|oxygen|sat)',
            'respiratory_rate': r'(\d{1,2})\s*(?:rr|respiratory rate|breathing rate)',
            'glucose': r'(\d{2,3})\s*(?:glucose|bg|blood sugar|sugar)',
            'gcs': r'(\d{1,2})\s*(?:gcs|glasgow)',
            'pain_scale': r'(\d{1,2})\s*(?:pain|scale)'
        }
    
    def _load_ems_stop_words(self) -> set:
        """Load EMS-specific stop words"""
        return {
            'patient', 'pt', 'ambulance', 'ems', 'paramedic', 'emergency', 'scene',
            'transport', 'hospital', 'vital', 'signs', 'assessment', 'treatment',
            'medication', 'dose', 'administered', 'protocol', 'guideline', 'standard',
            'procedure', 'documentation', 'reported', 'stated', 'complains', 'complaining',
            'found', 'discovered', 'observed', 'noted', 'appears', 'seems', 'looks'
        }
    
    def _initialize_nlp(self):
        """Initialize NLP components"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            logger.warning("NLTK downloads failed, continuing with available resources")
        
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')).union(self.ems_stop_words)
    
    def expand_medical_terms(self, text: str) -> str:
        """Expand medical abbreviations and synonyms"""
        expanded_text = text.lower()
        
        for term, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                # Replace synonyms with standard term
                expanded_text = re.sub(r'\b' + re.escape(synonym) + r'\b', term, expanded_text)
        
        return expanded_text
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with medical terminology expansion"""
        if not text:
            return ""
        
        # Expand medical terms first
        text = self.expand_medical_terms(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize the token
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def extract_keywords(self, text: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """Enhanced keyword extraction with medical context awareness"""
        if not text:
            return []
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return []
        
        # Create TF-IDF vectorizer if not exists
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 3),  # Include bigrams and trigrams
                min_df=1,
                max_df=1.0
            )
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform([processed_text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get scores
        scores = tfidf_matrix.toarray()[0]
        
        # Create keyword-score pairs
        keyword_scores = list(zip(feature_names, scores))
        
        # Sort by score and return top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_k]
    
    def assess_urgency_level(self, text: str) -> Tuple[str, float]:
        """Enhanced urgency assessment with medical standards"""
        text_lower = text.lower()
        urgency_scores = {'critical': 0, 'high': 0, 'moderate': 0, 'low': 0}
        
        # Check each urgency level and category
        for urgency_level, categories in self.urgency_keywords.items():
            for category, keywords in categories.items():
                for keyword in keywords:
                    # Use word boundary matching to avoid substring issues
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                        # Weight by urgency level
                        if urgency_level == 'critical':
                            urgency_scores['critical'] += 3
                        elif urgency_level == 'high':
                            urgency_scores['high'] += 2
                        elif urgency_level == 'moderate':
                            urgency_scores['moderate'] += 1
                        else:
                            urgency_scores['low'] += 0.5
        
        # Check vital signs for additional scoring
        vitals = self._extract_vital_signs(text)
        urgency_scores = self._adjust_urgency_with_vitals(urgency_scores, vitals)
        
        # Determine final urgency level
        max_score = max(urgency_scores.values())
        if max_score == 0:
            return 'low', 0.0
        
        # Find the urgency level with highest score
        for level in ['critical', 'high', 'moderate', 'low']:
            if urgency_scores[level] == max_score:
                # Return with proper capitalization for test framework
                if level == 'critical':
                    return 'Critical', max_score
                elif level == 'high':
                    return 'High', max_score
                elif level == 'moderate':
                    return 'Moderate', max_score
                else:
                    return 'Low', max_score
        
        return 'Low', 0.0
    
    def _extract_vital_signs(self, text: str) -> Dict[str, Any]:
        """Enhanced vital signs extraction"""
        vitals = {}
        
        for vital_type, pattern in self.vital_signs_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                vitals[vital_type] = matches[0] if len(matches) == 1 else matches
        
        return vitals
    
    def _adjust_urgency_with_vitals(self, urgency_scores: Dict[str, float], 
                                   vitals: Dict[str, Any]) -> Dict[str, float]:
        """Adjust urgency scores based on vital signs"""
        if 'blood_pressure' in vitals:
            bp = vitals['blood_pressure']
            if isinstance(bp, str) and '/' in bp:
                systolic, diastolic = bp.split('/')
                try:
                    sys_val, dia_val = int(systolic), int(diastolic)
                    if sys_val > 180 or dia_val > 110:
                        urgency_scores['critical'] += 2
                    elif sys_val > 160 or dia_val > 100:
                        urgency_scores['high'] += 1
                except ValueError:
                    pass
        
        if 'heart_rate' in vitals:
            try:
                hr = int(vitals['heart_rate'])
                if hr > 120:
                    urgency_scores['critical'] += 1
                elif hr > 100:
                    urgency_scores['high'] += 1
            except ValueError:
                pass
        
        if 'oxygen_saturation' in vitals:
            try:
                spo2 = int(vitals['oxygen_saturation'])
                if spo2 < 90:
                    urgency_scores['critical'] += 2
                elif spo2 < 95:
                    urgency_scores['high'] += 1
            except ValueError:
                pass
        
        if 'glucose' in vitals:
            try:
                glucose = int(vitals['glucose'])
                if glucose < 60 or glucose > 400:
                    urgency_scores['critical'] += 2
                elif glucose < 80 or glucose > 300:
                    urgency_scores['high'] += 1
            except ValueError:
                pass
        
        return urgency_scores
    
    def find_protocol_matches(self, text: str, protocols: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced protocol matching with fuzzy logic"""
        matches = []
        text_lower = text.lower()
        
        for protocol_name, protocol_data in protocols.items():
            match_score = 0.0
            matched_keywords = []
            
            # Check protocol keywords
            protocol_keywords = protocol_data.get('keywords', [])
            for keyword in protocol_keywords:
                # Exact match
                if keyword.lower() in text_lower:
                    match_score += 1.0
                    matched_keywords.append(keyword)
                # Fuzzy match for similar terms
                else:
                    similarity = self._calculate_similarity(keyword.lower(), text_lower)
                    if similarity > 0.8:
                        match_score += 0.8
                        matched_keywords.append(keyword)
            
            # Normalize score
            if protocol_keywords:
                match_score = match_score / len(protocol_keywords)
            
            if match_score > 0.1:  # Lower threshold for better recall
                matches.append({
                    'protocol_name': protocol_name,
                    'match_score': match_score,
                    'matched_keywords': matched_keywords
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches
    
    def _calculate_similarity(self, keyword: str, text: str) -> float:
        """Calculate similarity between keyword and text"""
        words = text.split()
        max_similarity = 0.0
        
        for word in words:
            similarity = SequenceMatcher(None, keyword, word).ratio()
            max_similarity = max(max_similarity, similarity)
        
        return max_similarity


class EnhancedEMSTriageSystem:
    """
    Enhanced EMS Triage System with medical standards optimization
    """
    
    def __init__(self):
        self.keyword_analyzer = EnhancedEMSKeywordAssociation()
        self.protocol_database = {}
        self.analysis_history = []
        
    def add_protocol_data(self, protocol_name: str, protocol_text: str, 
                          keywords: List[str] = None) -> None:
        """Add protocol data to the system"""
        self.protocol_database[protocol_name] = {
            'text': protocol_text,
            'keywords': keywords or [],
            'analysis': self.keyword_analyzer.analyze_narrative(protocol_text) if hasattr(self.keyword_analyzer, 'analyze_narrative') else {}
        }
    
    def analyze_ems_narrative(self, narrative: str, patient_id: str = None) -> Dict[str, Any]:
        """Enhanced narrative analysis with improved accuracy"""
        # Extract keywords
        keywords = self.keyword_analyzer.extract_keywords(narrative, top_k=15)
        
        # Assess urgency level
        urgency_level, urgency_score = self.keyword_analyzer.assess_urgency_level(narrative)
        
        # Find protocol matches
        protocol_matches = self.keyword_analyzer.find_protocol_matches(narrative, self.protocol_database)
        
        # Calculate triage score
        triage_score = self._calculate_enhanced_triage_score(narrative, keywords, urgency_score)
        
        # Generate recommendations
        recommendations = self._generate_enhanced_recommendations(narrative, keywords, protocol_matches, urgency_level)
        
        # Create comprehensive analysis
        analysis = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "narrative": narrative,
            "keywords": keywords,
            "urgency_level": urgency_level,
            "urgency_score": urgency_score,
            "protocol_matches": protocol_matches,
            "triage_score": triage_score,
            "recommendations": recommendations
        }
        
        # Store in history
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _calculate_enhanced_triage_score(self, narrative: str, keywords: List[Tuple[str, float]], 
                                       urgency_score: float) -> float:
        """Calculate enhanced triage score (0-10)"""
        score = 0.0
        
        # Base score from urgency
        score += urgency_score * 2
        
        # Add points for keyword density
        keyword_density = len(keywords) / max(len(narrative.split()), 1)
        score += keyword_density * 5
        
        # Add points for medical terminology
        medical_terms = sum(1 for kw, _ in keywords if any(term in kw for term in self.keyword_analyzer.medical_synonyms.keys()))
        score += medical_terms * 0.5
        
        # Add points for vital signs
        vitals = self.keyword_analyzer._extract_vital_signs(narrative)
        if vitals:
            score += len(vitals) * 0.3
        
        # Cap at 10
        return min(score, 10.0)
    
    def _generate_enhanced_recommendations(self, narrative: str, keywords: List[Tuple[str, float]], 
                                         protocol_matches: List[Dict[str, Any]], 
                                         urgency_level: str) -> List[str]:
        """Generate enhanced EMS protocol recommendations"""
        recommendations = []
        narrative_lower = narrative.lower()
        
        # Respiratory Distress Protocols
        respiratory_keywords = ['respiratory distress', 'wheezing', 'asthma', 'copd', 'shortness of breath', 'dyspnea', 'tripoding', 'accessory muscles', 'nasal flaring']
        if any(keyword in narrative_lower for keyword in respiratory_keywords):
            recommendations.append("🫁 **Oxygen Therapy**: Apply NRB mask at 15 LPM or nasal cannula at 2-6 LPM")
            recommendations.append("🫁 **CPAP Consideration**: Evaluate for CPAP if patient meets criteria")
            recommendations.append("🫁 **Assisted Inhaler**: Administer albuterol via nebulizer or MDI")
            recommendations.append("🫁 **Positioning**: Keep patient in position of comfort (tripoding)")
            recommendations.append("🫁 **Monitor**: Continuous pulse oximetry and respiratory rate")
            
            # Check for severe respiratory distress indicators
            severe_indicators = ['unable to speak', 'tripoding', 'accessory muscles', 'spo2 < 90%', 'respiratory rate > 30']
            if any(indicator in narrative_lower for indicator in severe_indicators):
                recommendations.append("🚨 **ALS Activation**: Consider immediate ALS for severe respiratory distress")
                recommendations.append("🚨 **Advanced Airway**: Prepare for potential intubation")
        
        # Cardiac Protocols
        cardiac_keywords = ['chest pain', 'cardiac', 'heart attack', 'mi', 'angina', 'palpitations', 'chest pressure']
        # Only trigger cardiac protocols if not primarily respiratory
        respiratory_indicators = ['respiratory distress', 'wheezing', 'asthma', 'copd', 'shortness of breath', 'dyspnea', 'tripoding', 'accessory muscles', 'nasal flaring']
        is_primary_respiratory = any(indicator in narrative_lower for indicator in respiratory_indicators)
        
        if any(keyword in narrative_lower for keyword in cardiac_keywords) and not is_primary_respiratory:
            recommendations.append("❤️ **12-Lead ECG**: Obtain immediately")
            recommendations.append("❤️ **Aspirin**: Administer 325mg PO if no contraindications")
            recommendations.append("❤️ **Nitroglycerin**: Consider if systolic BP > 100")
            recommendations.append("❤️ **IV Access**: Establish 18g or larger IV")
            recommendations.append("❤️ **Monitor**: Continuous cardiac monitoring")
        
        # Neurological Protocols
        neuro_keywords = ['stroke', 'cva', 'facial droop', 'arm drift', 'speech difficulty', 'seizure', 'unconscious']
        if any(keyword in narrative_lower for keyword in neuro_keywords):
            recommendations.append("🧠 **Stroke Assessment**: Perform FAST exam")
            recommendations.append("🧠 **Glucose Check**: Blood glucose level")
            recommendations.append("🧠 **IV Access**: Establish IV for potential thrombolytics")
            recommendations.append("🧠 **Time Critical**: Note exact time of onset")
        
        # Trauma Protocols
        trauma_keywords = ['trauma', 'injury', 'bleeding', 'fracture', 'laceration', 'head injury']
        if any(keyword in narrative_lower for keyword in trauma_keywords):
            recommendations.append("🩸 **C-Spine Precautions**: Maintain if indicated")
            recommendations.append("🩸 **Hemorrhage Control**: Direct pressure, tourniquet if needed")
            recommendations.append("🩸 **IV Access**: Large bore IV for potential shock")
            recommendations.append("🩸 **Trauma Assessment**: Full trauma survey")
        
        # Vital Signs Based Recommendations
        vitals = self.keyword_analyzer._extract_vital_signs(narrative)
        if vitals:
            if vitals.get('spo2', 100) < 90:
                recommendations.append("📊 **Oxygen Therapy**: SpO2 < 90% - immediate oxygen required")
            if vitals.get('hr', 0) > 100:
                recommendations.append("📊 **Tachycardia**: Monitor for cardiac issues")
            if vitals.get('bp_systolic', 0) < 90:
                recommendations.append("📊 **Hypotension**: Consider fluid resuscitation")
            if vitals.get('rr', 0) > 20:
                recommendations.append("📊 **Tachypnea**: Monitor respiratory status closely")
        
        # Urgency-based general recommendations
        if urgency_level == "Critical":
            recommendations.append("🚨 **ALS Activation**: Immediate ALS response required")
            recommendations.append("🚨 **Rapid Transport**: Expedited transport to appropriate facility")
            recommendations.append("🚨 **Advanced Life Support**: Prepare for advanced interventions")
        elif urgency_level == "High":
            recommendations.append("⚠️ **Expedited Care**: Monitor closely, prepare for escalation")
            recommendations.append("⚠️ **ALS Consideration**: Evaluate need for ALS")
        elif urgency_level == "Moderate":
            recommendations.append("📋 **Standard Care**: Routine monitoring and assessment")
        else:
            recommendations.append("✅ **Routine Care**: Standard BLS care appropriate")
        
        # Protocol-specific recommendations
        if protocol_matches:
            top_match = protocol_matches[0]
            if top_match['match_score'] > 0.3:
                recommendations.append(
                    f"📋 **Protocol Match**: Follow {top_match['protocol_name']} protocol "
                    f"(confidence: {top_match['match_score']:.1%})"
                )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Initialize enhanced system
    enhanced_system = EnhancedEMSTriageSystem()
    
    # Test with example narrative
    test_narrative = """
    Patient experiencing severe chest pain radiating to left arm, shortness of breath, diaphoretic.
    Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
    Patient has history of hypertension and diabetes.
    """
    
    # Analyze the narrative
    analysis_result = enhanced_system.analyze_ems_narrative(test_narrative, "PT001")
    
    # Print results
    print("=== Enhanced EMS AI Triage System Analysis ===")
    print(f"Patient ID: {analysis_result['patient_id']}")
    print(f"Urgency Level: {analysis_result['urgency_level']}")
    print(f"Urgency Score: {analysis_result['urgency_score']:.2f}")
    print(f"Triage Score: {analysis_result['triage_score']:.2f}")
    print(f"Keywords: {[kw[0] for kw in analysis_result['keywords'][:5]]}")
    print(f"Protocol Matches: {len(analysis_result['protocol_matches'])}")
    print("Recommendations:")
    for rec in analysis_result['recommendations']:
        print(f"  - {rec}") 