#!/usr/bin/env python3
"""
EMS AI Triage System - Keyword Association Model
================================================

This system provides machine learning-based keyword association analysis
for EMS protocol data and narrative analysis.

Features:
- Keyword extraction and association
- Natural language processing for EMS narratives
- Protocol-based triage recommendations
- Machine learning model training and inference
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

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud  # Optional - uncomment if needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMSKeywordAssociation:
    """
    Keyword association system for EMS protocol analysis
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.vectorizer = None
        self.keyword_model = None
        self.protocol_keywords = {}
        self.narrative_analyzer = None
        self.triage_classifier = None
        self.model_path = model_path or "ems_models/"
        
        # EMS-specific stop words
        self.ems_stop_words = {
            'patient', 'pt', 'patient', 'ambulance', 'ems', 'paramedic', 
            'emergency', 'scene', 'transport', 'hospital', 'vital', 'signs',
            'assessment', 'treatment', 'medication', 'dose', 'administered',
            'protocol', 'guideline', 'standard', 'procedure', 'documentation'
        }
        
        # Initialize NLP components
        self._initialize_nlp()
        
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
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for keyword analysis
        
        Args:
            text: Raw text input
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
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
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from text using TF-IDF
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            return []
        
        # Create TF-IDF vectorizer if not exists
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,  # Changed from 2 to 1 for single documents
                max_df=1.0  # Changed from 0.95 to 1.0 for single documents
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
    
    def find_keyword_associations(self, target_keyword: str, corpus: List[str], 
                                min_association: float = 0.3) -> Dict[str, float]:
        """
        Find associations for a specific keyword
        
        Args:
            target_keyword: The keyword to find associations for
            corpus: List of documents to analyze
            min_association: Minimum association strength
            
        Returns:
            Dictionary of associated keywords and their scores
        """
        if not corpus:
            return {}
        
        # Preprocess corpus
        processed_corpus = [self.preprocess_text(doc) for doc in corpus]
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        # Find target keyword index
        try:
            target_idx = list(feature_names).index(target_keyword.lower())
        except ValueError:
            logger.warning(f"Target keyword '{target_keyword}' not found in corpus")
            return {}
        
        # Calculate cosine similarities
        target_vector = tfidf_matrix[:, target_idx].toarray().flatten()
        similarities = cosine_similarity(tfidf_matrix, target_vector.reshape(1, -1)).flatten()
        
        # Find associated keywords
        associations = {}
        for i, doc_similarity in enumerate(similarities):
            if doc_similarity > min_association:
                doc_vector = tfidf_matrix[i].toarray().flatten()
                for j, score in enumerate(doc_vector):
                    if score > 0 and j != target_idx:
                        keyword = feature_names[j]
                        associations[keyword] = max(associations.get(keyword, 0), score * doc_similarity)
        
        return associations
    
    def train_protocol_classifier(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train a classifier for protocol-based triage
        
        Args:
            training_data: List of dictionaries with 'text' and 'protocol' keys
        """
        if not training_data:
            logger.error("No training data provided")
            return
        
        # Extract features and labels
        texts = [item['text'] for item in training_data]
        labels = [item['protocol'] for item in training_data]
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train multiple classifiers
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        best_score = 0
        best_classifier = None
        
        for name, classifier in classifiers.items():
            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)
            logger.info(f"{name} accuracy: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_classifier = classifier
        
        self.triage_classifier = best_classifier
        logger.info(f"Best classifier accuracy: {best_score:.3f}")
    
    def analyze_narrative(self, narrative: str) -> Dict[str, Any]:
        """
        Analyze EMS narrative for keyword associations and triage recommendations
        
        Args:
            narrative: EMS narrative text
            
        Returns:
            Analysis results dictionary
        """
        if not narrative:
            return {"error": "No narrative provided"}
        
        # Extract keywords
        keywords = self.extract_keywords(narrative, top_k=15)
        
        # Analyze sentiment and urgency indicators
        urgency_indicators = self._detect_urgency_indicators(narrative)
        
        # Extract vital signs and measurements
        vital_signs = self._extract_vital_signs(narrative)
        
        # Identify potential protocols
        suggested_protocols = self._suggest_protocols(narrative)
        
        # Create analysis result
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "keywords": keywords,
            "urgency_indicators": urgency_indicators,
            "vital_signs": vital_signs,
            "suggested_protocols": suggested_protocols,
            "confidence_score": self._calculate_confidence_score(narrative),
            "recommendations": self._generate_recommendations(narrative, keywords)
        }
        
        return analysis
    
    def _detect_urgency_indicators(self, text: str) -> Dict[str, Any]:
        """Detect urgency indicators in narrative"""
        urgency_keywords = {
            'critical': ['critical', 'severe', 'emergency', 'urgent', 'immediate'],
            'cardiac': ['chest pain', 'cardiac', 'heart attack', 'mi', 'stemi'],
            'trauma': ['trauma', 'injury', 'bleeding', 'fracture', 'laceration'],
            'neurological': ['stroke', 'seizure', 'altered', 'unresponsive', 'coma'],
            'respiratory': ['difficulty breathing', 'shortness of breath', 'respiratory distress']
        }
        
        indicators = {}
        text_lower = text.lower()
        
        for category, keywords in urgency_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                indicators[category] = found_keywords
        
        return indicators
    
    def _extract_vital_signs(self, text: str) -> Dict[str, Any]:
        """Extract vital signs from narrative"""
        vital_patterns = {
            'blood_pressure': r'(\d{2,3}/\d{2,3})\s*(?:bp|blood pressure)',
            'heart_rate': r'(\d{2,3})\s*(?:bpm|hr|heart rate)',
            'temperature': r'(\d{2,3}\.?\d*)\s*(?:f|fahrenheit|temp)',
            'oxygen_saturation': r'(\d{2,3})\s*(?:spo2|o2 sat|oxygen)',
            'respiratory_rate': r'(\d{1,2})\s*(?:rr|respiratory rate)'
        }
        
        vitals = {}
        for vital_type, pattern in vital_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                vitals[vital_type] = matches[0] if len(matches) == 1 else matches
        
        return vitals
    
    def _suggest_protocols(self, text: str) -> List[str]:
        """Suggest relevant protocols based on narrative"""
        protocol_keywords = {
            'cardiac': ['chest pain', 'cardiac', 'heart', 'ecg', 'ekg'],
            'stroke': ['stroke', 'facial droop', 'arm drift', 'speech'],
            'trauma': ['trauma', 'injury', 'bleeding', 'fracture'],
            'respiratory': ['breathing', 'respiratory', 'asthma', 'copd'],
            'pediatric': ['child', 'pediatric', 'infant', 'baby'],
            'obstetric': ['pregnant', 'labor', 'delivery', 'obstetric']
        }
        
        text_lower = text.lower()
        suggested = []
        
        for protocol, keywords in protocol_keywords.items():
            if any(kw in text_lower for kw in keywords):
                suggested.append(protocol)
        
        return suggested
    
    def _calculate_confidence_score(self, text: str) -> float:
        """Calculate confidence score for analysis"""
        # Simple heuristic based on text length and keyword density
        keywords = self.extract_keywords(text, top_k=20)
        keyword_density = len(keywords) / max(len(text.split()), 1)
        
        # Normalize to 0-1 scale
        confidence = min(keyword_density * 10, 1.0)
        return round(confidence, 3)
    
    def _generate_recommendations(self, text: str, keywords: List[Tuple[str, float]]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check for critical keywords
        critical_keywords = ['chest pain', 'stroke', 'trauma', 'bleeding', 'unresponsive']
        text_lower = text.lower()
        
        for keyword in critical_keywords:
            if keyword in text_lower:
                recommendations.append(f"Immediate assessment required for {keyword}")
        
        # Check vital signs
        vitals = self._extract_vital_signs(text)
        if 'blood_pressure' in vitals:
            bp = vitals['blood_pressure']
            if isinstance(bp, str) and '/' in bp:
                systolic, diastolic = bp.split('/')
                if int(systolic) > 180 or int(diastolic) > 110:
                    recommendations.append("Hypertensive crisis - monitor closely")
        
        # General recommendations
        if len(keywords) < 5:
            recommendations.append("Consider additional assessment for comprehensive evaluation")
        
        return recommendations
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'triage_classifier': self.triage_classifier,
            'protocol_keywords': self.protocol_keywords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.triage_classifier = model_data['triage_classifier']
        self.protocol_keywords = model_data['protocol_keywords']
        
        logger.info(f"Model loaded from {filepath}")


class EMSTriageSystem:
    """
    Main EMS Triage System that integrates keyword association with triage logic
    """
    
    def __init__(self):
        self.keyword_analyzer = EMSKeywordAssociation()
        self.protocol_database = {}
        self.analysis_history = []
        
    def add_protocol_data(self, protocol_name: str, protocol_text: str, 
                          keywords: List[str] = None) -> None:
        """
        Add protocol data to the system
        
        Args:
            protocol_name: Name of the protocol
            protocol_text: Full text of the protocol
            keywords: Optional list of key terms for the protocol
        """
        self.protocol_database[protocol_name] = {
            'text': protocol_text,
            'keywords': keywords or [],
            'analysis': self.keyword_analyzer.analyze_narrative(protocol_text)
        }
    
    def analyze_ems_narrative(self, narrative: str, patient_id: str = None) -> Dict[str, Any]:
        """
        Analyze EMS narrative and provide triage recommendations
        
        Args:
            narrative: EMS narrative text
            patient_id: Optional patient identifier
            
        Returns:
            Comprehensive analysis results
        """
        # Perform keyword analysis
        keyword_analysis = self.keyword_analyzer.analyze_narrative(narrative)
        
        # Match against protocols
        protocol_matches = self._match_protocols(narrative)
        
        # Generate triage score
        triage_score = self._calculate_triage_score(narrative, keyword_analysis)
        
        # Create comprehensive analysis
        analysis = {
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),
            "narrative": narrative,
            "keyword_analysis": keyword_analysis,
            "protocol_matches": protocol_matches,
            "triage_score": triage_score,
            "urgency_level": self._determine_urgency_level(triage_score),
            "recommendations": self._generate_triage_recommendations(
                narrative, keyword_analysis, protocol_matches
            )
        }
        
        # Store in history
        self.analysis_history.append(analysis)
        
        return analysis
    
    def _match_protocols(self, narrative: str) -> List[Dict[str, Any]]:
        """Match narrative against available protocols"""
        matches = []
        narrative_lower = narrative.lower()
        
        for protocol_name, protocol_data in self.protocol_database.items():
            # Calculate similarity score
            protocol_text = protocol_data['text'].lower()
            
            # Simple keyword matching
            protocol_keywords = protocol_data['keywords']
            keyword_matches = [kw for kw in protocol_keywords if kw.lower() in narrative_lower]
            
            if keyword_matches:
                match_score = len(keyword_matches) / len(protocol_keywords) if protocol_keywords else 0
                matches.append({
                    'protocol_name': protocol_name,
                    'match_score': match_score,
                    'matched_keywords': keyword_matches
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches
    
    def _calculate_triage_score(self, narrative: str, keyword_analysis: Dict[str, Any]) -> float:
        """Calculate triage urgency score (0-10)"""
        score = 0.0
        
        # Base score from keyword analysis confidence
        score += keyword_analysis.get('confidence_score', 0) * 2
        
        # Add points for urgency indicators
        urgency_indicators = keyword_analysis.get('urgency_indicators', {})
        for category, indicators in urgency_indicators.items():
            if category in ['critical', 'cardiac', 'trauma']:
                score += len(indicators) * 1.5
            else:
                score += len(indicators) * 0.5
        
        # Add points for vital signs abnormalities
        vitals = keyword_analysis.get('vital_signs', {})
        if vitals:
            score += 1.0
        
        # Cap at 10
        return min(score, 10.0)
    
    def _determine_urgency_level(self, triage_score: float) -> str:
        """Determine urgency level based on triage score"""
        if triage_score >= 8.0:
            return "CRITICAL"
        elif triage_score >= 6.0:
            return "URGENT"
        elif triage_score >= 4.0:
            return "MODERATE"
        elif triage_score >= 2.0:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_triage_recommendations(self, narrative: str, 
                                       keyword_analysis: Dict[str, Any],
                                       protocol_matches: List[Dict[str, Any]]) -> List[str]:
        """Generate triage recommendations"""
        recommendations = []
        
        # Add keyword analysis recommendations
        recommendations.extend(keyword_analysis.get('recommendations', []))
        
        # Add protocol-specific recommendations
        if protocol_matches:
            top_match = protocol_matches[0]
            if top_match['match_score'] > 0.5:
                recommendations.append(
                    f"Consider following {top_match['protocol_name']} protocol "
                    f"(match score: {top_match['match_score']:.2f})"
                )
        
        # Add urgency-based recommendations
        urgency_level = self._determine_urgency_level(
            keyword_analysis.get('triage_score', 0)
        )
        
        if urgency_level == "CRITICAL":
            recommendations.append("Immediate medical attention required")
            recommendations.append("Consider rapid transport to trauma center")
        elif urgency_level == "URGENT":
            recommendations.append("Expedited evaluation recommended")
        
        return recommendations
    
    def get_analysis_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        return self.analysis_history[-limit:]
    
    def export_analysis_report(self, analysis_id: str = None) -> Dict[str, Any]:
        """Export comprehensive analysis report"""
        if analysis_id:
            # Find specific analysis
            for analysis in self.analysis_history:
                if analysis.get('patient_id') == analysis_id:
                    return self._format_report(analysis)
            return {"error": "Analysis not found"}
        else:
            # Return summary of all analyses
            return self._generate_summary_report()
    
    def _format_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format analysis into report structure"""
        return {
            "report_id": f"EMS_{analysis.get('patient_id', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "analysis": analysis,
            "summary": {
                "urgency_level": analysis.get('urgency_level'),
                "triage_score": analysis.get('triage_score'),
                "protocol_matches": len(analysis.get('protocol_matches', [])),
                "keywords_found": len(analysis.get('keyword_analysis', {}).get('keywords', []))
            }
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of all analyses"""
        if not self.analysis_history:
            return {"error": "No analysis history available"}
        
        urgency_counts = Counter(
            analysis.get('urgency_level', 'UNKNOWN') 
            for analysis in self.analysis_history
        )
        
        return {
            "report_type": "summary",
            "total_analyses": len(self.analysis_history),
            "urgency_distribution": dict(urgency_counts),
            "average_triage_score": np.mean([
                analysis.get('triage_score', 0) 
                for analysis in self.analysis_history
            ]),
            "recent_analyses": self.get_analysis_history(10)
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    triage_system = EMSTriageSystem()
    
    # Example protocol data (you would load this from the NJ state protocol document)
    example_protocols = {
        "Cardiac Protocol": {
            "text": "Chest pain protocol includes ECG monitoring, aspirin administration, and rapid transport to cardiac center.",
            "keywords": ["chest pain", "cardiac", "ecg", "aspirin", "transport"]
        },
        "Stroke Protocol": {
            "text": "Stroke assessment includes FAST evaluation, glucose check, and rapid transport to stroke center.",
            "keywords": ["stroke", "fast", "facial droop", "arm drift", "speech"]
        }
    }
    
    # Add protocols to system
    for protocol_name, protocol_data in example_protocols.items():
        triage_system.add_protocol_data(
            protocol_name, 
            protocol_data["text"], 
            protocol_data["keywords"]
        )
    
    # Example narrative analysis
    example_narrative = """
    Patient presents with severe chest pain radiating to left arm. 
    Patient is diaphoretic and reports shortness of breath. 
    Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
    Patient has history of hypertension and diabetes.
    """
    
    # Analyze the narrative
    analysis_result = triage_system.analyze_ems_narrative(example_narrative, "PT001")
    
    # Print results
    print("=== EMS AI Triage System Analysis ===")
    print(f"Patient ID: {analysis_result['patient_id']}")
    print(f"Urgency Level: {analysis_result['urgency_level']}")
    print(f"Triage Score: {analysis_result['triage_score']:.2f}")
    print(f"Keywords: {[kw[0] for kw in analysis_result['keyword_analysis']['keywords'][:5]]}")
    print(f"Protocol Matches: {len(analysis_result['protocol_matches'])}")
    print("Recommendations:")
    for rec in analysis_result['recommendations']:
        print(f"  - {rec}") 