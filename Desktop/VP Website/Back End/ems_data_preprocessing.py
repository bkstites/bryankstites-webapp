#!/usr/bin/env python3
"""
EMS Data Preprocessing and Training Script
=========================================

This script handles preprocessing of EMS protocol data and training
of the keyword association models.
"""

import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Import the main system
from ems_ai_triage_system import EMSKeywordAssociation, EMSTriageSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EMSDataPreprocessor:
    """
    Preprocesses EMS protocol data for training
    """
    
    def __init__(self):
        self.processed_data = []
        self.protocol_categories = {}
        
    def load_protocol_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load protocol data from various file formats
        
        Args:
            file_path: Path to the protocol data file
            
        Returns:
            List of protocol dictionaries
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.json':
            return self._load_json_data(file_path)
        elif file_path.suffix.lower() in ['.csv', '.xlsx']:
            return self._load_tabular_data(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._load_text_data(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return []
    
    def _load_json_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON protocol data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                logger.error("Invalid JSON structure")
                return []
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            return []
    
    def _load_tabular_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load tabular protocol data (CSV/Excel)"""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Convert DataFrame to list of dictionaries
            protocols = []
            for _, row in df.iterrows():
                protocol = {}
                for col in df.columns:
                    if pd.notna(row[col]):
                        protocol[col.lower().replace(' ', '_')] = str(row[col])
                protocols.append(protocol)
            
            return protocols
        except Exception as e:
            logger.error(f"Error loading tabular data: {e}")
            return []
    
    def _load_text_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load text protocol data"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple text parsing - split by sections
            sections = re.split(r'\n\s*\n', content)
            protocols = []
            
            for i, section in enumerate(sections):
                if section.strip():
                    protocols.append({
                        'protocol_id': f'protocol_{i}',
                        'content': section.strip(),
                        'source_file': file_path.name
                    })
            
            return protocols
        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            return []
    
    def preprocess_protocols(self, protocols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess protocol data for training
        
        Args:
            protocols: Raw protocol data
            
        Returns:
            Preprocessed protocol data
        """
        processed = []
        
        for protocol in protocols:
            processed_protocol = self._preprocess_single_protocol(protocol)
            if processed_protocol:
                processed.append(processed_protocol)
        
        return processed
    
    def _preprocess_single_protocol(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single protocol"""
        try:
            # Extract text content
            text_content = self._extract_text_content(protocol)
            if not text_content:
                return None
            
            # Extract protocol metadata
            metadata = self._extract_metadata(protocol)
            
            # Extract keywords
            keywords = self._extract_protocol_keywords(text_content)
            
            # Categorize protocol
            category = self._categorize_protocol(text_content, keywords)
            
            processed = {
                'protocol_id': metadata.get('protocol_id', f'protocol_{len(self.processed_data)}'),
                'protocol_name': metadata.get('protocol_name', 'Unknown Protocol'),
                'category': category,
                'content': text_content,
                'keywords': keywords,
                'metadata': metadata,
                'processed_timestamp': datetime.now().isoformat()
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing protocol: {e}")
            return None
    
    def _extract_text_content(self, protocol: Dict[str, Any]) -> str:
        """Extract text content from protocol"""
        # Try different possible field names
        text_fields = ['content', 'text', 'description', 'protocol_text', 'body']
        
        for field in text_fields:
            if field in protocol and protocol[field]:
                return str(protocol[field])
        
        # If no specific text field, concatenate all string values
        text_parts = []
        for key, value in protocol.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
        
        return ' '.join(text_parts) if text_parts else ""
    
    def _extract_metadata(self, protocol: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from protocol"""
        metadata = {}
        
        # Common metadata fields
        metadata_fields = [
            'protocol_id', 'protocol_name', 'version', 'date', 'author',
            'category', 'subcategory', 'priority', 'status'
        ]
        
        for field in metadata_fields:
            if field in protocol:
                metadata[field] = protocol[field]
        
        return metadata
    
    def _extract_protocol_keywords(self, text: str) -> List[str]:
        """Extract keywords from protocol text"""
        # EMS-specific keyword patterns
        keyword_patterns = [
            r'\b(chest pain|cardiac|heart attack|mi|stemi)\b',
            r'\b(stroke|facial droop|arm drift|speech)\b',
            r'\b(trauma|injury|bleeding|fracture|laceration)\b',
            r'\b(respiratory|breathing|asthma|copd|dyspnea)\b',
            r'\b(pediatric|child|infant|baby)\b',
            r'\b(obstetric|pregnant|labor|delivery)\b',
            r'\b(diabetic|glucose|hypoglycemia|hyperglycemia)\b',
            r'\b(seizure|epilepsy|convulsion)\b',
            r'\b(overdose|poisoning|toxic)\b',
            r'\b(psychiatric|mental health|suicide)\b'
        ]
        
        keywords = []
        text_lower = text.lower()
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text_lower)
            keywords.extend(matches)
        
        # Remove duplicates and sort
        keywords = list(set(keywords))
        keywords.sort()
        
        return keywords
    
    def _categorize_protocol(self, text: str, keywords: List[str]) -> str:
        """Categorize protocol based on content and keywords"""
        text_lower = text.lower()
        
        # Define category keywords
        category_keywords = {
            'cardiac': ['cardiac', 'chest pain', 'heart', 'ecg', 'ekg', 'mi', 'stemi'],
            'stroke': ['stroke', 'facial droop', 'arm drift', 'speech', 'fast'],
            'trauma': ['trauma', 'injury', 'bleeding', 'fracture', 'laceration'],
            'respiratory': ['respiratory', 'breathing', 'asthma', 'copd', 'dyspnea'],
            'pediatric': ['pediatric', 'child', 'infant', 'baby'],
            'obstetric': ['obstetric', 'pregnant', 'labor', 'delivery'],
            'medical': ['diabetic', 'seizure', 'overdose', 'psychiatric']
        }
        
        # Calculate category scores
        category_scores = {}
        for category, cat_keywords in category_keywords.items():
            score = sum(1 for kw in cat_keywords if kw in text_lower or kw in keywords)
            if score > 0:
                category_scores[category] = score
        
        # Return the category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'
    
    def create_training_dataset(self, processed_protocols: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Create training dataset for the ML model
        
        Args:
            processed_protocols: Preprocessed protocol data
            
        Returns:
            Tuple of (texts, labels) for training
        """
        texts = []
        labels = []
        
        for protocol in processed_protocols:
            texts.append(protocol['content'])
            labels.append(protocol['category'])
        
        return texts, labels
    
    def save_processed_data(self, processed_data: List[Dict[str, Any]], output_path: str) -> None:
        """Save processed data to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed data saved to {output_path}")


class EMSTrainingPipeline:
    """
    Complete training pipeline for EMS keyword association model
    """
    
    def __init__(self):
        self.preprocessor = EMSDataPreprocessor()
        self.triage_system = EMSTriageSystem()
        self.keyword_analyzer = EMSKeywordAssociation()
        
    def train_model(self, protocol_data_path: str, output_model_path: str = None) -> None:
        """
        Complete training pipeline
        
        Args:
            protocol_data_path: Path to protocol data file
            output_model_path: Path to save trained model
        """
        logger.info("Starting EMS model training pipeline...")
        
        # Load and preprocess data
        logger.info("Loading protocol data...")
        raw_protocols = self.preprocessor.load_protocol_data(protocol_data_path)
        
        if not raw_protocols:
            logger.error("No protocol data loaded")
            return
        
        logger.info(f"Loaded {len(raw_protocols)} protocols")
        
        # Preprocess protocols
        logger.info("Preprocessing protocols...")
        processed_protocols = self.preprocessor.preprocess_protocols(raw_protocols)
        
        logger.info(f"Preprocessed {len(processed_protocols)} protocols")
        
        # Create training dataset
        logger.info("Creating training dataset...")
        texts, labels = self.preprocessor.create_training_dataset(processed_protocols)
        
        # Train the keyword association model
        logger.info("Training keyword association model...")
        self.keyword_analyzer.train_protocol_classifier([
            {'text': text, 'protocol': label} 
            for text, label in zip(texts, labels)
        ])
        
        # Add protocols to triage system
        logger.info("Adding protocols to triage system...")
        for protocol in processed_protocols:
            self.triage_system.add_protocol_data(
                protocol['protocol_name'],
                protocol['content'],
                protocol['keywords']
            )
        
        # Save processed data
        if output_model_path:
            output_path = Path(output_model_path)
            processed_data_path = output_path.parent / f"processed_protocols_{datetime.now().strftime('%Y%m%d')}.json"
            self.preprocessor.save_processed_data(processed_protocols, str(processed_data_path))
            
            # Save trained model
            model_path = output_path.parent / "ems_model.pkl"
            self.keyword_analyzer.save_model(str(model_path))
        
        logger.info("Training pipeline completed successfully!")
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            test_data_path: Path to test data
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating model...")
        
        # Load test data
        test_protocols = self.preprocessor.load_protocol_data(test_data_path)
        processed_test = self.preprocessor.preprocess_protocols(test_protocols)
        
        # Test narratives
        test_narratives = [
            "Patient presents with severe chest pain and shortness of breath",
            "Patient found unresponsive with facial droop and arm weakness",
            "Patient with multiple trauma injuries from motor vehicle accident",
            "Patient experiencing respiratory distress with wheezing",
            "Pediatric patient with fever and altered mental status"
        ]
        
        results = {
            'test_narratives': len(test_narratives),
            'protocol_matches': [],
            'keyword_extraction': [],
            'triage_scores': []
        }
        
        for narrative in test_narratives:
            analysis = self.triage_system.analyze_ems_narrative(narrative, f"TEST_{len(results['triage_scores'])}")
            
            results['protocol_matches'].append(len(analysis['protocol_matches']))
            results['keyword_extraction'].append(len(analysis['keyword_analysis']['keywords']))
            results['triage_scores'].append(analysis['triage_score'])
        
        # Calculate metrics
        results['avg_triage_score'] = np.mean(results['triage_scores'])
        results['avg_keywords'] = np.mean(results['keyword_extraction'])
        results['avg_protocol_matches'] = np.mean(results['protocol_matches'])
        
        logger.info(f"Evaluation completed. Average triage score: {results['avg_triage_score']:.2f}")
        
        return results


def create_sample_protocol_data():
    """Create sample protocol data for testing"""
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
    
    return sample_protocols


if __name__ == "__main__":
    # Create sample data
    sample_protocols = create_sample_protocol_data()
    
    # Save sample data to file
    with open('sample_protocols.json', 'w') as f:
        json.dump(sample_protocols, f, indent=2)
    
    # Initialize training pipeline
    pipeline = EMSTrainingPipeline()
    
    # Train model with sample data
    pipeline.train_model('sample_protocols.json', 'models/')
    
    # Test the system
    test_narrative = """
    Patient presents with severe chest pain radiating to left arm.
    Patient is diaphoretic and reports shortness of breath.
    Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
    """
    
    analysis = pipeline.triage_system.analyze_ems_narrative(test_narrative, "TEST001")
    
    print("\n=== Test Analysis Results ===")
    print(f"Urgency Level: {analysis['urgency_level']}")
    print(f"Triage Score: {analysis['triage_score']:.2f}")
    print(f"Keywords Found: {len(analysis['keyword_analysis']['keywords'])}")
    print(f"Protocol Matches: {len(analysis['protocol_matches'])}")
    print("Top Keywords:")
    for keyword, score in analysis['keyword_analysis']['keywords'][:5]:
        print(f"  - {keyword}: {score:.3f}") 