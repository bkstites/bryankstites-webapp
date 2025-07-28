# EMS AI Triage System - Keyword Association Model

A machine learning-based system for analyzing EMS narratives and extracting keyword associations for improved triage and protocol matching.

## Overview

This system provides advanced keyword association analysis for EMS protocol data, specifically designed to work with state protocol guidelines like the [NJ State EMS Clinical Practice Guidelines](https://www.nj.gov/health/ems/documents/2022%20Interim%20Clinical%20Practice%20Guidelines.pdf).

## Features

### 🔍 Keyword Association Analysis
- **TF-IDF based keyword extraction** from EMS narratives
- **Protocol matching** using cosine similarity
- **Urgency indicator detection** (cardiac, trauma, stroke, etc.)
- **Vital signs extraction** and analysis

### 🧠 Machine Learning Components
- **Multi-class classification** for protocol categorization
- **Random Forest, Naive Bayes, and Logistic Regression** models
- **Automatic model selection** based on performance
- **Model persistence** for deployment

### 📊 Narrative Analysis
- **Real-time triage scoring** (0-10 scale)
- **Urgency level classification** (CRITICAL, URGENT, MODERATE, LOW, MINIMAL)
- **Protocol recommendation** system
- **Comprehensive analysis reports**

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd bryankstites-webapp
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Quick Start

### Basic Usage

```python
from ems_ai_triage_system import EMSTriageSystem

# Initialize the system
triage_system = EMSTriageSystem()

# Analyze an EMS narrative
narrative = """
Patient presents with severe chest pain radiating to left arm.
Patient is diaphoretic and reports shortness of breath.
Vital signs: BP 180/110, HR 110, RR 24, O2 sat 92%.
"""

analysis = triage_system.analyze_ems_narrative(narrative, "PT001")

print(f"Urgency Level: {analysis['urgency_level']}")
print(f"Triage Score: {analysis['triage_score']:.2f}")
print(f"Keywords: {[kw[0] for kw in analysis['keyword_analysis']['keywords'][:5]]}")
```

### Training with Protocol Data

```python
from ems_data_preprocessing import EMSTrainingPipeline

# Initialize training pipeline
pipeline = EMSTrainingPipeline()

# Train with your protocol data
pipeline.train_model('your_protocol_data.json', 'models/')

# Evaluate the model
results = pipeline.evaluate_model('test_data.json')
print(f"Average triage score: {results['avg_triage_score']:.2f}")
```

## System Architecture

### Core Components

1. **EMSKeywordAssociation Class**
   - Keyword extraction using TF-IDF
   - Protocol matching algorithms
   - Association strength calculation

2. **EMSTriageSystem Class**
   - Main triage logic
   - Narrative analysis pipeline
   - Protocol recommendation engine

3. **EMSDataPreprocessor Class**
   - Data loading from multiple formats (JSON, CSV, Excel, TXT)
   - Protocol categorization
   - Training dataset creation

### Data Flow

```
Raw Protocol Data → Preprocessing → Keyword Extraction → Model Training → Analysis Engine
                                                                           ↓
EMS Narrative → Text Preprocessing → Keyword Analysis → Protocol Matching → Triage Score → Recommendations
```

## Working with NJ State Protocol Data

### Loading Protocol Data

The system can work with the NJ State EMS Clinical Practice Guidelines by:

1. **Converting PDF to structured data** (JSON/CSV format)
2. **Extracting protocol sections** and categorizing them
3. **Training the model** on the structured protocol data

### Example Protocol Structure

```json
{
  "protocol_id": "CARDIAC_001",
  "protocol_name": "Chest Pain Protocol",
  "category": "cardiac",
  "content": "CHEST PAIN PROTOCOL\nAssessment:\n- 12-lead ECG within 10 minutes\n- Aspirin 325mg PO...",
  "keywords": ["chest pain", "cardiac", "ecg", "aspirin", "nitroglycerin"]
}
```

## API Reference

### EMSKeywordAssociation

#### `extract_keywords(text: str, top_k: int = 10) -> List[Tuple[str, float]]`
Extract top keywords from text using TF-IDF analysis.

#### `find_keyword_associations(target_keyword: str, corpus: List[str], min_association: float = 0.3) -> Dict[str, float]`
Find keywords associated with a target keyword.

#### `train_protocol_classifier(training_data: List[Dict[str, Any]]) -> None`
Train a multi-class classifier for protocol categorization.

### EMSTriageSystem

#### `analyze_ems_narrative(narrative: str, patient_id: str = None) -> Dict[str, Any]`
Perform comprehensive analysis of EMS narrative.

#### `add_protocol_data(protocol_name: str, protocol_text: str, keywords: List[str] = None) -> None`
Add protocol data to the system for matching.

## Advanced Features

### Custom Protocol Integration

```python
# Add custom protocols
triage_system.add_protocol_data(
    "Custom Trauma Protocol",
    "TRAUMA ASSESSMENT\nPrimary survey: ABCDE...",
    ["trauma", "assessment", "primary survey", "abcde"]
)
```

### Model Persistence

```python
# Save trained model
keyword_analyzer.save_model('models/ems_model.pkl')

# Load trained model
keyword_analyzer.load_model('models/ems_model.pkl')
```

### Analysis Reports

```python
# Generate comprehensive report
report = triage_system.export_analysis_report("PT001")
print(json.dumps(report, indent=2))
```

## Performance Metrics

The system provides several performance indicators:

- **Triage Score**: 0-10 scale based on urgency indicators
- **Keyword Density**: Number of relevant keywords found
- **Protocol Match Score**: Similarity to known protocols
- **Confidence Score**: Analysis reliability metric

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Include tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Note**: This system is designed to assist EMS professionals but should not replace clinical judgment. Always follow local protocols and guidelines. 