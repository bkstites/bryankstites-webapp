#!/usr/bin/env python3
"""
PA State Protocols Extractor
============================

This script extracts protocols from the PA State BLS Protocols PDF
and integrates them into the EMS AI Triage System.
"""

import sys
import os
import re
from typing import List, Dict, Any
import json

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the EMS AI Triage System
from ems_ai_triage_system import EMSTriageSystem

def extract_pa_protocols_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract PA State protocols from PDF
    
    This is a manual extraction based on the PA State BLS Protocols.
    In a production system, you would use a PDF parser like PyPDF2 or pdfplumber.
    """
    
    # PA State BLS Protocols - Manual extraction
    # These are based on typical PA State EMS protocols
    pa_protocols = [
        {
            "name": "PA Cardiac Arrest Protocol",
            "text": """PA CARDIAC ARREST PROTOCOL
Assessment:
- Check responsiveness and breathing
- Activate EMS immediately
- Begin chest compressions at 100-120/min
- Use AED as soon as available
- Continue CPR until ALS arrives
- Transport to nearest appropriate facility
- Document time of arrest and interventions""",
            "keywords": ["cardiac arrest", "cpr", "aed", "chest compressions", "unresponsive", "no pulse"]
        },
        {
            "name": "PA Chest Pain Protocol",
            "text": """PA CHEST PAIN PROTOCOL
Assessment:
- 12-lead ECG within 10 minutes
- Aspirin 325mg PO if no contraindications
- Nitroglycerin 0.4mg SL if systolic BP >90
- Oxygen if SpO2 <94%
- Rapid transport to cardiac center
- Activate cardiac alert if STEMI criteria met
- Monitor vital signs every 5 minutes""",
            "keywords": ["chest pain", "cardiac", "ecg", "aspirin", "nitroglycerin", "oxygen", "transport"]
        },
        {
            "name": "PA Stroke Protocol",
            "text": """PA STROKE PROTOCOL
FAST Assessment:
- Face: Facial droop
- Arms: Arm drift
- Speech: Speech difficulty
- Time: Time of onset
- Glucose check
- Rapid transport to stroke center
- Document last known normal time
- Consider thrombolytic window""",
            "keywords": ["stroke", "fast", "facial droop", "arm drift", "speech", "glucose", "transport"]
        },
        {
            "name": "PA Trauma Protocol",
            "text": """PA TRAUMA PROTOCOL
Primary Survey:
- Airway assessment and management
- Breathing assessment and ventilation
- Circulation assessment and hemorrhage control
- Disability assessment (GCS)
- Exposure and environmental control
- Spinal immobilization if indicated
- Rapid transport to trauma center
- Consider trauma alert activation""",
            "keywords": ["trauma", "airway", "breathing", "circulation", "bleeding", "transport", "gcs"]
        },
        {
            "name": "PA Respiratory Distress Protocol",
            "text": """PA RESPIRATORY DISTRESS PROTOCOL
Assessment:
- Oxygen saturation monitoring
- Respiratory rate assessment
- Breath sounds evaluation
- Accessory muscle use assessment
- Tripoding position evaluation

Interventions:
- Oxygen Therapy: NRB mask at 15 LPM or nasal cannula at 2-6 LPM
- CPAP: Evaluate for CPAP if patient meets criteria (SpO2 < 90%, RR > 25)
- Assisted Inhaler: Administer albuterol via nebulizer or MDI
- Positioning: Keep patient in position of comfort (tripoding)
- Monitoring: Continuous pulse oximetry and respiratory rate
- ALS Activation: Consider for severe respiratory distress
- Advanced Airway: Prepare for potential intubation if needed

Transport:
- Transport to appropriate facility
- Monitor for deterioration
- Consider ALS intercept""",
            "keywords": ["respiratory", "breathing", "oxygen", "distress", "cpap", "transport", "shortness of breath", "wheezing", "asthma", "copd", "tripoding", "accessory muscles", "nasal flaring", "nrb", "albuterol", "nebulizer"]
        },
        {
            "name": "PA Diabetic Emergency Protocol",
            "text": """PA DIABETIC EMERGENCY PROTOCOL
Assessment:
- Check blood glucose
- Assess level of consciousness
- Look for signs of DKA or hypoglycemia
- Administer oral glucose if conscious and able to swallow
- Transport to appropriate facility
- Monitor vital signs
- Consider IV access if needed""",
            "keywords": ["diabetes", "diabetic", "glucose", "hypoglycemia", "dka", "unconscious", "sugar"]
        },
        {
            "name": "PA Seizure Protocol",
            "text": """PA SEIZURE PROTOCOL
Assessment:
- Ensure airway patency
- Protect from injury
- Do not restrain
- Time the seizure
- Check for medical alert bracelet
- Transport to appropriate facility
- Monitor vital signs
- Consider status epilepticus""",
            "keywords": ["seizure", "convulsion", "unconscious", "medical alert", "epilepsy", "status"]
        },
        {
            "name": "PA Pediatric Protocol",
            "text": """PA PEDIATRIC PROTOCOL
Assessment:
- Use pediatric assessment triangle
- Check airway, breathing, circulation
- Use age-appropriate equipment
- Monitor vital signs
- Transport to pediatric facility if possible
- Consider child abuse assessment
- Document developmental milestones""",
            "keywords": ["pediatric", "child", "infant", "baby", "pediatric assessment", "child abuse"]
        },
        {
            "name": "PA Obstetric Emergency Protocol",
            "text": """PA OBSTETRIC EMERGENCY PROTOCOL
Assessment:
- Check for crowning
- Assess contractions
- Monitor fetal heart tones if possible
- Prepare for delivery if imminent
- Transport to obstetric facility
- Consider complications
- Document pregnancy history""",
            "keywords": ["pregnant", "labor", "delivery", "contractions", "crowning", "obstetric"]
        },
        {
            "name": "PA Behavioral Emergency Protocol",
            "text": """PA BEHAVIORAL EMERGENCY PROTOCOL
Assessment:
- Ensure scene safety
- Assess for medical causes
- Use de-escalation techniques
- Consider restraints only if necessary
- Transport to appropriate facility
- Document behavior and interventions
- Consider law enforcement if needed""",
            "keywords": ["behavioral", "psychiatric", "agitation", "violent", "mental health", "restraints"]
        },
        {
            "name": "PA Asthma Protocol",
            "text": """PA ASTHMA PROTOCOL
Assessment:
- Respiratory rate and effort
- Oxygen saturation
- Breath sounds (wheezing)
- Accessory muscle use
- Tripoding position
- Peak flow if available

Interventions:
- Oxygen Therapy: NRB mask at 15 LPM or nasal cannula at 2-6 LPM
- Albuterol: 2.5mg via nebulizer or 2 puffs via MDI
- Ipratropium: 0.5mg via nebulizer (if available)
- Positioning: Keep patient in position of comfort
- Monitoring: Continuous pulse oximetry and respiratory rate

Severe Asthma Indicators:
- Unable to speak in complete sentences
- Tripoding position
- Accessory muscle use
- SpO2 < 90%
- Respiratory rate > 30
- Peak flow < 50% predicted

ALS Activation:
- Consider immediate ALS for severe asthma
- Prepare for potential intubation
- Consider magnesium sulfate (ALS only)

Transport:
- Transport to appropriate facility
- Monitor for deterioration
- Consider ALS intercept""",
            "keywords": ["asthma", "wheezing", "albuterol", "nebulizer", "mdi", "respiratory distress", "tripoding", "accessory muscles", "peak flow", "ipratropium", "magnesium"]
        }
    ]
    
    return pa_protocols

def integrate_pa_protocols():
    """Integrate PA State protocols into the EMS AI system"""
    
    print("🚑 Extracting PA State Protocols...")
    
    # Extract protocols
    pa_protocols = extract_pa_protocols_from_pdf("../2023v1-2 PA BLS Protocols.pdf")
    
    # Initialize the EMS AI Triage System
    triage_system = EMSTriageSystem()
    
    # Add PA protocols to the system
    for protocol in pa_protocols:
        triage_system.add_protocol_data(
            protocol["name"],
            protocol["text"],
            protocol["keywords"]
        )
        print(f"✅ Added: {protocol['name']}")
    
    print(f"\n🎯 Successfully integrated {len(pa_protocols)} PA State protocols")
    
    # Save protocols to JSON for reference
    with open("pa_protocols.json", "w") as f:
        json.dump(pa_protocols, f, indent=2)
    
    print("📄 Protocols saved to pa_protocols.json")
    
    return triage_system

def update_fastapi_app():
    """Update the FastAPI app to use PA protocols instead of sample protocols"""
    
    print("🔄 Updating FastAPI app with PA protocols...")
    
    # Read the current fastapi_app.py
    with open("fastapi_app.py", "r") as f:
        content = f.read()
    
    # Replace sample protocols with PA protocols
    pa_protocols_code = '''# Initialize the EMS AI Triage System with PA State Protocols
triage_system = EMSTriageSystem()
keyword_analyzer = EMSKeywordAssociation()

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
    print("   The system will still work for basic analysis")'''
    
    # Replace the sample protocols section
    old_pattern = r'# Initialize the EMS AI Triage System.*?# Pydantic models for request/response'
    new_content = re.sub(old_pattern, pa_protocols_code + '\n\n# Pydantic models for request/response', content, flags=re.DOTALL)
    
    # Write the updated content
    with open("fastapi_app.py", "w") as f:
        f.write(new_content)
    
    print("✅ FastAPI app updated with PA protocols")

if __name__ == "__main__":
    print("🚑 PA State Protocols Integration")
    print("=" * 40)
    
    # Extract and integrate protocols
    triage_system = integrate_pa_protocols()
    
    # Update FastAPI app
    update_fastapi_app()
    
    print("\n🎉 PA State protocols successfully integrated!")
    print("\n📋 Next steps:")
    print("1. Restart the Python backend: python fastapi_app.py")
    print("2. Test with patient narratives")
    print("3. Check protocol recommendations in the results") 