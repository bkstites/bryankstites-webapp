import { NextRequest, NextResponse } from 'next/server';

// Types for our prediction logic
interface VitalSigns {
  spo2: number;
  rr: number; // respiratory rate
  hr: number; // heart rate
  sbp: number; // systolic blood pressure
  gcs_eye: number;
  gcs_verbal: number;
  gcs_motor: number;
}

interface PredictionResult {
  rox_score: number;
  gcs_total: number;
  rpp_score: number;
  respiratory_risk: string;
  neurological_risk: string;
  cardiovascular_risk: string;
  overall_risk: string;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Critical';
  narrative_risk_score: number;
  narrative_insights: string[];
}

// Calculate ROX score (SpO2/FiO2 ratio / Respiratory Rate) - validated in emergency medicine
function calculateROX(spo2: number, rr: number): number {
  // FiO2 is 21% (room air) for pre-hospital setting
  const fio2 = 0.21;
  const spo2_fio2_ratio = spo2 / fio2;
  return spo2_fio2_ratio / rr;
}

// Calculate GCS total
function calculateGCSTotal(eye: number, verbal: number, motor: number): number {
  return eye + verbal + motor;
}

// Calculate RPP (Rate Pressure Product) - validated cardiovascular risk indicator
function calculateRPP(hr: number, sbp: number): number {
  return hr * sbp;
}

// NEWS2 (National Early Warning Score 2) - validated emergency medicine scoring system
function calculateNEWS2Score(vitals: VitalSigns): number {
  let score = 0;
  
  // Respiratory Rate scoring (NEWS2)
  if (vitals.rr <= 8) score += 3;
  else if (vitals.rr <= 11) score += 1;
  else if (vitals.rr >= 25) score += 3;
  else if (vitals.rr >= 21) score += 2;
  else if (vitals.rr >= 12) score += 0; // normal range
  
  // SpO2 scoring (NEWS2)
  if (vitals.spo2 <= 91) score += 3;
  else if (vitals.spo2 <= 93) score += 2;
  else if (vitals.spo2 <= 95) score += 1;
  else if (vitals.spo2 >= 96) score += 0; // normal range
  
  // Systolic BP scoring (NEWS2)
  if (vitals.sbp <= 90) score += 3;
  else if (vitals.sbp <= 100) score += 2;
  else if (vitals.sbp <= 110) score += 1;
  else if (vitals.sbp >= 220) score += 3;
  else if (vitals.sbp >= 111) score += 0; // normal range
  
  // Heart Rate scoring (NEWS2)
  if (vitals.hr <= 40) score += 3;
  else if (vitals.hr <= 50) score += 1;
  else if (vitals.hr >= 131) score += 3;
  else if (vitals.hr >= 111) score += 2;
  else if (vitals.hr >= 51) score += 0; // normal range
  
  // Level of consciousness (NEWS2)
  const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
  if (gcs_total <= 8) score += 3;
  else if (gcs_total <= 10) score += 2;
  else if (gcs_total <= 13) score += 1;
  else score += 0; // normal range
  
  return score;
}

// MEOWS (Modified Early Obstetric Warning System) - adapted for general emergency medicine
function calculateMEOWSScore(vitals: VitalSigns): number {
  let score = 0;
  
  // Respiratory Rate (MEOWS)
  if (vitals.rr < 10 || vitals.rr > 30) score += 2;
  else if (vitals.rr < 12 || vitals.rr > 20) score += 1;
  
  // SpO2 (MEOWS)
  if (vitals.spo2 < 95) score += 2;
  else if (vitals.spo2 < 97) score += 1;
  
  // Systolic BP (MEOWS)
  if (vitals.sbp < 90 || vitals.sbp > 160) score += 2;
  else if (vitals.sbp < 100 || vitals.sbp > 140) score += 1;
  
  // Heart Rate (MEOWS)
  if (vitals.hr < 50 || vitals.hr > 120) score += 2;
  else if (vitals.hr < 60 || vitals.hr > 100) score += 1;
  
  // Level of consciousness (MEOWS)
  const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
  if (gcs_total < 15) score += 2;
  else if (gcs_total < 13) score += 1;
  
  return score;
}

// Determine risk levels based on validated emergency medicine standards
function determineRiskLevels(vitals: VitalSigns, rox: number, gcs: number, rpp: number): {
  respiratory: string;
  neurological: string;
  cardiovascular: string;
  overall: string;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Critical';
} {
  // Calculate validated scores
  const news2_score = calculateNEWS2Score(vitals);
  const meows_score = calculateMEOWSScore(vitals);
  
  // Respiratory risk assessment (based on NEWS2 and ROX score)
  let respiratory = 'Low';
  if (vitals.spo2 < 90) {
    if (vitals.spo2 < 85) respiratory = 'Critical';
    else respiratory = 'High';
  } else if (vitals.rr > 25 || vitals.rr < 10) {
    respiratory = 'High';
  } else if (vitals.rr > 20 || vitals.rr < 12) {
    respiratory = 'Moderate';
  }
  
  // ROX score validation (emergency medicine standard) - adjusted thresholds
  if (rox < 4.88) respiratory = 'Critical';
  else if (rox < 7.5 && respiratory === 'Low') respiratory = 'Moderate';

  // Neurological risk based on GCS (emergency medicine standard) - adjusted thresholds
  let neurological = 'Low';
  if (gcs <= 8) neurological = 'Critical';
  else if (gcs <= 10) neurological = 'High'; // Adjusted from 12 to 10
  else if (gcs <= 13) neurological = 'Moderate'; // Adjusted from 14 to 13

  // Cardiovascular risk assessment (based on NEWS2 and RPP) - adjusted thresholds
  let cardiovascular = 'Low';
  if (vitals.hr < 50 || vitals.hr >= 120) {
    if (vitals.hr < 40 || vitals.hr > 150) cardiovascular = 'Critical';
    else cardiovascular = 'High';
  } else if (vitals.hr < 60 || vitals.hr > 100) {
    cardiovascular = 'Moderate';
  }
  
  // Blood pressure (NEWS2 standard) - adjusted thresholds
  if (vitals.sbp < 90) {
    if (vitals.sbp < 80) cardiovascular = 'Critical';
    else if (cardiovascular === 'Low') cardiovascular = 'High';
  } else if (vitals.sbp >= 140) {
    if (cardiovascular === 'Low') cardiovascular = 'Moderate';
  }
  
  // RPP validation (emergency medicine standard) - adjusted thresholds
  if (rpp > 20000) { // Increased threshold for critical
    if (cardiovascular === 'Low') cardiovascular = 'Critical';
  } else if (rpp >= 15000) { // Adjusted threshold for high
    if (cardiovascular === 'Low') cardiovascular = 'High';
  } else if (rpp < 4000) {
    if (cardiovascular === 'Low') cardiovascular = 'Moderate';
  }

  // Overall risk assessment using NEWS2 and MEOWS thresholds - adjusted for less sensitivity
  let overall = 'Low';
  let risk_level: 'Low' | 'Moderate' | 'High' | 'Critical' = 'Low';

  // NEWS2 thresholds for escalation - adjusted to be less sensitive
  if (news2_score >= 8) {
    overall = 'Critical';
    risk_level = 'Critical';
  } else if (news2_score >= 6) {
    overall = 'High';
    risk_level = 'High';
  } else if (news2_score >= 4) {
    overall = 'Moderate';
    risk_level = 'Moderate';
  }

  // MEOWS validation - adjusted to be less sensitive
  if (meows_score >= 7) {
    overall = 'Critical';
    risk_level = 'Critical';
  } else if (meows_score >= 5 && overall === 'Low') {
    overall = 'High';
    risk_level = 'High';
  } else if (meows_score >= 3 && overall === 'Low') {
    overall = 'Moderate';
    risk_level = 'Moderate';
  }

  // Nuanced critical system escalation - not every critical system should escalate overall
  const risks = [respiratory, neurological, cardiovascular];
  const criticalCount = risks.filter(r => r === 'Critical').length;
  
  // Only escalate to Critical if multiple systems are critical OR one critical with high scores
  if (criticalCount >= 2) {
    overall = 'Critical';
    risk_level = 'Critical';
  } else if (criticalCount === 1 && (news2_score >= 6 || meows_score >= 5)) {
    overall = 'Critical';
    risk_level = 'Critical';
  } else if (criticalCount === 1) {
    // Single critical system with moderate scores = High risk
    overall = 'High';
    risk_level = 'High';
  }

  // Final validation: if all individual systems are low risk, overall should be low
  const allLowRisk = risks.every(r => r === 'Low');
  if (allLowRisk && news2_score < 4 && meows_score < 3) {
    overall = 'Low';
    risk_level = 'Low';
  }

  // Additional validation: prevent over-escalation for borderline cases
  if (overall === 'Critical' && news2_score < 6 && meows_score < 5) {
    overall = 'High';
    risk_level = 'High';
  }

  return { respiratory, neurological, cardiovascular, overall, risk_level };
}

// Narrative analysis based on emergency medicine literature
function analyzeNarrative(narrative: string): { riskScore: number; insights: string[] } {
  const insights: string[] = [];
  const narrativeLower = narrative.toLowerCase();
  
  // Medical keyword categories with risk levels (based on emergency medicine literature)
  const keywordCategories = {
    // CRITICAL RISK KEYWORDS (immediate escalation)
    critical: {
      respiratory: [
        'can\'t breathe', 'stopped breathing', 'not breathing', 'respiratory arrest',
        'chest tightness', 'chest pressure', 'suffocating', 'choking',
        'blue lips', 'cyanosis', 'unable to speak', 'gasping'
      ],
      cardiovascular: [
        'chest pain', 'heart attack', 'cardiac arrest', 'heart stopped',
        'crushing chest pain', 'pressure in chest', 'pain radiating to arm',
        'irregular heartbeat', 'skipped beats', 'heart racing', 'palpitations'
      ],
      neurological: [
        'unconscious', 'passed out', 'fainted', 'seizure', 'convulsion',
        'stroke symptoms', 'facial droop', 'slurred speech', 'weakness on one side',
        'confusion', 'disoriented', 'altered mental status'
      ],
      trauma: [
        'major trauma', 'head injury', 'bleeding profusely', 'uncontrolled bleeding',
        'penetrating injury', 'gunshot', 'stab wound', 'amputation'
      ]
    },
    
    // HIGH RISK KEYWORDS (significant concern)
    high: {
      respiratory: [
        'shortness of breath', 'difficulty breathing', 'wheezing', 'coughing blood',
        'rapid breathing', 'shallow breathing', 'chest pain with breathing',
        'short of breath', 'can\'t breathe', 'trouble breathing'
      ],
      cardiovascular: [
        'dizziness', 'lightheaded', 'feeling faint', 'sweating profusely',
        'nausea with chest pain', 'arm pain', 'jaw pain', 'back pain',
        'sweating', 'sweat', 'palpitations', 'heart racing', 'irregular heartbeat'
      ],
      neurological: [
        'severe headache', 'worst headache', 'sudden headache', 'vision changes',
        'numbness', 'tingling', 'weakness', 'difficulty walking'
      ],
      medical_conditions: [
        'diabetes', 'diabetic', 'high blood pressure', 'heart disease',
        'copd', 'asthma', 'emphysema', 'lung disease'
      ]
    },
    
    // MODERATE RISK KEYWORDS (monitoring required)
    moderate: {
      respiratory: [
        'cough', 'sore throat', 'runny nose', 'congestion', 'mild shortness of breath'
      ],
      cardiovascular: [
        'mild chest discomfort', 'heartburn', 'indigestion', 'anxiety',
        'stress', 'feeling overwhelmed'
      ],
      neurological: [
        'mild headache', 'tired', 'fatigue', 'dizzy spells'
      ],
      medications: [
        'blood thinner', 'warfarin', 'coumadin', 'aspirin', 'plavix',
        'insulin', 'diabetes medication', 'heart medication'
      ]
    }
  };

  let riskScore = 0;
  const detectedKeywords = {
    critical: [] as string[],
    high: [] as string[],
    moderate: [] as string[]
  };
  
  // Check each category
  Object.entries(keywordCategories).forEach(([riskLevel, categories]) => {
    Object.entries(categories).forEach(([category, keywords]) => {
      keywords.forEach(keyword => {
        if (narrativeLower.includes(keyword)) {
          detectedKeywords[riskLevel as keyof typeof detectedKeywords].push(keyword);
          
          // Risk scoring based on keyword severity
          switch (riskLevel) {
            case 'critical':
              riskScore += 10;
              break;
            case 'high':
              riskScore += 5;
              break;
            case 'moderate':
              riskScore += 2;
              break;
          }
        }
      });
    });
  });
  
  // Generate insights based on detected keywords
  if (detectedKeywords.critical.length > 0) {
    insights.push('🚨 CRITICAL SYMPTOMS DETECTED: Immediate medical attention required');
    detectedKeywords.critical.forEach(keyword => {
      insights.push(`🚨 ${keyword.toUpperCase()}: Requires immediate assessment`);
    });
  }
  
  if (detectedKeywords.high.length > 0) {
    insights.push('⚠️ HIGH RISK SYMPTOMS: Significant medical concern');
    detectedKeywords.high.forEach(keyword => {
      insights.push(`⚠️ ${keyword}: Monitor closely, prepare for escalation`);
    });
  }
  
  if (detectedKeywords.moderate.length > 0) {
    insights.push('📋 MODERATE SYMPTOMS: Standard monitoring required');
    detectedKeywords.moderate.forEach(keyword => {
      insights.push(`📋 ${keyword}: Document and monitor`);
    });
  }

  // Specific medical condition alerts
  if (narrativeLower.includes('diabetes') || narrativeLower.includes('diabetic')) {
    insights.push('💉 Diabetes Alert: Check blood glucose, monitor for hypo/hyperglycemia');
  }
  
  if (narrativeLower.includes('heart') || narrativeLower.includes('cardiac')) {
    insights.push('❤️ Cardiac History: Prepare for cardiac assessment, consider ECG');
  }
  
  if (narrativeLower.includes('copd') || narrativeLower.includes('asthma') || narrativeLower.includes('lung')) {
    insights.push('🫁 Respiratory Condition: Monitor airway, prepare breathing treatments');
  }
  
  if (narrativeLower.includes('stroke') || narrativeLower.includes('cva')) {
    insights.push('🧠 Stroke History: Monitor for new symptoms, check FAST signs');
  }

  // Medication alerts
  if (narrativeLower.includes('blood thinner') || narrativeLower.includes('warfarin') || narrativeLower.includes('coumadin')) {
    insights.push('🩸 Blood Thinner Alert: Increased bleeding risk, check for bleeding');
  }
  
  if (narrativeLower.includes('insulin')) {
    insights.push('💉 Insulin Alert: Check blood glucose, watch for hypoglycemia');
  }

  // Overall risk assessment integration
  if (riskScore >= 15) {
    insights.push('🚨 HIGH NARRATIVE RISK SCORE: Multiple concerning symptoms detected');
  } else if (riskScore >= 8) {
    insights.push('⚠️ MODERATE NARRATIVE RISK SCORE: Several symptoms require attention');
  } else if (riskScore >= 3) {
    insights.push('📋 LOW NARRATIVE RISK SCORE: Minor symptoms noted');
  }

  return { riskScore, insights };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const vitals: VitalSigns = {
      spo2: Number(body.spo2),
      rr: Number(body.rr),
      hr: Number(body.hr),
      sbp: Number(body.sbp),
      gcs_eye: Number(body.gcs_eye),
      gcs_verbal: Number(body.gcs_verbal),
      gcs_motor: Number(body.gcs_motor),
    };

    // Validate inputs
    if (Object.values(vitals).some(val => isNaN(val) || val < 0)) {
      return NextResponse.json(
        { error: 'Invalid vital signs data' },
        { status: 400 }
      );
    }

    // Calculate validated emergency medicine scores
    const rox_score = calculateROX(vitals.spo2, vitals.rr);
    const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
    const rpp_score = calculateRPP(vitals.hr, vitals.sbp);

    // Determine risk levels using validated emergency medicine standards
    const riskAssessment = determineRiskLevels(vitals, rox_score, gcs_total, rpp_score);

    // Analyze narrative if provided
    const patientNarrative = body.patient_narrative || '';
    const narrativeAnalysis = analyzeNarrative(patientNarrative);

    // Integrate narrative risk with vital signs risk
    let finalRiskLevel = riskAssessment.risk_level;
    if (narrativeAnalysis.riskScore >= 15) {
      // High narrative risk can escalate overall risk
      if (finalRiskLevel === 'Low') finalRiskLevel = 'Moderate';
      else if (finalRiskLevel === 'Moderate') finalRiskLevel = 'High';
      else if (finalRiskLevel === 'High') finalRiskLevel = 'Critical';
    } else if (narrativeAnalysis.riskScore >= 8) {
      // Moderate narrative risk can escalate from Low to Moderate
      if (finalRiskLevel === 'Low') finalRiskLevel = 'Moderate';
    }

    const result: PredictionResult = {
      rox_score: Math.round(rox_score * 100) / 100, // Round to 2 decimal places
      gcs_total,
      rpp_score,
      respiratory_risk: riskAssessment.respiratory,
      neurological_risk: riskAssessment.neurological,
      cardiovascular_risk: riskAssessment.cardiovascular,
      overall_risk: riskAssessment.overall,
      risk_level: finalRiskLevel,
      narrative_risk_score: narrativeAnalysis.riskScore,
      narrative_insights: narrativeAnalysis.insights,
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { error: 'Failed to process prediction' },
      { status: 500 }
    );
  }
} 