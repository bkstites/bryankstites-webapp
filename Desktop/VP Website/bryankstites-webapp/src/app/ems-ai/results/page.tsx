'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';

function getRiskBadgeColor(risk: string) {
  switch (risk) {
    case 'Critical':
      return 'bg-red-100 text-red-800 border-red-200';
    case 'High':
      return 'bg-orange-100 text-orange-800 border-orange-200';
    case 'Moderate':
      return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    case 'Low':
      return 'bg-green-100 text-green-800 border-green-200';
    default:
      return 'bg-gray-100 text-gray-800 border-gray-200';
  }
}

function getClinicalRecommendations(riskLevel: string) {
  const recommendations = {
    transport_destination: '',
    monitoring_level: '',
    interventions: '',
    special_considerations: '',
  };

  switch (riskLevel) {
    case 'Critical':
      recommendations.transport_destination = 'Trauma Center or Cardiac Center';
      recommendations.monitoring_level = 'Continuous monitoring with ALS';
      recommendations.interventions = 'Immediate ALS interventions, consider advanced airway, IV access';
      recommendations.special_considerations = 'Prepare for rapid deterioration, notify receiving facility';
      break;
    case 'High':
      recommendations.transport_destination = 'Hospital ED';
      recommendations.monitoring_level = 'Frequent reassessment (every 5-10 minutes)';
      recommendations.interventions = 'ALS monitoring, IV access if needed, prepare for escalation';
      recommendations.special_considerations = 'Monitor for deterioration, consider ALS upgrade';
      break;
    case 'Moderate':
      recommendations.transport_destination = 'Hospital ED';
      recommendations.monitoring_level = 'Regular reassessment (every 15 minutes)';
      recommendations.interventions = 'BLS care with ALS consideration';
      recommendations.special_considerations = 'Monitor trends, prepare for escalation if needed';
      break;
    case 'Low':
      recommendations.transport_destination = 'Hospital ED or Urgent Care';
      recommendations.monitoring_level = 'Standard monitoring';
      recommendations.interventions = 'BLS care, comfort measures';
      recommendations.special_considerations = 'Routine transport, monitor for changes';
      break;
  }

  return recommendations;
}

// Enhanced narrative analysis based on emergency medicine literature
function extractNarrativeInsights(patientNarrative: string, riskLevel: string, respiratoryRisk: string, cardiovascularRisk: string, neurologicalRisk: string) {
  const insights = [];
  const narrative = patientNarrative.toLowerCase();
  
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

  // Function to detect keywords and calculate risk score
  function detectKeywords(narrative: string) {
    const detectedKeywords = {
      critical: [] as string[],
      high: [] as string[],
      moderate: [] as string[]
    };
    
    let riskScore = 0;
    
    // Check each category
    Object.entries(keywordCategories).forEach(([riskLevel, categories]) => {
      Object.entries(categories).forEach(([category, keywords]) => {
        keywords.forEach(keyword => {
          if (narrative.includes(keyword)) {
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
    
    return { detectedKeywords, riskScore };
  }

  // Analyze narrative for keywords
  const { detectedKeywords, riskScore } = detectKeywords(narrative);
  
  // Generate insights based on detected keywords
  if (detectedKeywords.critical.length > 0) {
    insights.push('🚨 **CRITICAL SYMPTOMS DETECTED**: Immediate medical attention required');
    detectedKeywords.critical.forEach(keyword => {
      insights.push(`🚨 **${keyword.toUpperCase()}**: Requires immediate assessment`);
    });
  }
  
  if (detectedKeywords.high.length > 0) {
    insights.push('⚠️ **HIGH RISK SYMPTOMS**: Significant medical concern');
    detectedKeywords.high.forEach(keyword => {
      insights.push(`⚠️ **${keyword}**: Monitor closely, prepare for escalation`);
    });
  }
  
  if (detectedKeywords.moderate.length > 0) {
    insights.push('📋 **MODERATE SYMPTOMS**: Standard monitoring required');
    detectedKeywords.moderate.forEach(keyword => {
      insights.push(`📋 **${keyword}**: Document and monitor`);
    });
  }

  // Specific medical condition alerts
  if (narrative.includes('diabetes') || narrative.includes('diabetic')) {
    insights.push('💉 **Diabetes Alert**: Check blood glucose, monitor for hypo/hyperglycemia');
  }
  
  if (narrative.includes('heart') || narrative.includes('cardiac')) {
    insights.push('❤️ **Cardiac History**: Prepare for cardiac assessment, consider ECG');
  }
  
  if (narrative.includes('copd') || narrative.includes('asthma') || narrative.includes('lung')) {
    insights.push('🫁 **Respiratory Condition**: Monitor airway, prepare breathing treatments');
  }
  
  if (narrative.includes('stroke') || narrative.includes('cva')) {
    insights.push('🧠 **Stroke History**: Monitor for new symptoms, check FAST signs');
  }

  // Medication alerts
  if (narrative.includes('blood thinner') || narrative.includes('warfarin') || narrative.includes('coumadin')) {
    insights.push('🩸 **Blood Thinner Alert**: Increased bleeding risk, check for bleeding');
  }
  
  if (narrative.includes('insulin')) {
    insights.push('💉 **Insulin Alert**: Check blood glucose, watch for hypoglycemia');
  }

  // Risk-specific insights based on vital signs
  if (respiratoryRisk === 'Critical' || respiratoryRisk === 'High') {
    insights.push('🫁 **Respiratory Distress Detected**: Prepare for airway management');
  }
  
  if (cardiovascularRisk === 'Critical' || cardiovascularRisk === 'High') {
    insights.push('❤️ **Cardiovascular Stress Detected**: Monitor ECG, prepare for cardiac care');
  }
  
  if (neurologicalRisk === 'Critical' || neurologicalRisk === 'High') {
    insights.push('🧠 **Neurological Concerns**: Monitor consciousness, check for stroke signs');
  }

  // Overall risk assessment integration
  if (riskScore >= 15) {
    insights.push('🚨 **HIGH NARRATIVE RISK SCORE**: Multiple concerning symptoms detected');
  } else if (riskScore >= 8) {
    insights.push('⚠️ **MODERATE NARRATIVE RISK SCORE**: Several symptoms require attention');
  } else if (riskScore >= 3) {
    insights.push('📋 **LOW NARRATIVE RISK SCORE**: Minor symptoms noted');
  }

  // General safety reminders
  if (riskLevel === 'Critical') {
    insights.push('🚨 **CRITICAL PATIENT**: Stay alert for rapid deterioration, prepare for emergency');
  } else if (riskLevel === 'High') {
    insights.push('⚠️ **HIGH RISK PATIENT**: Monitor closely, be ready to escalate care');
  }

  return insights;
}

function ResultsContent() {
  const searchParams = useSearchParams();
  
  const vitals = {
    spo2: searchParams.get('spo2') || '',
    rr: searchParams.get('rr') || '',
    hr: searchParams.get('hr') || '',
    sbp: searchParams.get('sbp') || '',
    gcs_eye: searchParams.get('gcs_eye') || '',
    gcs_verbal: searchParams.get('gcs_verbal') || '',
    gcs_motor: searchParams.get('gcs_motor') || '',
  };

  const patientNarrative = searchParams.get('patient_narrative') || '';

  const riskLevel = searchParams.get('risk_level') || 'Low';
  const respiratoryRisk = searchParams.get('respiratory_risk') || 'Low';
  const cardiovascularRisk = searchParams.get('cardiovascular_risk') || 'Low';
  const neurologicalRisk = searchParams.get('neurological_risk') || 'Low';
  
  const roxScore = searchParams.get('rox_score') || '0';
  const gcsTotal = searchParams.get('gcs_total') || '0';
  const rppScore = searchParams.get('rpp_score') || '0';
  const narrativeRiskScore = searchParams.get('narrative_risk_score') || '0';

  const recommendations = getClinicalRecommendations(riskLevel);
  
  // Get narrative insights from API if available, otherwise use local analysis
  const apiNarrativeInsights = searchParams.get('narrative_insights');
  let narrativeInsights: string[];
  
  if (apiNarrativeInsights) {
    try {
      narrativeInsights = JSON.parse(decodeURIComponent(apiNarrativeInsights));
    } catch {
      narrativeInsights = extractNarrativeInsights(
        patientNarrative,
        riskLevel,
        respiratoryRisk,
        cardiovascularRisk,
        neurologicalRisk
      );
    }
  } else {
    narrativeInsights = extractNarrativeInsights(
      patientNarrative,
      riskLevel,
      respiratoryRisk,
      cardiovascularRisk,
      neurologicalRisk
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">Risk Assessment Results</h1>
              <p className="text-gray-600 text-sm">Emergency Medical Services - Clinical Assessment</p>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Assessment ID</div>
              <div className="text-sm font-mono text-gray-700">EMS-{Date.now().toString().slice(-6)}</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Overall Risk Assessment */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">Risk Stratification</h2>
            </div>
            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getRiskBadgeColor(riskLevel)}`}>
                  {riskLevel} Risk
                </span>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-700">Response Priority</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {riskLevel === 'Critical' ? 'Immediate' : riskLevel === 'High' ? 'High' : riskLevel === 'Moderate' ? 'Moderate' : 'Routine'}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-700">Required Resources</div>
                  <div className="text-sm font-semibold text-gray-900">
                    {riskLevel === 'Critical' ? 'ALS Required' : riskLevel === 'High' ? 'ALS Consideration' : 'BLS'}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-700">Transport Destination</div>
                  <div className="text-sm font-semibold text-gray-900">
                    {recommendations.transport_destination}
                  </div>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="text-sm font-medium text-gray-700">Monitoring Level</div>
                  <div className="text-sm font-semibold text-gray-900">
                    {recommendations.monitoring_level}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Clinical Data */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">Clinical Data</h2>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Vital Signs */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Vital Signs</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">SpO₂:</span>
                      <span className="font-medium">{vitals.spo2}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Respiratory Rate:</span>
                      <span className="font-medium">{vitals.rr} bpm</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Heart Rate:</span>
                      <span className="font-medium">{vitals.hr} bpm</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Systolic BP:</span>
                      <span className="font-medium">{vitals.sbp} mmHg</span>
                    </div>
                  </div>
                </div>

                {/* Glasgow Coma Scale */}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Glasgow Coma Scale</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Eye Opening:</span>
                      <span className="font-medium">{vitals.gcs_eye}/4</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Verbal Response:</span>
                      <span className="font-medium">{vitals.gcs_verbal}/5</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Motor Response:</span>
                      <span className="font-medium">{vitals.gcs_motor}/6</span>
                    </div>
                    <div className="flex justify-between border-t pt-2">
                      <span className="text-gray-900 font-medium">Total:</span>
                      <span className="font-bold text-lg">{gcsTotal}/15</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Clinical Scores */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Clinical Assessment Scores</h3>
                                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                          {/* ROX Score */}
                          <div className="border border-gray-200 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-700">ROX Score</h4>
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getRiskBadgeColor(respiratoryRisk)}`}>
                                {respiratoryRisk}
                              </span>
                            </div>
                            <div className="text-2xl font-bold text-gray-900 mb-1">{roxScore}</div>
                            <p className="text-xs text-gray-500">SpO₂/FiO₂ ratio ÷ Respiratory Rate</p>
                          </div>
                          
                          {/* GCS Score */}
                          <div className="border border-gray-200 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-700">GCS Total</h4>
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getRiskBadgeColor(neurologicalRisk)}`}>
                                {neurologicalRisk}
                              </span>
                            </div>
                            <div className="text-2xl font-bold text-gray-900 mb-1">{gcsTotal}/15</div>
                            <p className="text-xs text-gray-500">Eye + Verbal + Motor Response</p>
                          </div>
                          
                          {/* RPP Score */}
                          <div className="border border-gray-200 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-700">RPP Score</h4>
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getRiskBadgeColor(cardiovascularRisk)}`}>
                                {cardiovascularRisk}
                              </span>
                            </div>
                            <div className="text-2xl font-bold text-gray-900 mb-1">{rppScore?.toLocaleString()}</div>
                            <p className="text-xs text-gray-500">Heart Rate × Systolic BP</p>
                          </div>
                          
                          {/* Narrative Risk Score */}
                          <div className="border border-gray-200 rounded-lg p-4">
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="text-sm font-medium text-gray-700">Narrative Risk</h4>
                              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${
                                parseInt(narrativeRiskScore) >= 15 ? 'bg-red-100 text-red-800 border-red-200' :
                                parseInt(narrativeRiskScore) >= 8 ? 'bg-orange-100 text-orange-800 border-orange-200' :
                                parseInt(narrativeRiskScore) >= 3 ? 'bg-yellow-100 text-yellow-800 border-yellow-200' :
                                'bg-green-100 text-green-800 border-green-200'
                              }`}>
                                {parseInt(narrativeRiskScore) >= 15 ? 'High' :
                                 parseInt(narrativeRiskScore) >= 8 ? 'Moderate' :
                                 parseInt(narrativeRiskScore) >= 3 ? 'Low' : 'None'}
                              </span>
                            </div>
                            <div className="text-2xl font-bold text-gray-900 mb-1">{narrativeRiskScore}</div>
                            <p className="text-xs text-gray-500">Keyword-based risk score</p>
                          </div>
                        </div>
        </div>

        {/* Clinical Recommendations */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Clinical Recommendations</h3>
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  <strong>Recommended Interventions:</strong> {recommendations.interventions}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Patient Narrative & Insights */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Patient Narrative */}
          {patientNarrative && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-900">Patient Narrative</h2>
              </div>
              <div className="p-6">
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-900 text-sm leading-relaxed">{patientNarrative}</p>
                </div>
              </div>
            </div>
          )}

          {/* Narrative Insights */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-900">Clinical Insights</h2>
            </div>
            <div className="p-6">
              {narrativeInsights.length > 0 ? (
                <div className="space-y-3">
                  {narrativeInsights.map((insight, index) => (
                    <div key={index} className="flex items-start">
                      <div className="flex-shrink-0 w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                        <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm text-blue-900" dangerouslySetInnerHTML={{ __html: insight }}></p>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-blue-700">Continue standard monitoring. No specific concerns identified from the narrative.</p>
              )}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link 
            href="/ems-ai/triage"
            className="bg-blue-600 text-white px-6 py-3 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors text-center"
          >
            New Assessment
          </Link>
          <Link 
            href="/ems-ai/behind-model" 
            className="bg-gray-600 text-white px-6 py-3 rounded-md font-medium hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors text-center"
          >
            About This Tool
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ResultsContent />
    </Suspense>
  );
} 