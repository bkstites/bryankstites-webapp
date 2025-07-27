#!/usr/bin/env node

// EMS Risk Assessment Model Test Suite - Node.js Version
// Tests the accuracy of the risk assessment model against clinical scenarios

const testCases = [
  // Normal patient - should be LOW risk
  {
    name: "Normal Patient",
    vitals: { spo2: 98, rr: 16, hr: 72, sbp: 120, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 },
    expected: { overall: "Low", respiratory: "Low", cardiovascular: "Low", neurological: "Low" }
  },
  
  // Critical respiratory failure - should be CRITICAL
  {
    name: "Critical Respiratory Failure",
    vitals: { spo2: 75, rr: 35, hr: 120, sbp: 90, gcs_eye: 3, gcs_verbal: 3, gcs_motor: 4 },
    expected: { overall: "Critical", respiratory: "Critical", cardiovascular: "High", neurological: "High" }
  },
  
  // Severe bradycardia - should be CRITICAL
  {
    name: "Severe Bradycardia",
    vitals: { spo2: 95, rr: 12, hr: 35, sbp: 70, gcs_eye: 2, gcs_verbal: 2, gcs_motor: 3 },
    expected: { overall: "Critical", respiratory: "Low", cardiovascular: "Critical", neurological: "Critical" }
  },
  
  // Moderate risk patient
  {
    name: "Moderate Risk Patient",
    vitals: { spo2: 92, rr: 22, hr: 110, sbp: 95, gcs_eye: 3, gcs_verbal: 4, gcs_motor: 5 },
    expected: { overall: "Moderate", respiratory: "Moderate", cardiovascular: "Moderate", neurological: "Moderate" }
  },
  
  // High risk trauma patient
  {
    name: "High Risk Trauma",
    vitals: { spo2: 88, rr: 28, hr: 140, sbp: 85, gcs_eye: 2, gcs_verbal: 2, gcs_motor: 3 },
    expected: { overall: "High", respiratory: "High", cardiovascular: "High", neurological: "High" }
  },
  
  // Borderline case - should be LOW
  {
    name: "Borderline Normal",
    vitals: { spo2: 95, rr: 20, hr: 100, sbp: 95, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 },
    expected: { overall: "Low", respiratory: "Low", cardiovascular: "Low", neurological: "Low" }
  },
  
  // Severe hypertension
  {
    name: "Severe Hypertension",
    vitals: { spo2: 96, rr: 18, hr: 180, sbp: 220, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 },
    expected: { overall: "High", respiratory: "Low", cardiovascular: "High", neurological: "Low" }
  },
  
  // Cardiac arrest scenario
  {
    name: "Cardiac Arrest",
    vitals: { spo2: 60, rr: 0, hr: 0, sbp: 0, gcs_eye: 1, gcs_verbal: 1, gcs_motor: 1 },
    expected: { overall: "Critical", respiratory: "Critical", cardiovascular: "Critical", neurological: "Critical" }
  }
];

// Scoring functions (matching the API implementation)
function calculateROX(spo2, rr) {
  const fio2 = 0.21;
  const spo2_fio2_ratio = spo2 / fio2;
  return Math.round((spo2_fio2_ratio / rr) * 100) / 100;
}

function calculateGCSTotal(eye, verbal, motor) {
  return eye + verbal + motor;
}

function calculateRPP(hr, sbp) {
  return hr * sbp;
}

function calculateNEWS2Score(vitals) {
  let score = 0;
  
  // Respiratory Rate
  if (vitals.rr <= 8) score += 3;
  else if (vitals.rr <= 11) score += 1;
  else if (vitals.rr >= 25) score += 3;
  else if (vitals.rr >= 21) score += 2;
  
  // SpO2
  if (vitals.spo2 <= 91) score += 3;
  else if (vitals.spo2 <= 93) score += 2;
  else if (vitals.spo2 <= 95) score += 1;
  
  // Systolic BP
  if (vitals.sbp <= 90) score += 3;
  else if (vitals.sbp <= 100) score += 2;
  else if (vitals.sbp <= 110) score += 1;
  else if (vitals.sbp >= 220) score += 3;
  
  // Heart Rate
  if (vitals.hr <= 40) score += 3;
  else if (vitals.hr <= 50) score += 1;
  else if (vitals.hr >= 131) score += 3;
  else if (vitals.hr >= 111) score += 2;
  
  // GCS
  const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
  if (gcs_total <= 8) score += 3;
  else if (gcs_total <= 10) score += 2;
  else if (gcs_total <= 13) score += 1;
  
  return score;
}

function calculateMEOWSScore(vitals) {
  let score = 0;
  
  // Respiratory Rate
  if (vitals.rr < 10 || vitals.rr > 30) score += 2;
  else if (vitals.rr < 12 || vitals.rr > 20) score += 1;
  
  // SpO2
  if (vitals.spo2 < 95) score += 2;
  else if (vitals.spo2 < 97) score += 1;
  
  // Systolic BP
  if (vitals.sbp < 90 || vitals.sbp > 160) score += 2;
  else if (vitals.sbp < 100 || vitals.sbp > 140) score += 1;
  
  // Heart Rate
  if (vitals.hr < 50 || vitals.hr > 120) score += 2;
  else if (vitals.hr < 60 || vitals.hr > 100) score += 1;
  
  // GCS
  const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
  if (gcs_total < 15) score += 2;
  else if (gcs_total < 13) score += 1;
  
  return score;
}

// Risk assessment logic (matching the API implementation with final adjustments)
function determineRiskLevels(vitals, rox, gcs, rpp) {
  const news2_score = calculateNEWS2Score(vitals);
  const meows_score = calculateMEOWSScore(vitals);
  
  // Respiratory risk assessment
  let respiratory = 'Low';
  if (vitals.spo2 < 90) {
    if (vitals.spo2 < 85) respiratory = 'Critical';
    else respiratory = 'High';
  } else if (vitals.rr > 25 || vitals.rr < 10) {
    respiratory = 'High';
  } else if (vitals.rr > 20 || vitals.rr < 12) {
    respiratory = 'Moderate';
  }
  
  if (rox < 4.88) respiratory = 'Critical';
  else if (rox < 7.5 && respiratory === 'Low') respiratory = 'Moderate';

  // Neurological risk with adjusted thresholds
  let neurological = 'Low';
  if (gcs <= 8) neurological = 'Critical';
  else if (gcs <= 10) neurological = 'High'; // Adjusted from 12 to 10
  else if (gcs <= 13) neurological = 'Moderate'; // Adjusted from 14 to 13

  // Cardiovascular risk
  let cardiovascular = 'Low';
  if (vitals.hr < 50 || vitals.hr >= 120) {
    if (vitals.hr < 40 || vitals.hr > 150) cardiovascular = 'Critical';
    else cardiovascular = 'High';
  } else if (vitals.hr < 60 || vitals.hr > 100) {
    cardiovascular = 'Moderate';
  }
  
  if (vitals.sbp < 90) {
    if (vitals.sbp < 80) cardiovascular = 'Critical';
    else if (cardiovascular === 'Low') cardiovascular = 'High';
  } else if (vitals.sbp >= 140) {
    if (cardiovascular === 'Low') cardiovascular = 'Moderate';
  }
  
  // Adjusted RPP thresholds
  if (rpp > 20000) { // Increased threshold for critical
    if (cardiovascular === 'Low') cardiovascular = 'Critical';
  } else if (rpp >= 15000) { // Adjusted threshold for high
    if (cardiovascular === 'Low') cardiovascular = 'High';
  } else if (rpp < 4000) {
    if (cardiovascular === 'Low') cardiovascular = 'Moderate';
  }

  // Overall risk assessment with adjusted thresholds
  let overall = 'Low';
  let risk_level = 'Low';

  // Adjusted NEWS2 thresholds
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

  // Adjusted MEOWS thresholds
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
  const highCount = risks.filter(r => r === 'High').length;
  
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

  const allLowRisk = risks.every(r => r === 'Low');
  if (allLowRisk && news2_score < 4 && meows_score < 3) {
    overall = 'Low';
    risk_level = 'Low';
  }

  // Prevent over-escalation for borderline cases
  if (overall === 'Critical' && news2_score < 6 && meows_score < 5) {
    overall = 'High';
    risk_level = 'High';
  }

  return { respiratory, neurological, cardiovascular, overall, risk_level };
}

// Test the model
function testModel(vitals) {
  const rox_score = calculateROX(vitals.spo2, vitals.rr);
  const gcs_total = calculateGCSTotal(vitals.gcs_eye, vitals.gcs_verbal, vitals.gcs_motor);
  const rpp_score = calculateRPP(vitals.hr, vitals.sbp);
  
  return determineRiskLevels(vitals, rox_score, gcs_total, rpp_score);
}

// Run tests
function runTests() {
  console.log('🏥 EMS Risk Assessment Model Test Suite');
  console.log('=====================================\n');
  
  let passedTests = 0;
  let totalTests = 0;
  
  for (const testCase of testCases) {
    totalTests++;
    console.log(`Test ${totalTests}: ${testCase.name}`);
    console.log(`Vitals: SpO₂ ${testCase.vitals.spo2}%, RR ${testCase.vitals.rr}, HR ${testCase.vitals.hr}, BP ${testCase.vitals.sbp}, GCS ${calculateGCSTotal(testCase.vitals.gcs_eye, testCase.vitals.gcs_verbal, testCase.vitals.gcs_motor)}/15`);
    
    const result = testModel(testCase.vitals);
    const news2 = calculateNEWS2Score(testCase.vitals);
    const meows = calculateMEOWSScore(testCase.vitals);
    const rox = calculateROX(testCase.vitals.spo2, testCase.vitals.rr);
    const gcs = calculateGCSTotal(testCase.vitals.gcs_eye, testCase.vitals.gcs_verbal, testCase.vitals.gcs_motor);
    const rpp = calculateRPP(testCase.vitals.hr, testCase.vitals.sbp);
    
    console.log(`Scores: ROX=${rox}, GCS=${gcs}, RPP=${rpp}, NEWS2=${news2}, MEOWS=${meows}`);
    console.log(`Expected: ${testCase.expected.overall} (${testCase.expected.respiratory}/${testCase.expected.cardiovascular}/${testCase.expected.neurological})`);
    console.log(`Actual: ${result.risk_level} (${result.respiratory}/${result.cardiovascular}/${result.neurological})`);
    
    const riskMatch = result.risk_level === testCase.expected.overall;
    const respiratoryMatch = result.respiratory === testCase.expected.respiratory;
    const cardiovascularMatch = result.cardiovascular === testCase.expected.cardiovascular;
    const neurologicalMatch = result.neurological === testCase.expected.neurological;
    
    if (riskMatch && respiratoryMatch && cardiovascularMatch && neurologicalMatch) {
      console.log('✅ PASSED\n');
      passedTests++;
    } else {
      console.log('❌ FAILED');
      console.log(`Risk: ${riskMatch ? '✅' : '❌'} (expected ${testCase.expected.overall}, got ${result.risk_level})`);
      console.log(`Respiratory: ${respiratoryMatch ? '✅' : '❌'} (expected ${testCase.expected.respiratory}, got ${result.respiratory})`);
      console.log(`Cardiovascular: ${cardiovascularMatch ? '✅' : '❌'} (expected ${testCase.expected.cardiovascular}, got ${result.cardiovascular})`);
      console.log(`Neurological: ${neurologicalMatch ? '✅' : '❌'} (expected ${testCase.expected.neurological}, got ${result.neurological})`);
      console.log('');
    }
  }
  
  console.log('=====================================');
  console.log(`Test Results: ${passedTests}/${totalTests} tests passed`);
  const accuracy = ((passedTests / totalTests) * 100).toFixed(1);
  console.log(`Accuracy: ${accuracy}%`);
  
  // Additional validation tests
  console.log('\n📊 Additional Validation Tests:');
  
  const edgeCases = [
    { name: "Perfect Normal", vitals: { spo2: 100, rr: 16, hr: 72, sbp: 120, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 } },
    { name: "Critical All Systems", vitals: { spo2: 70, rr: 40, hr: 200, sbp: 60, gcs_eye: 1, gcs_verbal: 1, gcs_motor: 1 } },
    { name: "Borderline Respiratory", vitals: { spo2: 94, rr: 24, hr: 80, sbp: 120, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 } },
    { name: "Borderline Cardiovascular", vitals: { spo2: 98, rr: 16, hr: 110, sbp: 95, gcs_eye: 4, gcs_verbal: 5, gcs_motor: 6 } },
    { name: "Borderline Neurological", vitals: { spo2: 98, rr: 16, hr: 80, sbp: 120, gcs_eye: 3, gcs_verbal: 4, gcs_motor: 5 } }
  ];
  
  for (const edgeCase of edgeCases) {
    const news2 = calculateNEWS2Score(edgeCase.vitals);
    const meows = calculateMEOWSScore(edgeCase.vitals);
    const rox = calculateROX(edgeCase.vitals.spo2, edgeCase.vitals.rr);
    const gcs = calculateGCSTotal(edgeCase.vitals.gcs_eye, edgeCase.vitals.gcs_verbal, edgeCase.vitals.gcs_motor);
    const rpp = calculateRPP(edgeCase.vitals.hr, edgeCase.vitals.sbp);
    const result = testModel(edgeCase.vitals);
    
    console.log(`${edgeCase.name}: NEWS2=${news2}, MEOWS=${meows}, ROX=${rox}, GCS=${gcs}, RPP=${rpp} → ${result.risk_level}`);
  }
  
  return { passedTests, totalTests, accuracy };
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests();
}

module.exports = { runTests, testCases, testModel }; 