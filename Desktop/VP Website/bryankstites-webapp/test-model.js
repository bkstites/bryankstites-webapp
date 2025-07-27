// EMS Risk Assessment Model Test Suite
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

// Mock API call function
async function testModel(vitals) {
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(vitals),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    return null;
  }
}

// Test scoring functions directly
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

// Run tests
async function runTests() {
  console.log('🏥 EMS Risk Assessment Model Test Suite');
  console.log('=====================================\n');
  
  let passedTests = 0;
  let totalTests = 0;
  
  for (const testCase of testCases) {
    totalTests++;
    console.log(`Test ${totalTests}: ${testCase.name}`);
    console.log(`Vitals: SpO₂ ${testCase.vitals.spo2}%, RR ${testCase.vitals.rr}, HR ${testCase.vitals.hr}, BP ${testCase.vitals.sbp}, GCS ${calculateGCSTotal(testCase.vitals.gcs_eye, testCase.vitals.gcs_verbal, testCase.vitals.gcs_motor)}/15`);
    
    // Calculate expected scores
    const expectedRox = calculateROX(testCase.vitals.spo2, testCase.vitals.rr);
    const expectedGcs = calculateGCSTotal(testCase.vitals.gcs_eye, testCase.vitals.gcs_verbal, testCase.vitals.gcs_motor);
    const expectedRpp = calculateRPP(testCase.vitals.hr, testCase.vitals.sbp);
    const expectedNews2 = calculateNEWS2Score(testCase.vitals);
    const expectedMeows = calculateMEOWSScore(testCase.vitals);
    
    console.log(`Expected Scores: ROX=${expectedRox}, GCS=${expectedGcs}, RPP=${expectedRpp}, NEWS2=${expectedNews2}, MEOWS=${expectedMeows}`);
    console.log(`Expected Risk: ${testCase.expected.overall} (${testCase.expected.respiratory}/${testCase.expected.cardiovascular}/${testCase.expected.neurological})`);
    
    // Test API if available
    if (typeof fetch !== 'undefined') {
      const result = await testModel(testCase.vitals);
      if (result) {
        console.log(`Actual Result: ${result.risk_level} (${result.respiratory_risk}/${result.cardiovascular_risk}/${result.neurological_risk})`);
        console.log(`Actual Scores: ROX=${result.rox_score}, GCS=${result.gcs_total}, RPP=${result.rpp_score}`);
        
        // Check accuracy
        const riskMatch = result.risk_level === testCase.expected.overall;
        const respiratoryMatch = result.respiratory_risk === testCase.expected.respiratory;
        const cardiovascularMatch = result.cardiovascular_risk === testCase.expected.cardiovascular;
        const neurologicalMatch = result.neurological_risk === testCase.expected.neurological;
        
        if (riskMatch && respiratoryMatch && cardiovascularMatch && neurologicalMatch) {
          console.log('✅ PASSED\n');
          passedTests++;
        } else {
          console.log('❌ FAILED');
          console.log(`Risk: ${riskMatch ? '✅' : '❌'} (expected ${testCase.expected.overall}, got ${result.risk_level})`);
          console.log(`Respiratory: ${respiratoryMatch ? '✅' : '❌'} (expected ${testCase.expected.respiratory}, got ${result.respiratory_risk})`);
          console.log(`Cardiovascular: ${cardiovascularMatch ? '✅' : '❌'} (expected ${testCase.expected.cardiovascular}, got ${result.cardiovascular_risk})`);
          console.log(`Neurological: ${neurologicalMatch ? '✅' : '❌'} (expected ${testCase.expected.neurological}, got ${result.neurological_risk})`);
          console.log('');
        }
      } else {
        console.log('⚠️  API test skipped (server not available)\n');
      }
    } else {
      console.log('⚠️  API test skipped (fetch not available)\n');
    }
  }
  
  console.log('=====================================');
  console.log(`Test Results: ${passedTests}/${totalTests} tests passed`);
  console.log(`Accuracy: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
  
  // Additional validation tests
  console.log('\n📊 Additional Validation Tests:');
  
  // Test edge cases
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
    
    console.log(`${edgeCase.name}: NEWS2=${news2}, MEOWS=${meows}, ROX=${rox}, GCS=${gcs}, RPP=${rpp}`);
  }
}

// Run tests if in Node.js environment
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runTests, testCases };
} else {
  // Run tests in browser environment
  runTests();
} 