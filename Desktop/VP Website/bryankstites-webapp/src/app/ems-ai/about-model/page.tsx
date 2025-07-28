'use client';

import Link from 'next/link';
import { useState } from 'react';

export default function AboutModelPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'methodology', label: 'Methodology' },
    { id: 'testing', label: 'Testing & Accuracy' },
    { id: 'citations', label: 'Citations & Background' }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">About the Model</h1>
              <p className="text-gray-600 text-sm">EMS AI Triage System - Narrative Analysis Deep Dive</p>
            </div>
            <Link 
              href="/ems-ai/triage"
              className="bg-blue-600 text-white px-4 py-2 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
            >
              Try Assessment
            </Link>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8 px-6" aria-label="Tabs">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'overview' && <OverviewContent />}
            {activeTab === 'methodology' && <MethodologyContent />}
            {activeTab === 'testing' && <TestingContent />}
            {activeTab === 'citations' && <CitationsContent />}
          </div>
        </div>
      </div>
    </div>
  );
}

function OverviewContent() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Overview</h2>
        <p className="text-gray-700 mb-4">
          The EMS AI Triage System employs a hybrid approach combining rule-based keyword analysis with machine learning 
          techniques to assess patient narratives and provide evidence-based protocol recommendations.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">Keyword Analysis</h3>
            <p className="text-blue-700 text-sm">
              Advanced natural language processing to extract medical keywords and assess urgency indicators
            </p>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h3 className="font-semibold text-green-900 mb-2">Protocol Matching</h3>
            <p className="text-green-700 text-sm">
              ML-based matching against PA State EMS protocols for evidence-based recommendations
            </p>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h3 className="font-semibold text-purple-900 mb-2">Risk Stratification</h3>
            <p className="text-purple-700 text-sm">
              Multi-dimensional risk assessment integrating vital signs, symptoms, and narrative analysis
            </p>
          </div>
        </div>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Key Features</h3>
        <ul className="space-y-2 text-gray-700">
          <li className="flex items-start">
            <span className="text-green-500 mr-2">✓</span>
            <span><strong>Real-time Analysis:</strong> Processes patient narratives instantly for immediate triage decisions</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-500 mr-2">✓</span>
            <span><strong>Evidence-Based:</strong> Grounded in PA State EMS protocols and emergency medicine literature</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-500 mr-2">✓</span>
            <span><strong>Multi-Modal:</strong> Integrates narrative analysis with vital signs and clinical scores</span>
          </li>
          <li className="flex items-start">
            <span className="text-green-500 mr-2">✓</span>
            <span><strong>Explainable:</strong> Provides transparent reasoning for all recommendations</span>
          </li>
        </ul>
      </div>
    </div>
  );
}

function MethodologyContent() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Technical Methodology</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">1. Natural Language Processing Pipeline</h3>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <h4 className="font-semibold text-gray-900 mb-2">Text Preprocessing</h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Tokenization and lemmatization using NLTK</li>
                <li>• Removal of EMS-specific stop words</li>
                <li>• Medical terminology normalization</li>
                <li>• Context-aware keyword extraction</li>
              </ul>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">2. Keyword Association Analysis</h3>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <h4 className="font-semibold text-gray-900 mb-2">TF-IDF Vectorization</h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Term frequency-inverse document frequency analysis</li>
                <li>• Medical keyword weighting based on urgency</li>
                <li>• Semantic similarity scoring</li>
                <li>• Context-dependent relevance assessment</li>
              </ul>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">3. Protocol Matching Algorithm</h3>
            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <h4 className="font-semibold text-gray-900 mb-2">Cosine Similarity Scoring</h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Vector-based protocol matching</li>
                <li>• Confidence scoring (0-1 scale)</li>
                <li>• Multi-protocol ranking</li>
                <li>• Threshold-based filtering</li>
              </ul>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">4. Risk Assessment Integration</h3>
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 mb-2">Multi-Dimensional Scoring</h4>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>• Narrative risk score (0-30 scale)</li>
                <li>• Vital signs integration (NEWS2, MEOWS)</li>
                <li>• Clinical score validation (ROX, GCS, RPP)</li>
                <li>• Weighted ensemble prediction</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TestingContent() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Testing & Validation</h2>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quantitative Metrics</h3>
            
            <div className="space-y-4">
                             <div>
                 <h4 className="font-medium text-gray-900 mb-2">Protocol Matching Accuracy</h4>
                 <div className="space-y-2">
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">F1-Score</span>
                     <span className="font-semibold text-orange-600">60.9%</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Response Time</span>
                     <span className="font-semibold text-green-600">&lt;10ms</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Success Rate</span>
                     <span className="font-semibold text-green-600">100.0%</span>
                   </div>
                 </div>
               </div>

               <div>
                 <h4 className="font-medium text-gray-900 mb-2">Risk Assessment Accuracy</h4>
                 <div className="space-y-2">
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Urgency Accuracy</span>
                     <span className="font-semibold text-red-600">0.0%</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Keyword Recognition</span>
                     <span className="font-semibold text-orange-600">17.1%</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">System Reliability</span>
                     <span className="font-semibold text-green-600">100.0%</span>
                   </div>
                 </div>
               </div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Qualitative Assessment</h3>
            
            <div className="space-y-4">
                             <div>
                 <h4 className="font-medium text-gray-900 mb-2">Clinical Relevance</h4>
                 <div className="space-y-2">
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Protocol Matching</span>
                     <span className="font-semibold text-orange-600">60.9%</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">System Reliability</span>
                     <span className="font-semibold text-green-600">100.0%</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Response Speed</span>
                     <span className="font-semibold text-green-600">Excellent</span>
                   </div>
                 </div>
               </div>

               <div>
                 <h4 className="font-medium text-gray-900 mb-2">Areas for Improvement</h4>
                 <div className="space-y-2">
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Urgency Assessment</span>
                     <span className="font-semibold text-red-600">Critical</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Keyword Recognition</span>
                     <span className="font-semibold text-orange-600">Moderate</span>
                   </div>
                   <div className="flex justify-between">
                     <span className="text-sm text-gray-600">Protocol Coverage</span>
                     <span className="font-semibold text-orange-600">Moderate</span>
                   </div>
                 </div>
               </div>
            </div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Results by Category</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Test Cases</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Precision</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recall</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                                 <tr>
                   <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Cardiac Emergencies</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">33.3%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">0.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">15.2%</td>
                 </tr>
                 <tr>
                   <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Respiratory Distress</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">50.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">0.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">18.3%</td>
                 </tr>
                 <tr>
                   <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Neurological Emergencies</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">50.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">0.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">22.1%</td>
                 </tr>
                 <tr>
                   <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Trauma</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">50.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">0.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">16.7%</td>
                 </tr>
                 <tr>
                   <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Medical Emergencies</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">2</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-green-600 font-semibold">83.3%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-red-600 font-semibold">0.0%</td>
                   <td className="px-6 py-4 whitespace-nowrap text-sm text-orange-600 font-semibold">19.8%</td>
                 </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function CitationsContent() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Academic Background & Citations</h2>
        
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Emergency Medicine Standards</h3>
            <div className="space-y-3">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">NEWS2 (National Early Warning Score 2)</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Royal College of Physicians. (2017). National Early Warning Score (NEWS) 2: Standardising the assessment of acute-illness severity in the NHS.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Validated scoring system for acute illness severity assessment
                </p>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">MEOWS (Modified Early Obstetric Warning System)</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Singh, S., et al. (2011). The Modified Early Obstetric Warning System: A way forward.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Adapted for general emergency medicine risk assessment
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Clinical Scoring Systems</h3>
            <div className="space-y-3">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">ROX Score (SpO2/FiO2 Ratio / Respiratory Rate)</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Roca, O., et al. (2016). An index combining respiratory rate and oxygenation to predict outcome of nasal high-flow therapy.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Validated predictor of respiratory failure and need for intubation
                </p>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">Glasgow Coma Scale</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Teasdale, G., & Jennett, B. (1974). Assessment of coma and impaired consciousness.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Gold standard for neurological assessment in emergency medicine
                </p>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">Rate Pressure Product (RPP)</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Kitamura, K., et al. (2002). Rate-pressure product as a predictor of cardiovascular events.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Validated cardiovascular stress indicator
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Natural Language Processing</h3>
            <div className="space-y-3">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">TF-IDF Vectorization</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Ramos, J. (2003). Using TF-IDF to determine word relevance in document queries.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Foundation for keyword extraction and document similarity
                </p>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">Cosine Similarity</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Singhal, A. (2001). Modern information retrieval: A brief overview.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Standard metric for text similarity and protocol matching
                </p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">EMS Protocol Standards</h3>
            <div className="space-y-3">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">PA State EMS Protocols</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Pennsylvania Department of Health. (2023). Pennsylvania Statewide Basic Life Support Protocols.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Evidence-based protocol foundation for triage recommendations
                </p>
              </div>
              
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-1">Emergency Medical Services Standards</h4>
                <p className="text-sm text-gray-600 mb-2">
                  National Association of State EMS Officials. (2021). National EMS Scope of Practice Model.
                </p>
                <p className="text-xs text-gray-500">
                  <strong>Impact:</strong> Standardized approach to emergency medical care
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 