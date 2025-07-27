import EMSLayout from './layout';
import Link from 'next/link';

export default function EMSHomePage() {
  return (
    <EMSLayout>
      <div className="min-h-screen bg-gray-50 py-8 px-4">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
            <div className="text-center">
              <h1 className="text-3xl font-semibold text-gray-900 mb-2">Emergency Medical Services</h1>
              <h2 className="text-xl text-gray-600 mb-4">Risk Assessment Tool</h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Clinical decision support system for pre-hospital emergency assessment and risk stratification.
              </p>
            </div>
          </div>

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            {/* Tool Description */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Clinical Assessment</h3>
              <p className="text-gray-700 mb-4">
                This tool provides real-time risk assessment based on patient vital signs and Glasgow Coma Scale scores. 
                Designed for emergency medical personnel to support clinical decision-making in the field.
              </p>
              <div className="space-y-3">
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                    <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Risk Stratification</p>
                    <p className="text-sm text-gray-600">Automated assessment of patient risk levels</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                    <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Resource Allocation</p>
                    <p className="text-sm text-gray-600">Guidance for appropriate resource deployment</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="flex-shrink-0 w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                    <svg className="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-900">Clinical Support</p>
                    <p className="text-sm text-gray-600">Evidence-based recommendations for field crews</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Assessment Parameters */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Assessment Parameters</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Vital Signs</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">SpO₂</span>
                      <span className="text-gray-600 ml-1">(% saturation)</span>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Respiratory Rate</span>
                      <span className="text-gray-600 ml-1">(breaths/min)</span>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Heart Rate</span>
                      <span className="text-gray-600 ml-1">(bpm)</span>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Systolic BP</span>
                      <span className="text-gray-600 ml-1">(mmHg)</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Glasgow Coma Scale</h4>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Eye Opening</span>
                      <span className="text-gray-600 ml-1">(1-4)</span>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Verbal</span>
                      <span className="text-gray-600 ml-1">(1-5)</span>
                    </div>
                    <div className="bg-gray-50 rounded p-2">
                      <span className="font-medium">Motor</span>
                      <span className="text-gray-600 ml-1">(1-6)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Call to Action */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
            <h3 className="text-xl font-medium text-gray-900 mb-4">Begin Patient Assessment</h3>
            <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
              Enter patient vital signs and Glasgow Coma Scale scores to receive real-time risk assessment and clinical recommendations.
            </p>
            <Link 
              href="/ems-ai/triage" 
              className="inline-flex items-center px-6 py-3 bg-blue-600 text-white font-medium rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
            >
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Assessment
            </Link>
          </div>

          {/* Footer Info */}
          <div className="mt-8 text-center text-xs text-gray-500">
            <p>This tool provides clinical decision support and should be used in conjunction with professional medical judgment.</p>
          </div>
        </div>
      </div>
    </EMSLayout>
  );
} 