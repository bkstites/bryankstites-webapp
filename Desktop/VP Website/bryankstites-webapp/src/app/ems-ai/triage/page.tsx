'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

interface FormData {
  spo2: string;
  rr: string;
  hr: string;
  sbp: string;
  gcs_eye: string;
  gcs_verbal: string;
  gcs_motor: string;
  // Simplified patient narrative
  patient_narrative: string;
}

export default function TriagePage() {
  const router = useRouter();
  const [form, setForm] = useState<FormData>({
    spo2: '',
    rr: '',
    hr: '',
    sbp: '',
    gcs_eye: '',
    gcs_verbal: '',
    gcs_motor: '',
    patient_narrative: '',
  });

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) {
    setForm({ ...form, [e.target.name]: e.target.value });
  }

  function handleGCSChange(component: string, value: string) {
    setForm({ ...form, [component]: value });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    
    try {
      // Call the API to get prediction with narrative analysis
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(form),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const prediction = await response.json();
      
      // Pass all form data and prediction results as URL parameters
      const params = new URLSearchParams({
        ...form,
        ...prediction,
        risk_level: prediction.risk_level,
        respiratory_risk: prediction.respiratory_risk,
        cardiovascular_risk: prediction.cardiovascular_risk,
        neurological_risk: prediction.neurological_risk,
        rox_score: prediction.rox_score.toString(),
        gcs_total: prediction.gcs_total.toString(),
        rpp_score: prediction.rpp_score.toString(),
        narrative_risk_score: prediction.narrative_risk_score.toString(),
      });
      
      router.push(`/ems-ai/results?${params.toString()}`);
    } catch (error) {
      console.error('Error getting prediction:', error);
      // Fallback to URL parameters only
      const params = new URLSearchParams(form as unknown as Record<string, string>);
      router.push(`/ems-ai/results?${params.toString()}`);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">Patient Assessment</h1>
              <p className="text-gray-600 text-sm">Emergency Medical Services - Risk Assessment Tool</p>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Assessment ID</div>
              <div className="text-sm font-mono text-gray-700">EMS-{Date.now().toString().slice(-6)}</div>
            </div>
          </div>

          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-blue-700">
                  Enter patient vital signs, Glasgow Coma Scale scores, and patient narrative. This assessment will provide risk stratification and resource allocation recommendations.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Assessment Form */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Patient Assessment</h2>
          </div>
          
          <form onSubmit={handleSubmit} className="p-6">
            {/* Patient Narrative Section */}
            <div className="mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Patient Narrative</h3>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Tell us what&apos;s going on
                </label>
                <textarea
                  name="patient_narrative"
                  value={form.patient_narrative}
                  onChange={handleChange}
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                  placeholder="What happened? What are the symptoms? Any medical conditions or medications we should know about?"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Include symptoms, what happened, medical conditions, medications, or anything else that might be important.
                </p>
              </div>
            </div>

            {/* Vital Signs Section */}
            <div className="mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Vital Signs</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* SpO2 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    SpO₂ (%)
                  </label>
                  <select
                    name="spo2"
                    required
                    value={form.spo2}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                  >
                    <option value="">Select SpO₂</option>
                    <option value="100">100% - Normal</option>
                    <option value="99">99% - Normal</option>
                    <option value="98">98% - Normal</option>
                    <option value="97">97% - Normal</option>
                    <option value="96">96% - Normal</option>
                    <option value="95">95% - Normal</option>
                    <option value="94">94% - Mild Hypoxemia</option>
                    <option value="93">93% - Mild Hypoxemia</option>
                    <option value="92">92% - Moderate Hypoxemia</option>
                    <option value="91">91% - Moderate Hypoxemia</option>
                    <option value="90">90% - Moderate Hypoxemia</option>
                    <option value="89">89% - Severe Hypoxemia</option>
                    <option value="88">88% - Severe Hypoxemia</option>
                    <option value="87">87% - Severe Hypoxemia</option>
                    <option value="86">86% - Severe Hypoxemia</option>
                    <option value="85">85% - Critical</option>
                    <option value="84">84% - Critical</option>
                    <option value="83">83% - Critical</option>
                    <option value="82">82% - Critical</option>
                    <option value="81">81% - Critical</option>
                    <option value="80">80% - Critical</option>
                    <option value="78">78% - Critical</option>
                    <option value="75">75% - Critical</option>
                    <option value="70">70% - Critical</option>
                    <option value="60">60% - Critical</option>
                  </select>
                </div>

                {/* Respiratory Rate */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Respiratory Rate (breaths/min)
                  </label>
                  <select
                    name="rr"
                    required
                    value={form.rr}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                  >
                    <option value="">Select Respiratory Rate</option>
                    <option value="8">8 - Bradypnea</option>
                    <option value="10">10 - Bradypnea</option>
                    <option value="12">12 - Normal</option>
                    <option value="14">14 - Normal</option>
                    <option value="16">16 - Normal</option>
                    <option value="18">18 - Normal</option>
                    <option value="20">20 - Normal</option>
                    <option value="22">22 - Tachypnea</option>
                    <option value="24">24 - Tachypnea</option>
                    <option value="26">26 - Tachypnea</option>
                    <option value="28">28 - Tachypnea</option>
                    <option value="30">30 - Tachypnea</option>
                    <option value="32">32 - Tachypnea</option>
                    <option value="35">35 - Tachypnea</option>
                    <option value="40">40 - Tachypnea</option>
                  </select>
                </div>

                {/* Heart Rate */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Heart Rate (bpm)
                  </label>
                  <select
                    name="hr"
                    required
                    value={form.hr}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                  >
                    <option value="">Select Heart Rate</option>
                    <option value="35">35 - Severe Bradycardia</option>
                    <option value="40">40 - Bradycardia</option>
                    <option value="45">45 - Bradycardia</option>
                    <option value="50">50 - Bradycardia</option>
                    <option value="55">55 - Bradycardia</option>
                    <option value="60">60 - Normal</option>
                    <option value="65">65 - Normal</option>
                    <option value="70">70 - Normal</option>
                    <option value="75">75 - Normal</option>
                    <option value="80">80 - Normal</option>
                    <option value="85">85 - Normal</option>
                    <option value="90">90 - Normal</option>
                    <option value="95">95 - Normal</option>
                    <option value="100">100 - Normal</option>
                    <option value="110">110 - Tachycardia</option>
                    <option value="120">120 - Tachycardia</option>
                    <option value="130">130 - Tachycardia</option>
                    <option value="140">140 - Tachycardia</option>
                    <option value="150">150 - Tachycardia</option>
                    <option value="160">160 - Tachycardia</option>
                    <option value="180">180 - Tachycardia</option>
                    <option value="200">200 - Tachycardia</option>
                  </select>
                </div>

                {/* Systolic BP */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Systolic BP (mmHg)
                  </label>
                  <select
                    name="sbp"
                    required
                    value={form.sbp}
                    onChange={handleChange}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                  >
                    <option value="">Select Systolic BP</option>
                    <option value="60">60 - Critical</option>
                    <option value="70">70 - Critical</option>
                    <option value="80">80 - Critical</option>
                    <option value="85">85 - Hypotension</option>
                    <option value="90">90 - Hypotension</option>
                    <option value="95">95 - Normal</option>
                    <option value="100">100 - Normal</option>
                    <option value="110">110 - Normal</option>
                    <option value="120">120 - Normal</option>
                    <option value="130">130 - Normal</option>
                    <option value="140">140 - Normal</option>
                    <option value="150">150 - Normal</option>
                    <option value="160">160 - Normal</option>
                    <option value="170">170 - Hypertension</option>
                    <option value="180">180 - Hypertension</option>
                    <option value="190">190 - Hypertension</option>
                    <option value="200">200 - Hypertension</option>
                    <option value="220">220 - Severe Hypertension</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Glasgow Coma Scale Section */}
            <div className="mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Glasgow Coma Scale Assessment</h3>
              
              {/* Eye Opening */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Eye Opening Response</h4>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="gcs_eye" value="4" checked={form.gcs_eye === '4'} onChange={(e) => handleGCSChange('gcs_eye', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>4 - Spontaneous:</strong> Eyes open without stimulation</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_eye" value="3" checked={form.gcs_eye === '3'} onChange={(e) => handleGCSChange('gcs_eye', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>3 - To Voice:</strong> Eyes open to verbal command</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_eye" value="2" checked={form.gcs_eye === '2'} onChange={(e) => handleGCSChange('gcs_eye', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>2 - To Pain:</strong> Eyes open to painful stimulus</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_eye" value="1" checked={form.gcs_eye === '1'} onChange={(e) => handleGCSChange('gcs_eye', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>1 - None:</strong> No eye opening to any stimulus</span>
                  </label>
                </div>
              </div>

              {/* Verbal Response */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Verbal Response</h4>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="gcs_verbal" value="5" checked={form.gcs_verbal === '5'} onChange={(e) => handleGCSChange('gcs_verbal', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>5 - Oriented:</strong> Patient knows who, where, when</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_verbal" value="4" checked={form.gcs_verbal === '4'} onChange={(e) => handleGCSChange('gcs_verbal', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>4 - Confused:</strong> Patient responds but disoriented</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_verbal" value="3" checked={form.gcs_verbal === '3'} onChange={(e) => handleGCSChange('gcs_verbal', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>3 - Inappropriate:</strong> Random words, no conversation</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_verbal" value="2" checked={form.gcs_verbal === '2'} onChange={(e) => handleGCSChange('gcs_verbal', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>2 - Incomprehensible:</strong> Moaning, no words</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_verbal" value="1" checked={form.gcs_verbal === '1'} onChange={(e) => handleGCSChange('gcs_verbal', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>1 - None:</strong> No verbal response</span>
                  </label>
                </div>
              </div>

              {/* Motor Response */}
              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Motor Response</h4>
                <div className="space-y-2">
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="6" checked={form.gcs_motor === '6'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>6 - Obeys Commands:</strong> Follows simple commands</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="5" checked={form.gcs_motor === '5'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>5 - Localizes Pain:</strong> Moves toward painful stimulus</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="4" checked={form.gcs_motor === '4'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>4 - Withdraws:</strong> Pulls away from painful stimulus</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="3" checked={form.gcs_motor === '3'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>3 - Flexion:</strong> Abnormal flexion to pain</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="2" checked={form.gcs_motor === '2'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>2 - Extension:</strong> Abnormal extension to pain</span>
                  </label>
                  <label className="flex items-center">
                    <input type="radio" name="gcs_motor" value="1" checked={form.gcs_motor === '1'} onChange={(e) => handleGCSChange('gcs_motor', e.target.value)} className="mr-3"/>
                    <span className="text-sm text-gray-700"><strong>1 - None:</strong> No motor response</span>
                  </label>
                </div>
              </div>

              {/* GCS Summary */}
              {form.gcs_eye && form.gcs_verbal && form.gcs_motor && (
                <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                  <h5 className="text-sm font-medium text-blue-900 mb-2">GCS Summary</h5>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div><span className="text-blue-700">Eye:</span> {form.gcs_eye}/4</div>
                    <div><span className="text-blue-700">Verbal:</span> {form.gcs_verbal}/5</div>
                    <div><span className="text-blue-700">Motor:</span> {form.gcs_motor}/6</div>
                  </div>
                  <div className="mt-2 text-sm font-medium text-blue-900">
                    Total: {parseInt(form.gcs_eye) + parseInt(form.gcs_verbal) + parseInt(form.gcs_motor)}/15
                  </div>
                </div>
              )}
            </div>

            <div className="mt-8 pt-6 border-t border-gray-200">
              <button
                type="submit"
                className="w-full bg-blue-600 text-white px-4 py-3 rounded-md font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors"
              >
                Generate Risk Assessment
              </button>
            </div>
          </form>
        </div>

        {/* Footer Info */}
        <div className="mt-6 text-center text-xs text-gray-500">
          <p>This assessment tool provides risk stratification and resource allocation recommendations based on clinical parameters.</p>
        </div>
      </div>
    </div>
  );
} 