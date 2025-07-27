'use client';
import EMSLayout from '../layout';

export default function BehindModelPage() {
  return (
    <EMSLayout>
      <section className="w-full max-w-2xl mx-auto py-8 px-4">
        <h2 className="text-3xl font-bold text-blue-900 dark:text-blue-200 mb-4 text-center">Behind the Model</h2>
        <p className="mb-6 text-lg text-gray-800 dark:text-gray-200 text-center">
          This EMS triage tool uses a Random Forest machine learning model trained on real-world pre-hospital data. The model predicts patient risk based on vital signs and GCS scores, helping paramedics and researchers make informed decisions in the field.
        </p>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow p-6 mb-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-2 text-blue-700 dark:text-blue-300">Feature Importance (Top 5)</h3>
          <ul className="list-disc list-inside text-gray-800 dark:text-gray-100">
            <li>SpO₂</li>
            <li>Respiratory Rate</li>
            <li>Heart Rate</li>
            <li>Systolic BP</li>
            <li>GCS Total</li>
          </ul>
        </div>
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg shadow p-6 mb-6 text-center border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold mb-2 text-blue-700 dark:text-blue-300">Model Confusion Matrix</h3>
          <div className="flex justify-center">
            <img src="/file.svg" alt="Confusion matrix placeholder" className="h-32 opacity-40" />
          </div>
        </div>
        <div className="text-center mt-8">
          <a href="/ems-ai" className="text-blue-700 dark:text-blue-300 hover:underline font-medium">Back to EMS AI Home</a>
        </div>
      </section>
    </EMSLayout>
  );
} 