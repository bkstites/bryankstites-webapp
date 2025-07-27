import React from 'react';

export default function AIVisionPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-gray-100 py-12 px-4 flex flex-col items-center justify-center">
      <div className="bg-white/80 rounded-xl shadow-lg p-8 max-w-3xl w-full">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-4 drop-shadow-lg tracking-tight text-center">AI Vision</h1>

        <p className="mb-6 text-lg text-gray-800 text-center">
          My vision is to demonstrate how AI tools can eliminate traditional barriers to development and innovation—showing how intelligent systems give humans super powers to build, deploy, and scale solutions that were previously out of reach.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6 border-l-4 border-green-500">
            <h3 className="text-xl font-bold text-green-900 mb-3">The AI Tool Revolution</h3>
            <p className="text-green-800 mb-3">
              Modern AI tools are transforming how we build and deploy solutions:
            </p>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• <strong>Agentic AI & LLMs:</strong> Code generation and problem-solving at human speed</li>
              <li>• <strong>Process Automation:</strong> Eliminate repetitive tasks and reduce overhead</li>
              <li>• <strong>Rapid Prototyping:</strong> Build complex systems in days, not months</li>
              <li>• <strong>Intelligent Interfaces:</strong> Create user experiences that adapt and learn</li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6 border-l-4 border-purple-500">
            <h3 className="text-xl font-bold text-purple-900 mb-3">Real Impact Examples</h3>
            <p className="text-purple-800 mb-3">
              Here&apos;s what AI-powered development delivers:
            </p>
            <ul className="text-sm text-purple-700 space-y-1">
              <li>• <strong>Significantly Faster</strong> development cycles with AI assistance</li>
              <li>• <strong>Reduced Development Time</strong> through intelligent code generation</li>
              <li>• <strong>Rapid Deployment</strong> from concept to production</li>
              <li>• <strong>Democratized Development</strong> - anyone can build powerful tools</li>
            </ul>
          </div>
        </div>

        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border-l-4 border-blue-500 mb-6">
          <h3 className="text-xl font-bold text-blue-900 mb-3">The AI Integration Approach</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-800 mb-1">1</div>
              <div className="font-semibold text-blue-900">Identify Bottlenecks</div>
              <div className="text-blue-700">Find where traditional processes create latency and overhead</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-800 mb-1">2</div>
              <div className="font-semibold text-blue-900">Leverage AI Tools</div>
              <div className="text-blue-700">Use Agentic AI, LLMs, and intelligent systems to accelerate development</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-800 mb-1">3</div>
              <div className="font-semibold text-blue-900">Deploy & Scale</div>
              <div className="text-blue-700">Rapidly build, test, and deploy solutions that give humans super powers</div>
            </div>
          </div>
        </div>

        <div className="text-center">
          <h3 className="text-2xl font-bold text-blue-900 mb-4">See It In Action</h3>
          <p className="text-gray-700 mb-4">
            This entire portfolio—including the EMS Risk Triage tool—was built and deployed in days using AI tools, demonstrating how intelligent systems can give anyone the super powers to create professional-grade applications with minimal traditional development experience.
          </p>
          <a 
            href="/ems-ai" 
            className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition"
          >
            Experience the AI-Built Demo
          </a>
        </div>
      </div>
    </main>
  );
} 