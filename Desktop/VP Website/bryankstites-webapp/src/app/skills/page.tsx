import React from 'react';

const skills = [
  'Business Insights',
  'Data-Driven Leadership',
  'Leadership',
  'Community Building',
  'Metrics & Analytics',
  'AI Empowerment',
  'Process Optimization',
  'Team Building & Mentorship',
  'Change Management',
  'Cloud & Automation',
  'Python', 'SQL', 'Snowflake', 'AWS',
];

export default function SkillsPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-gray-100 py-12 px-4 flex flex-col items-center justify-center">
      <div className="bg-white/80 rounded-xl shadow-lg p-8 max-w-2xl w-full">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-4 drop-shadow-lg tracking-tight text-center">My Strengths</h1>
        <p className="mb-6 text-lg text-gray-800 text-center">
          I specialize in translating data into actionable business insights and empowering teams with AI-driven tools. My strengths lie in building metrics that matter, optimizing processes, and helping associates unlock their full potential through technology.
        </p>
        <div className="flex flex-wrap justify-center gap-3 mb-6">
          {skills.map(skill => (
            <span key={skill} className="inline-block bg-blue-100 text-blue-900 px-4 py-2 rounded-full text-sm font-semibold shadow hover:bg-blue-200 transition cursor-pointer" tabIndex={0} aria-label={skill}>{skill}</span>
          ))}
        </div>
        <div className="mt-4 text-center">
          <h2 className="text-xl font-bold text-blue-800 mb-2 drop-shadow">Education</h2>
          <p className="text-gray-700">M.S. in Business Analytics, Rutgers University (Expected Dec 2025)</p>
          <p className="text-gray-700">B.S. in Psychology & Neuroscience, Rowan University</p>
        </div>
        <div className="mt-6 text-center">
          <h2 className="text-xl font-bold text-blue-800 mb-2 drop-shadow">Certifications</h2>
          <p className="text-gray-700">AWS Cloud Practitioner</p>
          <p className="text-gray-700">National Registry Paramedic</p>
          <p className="text-gray-700">Pennsylvania State Paramedic License</p>
        </div>
      </div>
    </main>
  );
} 