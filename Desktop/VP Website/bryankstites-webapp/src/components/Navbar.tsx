import React from 'react';
import Link from 'next/link';

const navLinks = [
  { href: '/ems-ai/triage', label: 'Assessment' },
];

export default function Navbar() {
  return (
    <nav className="bg-white shadow mb-8">
      <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
        <span className="font-bold text-xl text-gray-800">EMS AI Triage</span>
        <div className="space-x-4">
          {navLinks.map(link => (
            <Link key={link.href} href={link.href} className="text-gray-700 hover:text-blue-600 font-medium">
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
} 