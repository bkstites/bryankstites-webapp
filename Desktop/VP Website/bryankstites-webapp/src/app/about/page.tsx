import React from 'react';
import Image from 'next/image';

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-gray-100 py-12 px-4 flex flex-col items-center justify-center">
      <div className="bg-white/80 rounded-xl shadow-lg p-8 max-w-2xl w-full">
        <div className="flex flex-col items-center mb-6">
          <Image
            src="/headshot.jpg"
            alt="Bryan Stites headshot"
            width={144}
            height={144}
            className="rounded-full shadow-lg object-cover mb-4"
            priority
          />
        </div>
        <h1 className="text-4xl font-extrabold text-blue-900 mb-4 drop-shadow-lg tracking-tight text-center">About Bryan Stites</h1>
        <p className="mb-6 text-lg text-gray-800">
          I’m Bryan Stites, a business insights leader passionate about using data and AI to drive meaningful change. My focus is on unlocking actionable metrics, informing business processes, and empowering associates with technology that truly bridges the gap between people and platforms.
        </p>
        <section className="mb-6">
          <h2 className="text-2xl font-bold text-blue-800 mb-2 drop-shadow">Leadership Style</h2>
          <p className="text-gray-700">
            My approach is people-first and insight-driven. I believe in mentoring teams, fostering curiosity, and using data to illuminate opportunities for growth. By equipping associates with the right tools and AI-powered solutions, I help organizations make smarter, faster decisions.
          </p>
        </section>
        <section>
          <h2 className="text-2xl font-bold text-blue-800 mb-2 drop-shadow">Empowering People</h2>
          <p className="text-gray-700">
            I’m passionate about making technology accessible and impactful—bringing superpowers to people at every level of the organization.
          </p>
        </section>
      </div>
    </main>
  );
} 