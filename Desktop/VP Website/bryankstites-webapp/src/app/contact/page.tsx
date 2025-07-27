'use client';
import React, { useState } from 'react';

export default function ContactPage() {
  const [submitted, setSubmitted] = useState(false);

  function handleSubmit() {
    // Let the browser submit the form to Formsubmit, then show thank you message
    setTimeout(() => setSubmitted(true), 1000); // fallback in case redirect fails
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-gray-100 py-12 px-4 flex flex-col items-center justify-center">
      <div className="bg-white/80 rounded-xl shadow-lg p-8 max-w-2xl w-full">
        <h1 className="text-4xl font-extrabold text-blue-900 mb-4 drop-shadow-lg tracking-tight text-center">Let’s Connect</h1>
        <p className="mb-6 text-lg text-gray-800 text-center">
          Interested in building a more data-driven, empowered organization? I’m always open to connecting with leaders and teams who want to use business insights and AI to bridge technology gaps and unlock new potential for their people.
        </p>
        {submitted ? (
          <div className="bg-green-100 border-l-4 border-green-500 p-4 rounded shadow text-green-900 text-center font-semibold mb-6">
            Thank you for reaching out! I’ll get back to you soon.
          </div>
        ) : (
          <form
            action="https://formsubmit.co/bryankstites@gmail.com"
            method="POST"
            className="bg-blue-50 rounded-lg shadow p-6 space-y-4 max-w-md mx-auto"
            onSubmit={handleSubmit}
            target="_blank"
          >
            <input type="hidden" name="_captcha" value="false" />
            <input type="hidden" name="_next" value="/contact?success=true" />
            {/* Honeypot field for spam protection */}
            <div style={{ display: 'none' }}>
              <label>Leave this field blank</label>
              <input type="text" name="_honey" tabIndex={-1} autoComplete="off" />
            </div>
            <div>
              <label className="block mb-1 font-medium text-blue-900">Name</label>
              <input type="text" name="name" required className="w-full border border-blue-200 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-900" placeholder="Your name" />
            </div>
            <div>
              <label className="block mb-1 font-medium text-blue-900">Email</label>
              <input type="email" name="email" required className="w-full border border-blue-200 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-900" placeholder="Your email" />
            </div>
            <div>
              <label className="block mb-1 font-medium text-blue-900">Message</label>
              <textarea name="message" required className="w-full border border-blue-200 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-900" rows={4} placeholder="Your message" />
            </div>
            <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded font-semibold hover:bg-blue-700 transition">Send</button>
          </form>
        )}
        <div className="mt-6 text-center text-blue-800">
          <p>Email: <a href="mailto:bryankstites@gmail.com" className="underline">bryankstites@gmail.com</a></p>
          <p>LinkedIn: <a href="https://linkedin.com/in/bryankstites" className="underline" target="_blank" rel="noopener noreferrer">linkedin.com/in/bryankstites</a></p>
        </div>
      </div>
    </main>
  );
} 