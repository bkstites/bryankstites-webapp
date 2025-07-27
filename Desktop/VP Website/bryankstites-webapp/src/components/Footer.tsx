import React from 'react';

export default function Footer() {
  return (
    <footer className="bg-gray-100 dark:bg-gray-900 py-4 mt-8">
      <div className="max-w-3xl mx-auto text-center text-gray-600 dark:text-gray-300 text-sm flex flex-col items-center gap-2">
        <p>Bryan Stites &middot; Hammonton, NJ &middot; <a href="mailto:bryankstites@gmail.com" className="underline">bryankstites@gmail.com</a></p>
        <div className="flex flex-wrap gap-4 justify-center mt-1">
          <a href="https://github.com/bkstites/EMS-Analysis" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-700">EMS Analysis GitHub</a>
          <a href="/resume.pdf" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-700">Resume</a>
          <a href="https://linkedin.com/in/bryankstites" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-700">LinkedIn</a>
        </div>
        <p className="mt-1">&copy; {new Date().getFullYear()} Bryan Stites. All rights reserved.</p>
      </div>
    </footer>
  );
} 