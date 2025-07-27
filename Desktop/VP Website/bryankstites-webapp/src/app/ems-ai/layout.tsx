'use client';
import React from 'react';

export default function EMSLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen">
      <main className="flex flex-col items-center justify-center flex-1 w-full px-4 py-8 min-h-[80vh]">
        {children}
      </main>
    </div>
  );
} 