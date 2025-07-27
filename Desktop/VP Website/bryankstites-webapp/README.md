# EMS AI Triage System

A clinical decision support system for pre-hospital emergency assessment and risk stratification.

## Overview

This application provides real-time risk assessment based on patient vital signs and Glasgow Coma Scale scores. Designed for emergency medical personnel to support clinical decision-making in the field.

## Features

- **Risk Stratification**: Automated assessment of patient risk levels
- **Resource Allocation**: Guidance for appropriate resource deployment  
- **Clinical Support**: Evidence-based recommendations for field crews
- **Narrative Analysis**: AI-powered analysis of patient narratives for additional risk factors

## Assessment Parameters

### Vital Signs
- SpO₂ (% saturation)
- Respiratory Rate (breaths/min)
- Heart Rate (bpm)
- Systolic BP (mmHg)

### Glasgow Coma Scale
- Eye Opening (1-4)
- Verbal Response (1-5)
- Motor Response (1-6)

## Clinical Scores

The system calculates validated emergency medicine scores:
- **ROX Score**: SpO₂/FiO₂ ratio ÷ Respiratory Rate
- **GCS Total**: Eye + Verbal + Motor Response
- **RPP Score**: Heart Rate × Systolic BP
- **Narrative Risk Score**: Keyword-based risk assessment

## Technology Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Backend**: Next.js API Routes
- **AI/ML**: Custom risk assessment algorithms based on emergency medicine standards

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Clinical Disclaimer

This tool provides clinical decision support and should be used in conjunction with professional medical judgment. The system is designed to assist, not replace, clinical decision-making.
