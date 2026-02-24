"use client";

import React, { useState } from "react";
import { randomizeVitals, VitalsPayload } from "../utils/randomizer";
import { ModalApiClient } from "../../backend/api_client";
import { LocalTriageLogic, RiskLevel } from "../../orchestrator/triage_logic";

export default function SimulationForm() {
  const [vitals, setVitals] = useState<VitalsPayload>({
    age: 30,
    gestational_weeks: 34,
    systolic_bp: 120,
    diastolic_bp: 80,
    heart_rate: 80,
    temperature: 37.0,
    spo2: 98,
    fasting_glucose: 5.5,
  });
  
  const [patientContext, setPatientContext] = useState<any>({
    known_conditions: [],
    medications: []
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [responseLog, setResponseLog] = useState<string>("");
  const [preFlightRisk, setPreFlightRisk] = useState<{level: string, flags: string[]} | null>(null);

  const handleRandomize = () => {
    const randomVitals = randomizeVitals(true);
    setVitals(randomVitals);
    
    // Evaluate edge logic immediately
    const risk = LocalTriageLogic.evaluateImmediateRisk(randomVitals);
    setPreFlightRisk(risk);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setVitals(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponseLog("");

    // Initialize API Client. The NEXT_PUBLIC_MODAL_API_URL should be set in .env
    const apiClient = new ModalApiClient({
      apiUrl: process.env.NEXT_PUBLIC_MODAL_API_URL || "https://tayyabkhn343--momnitrix-api-v2-web.modal.run"
    });

    await apiClient.streamTriage(
      vitals,
      patientContext,
      { source: "simulation" }, // inputs
      (data) => {
        setResponseLog(prev => prev + JSON.stringify(data, null, 2) + "\\n");
      },
      (err) => {
        setError(err.message);
        setLoading(false);
      },
      () => {
        setLoading(false);
      }
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow rounded-lg mt-10">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800">MamaGuard Simulation Console</h1>
        <button
          onClick={handleRandomize}
          className="px-4 py-2 bg-blue-500 text-white font-semibold rounded hover:bg-blue-600 transition"
          type="button"
        >
          🎲 Randomize (Bounded)
        </button>
      </div>

      {preFlightRisk && preFlightRisk.level !== RiskLevel.GREEN && (
        <div className={`p-4 mb-6 rounded ${preFlightRisk.level === RiskLevel.CRITICAL ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}`}>
          <h3 className="font-bold">Edge Triage Flag: {preFlightRisk.level}</h3>
          <ul className="list-disc pl-5 mt-2">
            {preFlightRisk.flags.map((flag, i) => (
              <li key={i}>{flag}</li>
            ))}
          </ul>
        </div>
      )}

      <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Vitals Section */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-gray-700 border-b pb-2">Structured Vitals</h2>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-600">Age</label>
              <input type="number" name="age" value={vitals.age} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Gestational Weeks</label>
              <input type="number" name="gestational_weeks" value={vitals.gestational_weeks || ""} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Heart Rate (bpm)</label>
              <input type="number" name="heart_rate" value={vitals.heart_rate} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">SpO2 (%)</label>
              <input type="number" name="spo2" value={vitals.spo2} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Systolic BP</label>
              <input type="number" name="systolic_bp" value={vitals.systolic_bp} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Diastolic BP</label>
              <input type="number" name="diastolic_bp" value={vitals.diastolic_bp} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Temperature (°C)</label>
              <input type="number" step="0.1" name="temperature" value={vitals.temperature} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600">Fasting Glucose</label>
              <input type="number" step="0.1" name="fasting_glucose" value={vitals.fasting_glucose} onChange={handleChange} className="mt-1 p-2 w-full border rounded text-black" />
            </div>
          </div>
        </div>

        {/* Modal Results Section */}
        <div className="space-y-4 flex flex-col">
          <h2 className="text-xl font-semibold text-gray-700 border-b pb-2">Orchestrator Output</h2>
          <div className="flex-grow bg-gray-50 border border-gray-200 rounded p-4 overflow-y-auto font-mono text-sm min-h-[300px] text-gray-800 whitespace-pre-wrap">
            {loading && !responseLog && <span className="text-blue-500 animate-pulse">Sending to Modal...</span>}
            {error && <span className="text-red-500">Error: {error}</span>}
            {responseLog}
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-indigo-600 text-white font-bold rounded hover:bg-indigo-700 transition disabled:opacity-50"
          >
            {loading ? "Processing..." : "Submit to Triage Orchestrator"}
          </button>
        </div>
      </form>
    </div>
  );
}
