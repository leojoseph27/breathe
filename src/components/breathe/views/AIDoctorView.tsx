"use client";

import { useState } from "react";
import { useBreatheStore } from "@/lib/breathe-store";
import {
  Loader2,
  ChevronDown,
  ChevronUp,
  Stethoscope,
  Headphones,
  HeartPulse,
  CloudSun,
  Link2,
  FileText,
  Sparkles,
} from "lucide-react";

const yn = (v?: number) =>
  v === 1 ? "Yes" : v === 0 ? "No" : "--";
const genderStr = (v?: number) =>
  v === 0 ? "Male" : v === 1 ? "Female" : "--";
const smokeStr = (v?: number) =>
  v === 0 ? "Non-smoker" : v === 1 ? "Former smoker" : v === 2 ? "Current smoker" : "--";

export function AIDoctorView() {
  const patientData = useBreatheStore((s) => s.patientData);
  const audioAnalysis = useBreatheStore((s) => s.audioAnalysis);
  const asthmaAssessment = useBreatheStore((s) => s.asthmaAssessment);
  const environmentalData = useBreatheStore((s) => s.environmentalData);

  const [verdict, setVerdict] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);

  async function generateVerdict() {
    setLoading(true);
    setError("");
    setVerdict(null);
    try {
      const res = await fetch("/api/generate-ai-verdict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientData,
          audioAnalysis,
          asthmaAssessment,
          environmentalData,
        }),
      });
      const json = await res.json();
      if (json.success && json.verdict) {
        setVerdict(json.verdict);
      } else {
        setError(json.error || "Failed to generate AI verdict");
      }
    } catch (err) {
      setError("Failed to connect to AI service: " + String(err));
    } finally {
      setLoading(false);
    }
  }

  function buildSummary(): string {
    let s = "";
    if (patientData.age) s += `${patientData.age}-year-old `;
    if (patientData.gender !== undefined)
      s += patientData.gender === 0 ? "male " : "female ";
    if (patientData.bmi) s += `with BMI ${patientData.bmi}. `;
    if (patientData.smoking !== undefined) {
      s += `Smoking status: ${smokeStr(patientData.smoking).toLowerCase()}. `;
    }
    if (audioAnalysis.prediction)
      s += `Audio analysis suggests ${audioAnalysis.prediction}. `;
    if (asthmaAssessment.prediction)
      s += `Clinical assessment indicates ${asthmaAssessment.prediction}. `;
    if (environmentalData.aqi !== undefined) {
      s += `Environmental exposure shows ${
        environmentalData.aqi ? "moderate air quality" : "various conditions"
      }. `;
    }
    s +=
      "Findings suggest possible respiratory condition with environmental factors contributing to symptoms.";
    return s || "No assessment data yet. Complete the other sections first.";
  }

  const epa = environmentalData.epaIndex ?? 1;
  const correlation: string[] = [];
  if (Object.keys(environmentalData).length) {
    if (epa <= 1)
      correlation.push(
        "Air quality is good, unlikely to worsen respiratory symptoms"
      );
    else if (epa === 2)
      correlation.push(
        "Moderate air pollution may contribute to respiratory symptoms"
      );
    else
      correlation.push(
        "Higher pollution levels may significantly worsen respiratory symptoms"
      );
    if ((environmentalData.pm25 ?? 0) > 25)
      correlation.push(
        `PM2.5 levels (${environmentalData.pm25} µg/m³) may contribute to airway irritation`
      );
    if ((environmentalData.windSpeed ?? 0) > 25)
      correlation.push(
        `High wind speeds (${environmentalData.windSpeed} km/h) may spread allergens and irritants`
      );
    else correlation.push("Wind conditions unlikely to worsen symptoms");
  }

  const snapshot = [
    { label: "Age", value: patientData.age ?? "--" },
    { label: "Gender", value: genderStr(patientData.gender) },
    { label: "BMI", value: patientData.bmi ?? "--" },
    { label: "Smoking", value: smokeStr(patientData.smoking) },
    { label: "Allergy", value: yn(patientData.allergyHistory) },
    { label: "Family History", value: yn(patientData.familyHistory) },
  ];

  const symptoms: string[] = [];
  if (patientData.wheezing === 1) symptoms.push("Wheezing");
  if (patientData.shortnessOfBreath === 1) symptoms.push("Shortness of Breath");
  if (patientData.chestTightness === 1) symptoms.push("Chest Tightness");

  const hasData =
    Object.keys(patientData).length ||
    Object.keys(audioAnalysis).length ||
    Object.keys(asthmaAssessment).length ||
    Object.keys(environmentalData).length;

  const analysisCards = [
    {
      icon: Headphones,
      title: "Audio Respiratory Analysis",
      accent: "#0ea5e9",
      rows: [
        { label: "Detected Condition", value: audioAnalysis.prediction || "--" },
        { label: "Audio File", value: audioAnalysis.filename || "--" },
      ],
      footer: "AI-assisted signal analysis",
    },
    {
      icon: HeartPulse,
      title: "Clinical Asthma Assessment",
      accent: "#06b6d4",
      rows: [
        { label: "Diagnosis", value: asthmaAssessment.prediction || "--" },
        {
          label: "FEV1",
          value: patientData.lungFunctionFeV1
            ? `${patientData.lungFunctionFeV1}%`
            : "--",
        },
      ],
      footer: "ML-based clinical inference",
      extra:
        symptoms.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-1">
            {symptoms.map((sym) => (
              <span
                key={sym}
                className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600"
              >
                <span className="h-1 w-1 rounded-full bg-cyan-500" />
                {sym}
              </span>
            ))}
          </div>
        ) : null,
    },
    {
      icon: CloudSun,
      title: "Environmental Exposure",
      accent: "#10b981",
      rows: [
        {
          label: "Location",
          value: environmentalData.resolvedLocation?.name
            ? `${environmentalData.resolvedLocation.name}, ${environmentalData.resolvedLocation.region}`
            : "--",
        },
        {
          label: "AQI",
          value: environmentalData.epaIndex
            ? `EPA Index ${environmentalData.epaIndex}`
            : "--",
        },
        {
          label: "PM2.5",
          value: environmentalData.pm25
            ? `${environmentalData.pm25} µg/m³`
            : "--",
        },
        {
          label: "Weather",
          value: environmentalData.weatherDescription || "--",
        },
      ],
      footer: "Nearest available monitoring station",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          AI Doctor — Unified Respiratory Assessment
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Multimodal AI-assisted clinical decision support, aggregating data
          from all previous assessments.
        </p>
      </div>

      {!hasData && (
        <div className="flex items-start gap-3 rounded-xl border border-sky-100 bg-sky-50/70 p-4">
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
          <div className="text-[13px] leading-relaxed text-slate-600">
            Complete the Audio Analysis, Asthma Detection, and Safe Check
            sections first to populate this unified view, then generate an AI
            verdict.
          </div>
        </div>
      )}

      {/* Patient snapshot */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {snapshot.map((s) => (
          <div key={s.label} className="bd-card p-3 text-center">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
              {s.label}
            </div>
            <div className="mt-1 text-sm font-semibold text-slate-900">
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* Analysis panels */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {analysisCards.map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.title} className="bd-card bd-card-hover p-5">
              <div className="mb-3 flex items-center gap-2">
                <span
                  className="flex h-8 w-8 items-center justify-center rounded-lg text-white"
                  style={{ background: card.accent }}
                >
                  <Icon className="h-4 w-4" />
                </span>
                <h3 className="text-[13px] font-semibold text-slate-900">
                  {card.title}
                </h3>
              </div>
              <div className="space-y-2">
                {card.rows.map((r) => (
                  <div
                    key={r.label}
                    className="flex items-start justify-between gap-2 text-[13px]"
                  >
                    <span className="text-slate-500">{r.label}</span>
                    <span className="text-right font-medium text-slate-900">
                      {r.value}
                    </span>
                  </div>
                ))}
                {card.extra}
              </div>
              <div className="mt-3 border-t border-slate-100 pt-2 text-[11px] text-slate-400">
                {card.footer}
              </div>
            </div>
          );
        })}
      </div>

      {/* Correlation */}
      <div className="bd-card p-5">
        <div className="mb-3 flex items-center gap-2">
          <Link2 className="h-4 w-4 text-sky-600" />
          <h3 className="text-[13px] font-semibold text-slate-900">
            Symptom–Trigger Correlation
          </h3>
        </div>
        {correlation.length ? (
          <ul className="space-y-1.5">
            {correlation.map((c, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-[13px] text-slate-600"
              >
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-sky-500" />
                {c}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[13px] text-slate-500">
            Complete the Safe Check to see symptom–trigger correlations.
          </p>
        )}
      </div>

      {/* Case summary (expandable) */}
      <div className="bd-card overflow-hidden">
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          className="flex w-full items-center justify-between gap-2 p-5 text-left transition-colors hover:bg-slate-50/50"
          aria-expanded={expanded}
        >
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-sky-600" />
            <span className="text-[13px] font-semibold text-slate-900">
              Unified Case Summary
            </span>
            <span className="text-[11px] font-normal text-slate-400">
              for AI reasoning
            </span>
          </div>
          {expanded ? (
            <ChevronUp className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          )}
        </button>
        {expanded && (
          <div className="border-t border-slate-100 p-5">
            <p className="text-[13px] leading-relaxed text-slate-600">
              {buildSummary()}
            </p>
          </div>
        )}
      </div>

      {/* AI verdict */}
      <div
        className="rounded-2xl border border-sky-100 p-6 sm:p-7"
        style={{
          background:
            "linear-gradient(135deg, rgba(224,242,254,0.6), rgba(236,254,255,0.6))",
        }}
      >
        <div className="mb-4 flex items-center gap-2.5">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-xl text-white shadow-sm"
            style={{ background: "linear-gradient(135deg, #0ea5e9, #06b6d4)" }}
          >
            <Stethoscope className="h-5 w-5" />
          </span>
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              AI Doctor Verdict
            </h3>
            <p className="text-[11px] text-slate-500">
              Generated from your unified assessment data
            </p>
          </div>
        </div>

        <div className="min-h-[80px]">
          {loading ? (
            <div className="flex items-center gap-2 py-4 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin text-sky-600" />
              AI Doctor is analyzing the case…
            </div>
          ) : verdict ? (
            <p className="whitespace-pre-line text-[14px] leading-relaxed text-slate-700">
              {verdict}
            </p>
          ) : error ? (
            <p className="py-2 text-[13px] text-red-600">Error: {error}</p>
          ) : (
            <p className="py-2 text-[13px] text-slate-500">
              Review all assessments and click &quot;Generate AI Verdict&quot;
              to get the AI doctor&apos;s opinion.
            </p>
          )}
        </div>

        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={generateVerdict}
            disabled={loading}
            className="bd-btn bd-btn-primary bd-btn-lg w-full max-w-xs"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Generating…
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" /> Generate AI Verdict
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
