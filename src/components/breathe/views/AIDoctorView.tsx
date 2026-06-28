"use client";

import { useState } from "react";
import { useBreatheStore } from "@/lib/breathe-store";
import { ClinicalReport } from "@/components/breathe/ClinicalReport";
import { exportClinicalReport } from "@/components/breathe/export-report";
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
  Download,
  AlertCircle,
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

  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [reportSource, setReportSource] = useState<string>("");
  const [generatedAt, setGeneratedAt] = useState<string>("");
  const [summaryExpanded, setSummaryExpanded] = useState(false);

  async function generateVerdict() {
    setLoading(true);
    setError("");
    setReport(null);
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
        setReport(json.verdict);
        setReportSource(json.source || "gemini");
        setGeneratedAt(new Date().toISOString());
      } else {
        setError(json.error || "Failed to generate AI report");
      }
    } catch (err) {
      setError("Failed to connect to AI service: " + String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleExport() {
    if (!report) return;
    exportClinicalReport({
      markdown: report,
      generatedAt: generatedAt || new Date().toISOString(),
    });
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
        ...(audioAnalysis.confidence
          ? [
              {
                label: "Confidence",
                value: `${Math.round(audioAnalysis.confidence * 100)}%`,
              },
            ]
          : []),
      ],
      footer: "CNN model prediction",
    },
    {
      icon: HeartPulse,
      title: "Clinical Asthma Assessment",
      accent: "#06b6d4",
      rows: [
        { label: "Diagnosis", value: asthmaAssessment.prediction || "--" },
        ...(asthmaAssessment.confidence
          ? [{ label: "Confidence", value: `${asthmaAssessment.confidence}%` }]
          : []),
        {
          label: "FEV1",
          value: patientData.lungFunctionFeV1
            ? `${patientData.lungFunctionFeV1}%`
            : "--",
        },
      ],
      footer: "LightGBM model prediction",
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
      footer: "Nearest monitoring station",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          AI Doctor — Clinical Decision Support
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Generates a comprehensive, structured respiratory assessment report by
          analyzing all previous modules and correlating findings.
        </p>
      </div>

      {!hasData && (
        <div className="flex items-start gap-3 rounded-xl border border-sky-100 bg-sky-50/70 p-4">
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
          <div className="text-[13px] leading-relaxed text-slate-600">
            <span className="font-medium text-slate-900">
              Complete the assessments first.
            </span>{" "}
            Run the Audio Analysis, Asthma Detection, and Safe Check sections to
            populate the data this report needs. Then click{" "}
            <span className="font-medium">Generate Clinical Report</span> below.
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

      {/* Quick correlation preview */}
      {correlation.length > 0 && (
        <div className="bd-card p-5">
          <div className="mb-3 flex items-center gap-2">
            <Link2 className="h-4 w-4 text-sky-600" />
            <h3 className="text-[13px] font-semibold text-slate-900">
              Quick Symptom–Trigger Correlation
            </h3>
          </div>
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
        </div>
      )}

      {/* Expandable raw data summary */}
      <div className="bd-card overflow-hidden">
        <button
          type="button"
          onClick={() => setSummaryExpanded((e) => !e)}
          className="flex w-full items-center justify-between gap-2 p-5 text-left transition-colors hover:bg-slate-50/50"
          aria-expanded={summaryExpanded}
        >
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-sky-600" />
            <span className="text-[13px] font-semibold text-slate-900">
              Data Summary (for AI reasoning)
            </span>
          </div>
          {summaryExpanded ? (
            <ChevronUp className="h-4 w-4 text-slate-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-slate-400" />
          )}
        </button>
        {summaryExpanded && (
          <div className="border-t border-slate-100 p-5">
            <pre className="bd-scroll overflow-x-auto rounded-lg bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-600">
{JSON.stringify(
  {
    patientData,
    audioAnalysis,
    asthmaAssessment,
    environmentalData,
  },
  null,
  2
)}
            </pre>
          </div>
        )}
      </div>

      {/* Generate button */}
      <div className="flex flex-col items-center gap-3">
        <button
          type="button"
          onClick={generateVerdict}
          disabled={loading}
          className="bd-btn bd-btn-primary bd-btn-lg w-full max-w-sm"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Generating Report…
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" /> Generate Clinical Report
            </>
          )}
        </button>
        <p className="text-center text-xs text-slate-400">
          The AI analyzes all modules and generates an 800–1500 word structured
          report. This may take 10–20 seconds.
        </p>
      </div>

      {/* Error */}
      {error && (
        <div
          className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4"
          role="alert"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
          <div className="text-[13px] text-slate-700">
            <p className="font-medium text-slate-900">
              Could not generate the full AI report.
            </p>
            <p className="mt-1 text-slate-600">{error}</p>
            <p className="mt-1 text-slate-600">
              A structured fallback report is shown below with all available
              data.
            </p>
          </div>
        </div>
      )}

      {/* Report */}
      {report && (
        <div className="space-y-4 bd-fade-in">
          {/* Report toolbar */}
          <div className="flex flex-col gap-3 rounded-xl border border-sky-100 bg-sky-50/50 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2.5">
              <span
                className="flex h-9 w-9 items-center justify-center rounded-xl text-white shadow-sm"
                style={{
                  background: "linear-gradient(135deg, #0ea5e9, #06b6d4)",
                }}
              >
                <Stethoscope className="h-5 w-5" />
              </span>
              <div>
                <h3 className="text-[15px] font-semibold text-slate-900">
                  Clinical Decision Support Report
                </h3>
                <p className="text-[11px] text-slate-500">
                  {reportSource === "fallback"
                    ? "Fallback report (AI service unavailable)"
                    : "AI-generated · " +
                      new Date(generatedAt).toLocaleString()}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={handleExport}
              className="bd-btn bd-btn-secondary"
            >
              <Download className="h-4 w-4" /> Export Clinical Report
            </button>
          </div>

          {/* Report sections */}
          <ClinicalReport markdown={report} />
        </div>
      )}
    </div>
  );
}
