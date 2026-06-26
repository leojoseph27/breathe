"use client";

import { useState, type ReactNode } from "react";
import { useBreatheStore, type PatientData } from "@/lib/breathe-store";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { InfoTooltip } from "@/components/breathe/InfoTooltip";
import { Loader2, AlertCircle, CheckCircle2, Activity } from "lucide-react";

interface FieldDef {
  key: keyof PatientData;
  label: string;
  type: "number" | "select";
  options?: { value: string; label: string }[];
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
  tooltip: ReactNode;
}

const FIELDS: FieldDef[] = [
  {
    key: "age",
    label: "Age",
    type: "number",
    min: 0,
    max: 120,
    placeholder: "Example: 5–90 years",
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Age</div>
        <div>
          The patient&apos;s age in years. Asthma can occur at any age but is
          most commonly diagnosed in childhood. Risk factors and treatment
          approaches may vary by age group.
        </div>
        <div className="mt-2">
          <span className="font-semibold text-sky-300">Example: </span>35
        </div>
      </>
    ),
  },
  {
    key: "gender",
    label: "Gender",
    type: "select",
    options: [
      { value: "0", label: "Male" },
      { value: "1", label: "Female" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Gender</div>
        <div>
          Biological sex of the patient. Some studies suggest asthma prevalence
          and severity differ between males and females, particularly before
          and after puberty.
        </div>
      </>
    ),
  },
  {
    key: "bmi",
    label: "BMI",
    type: "number",
    min: 10,
    max: 50,
    step: 0.1,
    placeholder: "Example: 15–45 kg/m²",
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">
          Body Mass Index (BMI)
        </div>
        <div>Measures body weight relative to height.</div>
        <div className="mt-2">
          <span className="font-semibold text-sky-300">Normal: </span>
          18.5–24.9
        </div>
        <div>
          Higher BMI is associated with increased asthma severity and reduced
          lung function.
        </div>
        <div className="mt-2">
          <span className="font-semibold text-sky-300">Example: </span>22.3
        </div>
      </>
    ),
  },
  {
    key: "smoking",
    label: "Smoking Status",
    type: "select",
    options: [
      { value: "0", label: "Non-smoker" },
      { value: "1", label: "Former smoker" },
      { value: "2", label: "Current smoker" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Smoking Status</div>
        <div>
          <span className="font-semibold text-sky-300">Non-smoker</span> —
          Never smoked.
        </div>
        <div>
          <span className="font-semibold text-sky-300">Former Smoker</span> —
          Previously smoked but quit.
        </div>
        <div>
          <span className="font-semibold text-sky-300">Current Smoker</span> —
          Currently smokes cigarettes or other tobacco products.
        </div>
      </>
    ),
  },
  {
    key: "familyHistory",
    label: "Family History of Asthma",
    type: "select",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">
          Family History of Asthma
        </div>
        <div>
          Select &quot;Yes&quot; if an immediate family member (parent or
          sibling) has been diagnosed with asthma.
        </div>
        <div className="mt-2">
          A family history increases the likelihood of developing asthma.
        </div>
      </>
    ),
  },
  {
    key: "allergyHistory",
    label: "Allergy History",
    type: "select",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Allergies</div>
        <div>Includes allergies to:</div>
        <ul className="mt-1 list-disc pl-4">
          <li>Dust</li>
          <li>Pollen</li>
          <li>Pet dander</li>
          <li>Food</li>
          <li>Medication</li>
        </ul>
        <div className="mt-2">
          Allergic conditions are closely linked to asthma.
        </div>
      </>
    ),
  },
  {
    key: "lungFunctionFeV1",
    label: "Lung Function (FEV1 %)",
    type: "number",
    min: 20,
    max: 100,
    step: 0.1,
    placeholder: "Example: 30–120%",
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">
          Forced Expiratory Volume in 1 second (FEV1)
        </div>
        <div>Indicates lung function.</div>
        <div className="mt-2">
          <span className="font-semibold text-sky-300">
            Healthy adults:
          </span>{" "}
          80–120%
        </div>
        <div>Lower values may indicate airway obstruction.</div>
        <div className="mt-2">
          <span className="font-semibold text-sky-300">Example: </span>85%
        </div>
      </>
    ),
  },
  {
    key: "wheezing",
    label: "Wheezing",
    type: "select",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Wheezing</div>
        <div>
          A high-pitched whistling sound during breathing, caused by narrowed
          airways.
        </div>
        <div className="mt-2">
          Wheezing is a classic asthma symptom and indicates airway
          constriction.
        </div>
      </>
    ),
  },
  {
    key: "shortnessOfBreath",
    label: "Shortness of Breath",
    type: "select",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">
          Shortness of Breath
        </div>
        <div>
          Difficulty breathing or feeling unable to get enough air. Also known
          as dyspnea.
        </div>
        <div className="mt-2">
          A common symptom of asthma exacerbation.
        </div>
      </>
    ),
  },
  {
    key: "chestTightness",
    label: "Chest Tightness",
    type: "select",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes" },
    ],
    tooltip: (
      <>
        <div className="mb-1 font-semibold text-sky-300">Chest Tightness</div>
        <div>
          A feeling of pressure or squeezing in the chest. Often described as
          &quot;a band tightening around the chest.&quot;
        </div>
        <div className="mt-2">
          A characteristic symptom of asthma.
        </div>
      </>
    ),
  },
];

export function AsthmaDetectionView() {
  const setPatientData = useBreatheStore((s) => s.setPatientData);
  const setAsthmaAssessment = useBreatheStore((s) => s.setAsthmaAssessment);
  const storedAssessment = useBreatheStore((s) => s.asthmaAssessment);

  const [values, setValues] = useState<Record<string, string>>({});
  const [missing, setMissing] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    prediction: number;
    confidence: number;
  } | null>(
    storedAssessment.raw !== undefined && storedAssessment.confidence !== undefined
      ? { prediction: storedAssessment.raw, confidence: storedAssessment.confidence }
      : null
  );
  const [error, setError] = useState("");

  function setField(key: string, val: string) {
    setValues((v) => ({ ...v, [key]: val }));
    setMissing((m) => {
      const n = new Set(m);
      n.delete(key);
      return n;
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");

    const missingFields = FIELDS.filter((f) => !values[f.key]);
    if (missingFields.length) {
      setMissing(new Set(missingFields.map((f) => f.key)));
      setError("Please fill in all required fields");
      return;
    }

    const data: PatientData = {
      age: parseInt(values.age),
      gender: parseInt(values.gender),
      bmi: parseFloat(values.bmi),
      smoking: parseInt(values.smoking),
      familyHistory: parseInt(values.familyHistory),
      allergyHistory: parseInt(values.allergyHistory),
      lungFunctionFeV1: parseFloat(values.lungFunctionFeV1),
      wheezing: parseInt(values.wheezing),
      shortnessOfBreath: parseInt(values.shortnessOfBreath),
      chestTightness: parseInt(values.chestTightness),
    };

    setPatientData(data);
    setLoading(true);
    try {
      const res = await fetch("/api/predict-asthma", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const json = await res.json();
      if (!res.ok || json.error) {
        setError(json.error || "Could not make prediction");
        return;
      }
      const r = { prediction: json.prediction, confidence: json.confidence };
      setResult(r);
      setAsthmaAssessment({
        prediction: r.prediction === 1 ? "Asthma Detected" : "No Asthma Detected",
        confidence: r.confidence,
        raw: r.prediction,
      });
    } catch (err) {
      setError("An error occurred during prediction: " + String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          Asthma Prediction
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Enter patient information for comprehensive asthma diagnosis
          prediction using a LightGBM model trained on 10 clinical features.
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="bd-section p-5 sm:p-6">
        <h3 className="mb-5 text-[13px] font-semibold uppercase tracking-wide text-slate-400">
          Patient Information
        </h3>
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {FIELDS.map((f) => (
            <div key={f.key} className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <label
                  htmlFor={f.key}
                  className="text-[13px] font-medium text-slate-700"
                >
                  {f.label}
                </label>
                <InfoTooltip label={f.label}>{f.tooltip}</InfoTooltip>
              </div>
              {f.type === "number" ? (
                <input
                  id={f.key}
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  placeholder={f.placeholder}
                  value={values[f.key] ?? ""}
                  onChange={(e) => setField(f.key, e.target.value)}
                  className={`bd-input ${missing.has(f.key) ? "bd-input-error" : ""}`}
                />
              ) : (
                <Select
                  value={values[f.key] ?? ""}
                  onValueChange={(v) => setField(f.key, v)}
                >
                  <SelectTrigger
                    id={f.key}
                    className={`bd-select-trigger w-full ${
                      missing.has(f.key) ? "bd-input-error" : ""
                    }`}
                  >
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent>
                    {f.options!.map((o) => (
                      <SelectItem key={o.value} value={o.value}>
                        {o.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-center">
          <button
            type="submit"
            disabled={loading}
            className="bd-btn bd-btn-primary bd-btn-lg w-full max-w-xs"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Analyzing…
              </>
            ) : (
              <>
                <Activity className="h-4 w-4" /> Predict Asthma
              </>
            )}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div
          className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4"
          role="alert"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
          <div className="text-[13px] text-slate-700">{error}</div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="bd-card bd-scale-in p-6 text-center">
          <div className="mb-1 flex items-center justify-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            Diagnosis
          </div>
          <div
            className="mt-2 text-3xl font-bold"
            style={{
              color: result.prediction === 1 ? "#ef4444" : "#10b981",
            }}
          >
            {result.prediction === 1
              ? "Asthma Detected"
              : "No Asthma Detected"}
          </div>
          <div className="mt-2 text-sm text-slate-500">
            Confidence{" "}
            <span className="font-semibold text-slate-700">
              {result.confidence}%
            </span>
          </div>
          <div className="mt-4 rounded-lg bg-slate-50 p-3 text-left text-[13px] leading-relaxed">
            {result.prediction === 1 ? (
              <p className="text-red-600">
                ⚠️ Recommendation: Consult a pulmonologist for proper diagnosis
                and treatment. Consider pulmonary function tests and possible
                bronchodilator therapy.
              </p>
            ) : (
              <p className="text-emerald-600">
                ✅ Recommendation: Continue monitoring respiratory health.
                Maintain regular check-ups and avoid known triggers.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
