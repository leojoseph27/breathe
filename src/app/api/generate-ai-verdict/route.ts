import { NextRequest, NextResponse } from "next/server";
import { createHash } from "crypto";
import { db } from "@/lib/db";
import {
  getGeminiClient,
  GEMINI_MODEL,
  REPORT_GENERATION_CONFIG,
  SYSTEM_INSTRUCTION,
} from "@/lib/gemini";

export const runtime = "nodejs";
export const maxDuration = 120;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PatientData {
  age?: number;
  gender?: number;
  bmi?: number;
  smoking?: number;
  familyHistory?: number;
  allergyHistory?: number;
  lungFunctionFeV1?: number;
  wheezing?: number;
  shortnessOfBreath?: number;
  chestTightness?: number;
}
interface AudioAnalysis {
  prediction?: string;
  filename?: string;
  confidence?: number;
}
interface AsthmaAssessment {
  prediction?: string;
  confidence?: number;
}
interface EnvironmentalData {
  resolvedLocation?: { name?: string; region?: string; country?: string };
  aqi?: number;
  epaIndex?: number;
  pm25?: number;
  pm10?: number;
  no2?: number;
  weatherDescription?: string;
  temperature?: number;
  humidity?: number;
  windSpeed?: number;
  pressure?: number;
  cloudCover?: number;
  precip?: number;
}

interface VerdictBody {
  patientData?: PatientData;
  audioAnalysis?: AudioAnalysis;
  asthmaAssessment?: AsthmaAssessment;
  environmentalData?: EnvironmentalData;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const yn = (v?: number) =>
  v === 1 ? "Yes" : v === 0 ? "No" : "N/A";
const genderStr = (v?: number) =>
  v === 0 ? "Male" : v === 1 ? "Female" : "N/A";
const smokeStr = (v?: number) =>
  v === 0 ? "Non-smoker" : v === 1 ? "Former smoker" : v === 2 ? "Current smoker" : "N/A";

/**
 * Build the user prompt from the assessment data.
 * Optimized — only sends data that contributes to clinical reasoning.
 */
function buildPrompt(b: VerdictBody): string {
  const p = b.patientData || {};
  const a = b.audioAnalysis || {};
  const ast = b.asthmaAssessment || {};
  const e = b.environmentalData || {};
  return `Generate the clinical report for this patient.

PATIENT: ${p.age ?? "N/A"}yo ${genderStr(p.gender).toLowerCase()}, BMI ${p.bmi ?? "N/A"}, ${smokeStr(p.smoking).toLowerCase()}. Family hx asthma: ${yn(p.familyHistory)}. Allergies: ${yn(p.allergyHistory)}. FEV1: ${p.lungFunctionFeV1 ?? "N/A"}%. Symptoms: ${p.wheezing === 1 ? "wheezing" : "no wheezing"}, ${p.shortnessOfBreath === 1 ? "dyspnea" : "no dyspnea"}, ${p.chestTightness === 1 ? "chest tightness" : "no chest tightness"}.

AUDIO ANALYSIS: ${a.prediction || "N/A"}${a.confidence ? ` (${Math.round(a.confidence * 100)}% confidence)` : ""}.

ASTHMA ASSESSMENT: ${ast.prediction || "N/A"}${ast.confidence ? ` (${ast.confidence}% confidence)` : ""}.

ENVIRONMENT: ${e.resolvedLocation?.name || "N/A"}. AQI ${e.aqi ?? "N/A"} (EPA ${e.epaIndex ?? "N/A"}). PM2.5 ${e.pm25 ?? "N/A"} µg/m³. Weather: ${e.weatherDescription || "N/A"}, ${e.temperature ?? "N/A"}°C, humidity ${e.humidity ?? "N/A"}%, wind ${e.windSpeed ?? "N/A"} km/h.

Generate the full report now using the system instructions. Populate tables with actual values above.`;
}

/**
 * Compute a stable hash of the assessment data for cache lookup.
 * The cache is invalidated when ANY of the four data objects change.
 */
function computeDataHash(b: VerdictBody): string {
  const canonical = JSON.stringify({
    patientData: b.patientData || {},
    audioAnalysis: b.audioAnalysis || {},
    asthmaAssessment: b.asthmaAssessment || {},
    environmentalData: b.environmentalData || {},
  });
  return createHash("sha256").update(canonical).digest("hex");
}

// ---------------------------------------------------------------------------
// Gemini call via the official SDK
// ---------------------------------------------------------------------------

/**
 * Call Gemini using the official @google/genai SDK with a lazy singleton
 * client. Returns the generated text or throws on error.
 */
async function callGemini(
  prompt: string,
  timeoutMs: number
): Promise<string> {
  const client = getGeminiClient();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await client.models.generateContent({
      model: GEMINI_MODEL,
      contents: prompt,
      config: {
        ...REPORT_GENERATION_CONFIG,
        systemInstruction: SYSTEM_INSTRUCTION,
        abortSignal: controller.signal,
      },
    });

    const text = response.text?.trim();
    if (!text) {
      throw new Error("Gemini returned an empty response");
    }
    return text;
  } finally {
    clearTimeout(timeout);
  }
}

// ---------------------------------------------------------------------------
// Dynamic fallback (uses real patient data — zero N/A when data exists)
// ---------------------------------------------------------------------------

function buildDynamicFallback(b: VerdictBody, reason: string): string {
  const p = b.patientData || {};
  const a = b.audioAnalysis || {};
  const ast = b.asthmaAssessment || {};
  const e = b.environmentalData || {};

  const age = p.age ?? "N/A";
  const gender = genderStr(p.gender);
  const bmi = p.bmi ?? "N/A";
  const smoking = smokeStr(p.smoking);
  const familyHistory = yn(p.familyHistory);
  const allergies = yn(p.allergyHistory);
  const fev1 = p.lungFunctionFeV1 ? `${p.lungFunctionFeV1}%` : "N/A";
  const audioPred = a.prediction || "N/A";
  const asthmaPred = ast.prediction || "N/A";
  const aqi = e.aqi ?? "N/A";
  const pm25 = e.pm25 ?? "N/A";
  const weather = e.weatherDescription || "N/A";
  const location = e.resolvedLocation?.name
    ? `${e.resolvedLocation.name}, ${e.resolvedLocation.region || ""}`.trim()
    : "N/A";
  const temp = e.temperature ? `${e.temperature}°C` : "N/A";
  const humidity = e.humidity ? `${e.humidity}%` : "N/A";
  const wind = e.windSpeed ? `${e.windSpeed} km/h` : "N/A";
  const epaIdx = e.epaIndex ?? "N/A";

  const smokingRisk = p.smoking === 2 ? "High" : p.smoking === 1 ? "Moderate" : p.smoking === 0 ? "Low" : "Unknown";
  const familyRisk = p.familyHistory === 1 ? "High" : p.familyHistory === 0 ? "Low" : "Unknown";
  const bmiRisk = p.bmi && p.bmi >= 30 ? "High" : p.bmi && p.bmi >= 25 ? "Moderate" : p.bmi ? "Low" : "Unknown";
  const allergyRisk = p.allergyHistory === 1 ? "Moderate" : p.allergyHistory === 0 ? "Low" : "Unknown";
  const pollutionRisk = (e.epaIndex ?? 1) >= 4 ? "High" : (e.epaIndex ?? 1) >= 2 ? "Moderate" : "Low";

  const audioInterp = a.prediction
    ? a.prediction.toLowerCase() === "asthma"
      ? "Audio patterns consistent with asthma — wheezing suggests airway obstruction and bronchospasm."
      : a.prediction.toLowerCase() === "copd"
      ? "Audio patterns consistent with COPD — decreased breath sounds and prolonged expiration suggest chronic airflow limitation."
      : a.prediction.toLowerCase() === "pneumonia"
      ? "Audio patterns consistent with pneumonia — crackles suggest fluid in small airways."
      : a.prediction.toLowerCase() === "bronchial"
      ? "Audio patterns consistent with bronchitis — rhonchi suggest mucus in larger airways."
      : a.prediction.toLowerCase() === "healthy"
      ? "Normal respiratory sounds — no evidence of obstruction or fluid."
      : `Audio classified as ${a.prediction}.`
    : "No audio analysis performed.";

  const audioConfidence = a.confidence ? ` Confidence: ${Math.round(a.confidence * 100)}%.` : "";

  const asthmaInterp = ast.prediction
    ? ast.prediction === "Asthma Detected"
      ? `Clinical model predicts asthma${ast.confidence ? ` (${ast.confidence}% confidence)` : ""}. Driven by FEV1 ${fev1}${p.familyHistory === 1 ? ", positive family history" : ""}${p.allergyHistory === 1 ? ", allergies" : ""}${p.smoking === 2 ? ", current smoking" : ""}${p.wheezing === 1 ? ", wheezing" : ""}${p.shortnessOfBreath === 1 ? ", dyspnea" : ""}.`
      : `Clinical model does not detect asthma${ast.confidence ? ` (${ast.confidence}% confidence)` : ""}.`
    : "No clinical asthma assessment performed.";

  const envInterp = e.aqi
    ? `${location}: AQI ${aqi} (EPA ${epaIdx}), PM2.5 ${pm25} µg/m³, ${weather}, ${temp}, humidity ${humidity}, wind ${wind}. ${(e.epaIndex ?? 1) <= 1 ? "Good air quality." : (e.epaIndex ?? 1) <= 3 ? "Moderate air quality — sensitive groups should limit outdoor exertion." : "Poor air quality — stay indoors."}`
    : "No environmental data collected.";

  const audioSuggestsAsthma = a.prediction?.toLowerCase() === "asthma";
  const clinicalSuggestsAsthma = ast.prediction === "Asthma Detected";
  const bothSuggestAsthma = audioSuggestsAsthma && clinicalSuggestsAsthma;
  const modulesAgree = audioSuggestsAsthma === clinicalSuggestsAsthma;

  const correlation = bothSuggestAsthma
    ? "Audio and clinical findings CONVERGE on asthma. The audio model detected wheezing patterns and the clinical model independently predicted asthma — this convergence significantly increases diagnostic confidence."
    : !modulesAgree && a.prediction && ast.prediction
    ? `Audio and clinical findings DIVERGE: audio suggests ${a.prediction}, clinical model predicts ${ast.prediction}. This discrepancy warrants further investigation — possible mixed condition or atypical presentation.`
    : "Cross-module correlation limited — not all assessments completed. Interpret with caution.";

  const diffDiagnosis = [
    { condition: "Asthma", likelihood: audioSuggestsAsthma || clinicalSuggestsAsthma ? "More likely" : "Less likely", reason: audioSuggestsAsthma || clinicalSuggestsAsthma ? "Audio/clinical findings support" : "Not suggested" },
    { condition: "COPD", likelihood: a.prediction?.toLowerCase() === "copd" ? "More likely" : "Less likely", reason: a.prediction?.toLowerCase() === "copd" ? "Audio detected COPD" : p.smoking ? "Smoking is a risk factor" : "Not suggested" },
    { condition: "Bronchitis", likelihood: a.prediction?.toLowerCase() === "bronchial" ? "More likely" : "Less likely", reason: a.prediction?.toLowerCase() === "bronchial" ? "Audio detected bronchial patterns" : "Not assessed" },
    { condition: "Pneumonia", likelihood: a.prediction?.toLowerCase() === "pneumonia" ? "More likely" : "Unlikely", reason: a.prediction?.toLowerCase() === "pneumonia" ? "Audio detected pneumonia" : "Not suggested" },
    { condition: "Healthy", likelihood: a.prediction?.toLowerCase() === "healthy" ? "More likely" : "Unlikely", reason: a.prediction?.toLowerCase() === "healthy" ? "Audio normal" : "Abnormal findings" },
  ];

  return `## Patient Summary

${age !== "N/A" ? age + "-year-old" : "Patient"} ${gender !== "N/A" ? gender.toLowerCase() : ""} with BMI ${bmi}, ${smoking !== "N/A" ? smoking.toLowerCase() : "unknown smoking status"}. ${bothSuggestAsthma ? "Findings converge on likely asthma." : "Findings summarized below."}

| Parameter | Value |
|-----------|-------|
| Age | ${age} |
| Gender | ${gender} |
| BMI | ${bmi} |
| Smoking Status | ${smoking} |
| Family History | ${familyHistory} |
| Allergies | ${allergies} |
| Audio Prediction | ${audioPred} |
| Asthma Prediction | ${asthmaPred} |
| FEV1 | ${fev1} |
| AQI | ${aqi} |
| PM2.5 | ${pm25} |
| Weather | ${weather} |

## Clinical Findings

**Audio:** ${audioInterp}${audioConfidence}

**Asthma Assessment:** ${asthmaInterp}

**Environment:** ${envInterp}

## Cross-Module Correlation

${correlation} ${e.pm25 && e.pm25 > 25 ? `PM2.5 (${pm25} µg/m³) is elevated and may exacerbate symptoms.` : "PM2.5 is within acceptable range."} ${p.allergyHistory === 1 && audioSuggestsAsthma ? "Allergy history supports the asthma hypothesis." : ""}

## Risk & Differential Diagnosis

| Condition | Likelihood | Reasoning |
|-----------|------------|-----------|
${diffDiagnosis.map(d => `| ${d.condition} | ${d.likelihood} | ${d.reason} |`).join("\n")}

| Risk Factor | Level | Note |
|-------------|-------|------|
| Smoking | ${smokingRisk} | ${p.smoking === 2 ? "Current smoker" : p.smoking === 1 ? "Former smoker" : "Non-smoker"} |
| Family History | ${familyRisk} | ${p.familyHistory === 1 ? "Positive" : "Negative"} |
| BMI | ${bmiRisk} | ${p.bmi ? (p.bmi >= 30 ? "Obese" : p.bmi >= 25 ? "Overweight" : "Normal") : "Unknown"} |
| Allergies | ${allergyRisk} | ${p.allergyHistory === 1 ? "Present" : "Absent"} |
| Environment | ${pollutionRisk} | AQI ${aqi}, EPA ${epaIdx} |

## Recommendations

**Medical:**
- Consult a pulmonologist for comprehensive evaluation
- Consider spirometry for definitive diagnosis${asthmaPred === "Asthma Detected" ? "\n- Discuss bronchodilator therapy" : ""}

**Lifestyle:**
${p.smoking === 2 ? "- **Quit smoking** — most impactful intervention\n- Avoid secondhand smoke" : "- Maintain regular physical activity as tolerated"}
${p.bmi && p.bmi >= 25 ? "- Work toward healthy weight" : "- Maintain healthy weight"}

**Environmental:**
${pollutionRisk !== "Low" ? "- Limit outdoor activity during poor AQI\n- Use HEPA air purifiers indoors" : "- Air quality is good; no special precautions"}
${p.allergyHistory === 1 ? "- Identify and avoid known allergens" : ""}

**Follow-up:**
- Schedule follow-up within 1-2 weeks
- Re-run assessment if symptoms change
- Seek emergency care for severe dyspnea, blue lips, or inability to speak full sentences

## Limitations & Impression

This report is AI-assisted, not a medical diagnosis. Predictions require physician confirmation. Audio quality affects results. Environmental data reflects nearest monitoring station.

**Clinical Impression:** This ${age !== "N/A" ? age + "-year-old " : ""}${gender !== "N/A" ? gender.toLowerCase() + " " : ""}patient shows ${bothSuggestAsthma ? "convergent findings consistent with asthma" : modulesAgree ? "no strong evidence of acute respiratory disease" : "divergent findings warranting further investigation"}. ${audioPred !== "N/A" ? `Audio: ${audioPred}.` : ""} ${asthmaPred !== "N/A" ? `Clinical: ${asthmaPred}.` : ""} ${e.aqi ? `Environment: AQI ${aqi}.` : ""} ${pollutionRisk === "High" ? "Poor air quality is a contributing factor." : ""} Recommend pulmonology consultation with spirometry. This report does not replace professional medical advice.

${reason ? `> **Note:** ${reason}` : ""}`;
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

async function getCachedReport(hash: string) {
  try {
    return await db.clinicalReportCache.findUnique({
      where: { dataHash: hash },
    });
  } catch (err) {
    console.error("[/api/generate-ai-verdict] Cache read error:", err);
    return null;
  }
}

async function storeCachedReport(
  hash: string,
  verdict: string,
  source: string
) {
  try {
    await db.clinicalReportCache.upsert({
      where: { dataHash: hash },
      create: { dataHash: hash, verdict, source },
      update: { verdict, source },
    });
  } catch (err) {
    // Non-fatal — caching is an optimization, not a requirement
    console.error("[/api/generate-ai-verdict] Cache write error:", err);
  }
}

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

export async function POST(request: NextRequest) {
  let body: VerdictBody;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { success: false, error: "Invalid JSON body" },
      { status: 400 }
    );
  }

  const dataHash = computeDataHash(body);
  console.log("[/api/generate-ai-verdict] Data hash:", dataHash.slice(0, 16));

  // 1. Check cache first — if hit, return immediately (no Gemini call)
  const cached = await getCachedReport(dataHash);
  if (cached) {
    console.log("[/api/generate-ai-verdict] Cache HIT — returning cached report");
    return NextResponse.json({
      success: true,
      verdict: cached.verdict,
      source: cached.source,
      cached: true,
    });
  }

  console.log("[/api/generate-ai-verdict] Cache MISS — calling Gemini");

  // 2. Check API key
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.warn("[/api/generate-ai-verdict] GOOGLE_API_KEY not set");
    const fallback = buildDynamicFallback(body, "GOOGLE_API_KEY is not configured.");
    await storeCachedReport(dataHash, fallback, "fallback");
    return NextResponse.json({
      success: true,
      verdict: fallback,
      source: "fallback",
      error: "GOOGLE_API_KEY not configured",
    });
  }

  // 3. Call Gemini with retry logic (2 attempts, 60s each)
  const prompt = buildPrompt(body);
  const ATTEMPTS = 2;
  const TIMEOUT_MS = 60000;
  let lastError = "";

  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      console.log(`[/api/generate-ai-verdict] Attempt ${attempt}/${ATTEMPTS}`);
      const verdict = await callGemini(prompt, TIMEOUT_MS);
      console.log(`[/api/generate-ai-verdict] Success on attempt ${attempt}. ${verdict.length} chars, ${verdict.split(/\s+/).length} words`);

      // Cache the successful Gemini response
      await storeCachedReport(dataHash, verdict, "gemini");

      return NextResponse.json({ success: true, verdict, source: "gemini" });
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      const isTimeout = err instanceof Error && err.name === "AbortError";
      console.error(`[/api/generate-ai-verdict] Attempt ${attempt} failed: ${isTimeout ? "timeout" : lastError.slice(0, 120)}`);
      if (attempt < ATTEMPTS) {
        console.log("[/api/generate-ai-verdict] Retrying in 2s...");
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }

  // 4. All attempts failed — return dynamic fallback with real data
  console.error(`[/api/generate-ai-verdict] All ${ATTEMPTS} attempts failed. Returning dynamic fallback.`);
  const fallback = buildDynamicFallback(body, `Gemini failed after ${ATTEMPTS} attempts: ${lastError}`);
  await storeCachedReport(dataHash, fallback, "fallback");

  return NextResponse.json({
    success: true,
    verdict: fallback,
    source: "fallback",
    error: lastError,
  });
}
