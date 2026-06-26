import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";
export const maxDuration = 30;

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
}
interface AsthmaAssessment {
  prediction?: string;
}
interface EnvironmentalData {
  resolvedLocation?: { name?: string; region?: string };
  aqi?: number;
  epaIndex?: number;
  pm25?: number;
  weatherDescription?: string;
  humidity?: number;
  windSpeed?: number;
}

interface VerdictBody {
  patientData?: PatientData;
  audioAnalysis?: AudioAnalysis;
  asthmaAssessment?: AsthmaAssessment;
  environmentalData?: EnvironmentalData;
}

const yn = (v?: number) =>
  v === 1 ? "Yes" : v === 0 ? "No" : "N/A";
const genderStr = (v?: number) =>
  v === 0 ? "Male" : v === 1 ? "Female" : "N/A";
const smokeStr = (v?: number) =>
  v === 0 ? "Non-smoker" : v === 1 ? "Former smoker" : v === 2 ? "Current smoker" : "N/A";

const SYSTEM_INSTRUCTION =
  "You are an AI doctor specializing in respiratory medicine. Provide clear, professional, empathetic medical verdicts based on patient data. Always include a disclaimer that this is not a substitute for professional medical advice.";

const MODEL = "gemini-2.5-flash";
const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

function buildPrompt(b: VerdictBody): string {
  const p = b.patientData || {};
  const a = b.audioAnalysis || {};
  const ast = b.asthmaAssessment || {};
  const e = b.environmentalData || {};
  return `
As an AI doctor specializing in respiratory medicine, please provide a comprehensive medical verdict based on the following patient data:

PATIENT DEMOGRAPHICS & CLINICAL DATA:
- Age: ${p.age ?? "N/A"}
- Gender: ${genderStr(p.gender)}
- BMI: ${p.bmi ?? "N/A"}
- Smoking Status: ${smokeStr(p.smoking)}
- Family History of Asthma: ${yn(p.familyHistory)}
- Allergy History: ${yn(p.allergyHistory)}
- Lung Function (FEV1): ${p.lungFunctionFeV1 ?? "N/A"}%
- Wheezing: ${yn(p.wheezing)}
- Shortness of Breath: ${yn(p.shortnessOfBreath)}
- Chest Tightness: ${yn(p.chestTightness)}

AUDIO RESPIRATORY ANALYSIS:
- Detected Condition: ${a.prediction || "N/A"}

CLINICAL ASTHMA ASSESSMENT:
- Diagnosis: ${ast.prediction || "N/A"}

ENVIRONMENTAL EXPOSURE DATA:
- Location: ${e.resolvedLocation?.name || "N/A"}, ${e.resolvedLocation?.region || "N/A"}
- AQI: ${e.aqi ?? "N/A"} (EPA Index: ${e.epaIndex ?? "N/A"})
- PM2.5: ${e.pm25 ?? "N/A"} µg/m³
- Weather: ${e.weatherDescription || "N/A"}
- Humidity: ${e.humidity ?? "N/A"}%
- Wind Speed: ${e.windSpeed ?? "N/A"} km/h

Based on this comprehensive multimodal assessment, please provide a detailed medical verdict including:
1. Primary diagnosis consideration
2. Contributing factors (environmental, clinical, demographic)
3. Risk assessment
4. Recommended next steps for the patient
5. Any urgent concerns that require immediate attention

Please frame your response as if you're communicating directly with the patient, using clear but professional language. Keep your response concise but comprehensive, approximately 150-200 words.

Return the response as clean plain text.
Do not use HTML tags or Markdown.
Use paragraphs separated by line breaks only.
`.trim();
}

const FALLBACK_VERDICT =
  "Based on the comprehensive assessment including audio analysis, clinical data, and environmental factors: The patient shows signs that warrant medical attention. Please consult with a healthcare professional for proper evaluation and guidance. Consider avoiding known triggers and monitor symptoms closely.";

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

  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.warn(
      "[/api/generate-ai-verdict] GOOGLE_API_KEY is not set — returning fallback verdict."
    );
    return NextResponse.json({
      success: true,
      verdict: FALLBACK_VERDICT,
      source: "fallback",
      error: "GOOGLE_API_KEY not configured",
    });
  }

  const prompt = buildPrompt(body);

  // Call the Gemini REST API directly via fetch (no SDK) — this is the most
  // reliable approach for serverless runtimes (Vercel) and avoids any
  // SDK-level process-crash issues.
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 28000);

    const res = await fetch(`${API_ENDPOINT}?key=${encodeURIComponent(apiKey)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 600,
        },
      }),
    });

    clearTimeout(timeout);

    if (!res.ok) {
      const errText = await res.text();
      console.error(
        `[/api/generate-ai-verdict] Gemini API ${res.status}:`,
        errText.slice(0, 200)
      );
      return NextResponse.json({
        success: true,
        verdict: FALLBACK_VERDICT,
        source: "fallback",
        error: `Gemini API ${res.status}: ${errText.slice(0, 120)}`,
      });
    }

    const data = await res.json();
    const verdict =
      data?.candidates?.[0]?.content?.parts
        ?.map((p: { text?: string }) => p.text ?? "")
        .join("")
        .trim() ?? "";

    if (verdict) {
      return NextResponse.json({ success: true, verdict });
    }

    return NextResponse.json({
      success: true,
      verdict: FALLBACK_VERDICT,
      source: "fallback",
      error: "Empty response from Gemini",
    });
  } catch (err) {
    console.error("[/api/generate-ai-verdict] error:", err);
    const errMsg =
      err instanceof Error ? err.message : String(err);
    return NextResponse.json({
      success: true,
      verdict: FALLBACK_VERDICT,
      source: "fallback",
      error: errMsg,
    });
  }
}
