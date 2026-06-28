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

const yn = (v?: number) =>
  v === 1 ? "Yes" : v === 0 ? "No" : "N/A";
const genderStr = (v?: number) =>
  v === 0 ? "Male" : v === 1 ? "Female" : "N/A";
const smokeStr = (v?: number) =>
  v === 0 ? "Non-smoker" : v === 1 ? "Former smoker" : v === 2 ? "Current smoker" : "N/A";

const SYSTEM_INSTRUCTION = `You are an expert AI clinical decision support system specializing in respiratory medicine, designed to assist pulmonologists and primary care physicians.

Your task is to generate a COMPREHENSIVE, STRUCTURED CLINICAL DECISION SUPPORT REPORT from multimodal patient assessment data. This is NOT a chatbot response — it is a formal medical report suitable for academic demonstration.

REQUIREMENTS:
1. Analyze EVERY module (audio, clinical asthma assessment, environmental) SEPARATELY before combining them.
2. CORRELATE findings across modules — explain how evidence reinforces or contradicts.
3. Use professional medical language appropriate for an academic clinical report.
4. Clearly distinguish OBSERVATIONS from INTERPRETATIONS from RECOMMENDATIONS from LIMITATIONS.
5. Do not be generic. Reference the specific patient values provided.
6. If modules disagree, explain the disagreement — do not force a single conclusion.
7. Do not fabricate certainty. Express appropriate confidence levels.
8. The report should be 800–1500 words depending on available data.

OUTPUT FORMAT: Return well-structured Markdown with these exact section headings (## level), in this exact order:

## Patient Summary
A brief 2-3 sentence narrative introducing the patient profile, followed by a markdown table with columns "Parameter" and "Value" containing: Age, Gender, BMI, Smoking Status, Family History, Allergies, Audio Prediction, Asthma Prediction, FEV1, AQI, PM2.5, Weather.

## Audio Analysis Interpretation
Several paragraphs interpreting the audio respiratory prediction. Explain what the detected sound indicates physiologically (wheezing, crackles, obstruction, etc.), the confidence if available, possible physiological meaning, and limitations of audio-based analysis. Do not simply repeat the disease name.

## Clinical Asthma Assessment
Several paragraphs interpreting the clinical asthma prediction. Explain WHY the questionnaire-based model reached its conclusion — the role of FEV1, family history, allergy history, smoking status, BMI, and reported symptoms (wheezing, shortness of breath, chest tightness) in influencing the prediction.

## Environmental Risk Analysis
Several paragraphs analyzing AQI, PM2.5, weather, wind, humidity, and temperature. Explain effects on asthma and COPD, special considerations for children and elderly, and practical environmental recommendations.

## Cross-Module Correlation
The most important section. Correlate ALL modules. Explain how audio findings, clinical assessment, and environmental data reinforce or contradict each other. If findings converge, explain the increased confidence. If they diverge, explain the discrepancy honestly.

## Supporting Findings
A markdown table with columns "Finding", "Evidence", and "Interpretation". Populate dynamically from the actual patient data — do not use placeholder values.

## Risk Factor Analysis
A markdown table with columns "Risk Factor", "Level" (Low/Moderate/High), and "Explanation". Include: Smoking, Family history, Air pollution, BMI, Allergies, Environment. Assess each based on the actual patient values.

## Differential Diagnosis
A markdown table with columns "Possible Condition", "Reason", and "Likelihood" (More likely / Less likely / Unlikely). Discuss asthma, COPD, bronchitis, pneumonia, and healthy as alternatives. Explain why each is more or less likely based on the evidence.

## Recommendations
Organized into subsections with bullet lists:
### Lifestyle
### Medical
### Environmental
### Monitoring
### Follow-up
### Emergency Warning Signs

## Limitations
Explain clearly that this is AI-assisted, not a medical diagnosis, requires physician confirmation, predictions depend on audio quality, and environmental data is only one contributing factor.

## Overall Clinical Impression
A detailed conclusion of 300-500 words summarizing the patient profile, respiratory audio findings, asthma assessment, environmental contribution, overall confidence level, and recommended next steps. This should read like a formal clinical impression in a medical report.`;

const MODEL = "gemini-2.5-flash";
const API_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;

function buildPrompt(b: VerdictBody): string {
  const p = b.patientData || {};
  const a = b.audioAnalysis || {};
  const ast = b.asthmaAssessment || {};
  const e = b.environmentalData || {};
  return `Generate a comprehensive AI-assisted Clinical Decision Support Report based on the following multimodal patient assessment data.

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

AUDIO RESPIRATORY ANALYSIS (CNN model prediction):
- Detected Condition: ${a.prediction || "N/A"}
- Audio File: ${a.filename || "N/A"}
${a.confidence ? `- Model Confidence: ${Math.round(a.confidence * 100)}%` : ""}

CLINICAL ASTHMA ASSESSMENT (LightGBM model prediction):
- Diagnosis: ${ast.prediction || "N/A"}
${ast.confidence ? `- Model Confidence: ${ast.confidence}%` : ""}

ENVIRONMENTAL EXPOSURE DATA:
- Location: ${e.resolvedLocation?.name || "N/A"}, ${e.resolvedLocation?.region || "N/A"}, ${e.resolvedLocation?.country || "N/A"}
- AQI: ${e.aqi ?? "N/A"} (US EPA Index: ${e.epaIndex ?? "N/A"})
- PM2.5: ${e.pm25 ?? "N/A"} µg/m³
- PM10: ${e.pm10 ?? "N/A"} µg/m³
- NO₂: ${e.no2 ?? "N/A"} µg/m³
- Weather: ${e.weatherDescription || "N/A"}
- Temperature: ${e.temperature ?? "N/A"}°C
- Humidity: ${e.humidity ?? "N/A"}%
- Wind Speed: ${e.windSpeed ?? "N/A"} km/h
- Pressure: ${e.pressure ?? "N/A"} hPa
- Cloud Cover: ${e.cloudCover ?? "N/A"}%
- Precipitation: ${e.precip ?? "N/A"} mm

Now generate the full structured report following the system instructions exactly. Use the section headings specified. Populate all tables with the ACTUAL patient values above — never use placeholder or hardcoded values.`;
}

const FALLBACK_VERDICT = `## Patient Summary

This report presents a multimodal respiratory assessment combining audio analysis, clinical asthma prediction, and environmental exposure data. Due to an AI service limitation, the detailed narrative sections below could not be generated automatically; the structured data and recommendations are still available for physician review.

| Parameter | Value |
|-----------|-------|
| Age | ${"N/A"} |
| Gender | ${"N/A"} |
| BMI | ${"N/A"} |
| Smoking Status | ${"N/A"} |
| Family History | ${"N/A"} |
| Allergies | ${"N/A"} |
| Audio Prediction | ${"N/A"} |
| Asthma Prediction | ${"N/A"} |
| FEV1 | ${"N/A"} |
| AQI | ${"N/A"} |
| PM2.5 | ${"N/A"} |
| Weather | ${"N/A"} |

*Note: The patient-specific values above were not populated because the AI service was unavailable. Please refer to the other assessment tabs for the actual data.*

## Audio Analysis Interpretation

The audio respiratory analysis could not be interpreted in detail at this time. Please refer to the Audio Analysis tab for the CNN model's prediction and confidence score. A qualified clinician should review the original audio recording to confirm the presence of abnormal respiratory sounds such as wheezing, crackles, or decreased breath sounds.

## Clinical Asthma Assessment

The clinical asthma assessment could not be interpreted in detail at this time. Please refer to the Asthma Detection tab for the LightGBM model's prediction. The clinical questionnaire captures key variables including FEV1, family history, allergy history, smoking status, BMI, and respiratory symptoms (wheezing, shortness of breath, chest tightness), each of which contributes to the model's prediction.

## Environmental Risk Analysis

The environmental risk analysis could not be interpreted in detail at this time. Please refer to the Safe Check tab for the current AQI, PM2.5, weather conditions, and weather triggers. Elevated PM2.5 and poor AQI are known triggers for asthma exacerbations, particularly in children and elderly patients.

## Cross-Module Correlation

Cross-module correlation could not be performed automatically. The convergence of audio findings, clinical assessment, and environmental data typically strengthens diagnostic confidence. If the modules disagree, the discrepancy should be investigated clinically rather than resolved by majority vote.

## Supporting Findings

| Finding | Evidence | Interpretation |
|---------|----------|----------------|
| Audio | See Audio Analysis tab | Requires clinical correlation |
| Clinical | See Asthma Detection tab | Requires clinical correlation |
| Environmental | See Safe Check tab | Requires clinical correlation |

## Risk Factor Analysis

| Risk Factor | Level | Explanation |
|-------------|-------|-------------|
| Smoking | Unknown | Requires patient data |
| Family history | Unknown | Requires patient data |
| Air pollution | Unknown | Requires environmental data |
| BMI | Unknown | Requires patient data |
| Allergies | Unknown | Requires patient data |
| Environment | Unknown | Requires environmental data |

## Differential Diagnosis

| Possible Condition | Reason | Likelihood |
|--------------------|--------|------------|
| Asthma | Requires clinical correlation | Underdetermined |
| COPD | Requires clinical correlation | Underdetermined |
| Bronchitis | Requires clinical correlation | Underdetermined |
| Pneumonia | Requires clinical correlation | Underdetermined |
| Healthy | Requires clinical correlation | Underdetermined |

## Recommendations

### Lifestyle
- Maintain a healthy lifestyle with regular physical activity as tolerated
- Avoid known respiratory triggers including smoke, dust, and allergens

### Medical
- Consult a pulmonologist or primary care physician for a comprehensive evaluation
- Bring all assessment results from this application to your appointment

### Environmental
- Monitor local air quality reports and limit outdoor activity during poor AQI days
- Use air purifiers indoors if PM2.5 levels are elevated

### Monitoring
- Keep a symptom diary tracking wheezing, shortness of breath, and chest tightness
- Monitor peak flow readings regularly if you have a peak flow meter

### Follow-up
- Schedule a follow-up with a healthcare provider within 1-2 weeks
- Report any worsening symptoms promptly

### Emergency Warning Signs
- Seek immediate medical attention for severe shortness of breath, blue lips/fingernails, or inability to speak full sentences
- Call emergency services if symptoms worsen rapidly

## Limitations

This report is AI-assisted and does **not** constitute a medical diagnosis. It is intended for informational and educational purposes only. All predictions require confirmation by a qualified physician. The audio analysis depends on the quality and duration of the uploaded recording. Environmental data reflects the nearest monitoring station and may not represent the patient's exact exposure. The AI service was unavailable during this generation, so the narrative sections are limited — please re-run the assessment when the service is available.

## Overall Clinical Impression

This AI-assisted clinical decision support report combines multimodal respiratory assessment data including audio-based disease prediction, questionnaire-based asthma risk assessment, and real-time environmental exposure analysis. Due to a temporary limitation with the AI generation service, a detailed narrative impression could not be produced at this time. The structured data tables and recommendations above remain available for physician review.

The patient should consult with a qualified healthcare provider for a comprehensive clinical evaluation. All findings presented in this report should be interpreted in conjunction with a complete medical history, physical examination, and appropriate diagnostic testing. This report is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition.`;

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
          maxOutputTokens: 8192,
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
