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

/**
 * Build a DYNAMIC fallback report from the actual patient data.
 * This is used when Gemini is unavailable (geo-restriction, missing key, etc.)
 * but the assessment data IS available — so the report still shows real values
 * instead of hardcoded N/A placeholders.
 */
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

  // Determine risk levels from actual data
  const smokingRisk =
    p.smoking === 2 ? "High" : p.smoking === 1 ? "Moderate" : p.smoking === 0 ? "Low" : "Unknown";
  const familyRisk =
    p.familyHistory === 1 ? "High" : p.familyHistory === 0 ? "Low" : "Unknown";
  const bmiRisk =
    p.bmi && p.bmi >= 30 ? "High" : p.bmi && p.bmi >= 25 ? "Moderate" : p.bmi ? "Low" : "Unknown";
  const allergyRisk =
    p.allergyHistory === 1 ? "Moderate" : p.allergyHistory === 0 ? "Low" : "Unknown";
  const pollutionRisk =
    (e.epaIndex ?? 1) >= 4 ? "High" : (e.epaIndex ?? 1) >= 2 ? "Moderate" : "Low";
  const envRisk = pollutionRisk;

  // Audio interpretation
  const audioInterp = a.prediction
    ? a.prediction.toLowerCase() === "asthma"
      ? "The audio analysis detected patterns consistent with asthma, which typically presents as wheezing — a high-pitched whistling sound caused by narrowed airways during exhalation. This finding suggests airway obstruction and bronchospasm."
      : a.prediction.toLowerCase() === "copd"
      ? "The audio analysis detected patterns consistent with COPD, which often presents with decreased breath sounds, prolonged expiration, and occasionally wheezing. This finding suggests chronic airflow limitation."
      : a.prediction.toLowerCase() === "pneumonia"
      ? "The audio analysis detected patterns consistent with pneumonia, which typically presents with crackles (rales) — discontinuous popping sounds caused by fluid or secretions in the small airways and alveoli."
      : a.prediction.toLowerCase() === "bronchial"
      ? "The audio analysis detected patterns consistent with bronchitis, which may present with rhonchi (low-pitched continuous sounds) due to mucus in the larger airways."
      : a.prediction.toLowerCase() === "healthy"
      ? "The audio analysis detected normal respiratory sounds with no significant abnormalities. This suggests healthy airway function with no evidence of obstruction, fluid, or inflammation."
      : `The audio analysis detected patterns classified as ${a.prediction}.`
    : "No audio analysis was performed. Please complete the Audio Analysis module to obtain a CNN-based respiratory sound classification.";

  const audioConfidence = a.confidence
    ? ` The CNN model reported this prediction with ${Math.round(a.confidence * 100)}% confidence.`
    : "";

  // Asthma interpretation
  const asthmaInterp = ast.prediction
    ? ast.prediction === "Asthma Detected"
      ? `The clinical asthma assessment model predicted "Asthma Detected"${ast.confidence ? ` with ${ast.confidence}% confidence` : ""}. This prediction is driven by the patient's clinical features: FEV1 of ${fev1}${p.familyHistory === 1 ? ", positive family history of asthma" : ""}${p.allergyHistory === 1 ? ", presence of allergy history" : ""}${p.smoking === 2 ? ", current smoking status" : p.smoking === 1 ? ", former smoking history" : ""}, and reported symptoms${p.wheezing === 1 ? " including wheezing" : ""}${p.shortnessOfBreath === 1 ? ", shortness of breath" : ""}${p.chestTightness === 1 ? ", and chest tightness" : ""}.`
      : `The clinical asthma assessment model predicted "No Asthma Detected"${ast.confidence ? ` with ${ast.confidence}% confidence` : ""}. This suggests the patient's clinical features do not strongly indicate asthma at this time.`
    : "No clinical asthma assessment was performed. Please complete the Asthma Detection module to obtain a LightGBM-based risk prediction.";

  // Environmental interpretation
  const envInterp = e.aqi
    ? `The environmental monitoring data for ${location} shows an AQI of ${aqi} (US EPA Index: ${epaIdx}), PM2.5 of ${pm25} µg/m³, weather conditions of "${weather}", temperature ${temp}, humidity ${humidity}, and wind speed ${wind}. ${(e.epaIndex ?? 1) <= 1 ? "Air quality is good and unlikely to worsen respiratory symptoms." : (e.epaIndex ?? 1) <= 3 ? "Air quality is moderate; sensitive groups should limit prolonged outdoor exertion." : "Air quality is poor; asthma patients should stay indoors and use inhalers as needed."}`
    : "No environmental data was collected. Please complete the Safe Check module to obtain real-time air quality and weather data.";

  // Cross-module correlation
  const audioSuggestsAsthma = a.prediction?.toLowerCase() === "asthma";
  const clinicalSuggestsAsthma = ast.prediction === "Asthma Detected";
  const bothSuggestAsthma = audioSuggestsAsthma && clinicalSuggestsAsthma;
  const modulesAgree = audioSuggestsAsthma === clinicalSuggestsAsthma;

  const correlation = bothSuggestAsthma
    ? "The audio analysis and clinical asthma assessment CONVERGE on an asthma diagnosis. The CNN model detected respiratory sounds consistent with asthma, and the LightGBM clinical model independently predicted asthma based on the patient's questionnaire responses. This convergence significantly increases diagnostic confidence."
    : !modulesAgree && a.prediction && ast.prediction
    ? "The audio analysis and clinical asthma assessment DIVERGE. The audio model detected patterns consistent with " + a.prediction + ", while the clinical model predicted " + ast.prediction + ". This discrepancy should be investigated clinically — it may indicate a mixed condition, atypical presentation, or limitation in one of the models."
    : "Cross-module correlation is limited because not all assessments were completed. The available findings should be interpreted with caution and supplemented with clinical evaluation.";

  // Differential diagnosis based on audio prediction
  const diffDiagnosis = [
    { condition: "Asthma", reason: audioSuggestsAsthma || clinicalSuggestsAsthma ? "Audio/clinical findings support" : "Not suggested by current data", likelihood: audioSuggestsAsthma || clinicalSuggestsAsthma ? "More likely" : "Less likely" },
    { condition: "COPD", reason: a.prediction?.toLowerCase() === "copd" ? "Audio detected COPD patterns" : p.smoking ? "Smoking history is a risk factor" : "Not suggested", likelihood: a.prediction?.toLowerCase() === "copd" ? "More likely" : "Less likely" },
    { condition: "Bronchitis", reason: a.prediction?.toLowerCase() === "bronchial" ? "Audio detected bronchial patterns" : "Cough/sputum not assessed", likelihood: a.prediction?.toLowerCase() === "bronchial" ? "More likely" : "Less likely" },
    { condition: "Pneumonia", reason: a.prediction?.toLowerCase() === "pneumonia" ? "Audio detected pneumonia patterns" : "Fever/chest X-ray not assessed", likelihood: a.prediction?.toLowerCase() === "pneumonia" ? "More likely" : "Unlikely" },
    { condition: "Healthy", reason: a.prediction?.toLowerCase() === "healthy" ? "Audio analysis normal" : "Abnormal findings present", likelihood: a.prediction?.toLowerCase() === "healthy" ? "More likely" : "Unlikely" },
  ];

  return `## Patient Summary

This report presents a multimodal respiratory assessment for a ${age !== "N/A" ? age + "-year-old" : ""} ${gender !== "N/A" ? gender.toLowerCase() : "patient"} combining audio analysis, clinical asthma prediction, and environmental exposure data. ${bothSuggestAsthma ? "The findings converge on a likely asthma diagnosis." : "The findings are summarized below for clinical correlation."}

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

## Audio Analysis Interpretation

${audioInterp}${audioConfidence} The audio-based CNN model analyzes extracted features including zero-crossing rate, chroma STFT, MFCC coefficients, RMS energy, and mel spectrogram characteristics to classify the respiratory sound into one of five categories: Bronchial, Asthma, COPD, Healthy, or Pneumonia.

It is important to note that audio analysis alone has limitations — the model's prediction depends on recording quality, duration, and the presence of characteristic sounds. A single 2.5-second clip may not capture the full clinical picture. These findings should be correlated with the clinical assessment and environmental data below.

## Clinical Asthma Assessment

${asthmaInterp} The LightGBM model evaluates 10 clinical features (age, gender, BMI, smoking status, family history, allergy history, FEV1, wheezing, shortness of breath, and chest tightness) to predict asthma risk. ${p.lungFunctionFeV1 ? `An FEV1 of ${fev1} ${p.lungFunctionFeV1 < 80 ? "is below the normal range (80-120%), suggesting airflow obstruction" : "is within or above the normal range (80-120%), suggesting normal pulmonary function"}.` : ""} ${p.familyHistory === 1 ? "The positive family history increases the genetic predisposition to asthma." : ""} ${p.allergyHistory === 1 ? "The presence of allergies is a known comorbidity associated with asthma." : ""} ${p.smoking === 2 ? "Current smoking status is a significant risk factor that can worsen asthma symptoms and reduce treatment effectiveness." : ""}

## Environmental Risk Analysis

${envInterp} Elevated PM2.5 levels (above 25 µg/m³) can penetrate deep into the lungs and trigger airway inflammation, particularly affecting asthma and COPD patients. Children and elderly individuals are especially vulnerable to poor air quality — children have developing respiratory systems, and elderly patients may have reduced physiological reserve. Practical recommendations include monitoring local AQI, limiting outdoor activity during poor air quality days, and using air purifiers indoors.

## Cross-Module Correlation

${correlation} ${e.pm25 && e.pm25 > 25 ? `The environmental data shows elevated PM2.5 (${pm25} µg/m³), which may exacerbate respiratory symptoms and should be considered as a contributing environmental factor.` : "The environmental data does not show significantly elevated pollution levels."} ${p.allergyHistory === 1 && audioSuggestsAsthma ? "The patient's allergy history further supports the asthma hypothesis, as allergic conditions are closely linked to asthma pathophysiology." : ""}

## Supporting Findings

| Finding | Evidence | Interpretation |
|---------|----------|----------------|
| Audio Analysis | ${audioPred}${a.confidence ? ` (${Math.round(a.confidence * 100)}%)` : ""} | ${a.prediction ? a.prediction.toLowerCase() === "healthy" ? "Normal respiratory sounds" : "Abnormal respiratory pattern detected" : "Not assessed"} |
| Clinical Asthma | ${asthmaPred}${ast.confidence ? ` (${ast.confidence}%)` : ""} | ${asthmaPred === "Asthma Detected" ? "Supports airway obstruction" : "Does not support asthma"} |
| FEV1 | ${fev1} | ${p.lungFunctionFeV1 ? (p.lungFunctionFeV1 < 80 ? "Reduced pulmonary function" : "Normal pulmonary function") : "Not assessed"} |
| Allergy History | ${allergies} | ${p.allergyHistory === 1 ? "Increases asthma risk" : "No allergic predisposition"} |
| Family History | ${familyHistory} | ${p.familyHistory === 1 ? "Genetic predisposition" : "No family history"} |
| AQI | EPA ${epaIdx} | ${pollutionRisk === "High" ? "Poor environmental exposure" : pollutionRisk === "Moderate" ? "Moderate environmental exposure" : "Good air quality"} |
| PM2.5 | ${pm25} µg/m³ | ${e.pm25 && e.pm25 > 25 ? "Elevated particulate matter" : "Within acceptable range"} |

## Risk Factor Analysis

| Risk Factor | Level | Explanation |
|-------------|-------|-------------|
| Smoking | ${smokingRisk} | ${p.smoking === 2 ? "Current smoker — significantly increases respiratory risk" : p.smoking === 1 ? "Former smoker — elevated risk" : p.smoking === 0 ? "Non-smoker — low risk" : "Not assessed"} |
| Family History | ${familyRisk} | ${p.familyHistory === 1 ? "Positive family history of asthma" : p.familyHistory === 0 ? "No family history" : "Not assessed"} |
| Air Pollution | ${pollutionRisk} | ${e.epaIndex ? `EPA Index ${epaIdx}` : "Not assessed"} |
| BMI | ${bmiRisk} | ${p.bmi ? (p.bmi >= 30 ? "Obese — increases asthma severity" : p.bmi >= 25 ? "Overweight — moderate risk" : "Normal weight") : "Not assessed"} |
| Allergies | ${allergyRisk} | ${p.allergyHistory === 1 ? "Allergy history present" : p.allergyHistory === 0 ? "No known allergies" : "Not assessed"} |
| Environment | ${envRisk} | ${e.aqi ? `AQI ${aqi} at ${location}` : "Not assessed"} |

## Differential Diagnosis

| Possible Condition | Reason | Likelihood |
|--------------------|--------|------------|
${diffDiagnosis.map(d => `| ${d.condition} | ${d.reason} | ${d.likelihood} |`).join("\n")}

## Recommendations

### Lifestyle
${p.smoking === 2 ? "- **Quit smoking immediately** — this is the single most impactful lifestyle change for respiratory health\n- Avoid exposure to secondhand smoke" : "- Maintain a healthy lifestyle with regular physical activity as tolerated"}
${p.bmi && p.bmi >= 25 ? "- Work toward a healthy weight through diet and exercise" : "- Maintain current healthy weight"}
- Stay hydrated and practice diaphragmatic breathing exercises

### Medical
- Consult a pulmonologist or primary care physician for a comprehensive evaluation
- Bring all assessment results from this application to your appointment
${asthmaPred === "Asthma Detected" ? "- Discuss potential bronchodilator therapy and inhaled corticosteroids" : "- Discuss routine respiratory health monitoring"}
- Consider pulmonary function testing (spirometry) for definitive diagnosis

### Environmental
${pollutionRisk !== "Low" ? "- Monitor local air quality reports and limit outdoor activity during poor AQI days" : "- Air quality is currently good; no special precautions needed"}
${e.pm25 && e.pm25 > 25 ? "- Use HEPA air purifiers indoors to reduce PM2.5 exposure" : "- Maintain good indoor air ventilation"}
${p.allergyHistory === 1 ? "- Identify and avoid known allergens (dust, pollen, pet dander)" : ""}

### Monitoring
- Keep a symptom diary tracking wheezing, shortness of breath, and chest tightness
- Monitor peak flow readings regularly if you have a peak flow meter
- Track environmental conditions alongside symptoms to identify triggers

### Follow-up
- Schedule a follow-up with a healthcare provider within 1-2 weeks
- Report any worsening symptoms promptly
- Re-run this assessment if symptoms change

### Emergency Warning Signs
- Seek immediate medical attention for severe shortness of breath, blue lips/fingernails, or inability to speak full sentences
- Call emergency services if symptoms worsen rapidly or if rescue inhaler provides no relief
- Go to the emergency room for chest pain or confusion/altered consciousness

## Limitations

This report is AI-assisted and does **not** constitute a medical diagnosis. It is intended for educational and informational purposes only. All predictions require confirmation by a qualified physician. The audio analysis depends on the quality and duration of the uploaded recording. Environmental data reflects the nearest monitoring station and may not represent the patient's exact exposure. The clinical asthma model is a screening tool, not a diagnostic instrument.

${reason ? `> **Note:** ${reason} The structured report above was generated from the actual patient assessment data.` : ""}

## Overall Clinical Impression

This ${age !== "N/A" ? age + "-year-old " : ""}${gender !== "N/A" ? gender.toLowerCase() + " " : ""}patient underwent a multimodal respiratory assessment comprising audio-based disease prediction, questionnaire-based asthma risk assessment, and environmental exposure analysis. ${audioPred !== "N/A" ? `The audio analysis identified patterns consistent with ${audioPred}${a.confidence ? ` at ${Math.round(a.confidence * 100)}% confidence` : ""}.` : "Audio analysis was not completed."} ${asthmaPred !== "N/A" ? `The clinical asthma assessment ${asthmaPred === "Asthma Detected" ? "predicted asthma" : "did not detect asthma"}${ast.confidence ? ` at ${ast.confidence}% confidence` : ""}.` : "Clinical asthma assessment was not completed."} ${e.aqi ? `Environmental monitoring at ${location} showed AQI ${aqi} (EPA Index ${epaIdx}) and PM2.5 of ${pm25} µg/m³.` : "Environmental data was not collected."}

${bothSuggestAsthma ? "The convergence of audio and clinical findings toward an asthma diagnosis, combined with the patient's risk factor profile, suggests that respiratory symptoms may be associated with underlying asthma. However, this must be confirmed through formal pulmonary function testing and clinical evaluation by a qualified physician." : modulesAgree ? "The available assessment data does not strongly suggest an acute respiratory condition, but routine monitoring and follow-up are recommended." : "The divergence between audio and clinical findings warrants further investigation. A mixed or atypical presentation should be considered, and additional diagnostic workup including spirometry, imaging, and laboratory tests may be indicated."}

${pollutionRisk === "High" ? "The poor environmental air quality is a significant contributing factor that may exacerbate respiratory symptoms. Environmental modifications and exposure reduction are recommended alongside medical management." : pollutionRisk === "Moderate" ? "Moderate environmental air quality may contribute to respiratory symptoms in sensitive individuals. Standard precautions are advised." : "Current environmental conditions are favorable and unlikely to contribute to respiratory symptoms."}

**Recommended next steps:** The patient should consult with a pulmonologist for comprehensive evaluation, including spirometry and consideration of bronchodilator trial if clinically indicated. All findings from this multimodal assessment should be reviewed in conjunction with a complete medical history and physical examination. This report is not a substitute for professional medical advice, diagnosis, or treatment.`;
}

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

  // Debug logging (Task 3 & 6): log the exact payload received
  const p = body.patientData || {};
  const a = body.audioAnalysis || {};
  const ast = body.asthmaAssessment || {};
  const e = body.environmentalData || {};
  console.log("[/api/generate-ai-verdict] Received payload:", {
    patientData: { age: p.age, gender: p.gender, bmi: p.bmi, smoking: p.smoking, familyHistory: p.familyHistory, allergyHistory: p.allergyHistory, fev1: p.lungFunctionFeV1, wheezing: p.wheezing, sob: p.shortnessOfBreath, chest: p.chestTightness },
    audio: { prediction: a.prediction, confidence: a.confidence, filename: a.filename },
    asthma: { prediction: ast.prediction, confidence: ast.confidence },
    env: { location: e.resolvedLocation?.name, aqi: e.aqi, pm25: e.pm25, epa: e.epaIndex, weather: e.weatherDescription },
  });

  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    console.warn("[/api/generate-ai-verdict] GOOGLE_API_KEY not set — using dynamic fallback.");
    return NextResponse.json({
      success: true,
      verdict: buildDynamicFallback(body, "GOOGLE_API_KEY is not configured."),
      source: "fallback",
      error: "GOOGLE_API_KEY not configured",
    });
  }

  const prompt = buildPrompt(body);
  console.log("[/api/generate-ai-verdict] Prompt length:", prompt.length, "chars");
  console.log("[/api/generate-ai-verdict] Prompt preview (first 300 chars):", prompt.slice(0, 300));

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
      let errObj: { error?: { code?: number; message?: string; status?: string } } = {};
      try { errObj = JSON.parse(errText); } catch { /* not JSON */ }
      const errMsg = errObj.error?.message || errText.slice(0, 200);
      const errCode = errObj.error?.code || res.status;
      console.error(`[/api/generate-ai-verdict] Gemini API error ${errCode}:`, errMsg);

      // Task 5: Return a descriptive error (not a silent fallback)
      return NextResponse.json({
        success: true,
        verdict: buildDynamicFallback(body, `Gemini API returned ${errCode}: ${errMsg}`),
        source: "fallback",
        error: `Gemini API ${errCode}: ${errMsg}`,
      });
    }

    const data = await res.json();
    console.log("[/api/generate-ai-verdict] Gemini response keys:", Object.keys(data));
    console.log("[/api/generate-ai-verdict] Usage:", data.usageMetadata);

    const verdict =
      data?.candidates?.[0]?.content?.parts
        ?.map((part: { text?: string }) => part.text ?? "")
        .join("")
        .trim() ?? "";

    console.log("[/api/generate-ai-verdict] Verdict length:", verdict.length, "chars");

    if (verdict) {
      return NextResponse.json({ success: true, verdict });
    }

    // Gemini returned empty content (possibly blocked by safety filters)
    const finishReason = data?.candidates?.[0]?.finishReason;
    const blockReason = data?.promptFeedback?.blockReason;
    const emptyReason = finishReason
      ? `Gemini returned empty content (finishReason: ${finishReason})`
      : blockReason
      ? `Gemini blocked the request (blockReason: ${blockReason})`
      : "Gemini returned an empty response";

    return NextResponse.json({
      success: true,
      verdict: buildDynamicFallback(body, emptyReason),
      source: "fallback",
      error: emptyReason,
    });
  } catch (err) {
    console.error("[/api/generate-ai-verdict] Fetch error:", err);
    const errMsg = err instanceof Error ? err.message : String(err);
    const isTimeout = err instanceof Error && err.name === "AbortError";
    const reason = isTimeout
      ? "Gemini request timed out after 28 seconds"
      : `Network error: ${errMsg}`;
    return NextResponse.json({
      success: true,
      verdict: buildDynamicFallback(body, reason),
      source: "fallback",
      error: reason,
    });
  }
}
