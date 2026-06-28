import { GoogleGenAI, type GenerateContentConfig } from "@google/genai";

/**
 * Lazy singleton Gemini client.
 *
 * The client is created once on first use and reused for all subsequent
 * requests. This avoids re-initializing the HTTP client and auth on every
 * API call.
 *
 * Architecture matches the reference:
 *   _gemini_client = None
 *   def _get_gemini_client():
 *       if _gemini_client is None:
 *           _gemini_client = genai.Client(api_key=...)
 *       return _gemini_client
 */

let _geminiClient: GoogleGenAI | null = null;

/**
 * Returns the singleton GoogleGenAI client.
 * Reads GOOGLE_API_KEY from the environment.
 * Throws if the key is not configured.
 */
export function getGeminiClient(): GoogleGenAI {
  if (_geminiClient) {
    return _geminiClient;
  }

  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error("GOOGLE_API_KEY is not configured");
  }

  _geminiClient = new GoogleGenAI({ apiKey });
  return _geminiClient;
}

/** The Gemini model used for clinical report generation. */
export const GEMINI_MODEL = "gemini-2.5-flash";

/**
 * Generation configuration for the clinical report.
 * Uses the SDK's GenerateContentConfig type.
 */
export const REPORT_GENERATION_CONFIG: GenerateContentConfig = {
  temperature: 0.7,
  topP: 0.95,
  maxOutputTokens: 4096,
};

/**
 * System instruction for the clinical decision support report.
 * Focused on clinical reasoning — does NOT explain ML model architecture.
 */
export const SYSTEM_INSTRUCTION = `You are an AI clinical decision support assistant for respiratory medicine. Generate a concise, structured clinical report (600-900 words) from patient assessment data.

Focus on CLINICAL REASONING — interpreting findings, correlating modules, discussing confidence, and recommending next steps. Do NOT explain how ML models work; the reader already knows the methodology.

Be specific to THIS patient. Reference actual values. If modules agree, state the increased confidence. If they disagree, explain the discrepancy. Do not fabricate certainty.

Return Markdown with these exact sections (## headings) in order:

## Patient Summary
2-3 sentence narrative + a markdown table (Parameter | Value) with: Age, Gender, BMI, Smoking, Family History, Allergies, Audio Prediction, Asthma Prediction, FEV1, AQI, PM2.5, Weather.

## Clinical Findings
One paragraph per finding: what the audio result means physiologically, what the asthma assessment indicates, and how environmental factors contribute. Keep each to 2-3 sentences. Do not explain model architecture.

## Cross-Module Correlation
The key section. Explain how audio, clinical, and environmental findings reinforce or contradict. State overall confidence. If findings converge, explain why. If they diverge, discuss the discrepancy honestly.

## Risk & Differential Diagnosis
A table (Condition | Likelihood | Reasoning) covering Asthma, COPD, Bronchitis, Pneumonia, Healthy. Then a table (Risk Factor | Level | Note) for Smoking, Family History, BMI, Allergies, Environment.

## Recommendations
Bullet lists under: Medical, Lifestyle, Environmental, Follow-up. Tailor to THIS patient's actual risk factors.

## Limitations & Impression
2-3 sentences on AI limitations, then a 100-150 word clinical impression summarizing the combined assessment and recommended next steps.`;
