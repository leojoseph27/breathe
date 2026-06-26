import { NextRequest, NextResponse } from "next/server";

const ML_SERVICE_URL = "http://localhost:5001";

const REQUIRED = [
  "age",
  "gender",
  "bmi",
  "smoking",
  "familyHistory",
  "allergyHistory",
  "lungFunctionFeV1",
  "wheezing",
  "shortnessOfBreath",
  "chestTightness",
] as const;

export async function POST(request: NextRequest) {
  try {
    const data = await request.json().catch(() => null);
    if (!data || typeof data !== "object") {
      return NextResponse.json(
        { error: "Invalid JSON body" },
        { status: 400 }
      );
    }

    for (const field of REQUIRED) {
      if (data[field] === undefined || data[field] === null || data[field] === "") {
        return NextResponse.json(
          { error: `Missing required field: ${field}` },
          { status: 400 }
        );
      }
    }

    // Forward to the Python LightGBM service.
    const upstream = await fetch(`${ML_SERVICE_URL}/predict-asthma`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await upstream.json().catch(() => null);

    if (!upstream.ok || !result || result.error) {
      // Fallback heuristic mirroring the original service fallback.
      const fb = fallbackAsthma(data);
      return NextResponse.json({
        prediction: fb.prediction,
        confidence: fb.confidence,
        source: "fallback",
      });
    }

    return NextResponse.json({
      prediction: result.prediction,
      confidence: result.confidence,
      source: result.source ?? "model",
    });
  } catch (err) {
    console.error("[/api/predict-asthma] error:", err);
    return NextResponse.json(
      { error: "Error processing asthma prediction" },
      { status: 500 }
    );
  }
}

function fallbackAsthma(d: Record<string, number>) {
  let score = 0;
  score += (d.wheezing || 0) * 0.25;
  score += (d.shortnessOfBreath || 0) * 0.2;
  score += (d.chestTightness || 0) * 0.15;
  score += (d.familyHistory || 0) * 0.15;
  score += (d.allergyHistory || 0) * 0.1;
  score += (d.smoking > 0 ? 1 : 0) * 0.1;
  score += (d.lungFunctionFeV1 < 70 ? 1 : 0) * 0.2;
  const prediction = score >= 0.4 ? 1 : 0;
  const confidence = Math.round(Math.min(95, 55 + score * 60) * 100) / 100;
  return { prediction, confidence };
}
