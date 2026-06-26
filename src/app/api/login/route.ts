import { NextRequest, NextResponse } from "next/server";

// Hardcoded credentials — mirrors the original breathe Flask app exactly.
const VALID_EMAIL = "user@gmail.com";
const VALID_PASSWORD = "123456";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => null);
    const email = typeof body?.email === "string" ? body.email : "";
    const password = typeof body?.password === "string" ? body.password : "";

    if (email === VALID_EMAIL && password === VALID_PASSWORD) {
      return NextResponse.json({ success: true });
    }
    return NextResponse.json(
      { success: false, error: "Invalid email or password" },
      { status: 401 }
    );
  } catch {
    return NextResponse.json(
      { success: false, error: "Invalid request body" },
      { status: 400 }
    );
  }
}
