import { NextResponse } from "next/server";
import { db } from "@/lib/db";
import { seedFromPublicIfEmpty, CATEGORIES } from "@/lib/sample-storage";

export async function GET() {
  // Ensure DB is seeded (no-op if already populated).
  await seedFromPublicIfEmpty();

  const samples = await db.sampleAudio.findMany({
    orderBy: [{ category: "asc" }, { filename: "asc" }],
  });

  // Group by category for the frontend.
  const grouped: Record<string, typeof samples> = {};
  for (const cat of CATEGORIES) grouped[cat] = [];
  for (const s of samples) {
    if (!grouped[s.category]) grouped[s.category] = [];
    grouped[s.category].push(s);
  }

  return NextResponse.json(grouped);
}
