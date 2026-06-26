import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { existsSync } from "fs";
import { readFile } from "fs/promises";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const sample = await db.sampleAudio.findUnique({ where: { id } });
    if (!sample) {
      return NextResponse.json({ error: "Sample not found." }, { status: 404 });
    }
    if (!existsSync(sample.filepath)) {
      return NextResponse.json(
        { error: "Audio file missing on disk." },
        { status: 404 }
      );
    }
    const buf = await readFile(sample.filepath);
    const mime =
      sample.extension === "wav"
        ? "audio/wav"
        : sample.extension === "mp3"
        ? "audio/mpeg"
        : sample.extension === "m4a"
        ? "audio/mp4"
        : sample.extension === "flac"
        ? "audio/flac"
        : "application/octet-stream";
    return new NextResponse(buf, {
      status: 200,
      headers: {
        "Content-Type": mime,
        "Content-Length": String(buf.length),
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch (err) {
    console.error("[/api/samples/[id]/file] error:", err);
    return NextResponse.json({ error: "Failed to serve file." }, { status: 500 });
  }
}
