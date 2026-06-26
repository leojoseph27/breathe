import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { unlink, rename } from "fs/promises";
import { existsSync } from "fs";

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
    const { readFile } = await import("fs/promises");
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
    console.error("[/api/samples/[id]/file GET] error:", err);
    return NextResponse.json({ error: "Failed to serve file." }, { status: 500 });
  }
}

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const sample = await db.sampleAudio.findUnique({ where: { id } });
    if (!sample) {
      return NextResponse.json({ error: "Sample not found." }, { status: 404 });
    }
    // Remove file from disk.
    if (existsSync(sample.filepath)) {
      await unlink(sample.filepath);
    }
    // Remove DB record.
    await db.sampleAudio.delete({ where: { id } });
    return NextResponse.json({ success: true });
  } catch (err) {
    console.error("[/api/samples/[id] DELETE] error:", err);
    return NextResponse.json({ error: "Failed to delete sample." }, { status: 500 });
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const body = await req.json().catch(() => null);
    const newName = typeof body?.filename === "string" ? body.filename.trim() : "";
    if (!newName) {
      return NextResponse.json(
        { error: "New filename is required." },
        { status: 400 }
      );
    }
    // Sanitize new name (no extension, no path separators).
    const safeName = newName.replace(/[^a-zA-Z0-9_-]/g, "_");
    if (!safeName) {
      return NextResponse.json(
        { error: "Invalid filename." },
        { status: 400 }
      );
    }

    const sample = await db.sampleAudio.findUnique({ where: { id } });
    if (!sample) {
      return NextResponse.json({ error: "Sample not found." }, { status: 404 });
    }

    if (safeName === sample.filename) {
      return NextResponse.json({ success: true, sample });
    }

    // Compute new path and ensure it doesn't collide.
    const { dirname, join } = await import("path");
    const dir = dirname(sample.filepath);
    const newPath = join(dir, `${safeName}.${sample.extension}`);
    if (existsSync(newPath)) {
      return NextResponse.json(
        { error: "A sample with that name already exists in this category." },
        { status: 409 }
      );
    }

    // Rename file on disk.
    if (existsSync(sample.filepath)) {
      await rename(sample.filepath, newPath);
    }

    const updated = await db.sampleAudio.update({
      where: { id },
      data: { filename: safeName, filepath: newPath },
    });

    return NextResponse.json({ success: true, sample: updated });
  } catch (err) {
    console.error("[/api/samples/[id] PATCH] error:", err);
    return NextResponse.json({ error: "Failed to rename sample." }, { status: 500 });
  }
}
