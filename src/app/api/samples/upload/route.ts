import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import {
  ensureStorageDirs,
  isValidCategory,
  isValidExtension,
  categoryDir,
  uniqueFilename,
  getAudioDuration,
  ALLOWED_MIME,
} from "@/lib/sample-storage";
import { writeFile, stat } from "fs/promises";
import { join } from "path";

export async function POST(request: NextRequest) {
  try {
    await ensureStorageDirs();

    const formData = await request.formData();
    const file = formData.get("file");
    const category = formData.get("category");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "No audio file provided." },
        { status: 400 }
      );
    }
    if (typeof category !== "string" || !isValidCategory(category)) {
      return NextResponse.json(
        { error: "Invalid or missing category." },
        { status: 400 }
      );
    }

    // Validate file type.
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!isValidExtension(ext)) {
      return NextResponse.json(
        { error: "Invalid file type. Please upload a WAV, MP3, M4A, or FLAC file." },
        { status: 400 }
      );
    }
    // Also check MIME type when available (skip for empty/octet-stream).
    if (file.type && file.type !== "application/octet-stream") {
      if (!ALLOWED_MIME.includes(file.type)) {
        return NextResponse.json(
          { error: `Unsupported MIME type: ${file.type}` },
          { status: 400 }
        );
      }
    }

    // Validate file size (max 16 MB).
    if (file.size > 16 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File too large. Maximum size is 16 MB." },
        { status: 400 }
      );
    }

    // Sanitize base name (keep alphanumeric, underscore, dash).
    const rawBase = file.name.replace(/\.[^.]+$/, "");
    const safeBase = rawBase.replace(/[^a-zA-Z0-9_-]/g, "_") || "sample";
    const { filename, filepath } = await uniqueFilename(category, safeBase, ext);

    // Write file to disk.
    const bytes = await file.arrayBuffer();
    await writeFile(filepath, Buffer.from(bytes));

    const fileStat = await stat(filepath);
    const duration = await getAudioDuration(filepath, ext);

    const record = await db.sampleAudio.create({
      data: {
        category,
        filename,
        extension: ext,
        filepath,
        filesize: fileStat.size,
        duration,
      },
    });

    return NextResponse.json({
      success: true,
      sample: {
        id: record.id,
        category: record.category,
        filename: record.filename,
        extension: record.extension,
        filesize: record.filesize,
        duration: record.duration,
        createdAt: record.createdAt,
      },
    });
  } catch (err) {
    console.error("[/api/samples/upload] error:", err);
    return NextResponse.json(
      { error: "Failed to upload sample." },
      { status: 500 }
    );
  }
}
