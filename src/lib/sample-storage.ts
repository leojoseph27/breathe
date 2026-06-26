import { db } from "@/lib/db";
import { readdir, stat, rename, unlink, mkdir } from "fs/promises";
import { existsSync } from "fs";
import { join } from "path";

export const CATEGORIES = [
  "Bronchial",
  "Asthma",
  "COPD",
  "Healthy",
  "Pneumonia",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "flac"] as const;
export const ALLOWED_MIME = [
  "audio/wav",
  "audio/x-wav",
  "audio/wave",
  "audio/mpeg",
  "audio/mp3",
  "audio/mp4",
  "audio/x-m4a",
  "audio/m4a",
  "audio/flac",
  "audio/x-flac",
];

export const STORAGE_ROOT = join(process.cwd(), "storage", "sample-audio");

export function categoryDir(category: string): string {
  return join(STORAGE_ROOT, category);
}

export function isValidCategory(category: string): category is Category {
  return (CATEGORIES as readonly string[]).includes(category);
}

export function isValidExtension(ext: string): boolean {
  return (AUDIO_EXTENSIONS as readonly string[]).includes(
    ext.toLowerCase().replace(/^\./, "")
  );
}

/**
 * Ensure storage directories exist for all categories.
 */
export async function ensureStorageDirs() {
  for (const cat of CATEGORIES) {
    const dir = categoryDir(cat);
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }
  }
}

/**
 * Compute audio duration (seconds) by decoding with ffmpeg-style approach.
 * Falls back to 0 if decoding fails. Uses a lightweight WAV header parse first,
 * then the Web Audio API is not available server-side, so we estimate from
 * WAV header bytes for WAV files, otherwise 0.
 */
export async function getAudioDuration(
  filepath: string,
  ext: string
): Promise<number> {
  if (ext.toLowerCase() === "wav") {
    try {
      const { readFile } = await import("fs/promises");
      const buf = await readFile(filepath);
      // WAV header: bytes 24-27 = sample rate (little-endian uint32)
      // bytes 34-35 = bits per sample, bytes 40-43 = data chunk size
      if (buf.length > 44) {
        const sampleRate = buf.readUInt32LE(24);
        const bitsPerSample = buf.readUInt16LE(34);
        const dataChunkSize = buf.readUInt32LE(40);
        const numChannels = buf.readUInt16LE(22);
        if (sampleRate > 0 && bitsPerSample > 0 && numChannels > 0) {
          const bytesPerSample = bitsPerSample / 8;
          const duration =
            dataChunkSize / (sampleRate * numChannels * bytesPerSample);
          if (duration > 0 && duration < 3600) {
            return Math.round(duration * 100) / 100;
          }
        }
      }
    } catch {
      // ignore
    }
  }
  return 0;
}

/**
 * Seed existing files from public/sample-audio into the database on first run.
 * This migrates the placeholder demo files into the persistent storage system.
 * Only runs if the database has no SampleAudio records AND there are files in
 * public/sample-audio that haven't been migrated yet.
 */
export async function seedFromPublicIfEmpty() {
  await ensureStorageDirs();

  const count = await db.sampleAudio.count();
  if (count > 0) return;

  const publicRoot = join(process.cwd(), "public", "sample-audio");
  if (!existsSync(publicRoot)) return;

  const { copyFile } = await import("fs/promises");

  for (const cat of CATEGORIES) {
    const srcDir = join(publicRoot, cat);
    if (!existsSync(srcDir)) continue;
    const files = await readdir(srcDir).catch(() => []);
    for (const f of files) {
      const ext = f.split(".").pop()?.toLowerCase() ?? "";
      if (!isValidExtension(ext)) continue;
      const srcPath = join(srcDir, f);
      const filename = f.replace(/\.[^.]+$/, "");
      const dstPath = join(categoryDir(cat), f);
      // Copy into persistent storage
      if (!existsSync(dstPath)) {
        await copyFile(srcPath, dstPath);
      }
      const fileStat = await stat(dstPath);
      const duration = await getAudioDuration(dstPath, ext);
      await db.sampleAudio.create({
        data: {
          category: cat,
          filename,
          extension: ext,
          filepath: dstPath,
          filesize: fileStat.size,
          duration,
        },
      });
    }
  }
}

/**
 * Generate a unique filename in a category directory to avoid collisions.
 */
export async function uniqueFilename(
  category: string,
  baseName: string,
  ext: string
): Promise<{ filename: string; filepath: string }> {
  const dir = categoryDir(category);
  let candidate = baseName;
  let filepath = join(dir, `${candidate}.${ext}`);
  let counter = 1;
  while (existsSync(filepath)) {
    candidate = `${baseName}_${counter}`;
    filepath = join(dir, `${candidate}.${ext}`);
    counter++;
  }
  return { filename: candidate, filepath };
}

export { readdir, stat, rename, unlink, existsSync };
