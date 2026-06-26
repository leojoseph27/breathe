"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, X, FileAudio, Clock } from "lucide-react";

interface AudioPreviewProps {
  file: File;
  onRemove: () => void;
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * Safely close an AudioContext exactly once. Guards against the
 * "Cannot close a closed AudioContext" InvalidStateError by checking
 * the context state before closing.
 */
function safeCloseAudioContext(ctx: AudioContext | null) {
  if (ctx && ctx.state !== "closed") {
    // close() returns a Promise; catch it so a rejected close never throws
    // an unhandled rejection.
    ctx.close().catch(() => {});
  }
}

export function AudioPreview({ file, onRemove }: AudioPreviewProps) {
  // Derive the object URL from the file prop with useMemo — this avoids
  // setState-in-effect and guarantees the URL is never empty during render.
  // The URL is revoked when the file changes or the component unmounts.
  const url = useMemo(() => URL.createObjectURL(file), [file]);
  useEffect(() => {
    return () => URL.revokeObjectURL(url);
  }, [url]);

  const [duration, setDuration] = useState<string>("");
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  // If the browser has no AudioContext, skip the loading state entirely.
  const [waveformReady, setWaveformReady] = useState(() => {
    if (typeof window === "undefined") return false;
    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    return !AC; // true when unavailable → no "loading" spinner
  });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Reset derived state when the file prop changes — using the React-
  // recommended "adjust state during render" pattern (not an effect) so
  // there's no cascading render from setState-in-effect.
  const [prevFile, setPrevFile] = useState(file);
  if (prevFile !== file) {
    setPrevFile(file);
    const AC =
      typeof window !== "undefined"
        ? window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext
        : undefined;
    setWaveformReady(!AC);
    setDuration("");
    setProgress(0);
  }

  // Decode audio data and draw the waveform. The AudioContext is created per
  // effect run and closed exactly once in cleanup (Strict Mode safe).
  useEffect(() => {
    let cancelled = false;
    let ctx: AudioContext | null = null;

    const AC =
      typeof window !== "undefined"
        ? window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext
        : undefined;
    if (!AC) {
      // Already marked ready during render — nothing to do.
      return;
    }

    // Create the context lazily inside the effect so cleanup always owns it.
    ctx = new AC();
    const reader = new FileReader();

    reader.onload = async () => {
      if (cancelled) return;
      try {
        const buf = await ctx!.decodeAudioData(reader.result as ArrayBuffer);
        if (cancelled) return;
        setDuration(formatDuration(buf.duration));
        drawWaveform(canvasRef.current, buf);
        setWaveformReady(true);
      } catch {
        if (!cancelled) setWaveformReady(true);
      }
      // Close after decoding — guarded so it only closes once.
      if (!cancelled) safeCloseAudioContext(ctx);
    };

    reader.onerror = () => {
      if (!cancelled) setWaveformReady(true);
    };

    reader.readAsArrayBuffer(file);

    return () => {
      cancelled = true;
      // If decoding hasn't finished yet, the context is still open — close it.
      safeCloseAudioContext(ctx);
    };
  }, [file]);

  function togglePlay() {
    const a = audioRef.current;
    if (!a) return;
    if (playing) a.pause();
    else a.play();
  }

  return (
    <div className="bd-card bd-card-hover p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-white"
            style={{ background: "linear-gradient(135deg, #0ea5e9, #06b6d4)" }}
          >
            <FileAudio className="h-4 w-4" />
          </span>
          <div className="min-w-0">
            <div className="truncate text-sm font-medium text-slate-900">
              {file.name}
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-500">
              {duration && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" /> {duration}
                </span>
              )}
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove audio file"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-red-50 hover:text-red-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400/40"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Waveform + progress */}
      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          aria-label={playing ? "Pause" : "Play"}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-white shadow-sm transition-transform hover:scale-105"
          style={{ background: "linear-gradient(135deg, #0ea5e9, #06b6d4)" }}
        >
          {playing ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="ml-0.5 h-4 w-4" />
          )}
        </button>
        <div className="relative h-10 flex-1 overflow-hidden rounded-lg bg-slate-100/70">
          <canvas
            ref={canvasRef}
            width={600}
            height={40}
            className="h-full w-full"
            aria-label="Audio waveform visualization"
          />
          {!waveformReady && (
            <div className="absolute inset-0 flex items-center justify-center text-xs text-slate-400">
              Loading waveform…
            </div>
          )}
          {/* progress overlay */}
          <div
            className="pointer-events-none absolute inset-y-0 left-0 bg-sky-500/10"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </div>

      {/* Only render the <audio> element once a valid URL exists — never
          render it with an empty src (avoids the "empty src" warning). */}
      {url && (
        <audio
          ref={audioRef}
          src={url}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => {
            setPlaying(false);
            setProgress(0);
          }}
          onTimeUpdate={(e) => {
            const a = e.currentTarget;
            if (a.duration) setProgress(a.currentTime / a.duration);
          }}
          className="hidden"
        />
      )}
    </div>
  );
}

function drawWaveform(canvas: HTMLCanvasElement | null, buf: AudioBuffer) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const data = buf.getChannelData(0);
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const bars = 64;
  const barW = W / bars;
  const step = Math.floor(data.length / bars);
  const gap = 2;
  const mid = H / 2;

  ctx.fillStyle = "#0ea5e9";

  for (let i = 0; i < bars; i++) {
    let peak = 0;
    const start = i * step;
    for (let j = 0; j < step; j++) {
      const v = Math.abs(data[start + j] || 0);
      if (v > peak) peak = v;
    }
    const barH = Math.max(2, peak * H * 0.82);
    const x = i * barW;
    const y = mid - barH / 2;
    ctx.fillRect(x + gap / 2, y, barW - gap, barH);
  }
}
