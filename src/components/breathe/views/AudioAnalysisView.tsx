"use client";

import { useEffect, useState, useRef } from "react";
import { useBreatheStore } from "@/lib/breathe-store";
import { useToast } from "@/hooks/use-toast";
import { AudioPreview } from "@/components/breathe/AudioPreview";
import { InfoTooltip } from "@/components/breathe/InfoTooltip";
import {
  Loader2,
  Upload,
  AudioLines,
  Info,
  AlertCircle,
  Play,
  Pause,
  FileAudio,
  Inbox,
  CheckCircle2,
  Activity,
} from "lucide-react";

type Mode = "upload" | "sample";

interface Sample {
  id: string;
  category: string;
  filename: string;
  extension: string;
  filesize: number;
  duration: number;
}

const CATEGORIES = [
  { value: "Healthy", label: "Healthy", color: "#10b981" },
  { value: "Asthma", label: "Asthma", color: "#0ea5e9" },
  { value: "COPD", label: "COPD", color: "#6366f1" },
  { value: "Bronchial", label: "Bronchial", color: "#06b6d4" },
  { value: "Pneumonia", label: "Pneumonia", color: "#f43f5e" },
] as const;

function formatDuration(secs: number): string {
  if (!secs) return "--:--";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioAnalysisView() {
  const { toast } = useToast();
  const setAudioAnalysis = useBreatheStore((s) => s.setAudioAnalysis);
  const stored = useBreatheStore((s) => s.audioAnalysis);

  const [mode, setMode] = useState<Mode>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<{
    prediction: string;
    confidence?: number;
    source?: string;
  } | null>(stored?.prediction ? stored : null);
  const [error, setError] = useState("");

  const [samples, setSamples] = useState<Record<string, Sample[]>>({});
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [selectedSampleId, setSelectedSampleId] = useState<string>("");
  const [loadingSample, setLoadingSample] = useState(false);

  const [previewingId, setPreviewingId] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    fetch("/api/samples")
      .then((r) => r.json())
      .then((data) => {
        setSamples(data);
        const firstWithFiles = CATEGORIES.find(
          (c) => data[c.value]?.length > 0
        );
        if (firstWithFiles) setSelectedCategory(firstWithFiles.value);
      })
      .catch(() => setSamples({}));
  }, []);

  useEffect(() => {
    return () => {
      // Stop playback and release the audio element on unmount.
      if (previewAudioRef.current) {
        previewAudioRef.current.pause();
        previewAudioRef.current.onended = null;
        previewAudioRef.current.onerror = null;
        previewAudioRef.current.src = "";
        previewAudioRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!selectedSampleId) {
      setFile(null);
      return;
    }
    setLoadingSample(true);
    setError("");
    setResult(null);
    fetch(`/api/samples/${selectedSampleId}/file`)
      .then((r) => {
        if (!r.ok) throw new Error("fetch failed");
        return r.blob();
      })
      .then((blob) => {
        let found: Sample | undefined;
        for (const cat of Object.keys(samples)) {
          found = samples[cat]?.find((s) => s.id === selectedSampleId);
          if (found) break;
        }
        if (found) {
          const f = new File([blob], `${found.filename}.${found.extension}`, {
            type: "audio/wav",
          });
          setFile(f);
        }
      })
      .catch(() => setError("Could not load the selected sample."))
      .finally(() => setLoadingSample(false));
  }, [selectedSampleId, samples]);

  function togglePreview(sample: Sample) {
    if (previewingId === sample.id) {
      previewAudioRef.current?.pause();
      setPreviewingId(null);
      return;
    }
    // Clean up the previous audio element before creating a new one.
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current.onended = null;
      previewAudioRef.current.onerror = null;
    }
    const audio = new Audio(`/api/samples/${sample.id}/file`);
    audio.onended = () => setPreviewingId(null);
    audio.onerror = () => setPreviewingId(null);
    previewAudioRef.current = audio;
    audio.play();
    setPreviewingId(sample.id);
  }

  function handleUploadChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] ?? null;
    setError("");
    setResult(null);
    setFile(f);
    setSelectedSampleId("");
  }

  function handleModeChange(newMode: Mode) {
    setMode(newMode);
    setError("");
    setResult(null);
    setFile(null);
    setSelectedSampleId("");
    previewAudioRef.current?.pause();
    setPreviewingId(null);
    const input = document.getElementById("audioFile") as HTMLInputElement;
    if (input) input.value = "";
  }

  function handleRemove() {
    setFile(null);
    setResult(null);
    setError("");
    setSelectedSampleId("");
    previewAudioRef.current?.pause();
    setPreviewingId(null);
    const input = document.getElementById("audioFile") as HTMLInputElement;
    if (input) input.value = "";
  }

  async function handleAnalyze() {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }
    setAnalyzing(true);
    setError("");
    toast({
      title: "Analysis started",
      description: "The respiratory model is analyzing your audio…",
    });
    try {
      const formData = new FormData();
      formData.append("audio_file", file);

      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        setError(data.error || "Unable to analyze the selected audio.");
        toast({
          title: "Analysis failed",
          description: data.error || "Could not analyze the audio.",
          variant: "destructive",
        });
        return;
      }

      const r = {
        prediction: data.prediction,
        confidence: data.confidence,
        source: data.source,
      };
      setResult(r);
      setAudioAnalysis({ ...r, filename: file.name });
      toast({
        title: "Analysis completed",
        description: `Prediction: ${data.prediction}${
          data.confidence
            ? ` (${Math.round(data.confidence * 100)}% confidence)`
            : ""
        }`,
      });
    } catch {
      setError("Unable to analyze the selected audio.");
      toast({
        title: "Analysis failed",
        description: "A network error occurred.",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  }

  const confidencePct =
    typeof result?.confidence === "number"
      ? Math.round(result.confidence * 100) + "%"
      : null;

  const categoryFiles = selectedCategory
    ? samples[selectedCategory] ?? []
    : [];

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          Respiratory Disease Prediction
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload a recording or choose a sample to predict respiratory
          conditions using a CNN model trained on 5 disease classes.
        </p>
      </div>

      {/* Demo banner */}
      <div className="flex items-start gap-3 rounded-xl border border-sky-100 bg-sky-50/70 p-4">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-sky-600" />
        <div className="text-[13px] leading-relaxed text-slate-600">
          <span className="font-medium text-slate-900">
            No respiratory recording available?
          </span>{" "}
          Use one of our demo recordings to explore the prediction system —
          switch to <span className="font-medium">Use Demo Library</span> below,
          or manage samples in the Demo Library tab.
        </div>
      </div>

      {/* Audio source selector */}
      <div>
        <div className="mb-2.5 flex items-center gap-1.5">
          <span className="text-[13px] font-medium text-slate-700">
            Choose Audio Source
          </span>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => handleModeChange("upload")}
            className={`flex items-center gap-3 rounded-xl border p-4 text-left transition-all ${
              mode === "upload"
                ? "border-sky-500 bg-sky-50/50 ring-1 ring-sky-500/20"
                : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50"
            }`}
            aria-pressed={mode === "upload"}
          >
            <span
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                mode === "upload"
                  ? "bg-sky-600 text-white"
                  : "bg-slate-100 text-slate-500"
              }`}
            >
              <Upload className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-900">
                Upload My Own Recording
              </div>
              <div className="text-xs text-slate-500">
                WAV, MP3, M4A, or FLAC
              </div>
            </div>
          </button>

          <button
            type="button"
            onClick={() => handleModeChange("sample")}
            className={`flex items-center gap-3 rounded-xl border p-4 text-left transition-all ${
              mode === "sample"
                ? "border-sky-500 bg-sky-50/50 ring-1 ring-sky-500/20"
                : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50"
            }`}
            aria-pressed={mode === "sample"}
          >
            <span
              className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                mode === "sample"
                  ? "bg-sky-600 text-white"
                  : "bg-slate-100 text-slate-500"
              }`}
            >
              <AudioLines className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-900">
                Use Demo Library
              </div>
              <div className="text-xs text-slate-500">
                Pre-loaded demo recordings
              </div>
            </div>
          </button>
        </div>
      </div>

      {/* Upload mode */}
      {mode === "upload" && (
        <div className="bd-section p-5">
          <div className="flex flex-col items-center gap-3">
            <button
              type="button"
              onClick={() => document.getElementById("audioFile")?.click()}
              className="bd-btn bd-btn-primary bd-btn-lg"
            >
              <Upload className="h-4 w-4" /> Choose Audio File
            </button>
            <input
              id="audioFile"
              type="file"
              accept=".wav,.mp3,.m4a,.flac,audio/*"
              onChange={handleUploadChange}
              className="hidden"
            />
            <p className="text-xs text-slate-400">
              Supported formats: WAV, MP3, M4A, FLAC · Max 16 MB
            </p>
          </div>
        </div>
      )}

      {/* Sample mode */}
      {mode === "sample" && (
        <div className="bd-section space-y-4 p-5">
          {/* Category chips */}
          <div>
            <div className="mb-2 flex items-center gap-1.5">
              <span className="text-[13px] font-medium text-slate-700">
                Category
              </span>
              <InfoTooltip label="Category">
                <div className="mb-1 font-semibold text-sky-300">
                  Sample Category
                </div>
                <div>
                  Select the type of respiratory condition you want to analyze.
                  Each category contains pre-loaded demo recordings.
                </div>
              </InfoTooltip>
            </div>
            <div className="flex flex-wrap gap-2">
              {CATEGORIES.map((c) => {
                const active = selectedCategory === c.value;
                const count = samples[c.value]?.length ?? 0;
                return (
                  <button
                    key={c.value}
                    type="button"
                    onClick={() => {
                      setSelectedCategory(c.value);
                      setSelectedSampleId("");
                      setFile(null);
                    }}
                    className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-[13px] font-medium transition-all ${
                      active
                        ? "border-sky-500 bg-sky-50 text-sky-700"
                        : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                    }`}
                  >
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ background: c.color }}
                    />
                    {c.label}
                    <span className="text-xs text-slate-400">{count}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Available samples */}
          {selectedCategory && (
            <div>
              <div className="mb-2 text-[13px] font-medium text-slate-700">
                Available Samples
              </div>
              {categoryFiles.length === 0 ? (
                <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-slate-200 py-10 text-center">
                  <Inbox className="h-7 w-7 text-slate-300" />
                  <p className="text-sm text-slate-500">
                    No samples in this category yet.
                  </p>
                  <p className="text-xs text-slate-400">
                    Upload some in the Demo Library tab.
                  </p>
                </div>
              ) : (
                <ul className="bd-scroll max-h-72 space-y-1.5 overflow-y-auto pr-1">
                  {categoryFiles.map((s) => {
                    const isSelected = selectedSampleId === s.id;
                    const isPreviewing = previewingId === s.id;
                    return (
                      <li key={s.id}>
                        <button
                          type="button"
                          onClick={() => setSelectedSampleId(s.id)}
                          className={`flex w-full items-center gap-3 rounded-lg border p-2.5 text-left transition-all ${
                            isSelected
                              ? "border-sky-500 bg-sky-50/60"
                              : "border-slate-200 bg-white hover:border-slate-300"
                          }`}
                        >
                          <span
                            role="button"
                            tabIndex={0}
                            onClick={(e) => {
                              e.stopPropagation();
                              togglePreview(s);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.stopPropagation();
                                e.preventDefault();
                                togglePreview(s);
                              }
                            }}
                            aria-label={
                              isPreviewing
                                ? `Pause ${s.filename}`
                                : `Play ${s.filename}`
                            }
                            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-white transition-transform hover:scale-105"
                            style={{
                              background:
                                "linear-gradient(135deg, #0ea5e9, #06b6d4)",
                            }}
                          >
                            {isPreviewing ? (
                              <Pause className="h-3.5 w-3.5" />
                            ) : (
                              <Play className="ml-0.5 h-3.5 w-3.5" />
                            )}
                          </span>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5">
                              <FileAudio className="h-3.5 w-3.5 shrink-0 text-sky-500" />
                              <span className="truncate text-[13px] font-medium text-slate-900">
                                {s.filename}.{s.extension}
                              </span>
                            </div>
                            <div className="mt-0.5 text-xs text-slate-500">
                              {formatDuration(s.duration)} ·{" "}
                              {s.filesize < 1024
                                ? `${s.filesize} B`
                                : `${(s.filesize / 1024).toFixed(1)} KB`}
                            </div>
                          </div>
                          {isSelected && (
                            <CheckCircle2 className="h-4 w-4 shrink-0 text-sky-600" />
                          )}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}

          {loadingSample && (
            <div className="flex items-center justify-center gap-2 py-1 text-sm text-slate-500">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading sample…
            </div>
          )}
        </div>
      )}

      {/* Audio preview */}
      {file && <AudioPreview file={file} onRemove={handleRemove} />}

      {/* Analyze button */}
      {file && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || analyzing}
            className="bd-btn bd-btn-primary bd-btn-lg w-full max-w-xs"
          >
            {analyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Analyzing…
              </>
            ) : (
              <>
                <Activity className="h-4 w-4" />
                {mode === "sample" ? "Analyze Sample" : "Analyze Audio"}
              </>
            )}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4"
          role="alert"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div className="text-[13px] text-slate-700">
            <p className="font-medium text-slate-900">
              Unable to analyze the selected audio.
            </p>
            <p className="mt-1">Please ensure:</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-slate-600">
              <li>WAV, MP3, M4A, or FLAC format</li>
              <li>Recording is clear and audible</li>
              <li>Duration between 5–30 seconds</li>
            </ul>
            {error !== "Unable to analyze the selected audio." &&
              error !== "Please select an audio file first." &&
              error !== "Could not load the selected sample." && (
                <p className="mt-2 text-xs text-amber-700">Detail: {error}</p>
              )}
          </div>
        </div>
      )}

      {/* Result */}
      {result?.prediction && (
        <div className="bd-card bd-scale-in p-6 text-center">
          <div className="mb-1 flex items-center justify-center gap-2 text-xs font-medium uppercase tracking-wide text-slate-400">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            Prediction Result
          </div>
          <div className="mt-2 text-3xl font-bold text-slate-900">
            {result.prediction}
          </div>
          {confidencePct && (
            <div className="mt-2 text-sm text-slate-500">
              Confidence{" "}
              <span className="font-semibold text-slate-700">
                {confidencePct}
              </span>
            </div>
          )}
          {result.source && (
            <div className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  result.source === "model" ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              {result.source === "model" ? "CNN model" : "heuristic fallback"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
