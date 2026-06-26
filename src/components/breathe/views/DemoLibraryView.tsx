"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Upload,
  Play,
  Pause,
  Trash2,
  Pencil,
  FileAudio,
  Loader2,
  Inbox,
  Clock,
  HardDrive,
  Calendar,
  Library,
} from "lucide-react";

interface Sample {
  id: string;
  category: string;
  filename: string;
  extension: string;
  filepath: string;
  filesize: number;
  duration: number;
  createdAt: string;
  updatedAt: string;
}

const CATEGORIES = [
  { name: "Bronchial", color: "#06b6d4" },
  { name: "Asthma", color: "#0ea5e9" },
  { name: "COPD", color: "#6366f1" },
  { name: "Healthy", color: "#10b981" },
  { name: "Pneumonia", color: "#f43f5e" },
] as const;

function formatDuration(secs: number): string {
  if (!secs) return "--:--";
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return "--";
  }
}

export function DemoLibraryView() {
  const { toast } = useToast();
  const [samples, setSamples] = useState<Record<string, Sample[]>>({});
  const [loading, setLoading] = useState(true);

  const [uploadingCategory, setUploadingCategory] = useState<string | null>(
    null
  );

  const [deleteTarget, setDeleteTarget] = useState<Sample | null>(null);
  const [deleting, setDeleting] = useState(false);

  const [renameTarget, setRenameTarget] = useState<Sample | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);

  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const fetchSamples = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      setSamples(data);
    } catch {
      toast({
        title: "Failed to load library",
        description: "Could not fetch demo audio samples.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchSamples();
  }, [fetchSamples]);

  useEffect(() => {
    return () => {
      // Stop playback and fully release the audio element on unmount.
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.onended = null;
        audioRef.current.onerror = null;
        audioRef.current.src = "";
        audioRef.current = null;
      }
    };
  }, []);

  function togglePlay(sample: Sample) {
    if (playingId === sample.id) {
      audioRef.current?.pause();
      setPlayingId(null);
      return;
    }
    // Clean up the previous audio element before creating a new one.
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.onended = null;
      audioRef.current.onerror = null;
    }
    const audio = new Audio(`/api/samples/${sample.id}/file`);
    audio.onended = () => setPlayingId(null);
    audio.onerror = () => {
      setPlayingId(null);
      toast({
        title: "Playback failed",
        description: "Could not play this audio sample.",
        variant: "destructive",
      });
    };
    audioRef.current = audio;
    audio.play();
    setPlayingId(sample.id);
  }

  async function handleUpload(category: string, file: File) {
    setUploadingCategory(category);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("category", category);

      const res = await fetch("/api/samples/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        toast({
          title: "Upload failed",
          description: data.error || "Could not upload the audio file.",
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Upload successful",
        description: `${data.sample.filename}.${data.sample.extension} added to ${category}.`,
      });
      await fetchSamples();
    } catch {
      toast({
        title: "Upload failed",
        description: "A network error occurred during upload.",
        variant: "destructive",
      });
    } finally {
      setUploadingCategory(null);
    }
  }

  async function handleDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      if (playingId === deleteTarget.id) {
        audioRef.current?.pause();
        setPlayingId(null);
      }
      const res = await fetch(`/api/samples/${deleteTarget.id}`, {
        method: "DELETE",
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        toast({
          title: "Delete failed",
          description: data.error || "Could not delete the sample.",
          variant: "destructive",
        });
        return;
      }
      toast({
        title: "Deleted successfully",
        description: `${deleteTarget.filename}.${deleteTarget.extension} has been removed.`,
      });
      await fetchSamples();
    } catch {
      toast({
        title: "Delete failed",
        description: "A network error occurred.",
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
      setDeleteTarget(null);
    }
  }

  function openRename(sample: Sample) {
    setRenameTarget(sample);
    setRenameValue(sample.filename);
  }

  async function handleRename() {
    if (!renameTarget) return;
    setRenaming(true);
    try {
      const res = await fetch(`/api/samples/${renameTarget.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: renameValue }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        toast({
          title: "Rename failed",
          description: data.error || "Could not rename the sample.",
          variant: "destructive",
        });
        return;
      }
      toast({
        title: "Renamed successfully",
        description: `Now known as ${renameValue}.${renameTarget.extension}.`,
      });
      setRenameTarget(null);
      await fetchSamples();
    } catch {
      toast({
        title: "Rename failed",
        description: "A network error occurred.",
        variant: "destructive",
      });
    } finally {
      setRenaming(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          Demo Audio Library
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Manage sample respiratory recordings. Uploaded samples are available
          immediately in the Audio Analysis page for all users.
        </p>
      </div>

      {loading ? (
        <div className="flex items-center justify-center gap-2 py-16 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin text-sky-600" /> Loading
          library…
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {CATEGORIES.map((cat) => {
            const catSamples = samples[cat.name] ?? [];
            const isUploading = uploadingCategory === cat.name;
            return (
              <CategoryCard
                key={cat.name}
                name={cat.name}
                color={cat.color}
                samples={catSamples}
                playingId={playingId}
                isUploading={isUploading}
                onUpload={(file) => handleUpload(cat.name, file)}
                onPlay={togglePlay}
                onRename={openRename}
                onDelete={setDeleteTarget}
              />
            );
          })}
        </div>
      )}

      {/* Delete confirmation dialog */}
      <AlertDialog
        open={!!deleteTarget}
        onOpenChange={(o) => !o && setDeleteTarget(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete sample?</AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget && (
                <>
                  <span className="font-medium">
                    {deleteTarget.filename}.{deleteTarget.extension}
                  </span>{" "}
                  will be permanently removed from the{" "}
                  {deleteTarget.category} library.
                  <br />
                  <br />
                  This action cannot be undone.
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleting}
              className="bg-red-600 hover:bg-red-700 focus:ring-red-600"
            >
              {deleting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Deleting…
                </>
              ) : (
                "Delete"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Rename dialog */}
      <Dialog
        open={!!renameTarget}
        onOpenChange={(o) => !o && setRenameTarget(null)}
      >
        <DialogContent aria-describedby={undefined}>
          <DialogHeader>
            <DialogTitle>Rename sample</DialogTitle>
            <DialogDescription className="sr-only">
              Enter a new name for this audio sample.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2 py-2">
            <label
              htmlFor="rename-input"
              className="text-[13px] font-medium text-slate-700"
            >
              New name
            </label>
            <div className="flex items-center gap-2">
              <input
                id="rename-input"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && renameValue.trim()) {
                    handleRename();
                  }
                }}
                className="bd-input flex-1"
                autoFocus
              />
              {renameTarget && (
                <span className="text-sm text-slate-500">
                  .{renameTarget.extension}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400">
              Only letters, numbers, hyphens, and underscores are allowed.
            </p>
          </div>
          <DialogFooter>
            <button
              type="button"
              onClick={() => setRenameTarget(null)}
              disabled={renaming}
              className="bd-btn bd-btn-secondary"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleRename}
              disabled={renaming || !renameValue.trim()}
              className="bd-btn bd-btn-primary"
            >
              {renaming ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Saving…
                </>
              ) : (
                "Save"
              )}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

interface CategoryCardProps {
  name: string;
  color: string;
  samples: Sample[];
  playingId: string | null;
  isUploading: boolean;
  onUpload: (file: File) => void;
  onPlay: (s: Sample) => void;
  onRename: (s: Sample) => void;
  onDelete: (s: Sample) => void;
}

function CategoryCard({
  name,
  color,
  samples,
  playingId,
  isUploading,
  onUpload,
  onPlay,
  onRename,
  onDelete,
}: CategoryCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="bd-card flex flex-col overflow-hidden">
      {/* Card header */}
      <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <span
            className="flex h-8 w-8 items-center justify-center rounded-lg text-white"
            style={{ background: color }}
          >
            <FileAudio className="h-4 w-4" />
          </span>
          <div>
            <h3 className="text-[15px] font-semibold text-slate-900">
              {name}
            </h3>
            <p className="text-xs text-slate-500">
              {samples.length} {samples.length === 1 ? "sample" : "samples"}
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isUploading}
          className="bd-btn bd-btn-secondary bd-btn-sm"
        >
          {isUploading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Uploading…
            </>
          ) : (
            <>
              <Upload className="h-3.5 w-3.5" /> Upload
            </>
          )}
        </button>
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.m4a,.flac,audio/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onUpload(f);
            e.target.value = "";
          }}
        />
      </div>

      {/* Sample list */}
      <div className="bd-scroll max-h-80 flex-1 overflow-y-auto p-3">
        {samples.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8 text-center">
            <Inbox className="h-8 w-8 text-slate-300" />
            <div>
              <p className="text-sm font-medium text-slate-700">
                No demo recordings available.
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Upload a respiratory recording to build the demo library for
                future users.
              </p>
            </div>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={isUploading}
              className="bd-btn bd-btn-primary bd-btn-sm"
            >
              <Upload className="h-3.5 w-3.5" /> Upload Audio
            </button>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {samples.map((s) => {
              const isPlaying = playingId === s.id;
              return (
                <li
                  key={s.id}
                  className="group rounded-lg border border-slate-100 bg-white/60 p-2.5 transition-colors hover:border-slate-200 hover:bg-white"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2.5">
                      <button
                        type="button"
                        onClick={() => onPlay(s)}
                        aria-label={
                          isPlaying ? `Pause ${s.filename}` : `Play ${s.filename}`
                        }
                        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-white transition-transform hover:scale-105"
                        style={{
                          background: `linear-gradient(135deg, ${color}, ${color}dd)`,
                        }}
                      >
                        {isPlaying ? (
                          <Pause className="h-3.5 w-3.5" />
                        ) : (
                          <Play className="ml-0.5 h-3.5 w-3.5" />
                        )}
                      </button>
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <FileAudio className="h-3.5 w-3.5 shrink-0 text-sky-500" />
                          <span className="truncate text-[13px] font-medium text-slate-900">
                            {s.filename}.{s.extension}
                          </span>
                        </div>
                        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {formatDuration(s.duration)}
                          </span>
                          <span className="flex items-center gap-1">
                            <HardDrive className="h-3 w-3" />
                            {formatSize(s.filesize)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {formatDate(s.createdAt)}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-0.5 opacity-60 transition-opacity group-hover:opacity-100">
                      <button
                        type="button"
                        onClick={() => onRename(s)}
                        aria-label={`Rename ${s.filename}`}
                        className="flex h-7 w-7 items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-sky-50 hover:text-sky-600"
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(s)}
                        aria-label={`Delete ${s.filename}`}
                        className="flex h-7 w-7 items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-red-50 hover:text-red-500"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
