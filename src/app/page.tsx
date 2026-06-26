"use client";

import { useEffect, useState } from "react";
import { useBreatheStore } from "@/lib/breathe-store";
import { BackgroundShapes } from "@/components/breathe/BackgroundShapes";
import { NavigationBar } from "@/components/breathe/NavigationBar";
import { AudioAnalysisView } from "@/components/breathe/views/AudioAnalysisView";
import { AsthmaDetectionView } from "@/components/breathe/views/AsthmaDetectionView";
import { SafeCheckView } from "@/components/breathe/views/SafeCheckView";
import { AIDoctorView } from "@/components/breathe/views/AIDoctorView";
import { DemoLibraryView } from "@/components/breathe/views/DemoLibraryView";
import { Loader2 } from "lucide-react";

export default function Home() {
  const view = useBreatheStore((s) => s.view);
  const [mounted, setMounted] = useState(false);

  // The persisted store hydrates from localStorage on the client; render a
  // neutral splash until mounted to avoid hydration mismatches.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setMounted(true), []);

  if (!mounted) {
    return (
      <div className="bd-page flex min-h-screen items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-sky-600" />
      </div>
    );
  }

  return (
    <div className="bd-page">
      <BackgroundShapes />

      <NavigationBar />

      <main className="relative z-10 mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
        <div key={view} className="bd-fade-in">
          {view === "audio" && <AudioAnalysisView />}
          {view === "asthma" && <AsthmaDetectionView />}
          {view === "safecheck" && <SafeCheckView />}
          {view === "aidoctor" && <AIDoctorView />}
          {view === "library" && <DemoLibraryView />}
        </div>
      </main>

      <footer className="relative z-10 mx-auto w-full max-w-6xl px-4 pb-8 pt-2 sm:px-6">
        <p className="text-center text-xs text-slate-400">
          Breathe — Respiratory Disease Prediction System · For demonstration
          only · Not a substitute for professional medical advice
        </p>
      </footer>
    </div>
  );
}
