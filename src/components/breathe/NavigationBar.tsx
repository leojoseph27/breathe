"use client";

import { useBreatheStore, type ViewKey } from "@/lib/breathe-store";
import { Activity } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS: { key: ViewKey; label: string }[] = [
  { key: "audio", label: "Audio Analysis" },
  { key: "asthma", label: "Asthma Detection" },
  { key: "safecheck", label: "Safe Check" },
  { key: "aidoctor", label: "AI Doctor" },
  { key: "library", label: "Demo Library" },
];

export function NavigationBar() {
  const view = useBreatheStore((s) => s.view);
  const setView = useBreatheStore((s) => s.setView);

  return (
    <header
      className="sticky top-0 z-40 border-b border-slate-200/60 backdrop-blur-xl"
      style={{ background: "rgba(255,255,255,0.72)" }}
    >
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        {/* Logo + title */}
        <button
          type="button"
          onClick={() => setView("audio")}
          className="flex shrink-0 items-center gap-2.5 rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40"
          aria-label="Breathe home"
        >
          <span
            className="flex h-9 w-9 items-center justify-center rounded-xl text-white shadow-sm"
            style={{
              background: "linear-gradient(135deg, #0ea5e9, #06b6d4)",
              boxShadow: "0 2px 8px rgba(2,132,199,0.25)",
            }}
          >
            <Activity className="h-5 w-5" strokeWidth={2.2} />
          </span>
          <span className="hidden text-left sm:block">
            <span className="block text-[15px] font-semibold leading-tight text-slate-900">
              Breathe
            </span>
            <span className="block text-[11px] leading-tight text-slate-500">
              Respiratory Diagnostic System
            </span>
          </span>
        </button>

        {/* Nav links */}
        <nav
          className="bd-scroll flex items-center gap-0.5 overflow-x-auto"
          aria-label="Primary"
        >
          {NAV_ITEMS.map((item) => {
            const active = view === item.key;
            return (
              <button
                key={item.key}
                type="button"
                onClick={() => setView(item.key)}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "relative shrink-0 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                  active
                    ? "bg-sky-50 text-sky-700"
                    : "text-slate-600 hover:bg-slate-100/70 hover:text-slate-900"
                )}
              >
                {item.label}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
