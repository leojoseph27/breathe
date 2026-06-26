"use client";

import { useState, useRef, useCallback, type ReactNode } from "react";
import { Info } from "lucide-react";

interface InfoTooltipProps {
  /** Short label for the field, used in the icon's aria-label */
  label: string;
  /** Rich content inside the popover */
  children: ReactNode;
}

/**
 * Accessible info tooltip that opens on hover (desktop), tap/click (mobile),
 * and keyboard focus. Renders a dark popover above the trigger icon.
 */
export function InfoTooltip({ label, children }: InfoTooltipProps) {
  const [open, setOpen] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    setOpen(true);
  }, []);

  const hide = useCallback(() => {
    // Small delay so moving the mouse from trigger to popover doesn't close it
    timer.current = setTimeout(() => setOpen(false), 150);
  }, []);

  // Click always opens (never toggles) to avoid the focus→click race where
  // focus opens the tooltip and the subsequent click toggles it closed again.
  // On mobile, tapping the icon focuses (opens) + clicks (stays open); tapping
  // elsewhere blurs (closes). On desktop, hover/focus opens, leave/blur closes.
  const handleClick = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    setOpen(true);
  }, []);

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onClick={handleClick}
        aria-label={`More information about ${label}`}
        aria-expanded={open}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full text-[var(--portage)] transition-colors hover:text-[var(--perfume)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--portage)] focus-visible:ring-offset-1"
      >
        <Info className="h-4 w-4" />
      </button>
      {open && (
        <div
          role="tooltip"
          className="info-tooltip-popover"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          {children}
        </div>
      )}
    </span>
  );
}

/** Helper for a bold title line inside the tooltip */
export function TooltipTitle({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1 font-semibold text-cyan-300">{children}</div>
  );
}

/** Helper for a labelled section inside the tooltip */
export function TooltipSection({
  heading,
  children,
}: {
  heading: string;
  children: ReactNode;
}) {
  return (
    <div className="mt-2 first:mt-0">
      <span className="font-semibold text-cyan-300">{heading}: </span>
      <span>{children}</span>
    </div>
  );
}
