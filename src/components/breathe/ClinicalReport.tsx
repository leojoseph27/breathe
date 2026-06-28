"use client";

import { useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChevronDown,
  ChevronRight,
  User,
  AudioLines,
  HeartPulse,
  CloudSun,
  Link2,
  Table2,
  AlertTriangle,
  GitBranch,
  ClipboardList,
  ShieldAlert,
  FileText,
} from "lucide-react";

interface ReportSection {
  id: string;
  title: string;
  icon: typeof User;
  content: string;
  defaultOpen?: boolean;
}

/**
 * Split the AI-generated markdown report into sections by ## headings.
 * Returns an array of { id, title, icon, content } for collapsible rendering.
 */
function parseSections(markdown: string): ReportSection[] {
  const iconMap: Record<string, typeof User> = {
    "patient summary": User,
    "audio analysis interpretation": AudioLines,
    "clinical asthma assessment": HeartPulse,
    "environmental risk analysis": CloudSun,
    "cross-module correlation": Link2,
    "supporting findings": Table2,
    "risk factor analysis": AlertTriangle,
    "differential diagnosis": GitBranch,
    recommendations: ClipboardList,
    limitations: ShieldAlert,
    "overall clinical impression": FileText,
  };

  // Split on ## headings (but not ### which are subsections)
  const lines = markdown.split("\n");
  const sections: ReportSection[] = [];
  let currentTitle = "";
  let currentLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("## ") && !line.startsWith("### ")) {
      // Save previous section
      if (currentTitle) {
        sections.push(makeSection(currentTitle, currentLines.join("\n"), iconMap));
      }
      currentTitle = line.replace(/^##\s+/, "").trim();
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }
  if (currentTitle) {
    sections.push(makeSection(currentTitle, currentLines.join("\n"), iconMap));
  }

  return sections;
}

function makeSection(
  title: string,
  content: string,
  iconMap: Record<string, typeof User>
): ReportSection {
  const id = title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const icon = iconMap[title.toLowerCase()] ?? FileText;
  return {
    id,
    title,
    icon,
    content: content.trim(),
    defaultOpen: title.toLowerCase() === "patient summary",
  };
}

export function ClinicalReport({ markdown }: { markdown: string }) {
  const sections = parseSections(markdown);

  return (
    <div className="space-y-3">
      {sections.map((section) => (
        <CollapsibleSection
          key={section.id}
          title={section.title}
          icon={section.icon}
          defaultOpen={section.defaultOpen}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h3: ({ children }) => (
                <h4 className="mt-4 mb-2 text-[13px] font-semibold text-slate-700 first:mt-0">
                  {children}
                </h4>
              ),
              p: ({ children }) => (
                <p className="mb-3 text-[13.5px] leading-relaxed text-slate-600 last:mb-0">
                  {children}
                </p>
              ),
              ul: ({ children }) => (
                <ul className="mb-3 ml-4 list-disc space-y-1 text-[13.5px] leading-relaxed text-slate-600 last:mb-0">
                  {children}
                </ul>
              ),
              ol: ({ children }) => (
                <ol className="mb-3 ml-4 list-decimal space-y-1 text-[13.5px] leading-relaxed text-slate-600 last:mb-0">
                  {children}
                </ol>
              ),
              li: ({ children }) => <li>{children}</li>,
              strong: ({ children }) => (
                <strong className="font-semibold text-slate-800">
                  {children}
                </strong>
              ),
              em: ({ children }) => (
                <em className="italic text-slate-600">{children}</em>
              ),
              table: ({ children }) => (
                <div className="bd-scroll my-3 overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full border-collapse text-[13px]">
                    {children}
                  </table>
                </div>
              ),
              thead: ({ children }) => (
                <thead className="bg-sky-50">{children}</thead>
              ),
              th: ({ children }) => (
                <th className="border-b border-slate-200 px-3 py-2 text-left font-semibold text-slate-700">
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td className="border-b border-slate-100 px-3 py-2 text-slate-600 even:bg-slate-50/50">
                  {children}
                </td>
              ),
              tr: ({ children }) => <tr>{children}</tr>,
              hr: () => <hr className="my-4 border-slate-200" />,
              blockquote: ({ children }) => (
                <blockquote className="my-3 border-l-4 border-sky-300 bg-sky-50/50 py-2 pl-4 pr-3 text-[13px] italic text-slate-600">
                  {children}
                </blockquote>
              ),
            }}
          >
            {section.content}
          </ReactMarkdown>
        </CollapsibleSection>
      ))}
    </div>
  );
}

function CollapsibleSection({
  title,
  icon: Icon,
  defaultOpen = false,
  children,
}: {
  title: string;
  icon: typeof User;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="bd-card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-5 py-3.5 text-left transition-colors hover:bg-slate-50/60"
      >
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-sky-50 text-sky-600">
            <Icon className="h-4 w-4" />
          </span>
          <span className="text-[14px] font-semibold text-slate-900">
            {title}
          </span>
        </div>
        {open ? (
          <ChevronDown className="h-4 w-4 shrink-0 text-slate-400" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0 text-slate-400" />
        )}
      </button>
      {open && (
        <div className="border-t border-slate-100 px-5 py-4">{children}</div>
      )}
    </div>
  );
}
