"use client";

/**
 * Export the clinical report as a PDF using the browser's native print API.
 * Opens a new window with a professionally formatted HTML document and
 * triggers print → user selects "Save as PDF".
 */

interface ExportData {
  markdown: string;
  patientName?: string;
  generatedAt: string;
}

export function exportClinicalReport({
  markdown,
  generatedAt,
}: ExportData) {
  const printWindow = window.open("", "_blank", "width=900,height=700");
  if (!printWindow) {
    alert("Please allow pop-ups to export the clinical report.");
    return;
  }

  const html = buildPrintableHTML(markdown, generatedAt);
  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();

  // Wait for the window to load, then trigger print
  printWindow.onload = () => {
    setTimeout(() => {
      printWindow.focus();
      printWindow.print();
    }, 300);
  };
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Convert markdown to HTML for the printable report.
 * Supports: ## headings, ### subheadings, tables, lists, bold, italic, hr.
 */
function markdownToHtml(md: string): string {
  const lines = md.split("\n");
  const html: string[] = [];
  let inTable = false;
  let tableHeader = false;
  let inList = false;
  let listType: "ul" | "ol" = "ul";

  const inline = (text: string): string =>
    text
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`(.+?)`/g, "<code>$1</code>");

  const closeList = () => {
    if (inList) {
      html.push(`</${listType}>`);
      inList = false;
    }
  };
  const closeTable = () => {
    if (inTable) {
      html.push("</tbody></table>");
      inTable = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Table detection
    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
      closeList();
      const cells = line
        .trim()
        .slice(1, -1)
        .split("|")
        .map((c) => c.trim());
      // Check if next line is a separator (|---|---|)
      const nextLine = lines[i + 1] || "";
      if (
        nextLine.trim().match(/^\|[\s-:|]+\|$/) &&
        !tableHeader
      ) {
        // Header row
        if (inTable) closeTable();
        html.push('<table>');
        html.push("<thead><tr>");
        for (const cell of cells) {
          html.push(`<th>${inline(escapeHtml(cell))}</th>`);
        }
        html.push("</tr></thead><tbody>");
        inTable = true;
        tableHeader = true;
        i++; // skip separator line
        continue;
      } else if (inTable) {
        // Data row
        html.push("<tr>");
        for (const cell of cells) {
          html.push(`<td>${inline(escapeHtml(cell))}</td>`);
        }
        html.push("</tr>");
        continue;
      }
    }

    closeTable();
    tableHeader = false;

    // Headings
    if (line.startsWith("## ")) {
      closeList();
      html.push(`<h2>${inline(escapeHtml(line.slice(3)))}</h2>`);
    } else if (line.startsWith("### ")) {
      closeList();
      html.push(`<h3>${inline(escapeHtml(line.slice(4)))}</h3>`);
    } else if (line.startsWith("> ")) {
      closeList();
      html.push(
        `<blockquote>${inline(escapeHtml(line.slice(2)))}</blockquote>`
      );
    } else if (line.trim() === "---" || line.trim() === "***") {
      closeList();
      html.push("<hr />");
    } else if (/^\s*[-*]\s+/.test(line)) {
      if (!inList || listType !== "ul") {
        closeList();
        html.push("<ul>");
        inList = true;
        listType = "ul";
      }
      html.push(`<li>${inline(escapeHtml(line.replace(/^\s*[-*]\s+/, "")))}</li>`);
    } else if (/^\s*\d+\.\s+/.test(line)) {
      if (!inList || listType !== "ol") {
        closeList();
        html.push("<ol>");
        inList = true;
        listType = "ol";
      }
      html.push(
        `<li>${inline(escapeHtml(line.replace(/^\s*\d+\.\s+/, "")))}</li>`
      );
    } else if (line.trim() === "") {
      closeList();
    } else {
      closeList();
      html.push(`<p>${inline(escapeHtml(line))}</p>`);
    }
  }
  closeList();
  closeTable();

  return html.join("\n");
}

function buildPrintableHTML(markdown: string, generatedAt: string): string {
  const date = new Date(generatedAt).toLocaleString(undefined, {
    dateStyle: "long",
    timeStyle: "short",
  });
  const body = markdownToHtml(markdown);

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Clinical Decision Support Report — Breathe</title>
<style>
  @page {
    margin: 2cm 1.8cm;
    size: A4;
  }
  * { box-sizing: border-box; }
  body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1e293b;
    margin: 0;
    padding: 24px;
    background: #fff;
  }
  .report-header {
    border-bottom: 3px solid #0284c7;
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .report-header h1 {
    font-size: 18pt;
    color: #0284c7;
    margin: 0 0 4px 0;
    font-family: Arial, Helvetica, sans-serif;
  }
  .report-header .subtitle {
    font-size: 10pt;
    color: #64748b;
    margin: 0;
  }
  .report-header .meta {
    text-align: right;
    font-size: 9pt;
    color: #64748b;
    font-family: Arial, Helvetica, sans-serif;
  }
  h2 {
    font-size: 13pt;
    color: #0c4a6e;
    margin-top: 24px;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e0f2fe;
    font-family: Arial, Helvetica, sans-serif;
    page-break-after: avoid;
  }
  h3 {
    font-size: 11pt;
    color: #0369a1;
    margin-top: 16px;
    margin-bottom: 6px;
    font-family: Arial, Helvetica, sans-serif;
    page-break-after: avoid;
  }
  p { margin: 0 0 8px 0; }
  ul, ol { margin: 0 0 8px 0; padding-left: 20px; }
  li { margin-bottom: 3px; }
  strong { color: #0f172a; font-weight: bold; }
  em { color: #475569; }
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 10pt;
    page-break-inside: avoid;
  }
  th {
    background: #e0f2fe;
    color: #0c4a6e;
    text-align: left;
    padding: 6px 10px;
    border: 1px solid #bae6fd;
    font-family: Arial, Helvetica, sans-serif;
    font-weight: bold;
  }
  td {
    padding: 5px 10px;
    border: 1px solid #e2e8f0;
    vertical-align: top;
  }
  tr:nth-child(even) td {
    background: #f8fafc;
  }
  blockquote {
    border-left: 3px solid #0284c7;
    margin: 8px 0;
    padding: 4px 12px;
    background: #f0f9ff;
    color: #475569;
    font-style: italic;
  }
  hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
  }
  .report-footer {
    margin-top: 32px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    font-size: 8.5pt;
    color: #94a3b8;
    text-align: center;
    font-family: Arial, Helvetica, sans-serif;
  }
  @media print {
    body { padding: 0; }
    .report-header { page-break-after: avoid; }
    h2, h3 { page-break-after: avoid; }
    table, blockquote { page-break-inside: avoid; }
  }
</style>
</head>
<body>
  <div class="report-header">
    <div>
      <h1>Clinical Decision Support Report</h1>
      <p class="subtitle">Breathe — AI-Assisted Respiratory Diagnostic System</p>
    </div>
    <div class="meta">
      Generated: ${date}<br />
      Report ID: ${Date.now().toString(36).toUpperCase()}
    </div>
  </div>
  ${body}
  <div class="report-footer">
    This report was generated by the Breathe AI-Assisted Respiratory Diagnostic System and is intended for
    educational and informational purposes only. It does not constitute a medical diagnosis and must be
    reviewed by a qualified healthcare professional. &copy; ${new Date().getFullYear()} Breathe.
  </div>
</body>
</html>`;
}
