"use client";

import { useState } from "react";
import { useBreatheStore, type EnvironmentalData } from "@/lib/breathe-store";
import {
  Loader2,
  MapPin,
  Wind,
  Droplets,
  Thermometer,
  CloudRain,
  Cloud,
  Gauge,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";

type RiskLevel = "low" | "moderate" | "high";

function riskFromEpa(epaIndex?: number): RiskLevel {
  const e = epaIndex ?? 1;
  if (e <= 1) return "low";
  if (e <= 3) return "moderate";
  return "high";
}

const RISK_STYLES: Record<
  RiskLevel,
  { bg: string; color: string; label: string; Icon: typeof CheckCircle2 }
> = {
  low: {
    bg: "bg-emerald-50 border-emerald-200",
    color: "text-emerald-700",
    label: "Low Risk",
    Icon: CheckCircle2,
  },
  moderate: {
    bg: "bg-amber-50 border-amber-200",
    color: "text-amber-700",
    label: "Moderate Risk",
    Icon: AlertTriangle,
  },
  high: {
    bg: "bg-red-50 border-red-200",
    color: "text-red-700",
    label: "High Risk",
    Icon: AlertCircle,
  },
};

function adviceFromEpa(epaIndex?: number): string {
  const e = epaIndex ?? 1;
  switch (e) {
    case 1:
      return "Air quality is good. Safe for asthma patients.";
    case 2:
      return "Moderate air quality. Avoid long outdoor exposure.";
    case 3:
      return "Unhealthy for sensitive groups. Stay indoors if possible.";
    default:
      return "Poor air quality. High asthma risk. Use inhaler and stay indoors.";
  }
}

function buildTriggers(d: EnvironmentalData): string[] {
  const t: string[] = [];
  if ((d.humidity ?? 0) > 70) t.push("High humidity may trigger asthma symptoms.");
  if ((d.temperature ?? 99) < 15) t.push("Cold air can cause breathing difficulty.");
  const w = d.windSpeed ?? 0;
  if (w <= 15) t.push("Wind conditions are safe.");
  else if (w <= 25) t.push("Moderate wind may spread dust and pollen.");
  else t.push("High wind may significantly worsen asthma symptoms.");
  if ((d.pressure ?? 1010) < 1000) t.push("Low pressure may worsen respiratory issues.");
  if ((d.cloudCover ?? 0) > 60) t.push("Pollution may remain trapped in the air.");
  return t;
}

export function SafeCheckView() {
  const setEnvironmentalData = useBreatheStore((s) => s.setEnvironmentalData);
  const stored = useBreatheStore((s) => s.environmentalData);

  const [data, setData] = useState<EnvironmentalData | null>(
    Object.keys(stored).length ? stored : null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function fetchWeather(lat: number, lon: number) {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`/api/weather?lat=${lat}&lon=${lon}`);
      const json = await res.json();
      if (!res.ok || json.error) {
        setError(
          typeof json.error === "string"
            ? json.error
            : json.error?.info || "Failed to fetch location data"
        );
        return;
      }
      setData(json);
      setEnvironmentalData(json);
    } catch {
      setError("Server error while fetching location data");
    } finally {
      setLoading(false);
    }
  }

  function handleLocation() {
    if (!navigator.geolocation) {
      setError("Geolocation is not supported by your browser.");
      return;
    }
    setLoading(true);
    setError("");
    navigator.geolocation.getCurrentPosition(
      (pos) => fetchWeather(pos.coords.latitude, pos.coords.longitude),
      () => {
        setError("Location access denied. Please allow location access.");
        setLoading(false);
      }
    );
  }

  const risk = riskFromEpa(data?.epaIndex);
  const riskStyle = RISK_STYLES[risk];
  const triggers = data ? buildTriggers(data) : [];
  const RiskIcon = riskStyle.Icon;

  const cards = data
    ? [
        { label: "AQI", value: data.aqi ?? "--", icon: Gauge },
        { label: "PM2.5", value: data.pm25 ?? "--", icon: Gauge },
        { label: "PM10", value: data.pm10 ?? "--", icon: Gauge },
        { label: "NO₂", value: data.no2 ?? "--", icon: Gauge },
        { label: "Temperature", value: `${data.temperature ?? "--"}°C`, icon: Thermometer },
        { label: "Humidity", value: `${data.humidity ?? "--"}%`, icon: Droplets },
        { label: "Weather", value: data.weatherDescription ?? "--", icon: Cloud },
        { label: "Wind Speed", value: `${data.windSpeed ?? "--"} km/h`, icon: Wind },
        { label: "Rainfall", value: `${data.precip ?? "--"} mm`, icon: CloudRain },
      ]
    : [];

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h2 className="text-xl font-semibold tracking-tight text-slate-900">
          Safe Check
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          Real-time air quality and weather monitoring for asthma patients with
          personalized health alerts.
        </p>
      </div>

      {/* Location input */}
      <div className="bd-section p-5">
        <h3 className="mb-3 text-[13px] font-semibold uppercase tracking-wide text-slate-400">
          Location & Monitoring
        </h3>
        <div className="flex justify-center">
          <button
            type="button"
            onClick={handleLocation}
            disabled={loading}
            className="bd-btn bd-btn-primary bd-btn-lg w-full max-w-xs"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Detecting…
              </>
            ) : (
              <>
                <MapPin className="h-4 w-4" /> Use Current Location
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4"
          role="alert"
        >
          <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div className="text-[13px] text-slate-700">{error}</div>
        </div>
      )}

      {/* Result */}
      {data && (
        <div className="space-y-4 bd-fade-in">
          {/* Location + risk */}
          <div className="bd-card p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <MapPin className="h-4 w-4 shrink-0 text-sky-600" />
                <span>
                  <span className="font-medium text-slate-900">
                    {data.resolvedLocation?.name ?? "--"}
                  </span>
                  {data.resolvedLocation?.region
                    ? `, ${data.resolvedLocation.region}`
                    : ""}
                  {data.resolvedLocation?.country
                    ? `, ${data.resolvedLocation.country}`
                    : ""}
                </span>
              </div>
              <div
                className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[13px] font-semibold ${riskStyle.bg} ${riskStyle.color}`}
              >
                <RiskIcon className="h-3.5 w-3.5" />
                {riskStyle.label}
              </div>
            </div>
          </div>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            {cards.map((c) => {
              const Icon = c.icon;
              return (
                <div
                  key={c.label}
                  className="bd-card p-4"
                >
                  <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
                    <Icon className="h-3 w-3" />
                    {c.label}
                  </div>
                  <div className="mt-1.5 text-lg font-semibold text-slate-900">
                    {c.value}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Recommendations */}
          <div className="bd-card p-5">
            <h3 className="mb-2 text-[13px] font-semibold uppercase tracking-wide text-slate-400">
              Health Recommendations
            </h3>
            <p className="text-sm leading-relaxed text-slate-700">
              {adviceFromEpa(data.epaIndex)}
            </p>
            {triggers.length > 0 && (
              <div className="mt-3 border-t border-slate-100 pt-3">
                <div className="mb-1.5 text-[13px] font-medium text-slate-700">
                  Weather Triggers
                </div>
                <ul className="space-y-1">
                  {triggers.map((t, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2 text-[13px] text-slate-600"
                    >
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-sky-500" />
                      {t}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <p className="mt-3 text-center text-[11px] text-slate-400">
              *Air quality data is derived from the nearest available monitoring
              station.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
