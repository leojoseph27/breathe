import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ViewKey =
  | "audio"
  | "asthma"
  | "safecheck"
  | "aidoctor"
  | "library";

export interface PatientData {
  age?: number;
  gender?: number; // 0 male, 1 female
  bmi?: number;
  smoking?: number; // 0 non, 1 former, 2 current
  familyHistory?: number; // 0 no, 1 yes
  allergyHistory?: number; // 0 no, 1 yes
  lungFunctionFeV1?: number;
  wheezing?: number; // 0 no, 1 yes
  shortnessOfBreath?: number; // 0 no, 1 yes
  chestTightness?: number; // 0 no, 1 yes
}

export interface AudioAnalysis {
  prediction?: string;
  filename?: string;
  confidence?: number;
  source?: string;
}

export interface AsthmaAssessment {
  prediction?: string; // "Asthma Detected" | "No Asthma Detected"
  confidence?: number;
  raw?: number; // 0 | 1
}

export interface EnvironmentalData {
  resolvedLocation?: {
    name?: string;
    region?: string;
    country?: string;
    lat?: number;
    lon?: number;
  };
  temperature?: number;
  humidity?: number;
  windSpeed?: number;
  pressure?: number;
  cloudCover?: number;
  precip?: number;
  weatherDescription?: string;
  pm25?: number;
  pm10?: number;
  no2?: number;
  so2?: number;
  o3?: number;
  co?: number;
  aqi?: number;
  epaIndex?: number;
}

interface BreatheState {
  // navigation (single-route SPA — no auth required)
  view: ViewKey;
  setView: (v: ViewKey) => void;

  // session data (mirrors the original Flask session storage)
  patientData: PatientData;
  audioAnalysis: AudioAnalysis;
  asthmaAssessment: AsthmaAssessment;
  environmentalData: EnvironmentalData;
  setPatientData: (d: PatientData) => void;
  setAudioAnalysis: (d: AudioAnalysis) => void;
  setAsthmaAssessment: (d: AsthmaAssessment) => void;
  setEnvironmentalData: (d: EnvironmentalData) => void;
  resetSessionData: () => void;
}

export const useBreatheStore = create<BreatheState>()(
  persist(
    (set) => ({
      view: "audio",
      setView: (v) => set({ view: v }),

      patientData: {},
      audioAnalysis: {},
      asthmaAssessment: {},
      environmentalData: {},
      setPatientData: (d) => set({ patientData: d }),
      setAudioAnalysis: (d) => set({ audioAnalysis: d }),
      setAsthmaAssessment: (d) => set({ asthmaAssessment: d }),
      setEnvironmentalData: (d) => set({ environmentalData: d }),
      resetSessionData: () =>
        set({
          patientData: {},
          audioAnalysis: {},
          asthmaAssessment: {},
          environmentalData: {},
        }),
    }),
    {
      name: "breathe-session-v3",
      partialize: (state) => ({
        view: state.view,
        patientData: state.patientData,
        audioAnalysis: state.audioAnalysis,
        asthmaAssessment: state.asthmaAssessment,
        environmentalData: state.environmentalData,
      }),
    }
  )
);
