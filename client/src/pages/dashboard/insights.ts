/**
 * Dashboard insights derived from saved assessments — marker history,
 * focus-area tips and the next-check suggestion. Display only; nothing here
 * changes what is stored or how the prediction is made.
 */
import type { AssessmentHistoryItem } from '../../types/types';
import type { FactorKey } from '../../theme/tokens';

export interface MarkerPoint {
  date: string;
  value: number;
  display: string;
}

export interface MarkerSeries {
  key: FactorKey;
  unit: string;
  points: MarkerPoint[];
  /** last − first (systolic for blood pressure) */
  delta: number;
  /** true when the change is an improvement (HDL: up; everything else: down) */
  improving: boolean | null;
}

const UNIT: Record<FactorKey, string> = {
  waist: 'cm',
  bloodPressure: 'mmHg',
  glucose: 'mg/dL',
  triglycerides: 'mg/dL',
  hdl: 'mg/dL',
};

const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

/** Per-marker series across assessments (oldest first), skipping missing values. */
export const buildMarkerSeries = (sorted: AssessmentHistoryItem[]): Partial<Record<FactorKey, MarkerSeries>> => {
  const out: Partial<Record<FactorKey, MarkerSeries>> = {};

  const push = (key: FactorKey, date: string, value: unknown, display?: string) => {
    if (!isNum(value)) return;
    const s = (out[key] ??= { key, unit: UNIT[key], points: [], delta: 0, improving: null });
    s.points.push({ date, value, display: display ?? String(value) });
  };

  sorted.forEach((a) => {
    const p = a.inputParameters;
    if (!p) return;
    push('waist', a.date, p.waistCircumference);
    push('bloodPressure', a.date, p.systolicBP, `${p.systolicBP}/${p.diastolicBP}`);
    push('glucose', a.date, p.fpg);
    push('triglycerides', a.date, p.triglyceride);
    push('hdl', a.date, p.hdlCholesterol);
  });

  Object.values(out).forEach((s) => {
    if (!s || s.points.length < 2) return;
    s.delta = s.points[s.points.length - 1].value - s.points[0].value;
    s.improving = s.delta === 0 ? null : s.key === 'hdl' ? s.delta > 0 : s.delta < 0;
  });

  return out;
};

/** One practical, general lifestyle tip per marker (not medical advice). */
export const FOCUS_TIPS: Record<FactorKey, string> = {
  waist: 'Aim for 150 minutes of brisk activity a week and fewer refined carbs — waist size often responds within weeks.',
  bloodPressure: 'Cut back on salt and processed food and stay active. Recheck at the same time of day for a fair comparison.',
  glucose: 'Swap sugary drinks and white bread for whole grains; a 10-minute walk after meals helps steady blood sugar.',
  triglycerides: 'Limit sugar, refined carbs and alcohol. Oily fish twice a week can help bring triglycerides down.',
  hdl: 'Regular exercise and healthy fats such as olive oil and nuts help raise "good" HDL cholesterol.',
};

/** Suggested interval between checks. */
export const RECHECK_DAYS = 90;

export const nextCheck = (lastDate: string) => {
  const due = new Date(new Date(lastDate).getTime() + RECHECK_DAYS * 86_400_000);
  const daysLeft = Math.ceil((due.getTime() - Date.now()) / 86_400_000);
  return { due, daysLeft };
};
