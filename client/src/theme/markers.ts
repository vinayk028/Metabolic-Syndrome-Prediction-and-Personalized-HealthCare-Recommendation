/**
 * Marker helpers for the assessment page and dashboard — status + reference range for each
 * of the five markers, derived from the shared thresholds in theme/factors.ts.
 * Display only; the prediction itself comes from the backend.
 */
import { getFactorStatus, STATUS_TO_RISK, type FactorStatus } from './factors';
import type { FactorKey, RiskKey } from './tokens';
import type { RangeZone } from '../components/ui/RangeBar';

export const STATUS_LABEL: Record<FactorStatus, string> = {
  normal: 'In range',
  borderline: 'Borderline',
  elevated: 'Out of range',
};

export interface MarkerRow {
  key: FactorKey;
  value: number;
  display: string;
  unit: string;
  status: FactorStatus;
  risk: RiskKey;
  min: number;
  max: number;
  zones: RangeZone[];
}

const isNum = (n: number) => Number.isFinite(n);

/** Reference range zones per marker (ascending). HDL is "lower is worse". */
const rangeFor = (key: FactorKey, gender: string): { min: number; max: number; zones: RangeZone[] } => {
  const woman = gender === 'Women';
  switch (key) {
    case 'waist': {
      const cut = woman ? 88 : 102;
      return {
        min: 60,
        max: 140,
        zones: [
          { to: cut - 8, risk: 'low' },
          { to: cut, risk: 'medium' },
          { to: 140, risk: 'high' },
        ],
      };
    }
    case 'bloodPressure':
      return {
        min: 90,
        max: 180,
        zones: [
          { to: 120, risk: 'low' },
          { to: 130, risk: 'medium' },
          { to: 180, risk: 'high' },
        ],
      };
    case 'glucose':
      return {
        min: 60,
        max: 160,
        zones: [
          { to: 90, risk: 'low' },
          { to: 100, risk: 'medium' },
          { to: 160, risk: 'high' },
        ],
      };
    case 'triglycerides':
      return {
        min: 50,
        max: 300,
        zones: [
          { to: 120, risk: 'low' },
          { to: 150, risk: 'medium' },
          { to: 300, risk: 'high' },
        ],
      };
    case 'hdl': {
      const cut = woman ? 50 : 40;
      return {
        min: 20,
        max: 90,
        zones: [
          { to: cut, risk: 'high' },
          { to: cut + 10, risk: 'medium' },
          { to: 90, risk: 'low' },
        ],
      };
    }
  }
};

/** Status for a single marker, or null while the input is empty/invalid. */
const markerStatus = (key: FactorKey, value: number, gender: string, secondary?: number): FactorStatus | null => {
  if (!isNum(value) || (secondary !== undefined && !isNum(secondary))) return null;
  return getFactorStatus(key, value, gender, secondary);
};

interface Inputs {
  gender: string;
  systolicBP: number;
  diastolicBP: number;
  waistCircumference: number;
  hdlCholesterol?: number;
  triglyceride?: number;
  fpg?: number;
}

/** Rows for every marker we have a value for, in the canonical order. */
export const buildMarkerRows = (i: Inputs): MarkerRow[] => {
  const raw: { key: FactorKey; value?: number; display?: string; unit: string; secondary?: number }[] = [
    { key: 'waist', value: i.waistCircumference, unit: 'cm' },
    {
      key: 'bloodPressure',
      value: i.systolicBP,
      secondary: i.diastolicBP,
      display: `${i.systolicBP}/${i.diastolicBP}`,
      unit: 'mmHg',
    },
    { key: 'glucose', value: i.fpg, unit: 'mg/dL' },
    { key: 'triglycerides', value: i.triglyceride, unit: 'mg/dL' },
    { key: 'hdl', value: i.hdlCholesterol, unit: 'mg/dL' },
  ];

  return raw.flatMap((r) => {
    if (r.value === undefined) return [];
    const status = markerStatus(r.key, r.value, i.gender, r.secondary);
    if (!status) return [];
    return [
      {
        key: r.key,
        value: r.value,
        display: r.display ?? String(r.value),
        unit: r.unit,
        status,
        risk: STATUS_TO_RISK[status],
        ...rangeFor(r.key, i.gender),
      },
    ];
  });
};
