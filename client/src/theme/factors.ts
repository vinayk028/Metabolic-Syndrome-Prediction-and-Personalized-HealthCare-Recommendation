/**
 * Metabolic-syndrome risk factor metadata.
 *
 * Every page that talks about a factor (landing body map, assessment steps,
 * dashboard ring, recommendation tags) reads label / icon / colour from here so
 * the five factors look identical everywhere.
 *
 * Thresholds follow the 2009 Joint Interim Statement ("harmonized" criteria),
 * using ATP III waist cut-offs. They are for display/education only — the
 * prediction itself comes from the backend model.
 */
import type { SvgIconComponent } from '@mui/icons-material';
import StraightenRoundedIcon from '@mui/icons-material/StraightenRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import BloodtypeRoundedIcon from '@mui/icons-material/BloodtypeRounded';
import WaterDropRoundedIcon from '@mui/icons-material/WaterDropRounded';
import ShieldRoundedIcon from '@mui/icons-material/ShieldRounded';
import { factorColors, type FactorKey, type RiskKey } from './tokens';

export interface FactorMeta {
  key: FactorKey;
  label: string;
  shortLabel: string;
  unit: string;
  /** Plain-language threshold shown in tooltips / body map */
  threshold: string;
  description: string;
  icon: SvgIconComponent;
  color: string;
}

export const FACTOR_ORDER: FactorKey[] = ['waist', 'bloodPressure', 'glucose', 'triglycerides', 'hdl'];

export const FACTORS: Record<FactorKey, FactorMeta> = {
  waist: {
    key: 'waist',
    label: 'Waist circumference',
    shortLabel: 'Waist',
    unit: 'cm',
    threshold: '≥ 102 cm (men) · ≥ 88 cm (women)',
    description: 'Abdominal fat is the strongest visible marker of insulin resistance.',
    icon: StraightenRoundedIcon,
    color: factorColors.waist.main,
  },
  bloodPressure: {
    key: 'bloodPressure',
    label: 'Blood pressure',
    shortLabel: 'BP',
    unit: 'mmHg',
    threshold: '≥ 130 / 85 mmHg',
    description: 'Elevated pressure strains the heart and blood vessels over time.',
    icon: MonitorHeartRoundedIcon,
    color: factorColors.bloodPressure.main,
  },
  glucose: {
    key: 'glucose',
    label: 'Fasting glucose',
    shortLabel: 'Glucose',
    unit: 'mg/dL',
    threshold: '≥ 100 mg/dL',
    description: 'High fasting sugar is an early signal on the path to type 2 diabetes.',
    icon: BloodtypeRoundedIcon,
    color: factorColors.glucose.main,
  },
  triglycerides: {
    key: 'triglycerides',
    label: 'Triglycerides',
    shortLabel: 'TG',
    unit: 'mg/dL',
    threshold: '≥ 150 mg/dL',
    description: 'Blood fats that rise with excess sugar, refined carbs and alcohol.',
    icon: WaterDropRoundedIcon,
    color: factorColors.triglycerides.main,
  },
  hdl: {
    key: 'hdl',
    label: 'HDL cholesterol',
    shortLabel: 'HDL',
    unit: 'mg/dL',
    threshold: '< 40 mg/dL (men) · < 50 mg/dL (women)',
    description: '"Good" cholesterol that protects arteries — low levels raise risk.',
    icon: ShieldRoundedIcon,
    color: factorColors.hdl.main,
  },
};

export type FactorStatus = 'normal' | 'borderline' | 'elevated';

type Gender = 'Men' | 'Women' | string;

/**
 * Classify a single factor value for UI display (ring segments, badges).
 * `value` for bloodPressure is systolic; pass diastolic as `secondary`.
 */
export function getFactorStatus(key: FactorKey, value: number, gender: Gender = 'Men', secondary?: number): FactorStatus {
  const isWoman = gender === 'Women';
  switch (key) {
    case 'waist': {
      const cut = isWoman ? 88 : 102;
      return value >= cut ? 'elevated' : value >= cut - 8 ? 'borderline' : 'normal';
    }
    case 'bloodPressure': {
      const dia = secondary ?? 0;
      if (value >= 130 || dia >= 85) return 'elevated';
      if (value >= 120 || dia >= 80) return 'borderline';
      return 'normal';
    }
    case 'glucose':
      return value >= 100 ? 'elevated' : value >= 90 ? 'borderline' : 'normal';
    case 'triglycerides':
      return value >= 150 ? 'elevated' : value >= 120 ? 'borderline' : 'normal';
    case 'hdl': {
      const cut = isWoman ? 50 : 40;
      return value < cut ? 'elevated' : value < cut + 10 ? 'borderline' : 'normal';
    }
  }
}

/** Map backend risk strings ("Low Severity", "Medium Severity", "High Severity") to a token key. */
export function toRiskKey(riskLevel?: string | null): RiskKey | null {
  if (!riskLevel) return null;
  const v = riskLevel.toLowerCase();
  if (v.startsWith('low')) return 'low';
  if (v.startsWith('med') || v.startsWith('mod')) return 'medium';
  if (v.startsWith('high')) return 'high';
  return null;
}

export const RISK_LABEL: Record<RiskKey, string> = {
  low: 'Low risk',
  medium: 'Moderate risk',
  high: 'High risk',
};

export const STATUS_TO_RISK: Record<FactorStatus, RiskKey> = {
  normal: 'low',
  borderline: 'medium',
  elevated: 'high',
};
