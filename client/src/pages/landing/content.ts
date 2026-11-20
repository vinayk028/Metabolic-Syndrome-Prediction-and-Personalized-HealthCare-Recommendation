/**
 * Landing page copy & data — edit wording here without touching layout.
 */
import type { SvgIconComponent } from '@mui/icons-material';
import PictureAsPdfRoundedIcon from '@mui/icons-material/PictureAsPdfRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import SignalCellularAltRoundedIcon from '@mui/icons-material/SignalCellularAltRounded';
import VolunteerActivismRoundedIcon from '@mui/icons-material/VolunteerActivismRounded';
import SpaceDashboardRoundedIcon from '@mui/icons-material/SpaceDashboardRounded';
import NewspaperRoundedIcon from '@mui/icons-material/NewspaperRounded';

export interface IconCopy {
  icon: SvgIconComponent;
  title: string;
  text: string;
}

/* ── How it works ──────────────────────────────────────────────────────── */
export const STEPS: { title: string; text: string }[] = [
  {
    title: 'Enter five markers',
    text: 'Waist, blood pressure, fasting glucose, triglycerides and HDL — all on a routine check-up — plus age, sex and existing diagnoses.',
  },
  {
    title: 'Get your risk and severity',
    text: 'See your chance of metabolic syndrome and how severe it is — in plain language, marker by marker.',
  },
  {
    title: 'Follow a personal plan',
    text: 'Diet, exercise and yoga recommendations matched to your severity — saved to your dashboard and downloadable as a PDF.',
  },
];

/* ── What we provide ───────────────────────────────────────────────────── */
export interface Feature extends IconCopy {
  /** Gradient stops for the icon tile */
  from: string;
  to: string;
}

export const FEATURES: Feature[] = [
  {
    icon: MonitorHeartRoundedIcon,
    title: 'Metabolic Syndrome Prediction',
    text: 'A highly accurate estimate of your chance of metabolic syndrome from five routine check-up markers.',
    from: '#0f9d8f',
    to: '#2dd4bf',
  },
  {
    icon: SignalCellularAltRoundedIcon,
    title: 'Severity Classification',
    text: 'See whether your result is low, moderate or high, so you know how much attention it needs.',
    from: '#f59e0b',
    to: '#fb923c',
  },
  {
    icon: VolunteerActivismRoundedIcon,
    title: 'Health Recommendations',
    text: 'Diet, exercise and yoga guidance matched to your result.',
    from: '#16a34a',
    to: '#4ade80',
  },
  {
    icon: PictureAsPdfRoundedIcon,
    title: 'Doctor-Ready Report',
    text: 'Download a clean PDF of your assessment to share with your doctor.',
    from: '#2563eb',
    to: '#6366f1',
  },
  {
    icon: NewspaperRoundedIcon,
    title: 'News & Research',
    text: 'Fresh metabolic-health stories and guides from trusted sources.',
    from: '#e11d48',
    to: '#fb7185',
  },
  {
    icon: SpaceDashboardRoundedIcon,
    title: 'Personalized Dashboard',
    text: 'Save your assessments and track your progress over time.',
    from: '#7c3aed',
    to: '#a78bfa',
  },
];

/* ── FAQ ───────────────────────────────────────────────────────────────── */
export const FAQS: { q: string; a: string }[] = [
  {
    q: 'How does it work?',
    a: 'You enter a few numbers from a routine check-up. We estimate how likely metabolic syndrome is, work out how severe it is, and match a diet, exercise and yoga plan to your result.',
  },
  {
    q: 'What information do I need?',
    a: 'Your age and sex, waist circumference, blood pressure, fasting plasma glucose, triglycerides and HDL cholesterol, plus whether you have been diagnosed with fatty liver, hypertension or diabetes. These are all on a standard blood test and check-up.',
  },
  {
    q: 'Do I need an account?',
    a: 'No — anyone can take an assessment and download the PDF report. Create a free account if you want to save your results and follow your progress on the dashboard.',
  },
  {
    q: 'Is this a medical diagnosis?',
    a: 'No. MetS Health is an educational tool that helps you understand your risk. Always discuss your results and any changes to your treatment with a qualified healthcare professional.',
  },
];
