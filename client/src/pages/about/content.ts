/**
 * About page copy — edit wording here without touching layout.
 */
import type { SvgIconComponent } from '@mui/icons-material';
import BloodtypeRoundedIcon from '@mui/icons-material/BloodtypeRounded';
import StraightenRoundedIcon from '@mui/icons-material/StraightenRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import BadgeRoundedIcon from '@mui/icons-material/BadgeRounded';
import PersonOffRoundedIcon from '@mui/icons-material/PersonOffRounded';
import CloudDoneRoundedIcon from '@mui/icons-material/CloudDoneRounded';
import DeleteSweepRoundedIcon from '@mui/icons-material/DeleteSweepRounded';
import type { FactorKey } from '../../theme/tokens';

export const RISK_FACTORS = [
  'Age (risk increases with age)',
  'Obesity, particularly around the waist',
  'Physical inactivity',
  'Insulin resistance',
  'Genetics and family history',
  'Hormonal imbalances',
  'Diet high in processed foods',
  'Chronic stress',
];

export const PREVENTION = [
  'Regular physical activity (30 min a day)',
  'A diet rich in fruit, vegetables and whole grains',
  'Weight loss, if overweight',
  'Stopping smoking',
  'Limiting alcohol',
  'Regular health check-ups',
  'Managing stress',
  'Enough sleep (7–9 hours)',
];

/* ── Getting your numbers ───────────────────────────────────────────────── */
export interface NumberGuide {
  icon: SvgIconComponent;
  /** Factor colour for the icon tile (optional) */
  factor?: FactorKey;
  title: string;
  gives: string;
  tips: string[];
}

export const NUMBER_GUIDES: NumberGuide[] = [
  {
    icon: BloodtypeRoundedIcon,
    factor: 'glucose',
    title: 'Fasting blood test',
    gives: 'HDL cholesterol, triglycerides and fasting glucose',
    tips: [
      'Ask for a lipid panel and a fasting glucose test — one blood draw covers both.',
      'Fast for 8–12 hours beforehand (water is fine).',
      'Use your most recent results, ideally from the last few months.',
    ],
  },
  {
    icon: StraightenRoundedIcon,
    factor: 'waist',
    title: 'Waist circumference',
    gives: 'Your waist measurement in cm',
    tips: [
      'Stand up straight and measure at the level of your belly button.',
      'Keep the tape snug but not tight, and level all the way round.',
      'Breathe out normally, then read the tape.',
    ],
  },
  {
    icon: MonitorHeartRoundedIcon,
    factor: 'bloodPressure',
    title: 'Blood pressure',
    gives: 'Systolic (top) and diastolic (bottom) numbers',
    tips: [
      'Sit quietly for 5 minutes with your back supported and feet flat.',
      'Rest your arm at heart level; avoid caffeine and exercise for 30 minutes before.',
      'Take two readings a minute apart and use the average.',
    ],
  },
  {
    icon: BadgeRoundedIcon,
    title: 'About you',
    gives: 'Age, sex and existing diagnoses',
    tips: [
      'Have any diagnosis of fatty liver, hypertension or diabetes to hand.',
      'Thresholds for waist and HDL differ by sex, so choose the right one.',
    ],
  },
];

/* ── Your data & privacy ────────────────────────────────────────────────── */
export const PRIVACY_POINTS: { icon: SvgIconComponent; title: string; text: string }[] = [
  {
    icon: PersonOffRoundedIcon,
    title: 'Guests: nothing is saved',
    text: 'Without an account, your results stay on screen and are gone when you leave. Download the PDF if you want to keep them.',
  },
  {
    icon: CloudDoneRoundedIcon,
    title: 'With an account: your latest 7',
    text: 'Your most recent seven assessments are saved so you can follow your progress on the dashboard.',
  },
  {
    icon: DeleteSweepRoundedIcon,
    title: 'Delete any time',
    text: 'You can delete your account and all saved assessments from Profile → Security whenever you like.',
  },
];

/* ── FAQ (moved here from the landing page) ─────────────────────────────── */
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
