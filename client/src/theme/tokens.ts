/**
 * Design Tokens — "Vital Balance"
 *
 * Single source of truth for the UI. The MUI theme (theme.ts) is built from
 * these values and MUI exposes them as CSS variables (`--mui-palette-*`), which
 * global.css re-maps to short app aliases (`--vb-*`) for plain CSS files.
 *
 * Contrast notes (WCAG AA on white):
 *   brand.teal[700] #0B7F74  → 4.9:1  (safe for button text / links)
 *   brand.coral[600] #D63B42 → 4.6:1  (safe for button text)
 *   brand.teal[500] / coral[400] are decorative only (illustrations, gradients).
 */

// ─── Brand palette ────────────────────────────────────────────────────────────
export const brand = {
  teal: {
    50: '#ECFDF9',
    100: '#CCF5EC',
    200: '#99E9DA',
    300: '#5DD6C4',
    400: '#2BC4B4', // primary on dark surfaces
    500: '#0F9D8F', // "Vital Teal" — brand / illustration colour
    600: '#0C8C80',
    700: '#0B7F74', // primary.main on light surfaces (AA)
    800: '#0A6359',
    900: '#084A43',
  },
  coral: {
    50: '#FFF1F1',
    100: '#FFE0E1',
    200: '#FFC2C4',
    300: '#FF9497',
    400: '#FF5A5F', // "Pulse Coral" — highlights, heartbeat line
    500: '#EF4A50',
    600: '#D63B42', // accent.main (AA)
    700: '#B32A31',
  },
  slate: {
    50: '#F8FAFC',
    100: '#F1F5F9',
    200: '#E2E8F0',
    300: '#CBD5E1',
    400: '#94A3B8',
    500: '#64748B',
    600: '#475569',
    700: '#334155',
    800: '#1E293B',
    900: '#0F172A', // "Deep Slate"
    950: '#0B1220',
  },
} as const;

// ─── The 5 metabolic-syndrome risk factors ───────────────────────────────────
// Identity colours: used for icons, chart series, step headers and tags.
// They never encode risk status — that is what `risk` below is for.
export type FactorKey = 'waist' | 'bloodPressure' | 'glucose' | 'triglycerides' | 'hdl';

export const factorColors: Record<FactorKey, { main: string; soft: string; softDark: string }> = {
  waist: { main: '#6366F1', soft: '#EEF2FF', softDark: 'rgba(99,102,241,0.16)' },
  bloodPressure: { main: '#E11D48', soft: '#FFF1F2', softDark: 'rgba(225,29,72,0.16)' },
  glucose: { main: '#0284C7', soft: '#F0F9FF', softDark: 'rgba(2,132,199,0.18)' },
  triglycerides: { main: '#EA580C', soft: '#FFF7ED', softDark: 'rgba(234,88,12,0.16)' },
  hdl: { main: '#9333EA', soft: '#FAF5FF', softDark: 'rgba(147,51,234,0.16)' },
};

// ─── Risk status (always paired with a text label, never colour alone) ───────
export type RiskKey = 'low' | 'medium' | 'high';

export const riskColors: Record<
  RiskKey,
  { main: string; light: string; soft: string; softDark: string; contrastText: string }
> = {
  low: { main: '#15803D', light: '#4ADE80', soft: '#F0FDF4', softDark: 'rgba(34,197,94,0.14)', contrastText: '#FFFFFF' },
  medium: { main: '#B45309', light: '#FBBF24', soft: '#FFFBEB', softDark: 'rgba(245,158,11,0.14)', contrastText: '#FFFFFF' },
  high: { main: '#B91C1C', light: '#F87171', soft: '#FEF2F2', softDark: 'rgba(239,68,68,0.14)', contrastText: '#FFFFFF' },
};

// ─── Typography ──────────────────────────────────────────────────────────────
export const fonts = {
  display: '"Plus Jakarta Sans", "Inter", system-ui, -apple-system, "Segoe UI", sans-serif',
  body: '"Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
} as const;

/** Fluid type scale — scales smoothly from 320px phones up to 2560px displays. */
export const fluidType = {
  display: 'clamp(2.5rem, 1.6rem + 4.2vw, 4.75rem)',
  h1: 'clamp(2.125rem, 1.55rem + 2.6vw, 3.5rem)',
  h2: 'clamp(1.75rem, 1.4rem + 1.6vw, 2.625rem)',
  h3: 'clamp(1.4rem, 1.2rem + 0.9vw, 1.875rem)',
  h4: 'clamp(1.2rem, 1.1rem + 0.5vw, 1.5rem)',
  h5: 'clamp(1.075rem, 1.03rem + 0.25vw, 1.25rem)',
  h6: '1rem',
  lead: 'clamp(1.05rem, 1rem + 0.35vw, 1.25rem)',
} as const;

// ─── Layout ──────────────────────────────────────────────────────────────────
export const breakpoints = { xs: 0, sm: 600, md: 900, lg: 1200, xl: 1536, xxl: 1920 } as const;

export const radius = { xs: 6, sm: 10, md: 14, lg: 20, xl: 28, pill: 999 } as const;

export const zIndex = {
  base: 0,
  raised: 10,
  sticky: 1100,
  navbar: 1200,
  bottomNav: 1250,
  chat: 1300,
  modal: 1400,
  toast: 1500,
} as const;

// ─── Motion ──────────────────────────────────────────────────────────────────
export const motion = {
  duration: { fast: 0.15, base: 0.25, slow: 0.45, slower: 0.7 },
  ease: {
    standard: [0.4, 0, 0.2, 1] as const,
    emphasized: [0.2, 0, 0, 1] as const,
    out: [0, 0, 0.2, 1] as const,
  },
  /** CSS string versions for plain CSS / MUI transitions */
  css: {
    standard: 'cubic-bezier(0.4, 0, 0.2, 1)',
    emphasized: 'cubic-bezier(0.2, 0, 0, 1)',
  },
} as const;

// ─── Gradients ───────────────────────────────────────────────────────────────
export const gradients = {
  brand: `linear-gradient(135deg, ${brand.teal[500]} 0%, ${brand.teal[700]} 100%)`,
  vital: `linear-gradient(120deg, ${brand.teal[500]} 0%, #0EA5A0 45%, ${brand.coral[400]} 100%)`,
  heroLight: `radial-gradient(1200px 600px at 85% -10%, ${brand.teal[100]} 0%, transparent 60%),
              radial-gradient(900px 500px at -10% 110%, ${brand.coral[50]} 0%, transparent 55%)`,
  heroDark: `radial-gradient(1200px 600px at 85% -10%, rgba(43,196,180,0.18) 0%, transparent 60%),
             radial-gradient(900px 500px at -10% 110%, rgba(255,90,95,0.12) 0%, transparent 55%)`,
} as const;
