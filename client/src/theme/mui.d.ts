/**
 * MUI module augmentation for the "Vital Balance" design system.
 */
import '@mui/material/styles';
import '@mui/material/Button';
import '@mui/material/Chip';
import '@mui/material/Typography';
import type { CSSProperties } from 'react';
import type { FactorKey, RiskKey } from './tokens';

interface SoftColor {
  main: string;
  soft: string;
}

interface RiskColor extends SoftColor {
  light: string;
  contrastText: string;
}

declare module '@mui/material/styles' {
  interface BreakpointOverrides {
    xxl: true;
  }

  interface Palette {
    accent: Palette['primary'];
    factor: Record<FactorKey, SoftColor>;
    risk: Record<RiskKey, RiskColor>;
    surface: { raised: string; sunken: string; glass: string; border: string };
  }
  interface PaletteOptions {
    accent?: PaletteOptions['primary'];
    factor?: Record<FactorKey, SoftColor>;
    risk?: Record<RiskKey, RiskColor>;
    surface?: { raised: string; sunken: string; glass: string; border: string };
  }

  interface TypographyVariants {
    display: CSSProperties;
    eyebrow: CSSProperties;
    lead: CSSProperties;
  }
  interface TypographyVariantsOptions {
    display?: CSSProperties;
    eyebrow?: CSSProperties;
    lead?: CSSProperties;
  }
}

declare module '@mui/material/Typography' {
  interface TypographyPropsVariantOverrides {
    display: true;
    eyebrow: true;
    lead: true;
  }
}

declare module '@mui/material/Button' {
  interface ButtonPropsVariantOverrides {
    gradient: true;
  }
  interface ButtonPropsColorOverrides {
    accent: true;
  }
}

declare module '@mui/material/Chip' {
  interface ChipPropsColorOverrides {
    accent: true;
  }
}
