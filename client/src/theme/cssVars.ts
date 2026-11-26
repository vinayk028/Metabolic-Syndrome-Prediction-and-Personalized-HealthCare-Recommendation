/**
 * Typed helpers that return CSS-variable references for theme colours.
 * They follow the active colour scheme automatically, so components never need
 * to branch on light/dark themselves.
 */
import type { FactorKey, RiskKey } from './tokens';

export const factorVar = (key: FactorKey, tone: 'main' | 'soft' = 'main') =>
  `var(--mui-palette-factor-${key}-${tone})`;

export const riskVar = (key: RiskKey, tone: 'main' | 'light' | 'soft' | 'contrastText' = 'main') =>
  `var(--mui-palette-risk-${key}-${tone})`;
