/**
 * MetabolicRing — the signature visual. A 5-segment ring, one segment per
 * metabolic-syndrome factor.
 *
 *  colorBy="status" (dashboard): segment colour = that factor's status
 *                                (in range / borderline / above threshold)
 *  colorBy="factor" (landing, illustrations): segment colour = factor identity colour
 */
import type { ReactNode } from 'react';
import { Box, Stack, Typography } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';
import { FACTORS, FACTOR_ORDER, STATUS_TO_RISK, type FactorStatus } from '../../theme/factors';
import { factorVar, riskVar } from '../../theme/cssVars';
import type { FactorKey } from '../../theme/tokens';
import FactorBadge from './FactorBadge';
import RiskBadge from './RiskBadge';

interface MetabolicRingProps {
  /** Status per factor; missing / null = not measured yet */
  factors?: Partial<Record<FactorKey, FactorStatus | null>>;
  colorBy?: 'status' | 'factor';
  /** Rendered diameter in px (the SVG scales to the container up to this). */
  size?: number;
  thickness?: number;
  /** Content placed in the centre (score, label …). Defaults to "n / 5 in range". */
  children?: ReactNode;
  showLegend?: boolean;
  /** Animate segments drawing in */
  animate?: boolean;
  /** Highlight one factor (others dim) — used by interactive explainers */
  activeFactor?: FactorKey | null;
  /** Makes segments clickable */
  onFactorSelect?: (factor: FactorKey) => void;
}

const STATUS_LABEL: Record<FactorStatus, string> = {
  normal: 'In range',
  borderline: 'Borderline',
  elevated: 'Above threshold',
};

const VB = 220;
const C = VB / 2;
const SEGMENT = 360 / FACTOR_ORDER.length;
const GAP = 16; // degrees between segments (accounts for round caps)

const polar = (r: number, deg: number) => {
  const rad = (deg * Math.PI) / 180;
  return { x: C + r * Math.cos(rad), y: C + r * Math.sin(rad) };
};

const arcPath = (r: number, startDeg: number, endDeg: number) => {
  const s = polar(r, startDeg);
  const e = polar(r, endDeg);
  const large = endDeg - startDeg > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
};

const MetabolicRing = ({
  factors = {},
  colorBy = 'status',
  size = 260,
  thickness = 18,
  children,
  showLegend = false,
  animate = true,
  activeFactor = null,
  onFactorSelect,
}: MetabolicRingProps) => {
  const reduce = useReducedMotion();
  const r = C - thickness / 2 - 4;
  const inRange = FACTOR_ORDER.filter((k) => factors[k] === 'normal').length;
  const measured = FACTOR_ORDER.filter((k) => factors[k]).length;

  const summary = FACTOR_ORDER.map((k) => `${FACTORS[k].label}: ${factors[k] ? STATUS_LABEL[factors[k]!] : 'not measured'}`).join(
    ', ',
  );

  const segmentColor = (k: FactorKey) => {
    if (colorBy === 'factor') return factorVar(k);
    const status = factors[k];
    return status ? riskVar(STATUS_TO_RISK[status]) : 'var(--vb-border)';
  };

  return (
    <Stack
      direction={{ xs: 'column', md: showLegend ? 'row' : 'column' }}
      spacing={{ xs: 3, md: 5 }}
      sx={{ alignItems: 'center' }}
    >
      <Box sx={{ position: 'relative', width: '100%', maxWidth: size, aspectRatio: '1 / 1', flexShrink: 0 }}>
        <svg viewBox={`0 0 ${VB} ${VB}`} width="100%" height="100%" role="img" aria-label={`Metabolic ring — ${summary}`}>
          {FACTOR_ORDER.map((k, i) => {
            const start = -90 + i * SEGMENT + GAP / 2;
            const end = start + SEGMENT - GAP;
            const d = arcPath(r, start, end);
            const dimmed = activeFactor !== null && activeFactor !== k;
            const isActive = activeFactor === k;
            return (
              <g
                key={k}
                onClick={onFactorSelect ? () => onFactorSelect(k) : undefined}
                style={{
                  cursor: onFactorSelect ? 'pointer' : undefined,
                  opacity: dimmed ? 0.22 : 1,
                  transition: 'opacity 250ms cubic-bezier(0.4,0,0.2,1)',
                }}
              >
                <path d={d} fill="none" strokeWidth={thickness} strokeLinecap="round" style={{ stroke: 'var(--vb-surface-sunken)' }} />
                <motion.path
                  d={d}
                  fill="none"
                  strokeWidth={isActive ? thickness + 6 : thickness}
                  strokeLinecap="round"
                  style={{ stroke: segmentColor(k), transition: 'stroke-width 250ms cubic-bezier(0.4,0,0.2,1)' }}
                  initial={animate && !reduce ? { pathLength: 0 } : false}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.8, delay: 0.15 + i * 0.12, ease: [0.2, 0, 0, 1] }}
                >
                  <title>{`${FACTORS[k].label}${
                    colorBy === 'status' ? ` — ${factors[k] ? STATUS_LABEL[factors[k]!] : 'not measured'}` : ''
                  }`}</title>
                </motion.path>
              </g>
            );
          })}
        </svg>

        <Box
          sx={{
            position: 'absolute',
            inset: `${(thickness / VB) * 100 + 10}%`,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
          }}
        >
          {children ?? (
            <>
              <Typography
                sx={{ fontFamily: 'var(--vb-font-display)', fontWeight: 800, fontSize: 'clamp(1.75rem, 1.2rem + 2vw, 2.5rem)', lineHeight: 1, color: 'text.primary' }}
              >
                {inRange}
                <Box component="span" sx={{ color: 'text.disabled', fontWeight: 600 }}>
                  /5
                </Box>
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
                {measured ? 'factors in range' : 'not assessed yet'}
              </Typography>
            </>
          )}
        </Box>
      </Box>

      {showLegend && (
        <Stack component="ul" spacing={1.25} sx={{ listStyle: 'none', p: 0, m: 0, width: '100%', maxWidth: 360 }}>
          {FACTOR_ORDER.map((k) => {
            const status = factors[k];
            return (
              <Stack
                component="li"
                key={k}
                direction="row"
                sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 2 }}
              >
                <Stack direction="row" spacing={1.5} sx={{ alignItems: 'center', minWidth: 0 }}>
                  <FactorBadge factor={k} variant="icon" size="small" />
                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }} noWrap>
                    {FACTORS[k].label}
                  </Typography>
                </Stack>
                {status ? (
                  <RiskBadge level={STATUS_TO_RISK[status]} label={STATUS_LABEL[status]} size="small" />
                ) : (
                  <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                    —
                  </Typography>
                )}
              </Stack>
            );
          })}
        </Stack>
      )}
    </Stack>
  );
};

export default MetabolicRing;
