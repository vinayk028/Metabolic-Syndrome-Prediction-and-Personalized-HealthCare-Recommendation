/**
 * RiskGauge — semicircular gauge for MetS probability (0–1).
 * Zones: low < 40 % ≤ moderate < 65 % ≤ high  (65 % matches the backend
 * METS_THRESHOLD). Pass `risk` to override the derived zone with the
 * backend's severity level.
 */
import { useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from 'framer-motion';
import { riskVar } from '../../theme/cssVars';
import { toRiskKey } from '../../theme/factors';
import type { RiskKey } from '../../theme/tokens';
import RiskBadge from './RiskBadge';

interface RiskGaugeProps {
  /** Probability between 0 and 1 */
  value: number;
  /** Override zone (token key or backend string, e.g. "High Severity") */
  risk?: RiskKey | string | null;
  label?: string;
  size?: number;
  showBadge?: boolean;
}

const ZONES: { key: RiskKey; from: number; to: number }[] = [
  { key: 'low', from: 0, to: 0.4 },
  { key: 'medium', from: 0.4, to: 0.65 },
  { key: 'high', from: 0.65, to: 1 },
];

const W = 240;
const CX = W / 2;
const CY = 128;
const R = 100;
const STROKE = 18;

const point = (t: number, r = R) => {
  const a = Math.PI + Math.PI * t; // 180° → 360°
  return { x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) };
};

const arc = (from: number, to: number) => {
  const s = point(from);
  const e = point(to);
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${R} ${R} 0 0 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
};

const zoneFor = (v: number): RiskKey => (v >= 0.65 ? 'high' : v >= 0.4 ? 'medium' : 'low');

const RiskGauge = ({ value, risk, label = 'MetS probability', size = 320, showBadge = true }: RiskGaugeProps) => {
  const reduce = useReducedMotion();
  const v = Math.min(1, Math.max(0, value || 0));
  const key: RiskKey =
    (risk === 'low' || risk === 'medium' || risk === 'high' ? risk : toRiskKey(risk as string | undefined)) ?? zoneFor(v);
  const color = riskVar(key);

  const progress = useMotionValue(reduce ? v : 0);
  const percent = useTransform(progress, (p) => `${Math.round(p * 100)}`);
  const dotX = useTransform(progress, (p) => point(p).x);
  const dotY = useTransform(progress, (p) => point(p).y);

  useEffect(() => {
    if (reduce) {
      progress.set(v);
      return;
    }
    const controls = animate(progress, v, { duration: 1.2, ease: [0.2, 0, 0, 1] });
    return () => controls.stop();
  }, [v, reduce, progress]);

  return (
    <Box sx={{ width: '100%', maxWidth: size, mx: 'auto', textAlign: 'center' }}>
      <Box sx={{ position: 'relative' }}>
        <svg
          viewBox={`0 0 ${W} ${CY + STROKE}`}
          width="100%"
          role="img"
          aria-label={`${label}: ${Math.round(v * 100)} percent`}
        >
          {ZONES.map((z) => (
            <path
              key={z.key}
              d={arc(z.from + (z.from ? 0.006 : 0), z.to - (z.to < 1 ? 0.006 : 0))}
              fill="none"
              strokeWidth={STROKE}
              style={{ stroke: riskVar(z.key, 'soft') }}
            />
          ))}
          <motion.path
            d={arc(0, 1)}
            fill="none"
            strokeWidth={STROKE}
            strokeLinecap="round"
            style={{ stroke: color, pathLength: progress }}
          />
          <motion.circle
            r={STROKE / 2 + 4}
            style={{ cx: dotX, cy: dotY, fill: 'var(--vb-surface)', stroke: color }}
            strokeWidth={4}
          />
        </svg>

        <Box sx={{ position: 'absolute', left: 0, right: 0, bottom: 4 }}>
          <Typography
            component="div"
            sx={{
              fontFamily: 'var(--vb-font-display)',
              fontWeight: 800,
              fontSize: 'clamp(2.25rem, 1.6rem + 2.5vw, 3.25rem)',
              lineHeight: 1,
              color: 'text.primary',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            <motion.span>{percent}</motion.span>
            <Box component="span" sx={{ fontSize: '0.5em', color: 'text.secondary', ml: 0.25 }}>
              %
            </Box>
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {label}
          </Typography>
        </Box>
      </Box>
      {showBadge && (
        <Box sx={{ mt: 2 }}>
          <RiskBadge level={key} size="large" />
        </Box>
      )}
    </Box>
  );
};

export default RiskGauge;
