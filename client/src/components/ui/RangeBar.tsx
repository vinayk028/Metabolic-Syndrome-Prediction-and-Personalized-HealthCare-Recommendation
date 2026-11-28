/**
 * RangeBar — lab-report style reference range.
 * Zones are drawn as segments (in-range / borderline / above threshold) and a
 * marker shows where the value sits. Works for "lower is worse" markers (HDL)
 * because zones are passed explicitly, in ascending order.
 */
import { Box } from '@mui/material';
import { riskVar } from '../../theme/cssVars';
import type { RiskKey } from '../../theme/tokens';

export interface RangeZone {
  /** Upper bound of this zone (the last zone ends at `max`) */
  to: number;
  risk: RiskKey;
}

interface RangeBarProps {
  min: number;
  max: number;
  zones: RangeZone[];
  value: number;
  /** Accessible description, e.g. "Waist 104 cm, above threshold" */
  label: string;
  /** Bar thickness in px */
  thickness?: number;
}

const clampPct = (v: number, min: number, max: number) => Math.min(100, Math.max(0, ((v - min) / (max - min)) * 100));

const RangeBar = ({ min, max, zones, value, label, thickness = 8 }: RangeBarProps) => {
  const segments = zones.map((z, i) => ({
    from: clampPct(i === 0 ? min : zones[i - 1].to, min, max),
    to: clampPct(z.to, min, max),
    risk: z.risk,
  }));
  const markerRisk = zones.find((z) => value < z.to)?.risk ?? zones[zones.length - 1].risk;
  const pos = clampPct(value, min, max);

  return (
    <Box
      role="img"
      aria-label={label}
      sx={{ position: 'relative', height: thickness + 10, display: 'flex', alignItems: 'center' }}
    >
      <Box sx={{ position: 'relative', width: '100%', height: thickness, display: 'flex', gap: '3px' }}>
        {segments.map((s, i) => (
          <Box
            key={i}
            sx={{
              width: `${s.to - s.from}%`,
              height: '100%',
              borderRadius: 'var(--vb-radius-pill)',
              backgroundColor: riskVar(s.risk, 'soft'),
              boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${riskVar(s.risk)} 22%, transparent)`,
            }}
          />
        ))}
      </Box>
      <Box
        aria-hidden
        sx={{
          position: 'absolute',
          left: `${pos}%`,
          top: '50%',
          width: thickness + 8,
          height: thickness + 8,
          transform: 'translate(-50%, -50%)',
          borderRadius: '50%',
          backgroundColor: 'var(--vb-paper)',
          border: `3px solid ${riskVar(markerRisk)}`,
          boxShadow: 'var(--vb-shadow-sm)',
        }}
      />
    </Box>
  );
};

export default RangeBar;
