/**
 * StatTile — compact metric card (value + label + optional trend).
 * Pass `factor` to theme it with a risk-factor identity colour.
 */
import type { ReactNode } from 'react';
import { Box, Skeleton, Stack, Typography } from '@mui/material';
import ArrowDownwardRoundedIcon from '@mui/icons-material/ArrowDownwardRounded';
import ArrowUpwardRoundedIcon from '@mui/icons-material/ArrowUpwardRounded';
import { FACTORS } from '../../theme/factors';
import { factorVar, riskVar } from '../../theme/cssVars';
import type { FactorKey } from '../../theme/tokens';
import GlassCard from './GlassCard';

interface StatTileProps {
  label: string;
  value: ReactNode;
  unit?: string;
  caption?: ReactNode;
  icon?: ReactNode;
  factor?: FactorKey;
  /** Change since last reading. `goodWhen` decides green vs red. */
  trend?: { delta: number; goodWhen: 'up' | 'down'; suffix?: string };
  loading?: boolean;
  onClick?: () => void;
}

const StatTile = ({ label, value, unit, caption, icon, factor, trend, loading, onClick }: StatTileProps) => {
  const FactorIcon = factor ? FACTORS[factor].icon : null;
  const tint = factor ? factorVar(factor) : 'var(--vb-primary)';
  const tintSoft = factor ? factorVar(factor, 'soft') : 'rgba(var(--mui-palette-primary-mainChannel) / 0.1)';

  const trendGood = trend ? (trend.delta === 0 ? null : (trend.delta < 0) === (trend.goodWhen === 'down')) : null;

  return (
    <GlassCard
      interactive={!!onClick}
      onClick={onClick}
      tabIndex={onClick ? 0 : undefined}
      padding={{ xs: 2, md: 2.5 }}
      sx={{ height: '100%' }}
    >
      <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 1.5 }}>
        <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600 }}>
          {label}
        </Typography>
        {(icon || FactorIcon) && (
          <Box
            aria-hidden
            sx={{
              width: 36,
              height: 36,
              flexShrink: 0,
              borderRadius: 'var(--vb-radius-sm)',
              display: 'grid',
              placeItems: 'center',
              color: tint,
              backgroundColor: tintSoft,
              '& svg': { fontSize: 20 },
            }}
          >
            {icon ?? (FactorIcon && <FactorIcon />)}
          </Box>
        )}
      </Stack>

      {loading ? (
        <Skeleton variant="text" width="60%" sx={{ fontSize: '2rem', mt: 1 }} />
      ) : (
        <Stack direction="row" sx={{ alignItems: 'baseline', gap: 0.75, mt: 1, flexWrap: 'wrap' }}>
          <Typography
            component="span"
            sx={{
              fontFamily: 'var(--vb-font-display)',
              fontWeight: 800,
              fontSize: 'clamp(1.5rem, 1.25rem + 1vw, 2rem)',
              lineHeight: 1.1,
              color: 'text.primary',
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {value}
          </Typography>
          {unit && (
            <Typography component="span" variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
              {unit}
            </Typography>
          )}
        </Stack>
      )}

      {(trend || caption) && (
        <Stack direction="row" sx={{ alignItems: 'center', gap: 1, mt: 1, flexWrap: 'wrap' }}>
          {trend && trend.delta !== 0 && (
            <Box
              component="span"
              sx={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 0.25,
                fontSize: '0.8125rem',
                fontWeight: 700,
                color: trendGood ? riskVar('low') : riskVar('high'),
              }}
            >
              {trend.delta > 0 ? <ArrowUpwardRoundedIcon sx={{ fontSize: 16 }} /> : <ArrowDownwardRoundedIcon sx={{ fontSize: 16 }} />}
              {Math.abs(trend.delta)}
              {trend.suffix}
              <span className="visually-hidden">{trendGood ? '(improved)' : '(worsened)'}</span>
            </Box>
          )}
          {caption && (
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              {caption}
            </Typography>
          )}
        </Stack>
      )}
    </GlassCard>
  );
};

export default StatTile;
