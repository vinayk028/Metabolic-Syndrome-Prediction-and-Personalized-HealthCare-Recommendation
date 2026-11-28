/**
 * FactorBadge — icon + label for one of the 5 metabolic risk factors,
 * always in that factor's identity colour.
 */
import { Box, Tooltip, Typography } from '@mui/material';
import { FACTORS } from '../../theme/factors';
import { factorVar } from '../../theme/cssVars';
import type { FactorKey } from '../../theme/tokens';

interface FactorBadgeProps {
  factor: FactorKey;
  /** "chip" = pill with label, "icon" = rounded icon tile only, "inline" = icon + text, no background */
  variant?: 'chip' | 'icon' | 'inline';
  size?: 'small' | 'medium' | 'large';
  /** Use the short label (BP, TG, HDL…) */
  short?: boolean;
  /** Show threshold tooltip on hover/focus */
  showThreshold?: boolean;
}

const SIZES = {
  small: { tile: 28, icon: 16, font: '0.75rem' },
  medium: { tile: 40, icon: 22, font: '0.8125rem' },
  large: { tile: 56, icon: 30, font: '0.875rem' },
};

const FactorBadge = ({ factor, variant = 'chip', size = 'medium', short = false, showThreshold = false }: FactorBadgeProps) => {
  const meta = FACTORS[factor];
  const Icon = meta.icon;
  const s = SIZES[size];
  const label = short ? meta.shortLabel : meta.label;

  const content =
    variant === 'icon' ? (
      <Box
        role="img"
        aria-label={meta.label}
        tabIndex={showThreshold ? 0 : undefined}
        sx={{
          width: s.tile,
          height: s.tile,
          borderRadius: 'var(--vb-radius-sm)',
          display: 'grid',
          placeItems: 'center',
          color: factorVar(factor),
          backgroundColor: factorVar(factor, 'soft'),
          flexShrink: 0,
        }}
      >
        <Icon sx={{ fontSize: s.icon }} />
      </Box>
    ) : (
      <Box
        component="span"
        tabIndex={showThreshold ? 0 : undefined}
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 0.75,
          color: factorVar(factor),
          ...(variant === 'chip' && {
            px: 1.25,
            py: 0.5,
            borderRadius: 'var(--vb-radius-pill)',
            backgroundColor: factorVar(factor, 'soft'),
          }),
        }}
      >
        <Icon sx={{ fontSize: s.icon * 0.8 }} aria-hidden />
        <Typography component="span" sx={{ fontSize: s.font, fontWeight: 650, color: 'inherit', lineHeight: 1.2 }}>
          {label}
        </Typography>
      </Box>
    );

  if (!showThreshold) return content;

  return (
    <Tooltip
      title={
        <>
          <strong>{meta.label}</strong>
          <br />
          Risk threshold: {meta.threshold}
        </>
      }
    >
      {content}
    </Tooltip>
  );
};

export default FactorBadge;
