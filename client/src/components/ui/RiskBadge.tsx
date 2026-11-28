/**
 * RiskBadge — status pill for Low / Moderate / High risk.
 * Always shows a text label + icon (never colour alone).
 * Accepts either a token key or the backend string ("Medium Severity").
 */
import { Box } from '@mui/material';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import ErrorOutlineRoundedIcon from '@mui/icons-material/ErrorOutlineRounded';
import WarningAmberRoundedIcon from '@mui/icons-material/WarningAmberRounded';
import { RISK_LABEL, toRiskKey } from '../../theme/factors';
import { riskVar } from '../../theme/cssVars';
import type { RiskKey } from '../../theme/tokens';

interface RiskBadgeProps {
  /** 'low' | 'medium' | 'high' or a backend riskLevel string */
  level: RiskKey | string | null | undefined;
  label?: string;
  size?: 'small' | 'medium' | 'large';
  /** solid = filled colour, soft = tinted background */
  tone?: 'soft' | 'solid';
}

const ICONS: Record<RiskKey, typeof CheckCircleRoundedIcon> = {
  low: CheckCircleRoundedIcon,
  medium: WarningAmberRoundedIcon,
  high: ErrorOutlineRoundedIcon,
};

const SIZE = {
  small: { px: 1, py: 0.25, font: '0.75rem', icon: 14 },
  medium: { px: 1.5, py: 0.5, font: '0.8125rem', icon: 16 },
  large: { px: 2, py: 0.75, font: '0.9375rem', icon: 20 },
};

const RiskBadge = ({ level, label, size = 'medium', tone = 'soft' }: RiskBadgeProps) => {
  const key: RiskKey | null =
    level === 'low' || level === 'medium' || level === 'high' ? level : toRiskKey(level as string | undefined);
  if (!key) return null;

  const Icon = ICONS[key];
  const s = SIZE[size];

  return (
    <Box
      component="span"
      sx={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.75,
        px: s.px,
        py: s.py,
        borderRadius: 'var(--vb-radius-pill)',
        fontSize: s.font,
        fontWeight: 700,
        fontFamily: 'var(--vb-font-display)',
        lineHeight: 1.4,
        whiteSpace: 'nowrap',
        ...(tone === 'solid'
          ? { color: riskVar(key, 'contrastText'), backgroundColor: riskVar(key) }
          : {
              color: riskVar(key),
              backgroundColor: riskVar(key, 'soft'),
              boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${riskVar(key)} 25%, transparent)`,
            }),
      }}
    >
      <Icon sx={{ fontSize: s.icon }} aria-hidden />
      {label ?? RISK_LABEL[key]}
    </Box>
  );
};

export default RiskBadge;
