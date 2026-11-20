/**
 * GlassCard — the base surface. Solid by default; `glass` adds a translucent
 * blurred background (for use over gradients / imagery). `interactive` adds a
 * hover lift for clickable cards.
 */
import { forwardRef, type ReactNode } from 'react';
import { Card, type CardProps } from '@mui/material';

interface GlassCardProps extends Omit<CardProps, 'variant'> {
  children?: ReactNode;
  glass?: boolean;
  interactive?: boolean;
  /** Optional top accent bar colour (e.g. a factor var). */
  accent?: string;
  padding?: number | { xs?: number; sm?: number; md?: number; lg?: number };
}

const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
  ({ children, glass = false, interactive = false, accent, padding = { xs: 2.5, md: 3.5 }, sx, ...rest }, ref) => (
    <Card
      ref={ref}
      sx={[
        {
          position: 'relative',
          overflow: 'hidden',
          p: padding,
          transition:
            'transform var(--vb-dur-base) var(--vb-ease), box-shadow var(--vb-dur-base) var(--vb-ease), border-color var(--vb-dur-base) var(--vb-ease)',
        },
        glass && {
          backgroundColor: 'var(--vb-glass)',
          backdropFilter: 'saturate(180%) blur(16px)',
          WebkitBackdropFilter: 'saturate(180%) blur(16px)',
        },
        interactive && {
          cursor: 'pointer',
          '&:hover': { transform: 'translateY(-4px)', boxShadow: 'var(--vb-shadow-md)', borderColor: 'transparent' },
          '&:focus-visible': { boxShadow: 'var(--vb-ring-focus)' },
        },
        !!accent && {
          '&::before': {
            content: '""',
            position: 'absolute',
            inset: '0 0 auto 0',
            height: 4,
            backgroundColor: accent,
          },
        },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
      {...rest}
    >
      {children}
    </Card>
  ),
);

GlassCard.displayName = 'GlassCard';

export default GlassCard;
