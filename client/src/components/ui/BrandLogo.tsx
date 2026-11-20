/**
 * BrandLogo — heart + heartbeat mark with the "MetS Health" wordmark.
 */
import { Box, Typography } from '@mui/material';

interface BrandLogoProps {
  /** Mark size in px */
  size?: number;
  showWordmark?: boolean;
  showTagline?: boolean;
  /** Use white text (for dark/brand backgrounds) */
  inverted?: boolean;
}

export const BrandMark = ({ size = 36 }: { size?: number }) => (
  <Box
    component="span"
    aria-hidden
    sx={{
      width: size,
      height: size,
      flexShrink: 0,
      display: 'grid',
      placeItems: 'center',
      borderRadius: `${Math.round(size * 0.3)}px`,
      backgroundImage: 'var(--vb-gradient-brand)',
      boxShadow: 'var(--vb-shadow-brand)',
    }}
  >
    <svg viewBox="0 0 32 32" width={size * 0.72} height={size * 0.72}>
      <path
        fill="rgba(255,255,255,0.22)"
        d="M16 28.6l-1.9-1.7C7.2 20.7 2.7 16.6 2.7 11.6 2.7 7.5 5.9 4.3 10 4.3c2.3 0 4.5 1.1 6 2.8 1.5-1.7 3.7-2.8 6-2.8 4.1 0 7.3 3.2 7.3 7.3 0 5-4.5 9.1-11.4 15.3L16 28.6z"
      />
      <path
        fill="none"
        stroke="#fff"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4.5 15h6l2-4 3 8 2.5-6 1.5 2h8"
      />
    </svg>
  </Box>
);

const BrandLogo = ({ size = 36, showWordmark = true, showTagline = false, inverted = false }: BrandLogoProps) => (
  <Box component="span" sx={{ display: 'inline-flex', alignItems: 'center', gap: 1.25 }}>
    <BrandMark size={size} />
    {showWordmark && (
      <Box component="span" sx={{ display: 'flex', flexDirection: 'column', lineHeight: 1 }}>
        <Typography
          component="span"
          sx={{
            fontFamily: 'var(--vb-font-display)',
            fontWeight: 800,
            fontSize: size * 0.5,
            letterSpacing: '-0.02em',
            lineHeight: 1.1,
            color: inverted ? '#fff' : 'text.primary',
          }}
        >
          MetS
          <Box component="span" sx={{ color: inverted ? 'rgba(255,255,255,0.85)' : 'primary.main' }}>
            {' '}
            Health
          </Box>
        </Typography>
        {showTagline && (
          <Typography
            component="span"
            sx={{
              fontSize: Math.max(10, size * 0.28),
              fontWeight: 600,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: inverted ? 'rgba(255,255,255,0.7)' : 'text.secondary',
              mt: 0.25,
            }}
          >
            Metabolic risk, decoded
          </Typography>
        )}
      </Box>
    )}
  </Box>
);

export default BrandLogo;
