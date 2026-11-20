/**
 * PulseLoader — on-brand loading indicator (looping heartbeat).
 */
import { Box, Typography } from '@mui/material';
import PulseLine from './PulseLine';

interface PulseLoaderProps {
  label?: string;
  /** Fill the viewport height (route-level loading). */
  fullPage?: boolean;
  width?: number;
}

const PulseLoader = ({ label = 'Loading…', fullPage = false, width = 180 }: PulseLoaderProps) => (
  <Box
    role="status"
    aria-live="polite"
    sx={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: 1.5,
      py: 6,
      minHeight: fullPage ? '60vh' : undefined,
    }}
  >
    <Box sx={{ width }}>
      <PulseLine loop height={40} duration={1.6} color="var(--vb-primary)" />
    </Box>
    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
      {label}
    </Typography>
  </Box>
);

export default PulseLoader;
