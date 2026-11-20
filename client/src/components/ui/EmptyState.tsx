/**
 * EmptyState — friendly placeholder for "no data yet" screens
 * (no assessments, no news, no chat history …).
 */
import type { ReactNode } from 'react';
import { Box, Stack, Typography } from '@mui/material';
import PulseLine from './PulseLine';

interface EmptyStateProps {
  title: string;
  description?: ReactNode;
  /** Icon element; defaults to the heartbeat line */
  icon?: ReactNode;
  action?: ReactNode;
  compact?: boolean;
}

const EmptyState = ({ title, description, icon, action, compact = false }: EmptyStateProps) => (
  <Stack
    sx={{
      alignItems: 'center',
      textAlign: 'center',
      py: compact ? 4 : { xs: 6, md: 10 },
      px: 2,
      gap: 2,
      border: '1.5px dashed var(--vb-border)',
      borderRadius: 'var(--vb-radius-lg)',
      backgroundColor: 'var(--vb-surface-sunken)',
    }}
  >
    {icon ? (
      <Box
        aria-hidden
        sx={{
          width: 64,
          height: 64,
          borderRadius: '50%',
          display: 'grid',
          placeItems: 'center',
          color: 'primary.main',
          backgroundColor: 'rgba(var(--mui-palette-primary-mainChannel) / 0.1)',
          '& svg': { fontSize: 32 },
        }}
      >
        {icon}
      </Box>
    ) : (
      <Box sx={{ width: 160 }}>
        <PulseLine height={40} color="var(--vb-primary)" />
      </Box>
    )}
    <Typography variant="h5" component="h3" sx={{ color: 'text.primary' }}>
      {title}
    </Typography>
    {description && (
      <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 420 }}>
        {description}
      </Typography>
    )}
    {action && <Box sx={{ mt: 1 }}>{action}</Box>}
  </Stack>
);

export default EmptyState;
