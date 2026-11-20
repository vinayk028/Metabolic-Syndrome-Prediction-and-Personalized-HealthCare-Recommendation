/**
 * PageHeader — top-of-page title block for app pages (Dashboard, Profile …).
 * Stacks on mobile, puts actions on the right from `sm` up.
 */
import type { ReactNode } from 'react';
import { Box, Stack, Typography } from '@mui/material';

interface PageHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  eyebrow?: string;
  icon?: ReactNode;
  actions?: ReactNode;
}

const PageHeader = ({ title, subtitle, eyebrow, icon, actions }: PageHeaderProps) => (
  <Stack
    direction={{ xs: 'column', sm: 'row' }}
    spacing={{ xs: 2, sm: 3 }}
    sx={{ alignItems: { xs: 'flex-start', sm: 'flex-end' }, justifyContent: 'space-between', mb: { xs: 3, md: 5 } }}
  >
    <Stack direction="row" spacing={2} sx={{ alignItems: 'center', minWidth: 0 }}>
      {icon && (
        <Box
          aria-hidden
          sx={{
            flexShrink: 0,
            width: { xs: 44, md: 52 },
            height: { xs: 44, md: 52 },
            borderRadius: 'var(--vb-radius-md)',
            display: 'grid',
            placeItems: 'center',
            color: '#fff',
            backgroundImage: 'var(--vb-gradient-brand)',
            boxShadow: 'var(--vb-shadow-brand)',
            '& svg': { fontSize: { xs: 24, md: 28 } },
          }}
        >
          {icon}
        </Box>
      )}
      <Box sx={{ minWidth: 0 }}>
        {eyebrow && (
          <Typography variant="eyebrow" sx={{ color: 'primary.main', display: 'block', mb: 0.5 }}>
            {eyebrow}
          </Typography>
        )}
        <Typography variant="h2" component="h1" sx={{ color: 'text.primary' }}>
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body1" sx={{ color: 'text.secondary', mt: 0.75 }}>
            {subtitle}
          </Typography>
        )}
      </Box>
    </Stack>
    {actions && (
      <Stack direction="row" spacing={1.5} sx={{ flexShrink: 0, width: { xs: '100%', sm: 'auto' }, '& > *': { flex: { xs: 1, sm: 'none' } } }}>
        {actions}
      </Stack>
    )}
  </Stack>
);

export default PageHeader;
