/**
 * SectionHeader — eyebrow + title + supporting copy, used to open every
 * landing / page section consistently.
 */
import type { ReactNode } from 'react';
import { Box, Typography, type SxProps, type Theme } from '@mui/material';

interface SectionHeaderProps {
  eyebrow?: string;
  title: ReactNode;
  subtitle?: ReactNode;
  align?: 'left' | 'center';
  /** Heading level for the title (keeps document outline correct). */
  as?: 'h1' | 'h2' | 'h3';
  /** Eyebrow colour; defaults to primary. Pass a factor/risk var for themed sections. */
  accentColor?: string;
  maxWidth?: number | string;
  sx?: SxProps<Theme>;
}

const SectionHeader = ({
  eyebrow,
  title,
  subtitle,
  align = 'left',
  as = 'h2',
  accentColor = 'var(--vb-primary)',
  maxWidth = 720,
  sx,
}: SectionHeaderProps) => (
  <Box
    sx={[
      {
        textAlign: align,
        maxWidth,
        mx: align === 'center' ? 'auto' : 0,
        mb: { xs: 4, md: 6 },
      },
      ...(Array.isArray(sx) ? sx : [sx]),
    ]}
  >
    {eyebrow && (
      <Typography
        variant="eyebrow"
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 1,
          color: accentColor,
          mb: 1.5,
          '&::before': {
            content: '""',
            width: 18,
            height: 2,
            borderRadius: 2,
            backgroundColor: 'currentColor',
          },
        }}
      >
        {eyebrow}
      </Typography>
    )}
    <Typography variant={as === 'h1' ? 'h1' : as === 'h3' ? 'h3' : 'h2'} component={as} sx={{ color: 'text.primary' }}>
      {title}
    </Typography>
    {subtitle && (
      <Typography variant="lead" sx={{ color: 'text.secondary', mt: 2 }}>
        {subtitle}
      </Typography>
    )}
  </Box>
);

export default SectionHeader;
