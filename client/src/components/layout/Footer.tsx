/**
 * Footer — brand + link columns + medical disclaimer.
 */
import { Link } from 'react-router-dom';
import { Box, Container, Grid, IconButton, Stack, Tooltip, Typography } from '@mui/material';
import GitHubIcon from '@mui/icons-material/GitHub';
import { useAuthStore } from '../../stores';
import BrandLogo from '../ui/BrandLogo';
import PulseLine from '../ui/PulseLine';
import { FACTORS, FACTOR_ORDER } from '../../theme/factors';
import { factorVar } from '../../theme/cssVars';
import { REPO_URL } from './navConfig';
import './Footer.css';

interface FooterLink {
  label: string;
  to: string;
}

const Footer = () => {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);

  const columns: { title: string; links: FooterLink[] }[] = [
    {
      title: 'Product',
      links: [
        { label: 'Risk assessment', to: '/assessment' },
        { label: 'Dashboard', to: '/dashboard' },
        { label: 'Health resources', to: '/resources' },
      ],
    },
    {
      title: 'Learn',
      links: [
        { label: 'About MetS Health', to: '/about' },
        { label: 'Frequently asked questions', to: '/#faq' },
      ],
    },
    {
      title: 'Account',
      links: isAuthenticated
        ? [
            { label: 'Profile', to: '/profile' },
            { label: 'My dashboard', to: '/dashboard' },
          ]
        : [
            { label: 'Sign in', to: '/login' },
            { label: 'Create account', to: '/signup' },
          ],
    },
  ];

  return (
    <Box component="footer" className="vb-footer">
      <PulseLine
        height={28}
        strokeWidth={2}
        color="var(--vb-primary)"
        sx={{ position: 'absolute', top: -14, left: 0, right: 0, opacity: 0.35, pointerEvents: 'none' }}
      />

      <Container maxWidth="xl">
        <Grid container spacing={{ xs: 4, md: 6 }} className="vb-footer__top">
          <Grid size={{ xs: 12, md: 4 }}>
            <BrandLogo size={36} showTagline />
            <Typography variant="body2" className="vb-footer__blurb">
              Metabolic syndrome risk prediction and personalised lifestyle plans, built around five key health
              markers.
            </Typography>
            <Stack direction="row" spacing={1} className="vb-footer__factors" aria-label="The five risk factors">
              {FACTOR_ORDER.map((k) => {
                const Icon = FACTORS[k].icon;
                return (
                  <Tooltip key={k} title={FACTORS[k].label}>
                    <Box
                      className="vb-footer__factor"
                      role="img"
                      aria-label={FACTORS[k].label}
                      sx={{ color: factorVar(k), backgroundColor: factorVar(k, 'soft') }}
                    >
                      <Icon />
                    </Box>
                  </Tooltip>
                );
              })}
            </Stack>
          </Grid>

          {columns.map((col) => (
            <Grid key={col.title} size={{ xs: 6, sm: 4, md: 'grow' }}>
              <Typography component="h2" className="vb-footer__heading">
                {col.title}
              </Typography>
              <ul className="vb-footer__links">
                {col.links.map((l) => (
                  <li key={l.label}>
                    <Link to={l.to}>{l.label}</Link>
                  </li>
                ))}
              </ul>
            </Grid>
          ))}
        </Grid>

        <Box className="vb-footer__bottom">
          <Typography variant="caption" className="vb-footer__disclaimer">
            For educational purposes only — not a substitute for professional medical advice, diagnosis or treatment.
          </Typography>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <Typography variant="caption" className="vb-footer__copy">
              © {new Date().getFullYear()} MetS Health
            </Typography>
            <Tooltip title="Source on GitHub">
              <IconButton component="a" href={REPO_URL} target="_blank" rel="noopener noreferrer" aria-label="Source on GitHub" size="small">
                <GitHubIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Stack>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
