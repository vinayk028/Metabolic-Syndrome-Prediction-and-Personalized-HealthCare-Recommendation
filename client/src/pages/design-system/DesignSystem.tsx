/**
 * Design System preview — DEV ONLY (route registered only when import.meta.env.DEV).
 * Living style guide for the "Vital Balance" tokens and UI kit.
 */
import { Box, Button, Grid, Stack, TextField, ToggleButton, ToggleButtonGroup, Typography, useColorScheme } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import DashboardRoundedIcon from '@mui/icons-material/DashboardRounded';
import InsightsRoundedIcon from '@mui/icons-material/InsightsRounded';
import {
  AnimatedReveal,
  EmptyState,
  FactorBadge,
  GlassCard,
  MetabolicRing,
  PageHeader,
  PulseLine,
  PulseLoader,
  RiskBadge,
  RiskGauge,
  SectionHeader,
  StatTile,
} from '../../components/ui';
import { FACTOR_ORDER } from '../../theme/factors';

const Swatch = ({ name, cssVar }: { name: string; cssVar: string }) => (
  <Stack spacing={0.75}>
    <Box sx={{ height: 56, borderRadius: 'var(--vb-radius-md)', backgroundColor: cssVar, border: '1px solid var(--vb-border)' }} />
    <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.primary' }}>
      {name}
    </Typography>
  </Stack>
);

const DesignSystem = () => {
  const { mode, setMode } = useColorScheme();

  return (
    <Box sx={{ py: { xs: 4, md: 6 } }}>
      <PageHeader
        eyebrow="Dev only"
        title="Vital Balance"
        subtitle="Design tokens and shared UI components."
        icon={<InsightsRoundedIcon />}
        actions={
          <ToggleButtonGroup size="small" exclusive value={mode ?? 'light'} onChange={(_, v) => v && setMode(v)}>
            <ToggleButton value="light">Light</ToggleButton>
            <ToggleButton value="dark">Dark</ToggleButton>
            <ToggleButton value="system">System</ToggleButton>
          </ToggleButtonGroup>
        }
      />

      {/* Typography */}
      <GlassCard sx={{ mb: 4 }}>
        <Typography variant="eyebrow" sx={{ color: 'primary.main' }}>
          Typography
        </Typography>
        <Typography variant="display" sx={{ mt: 1 }}>
          Know your <span className="gradient-text">metabolic risk</span>
        </Typography>
        <Typography variant="h2" component="p" sx={{ mt: 2 }}>
          Heading 2 — Plus Jakarta Sans
        </Typography>
        <Typography variant="h4" component="p" sx={{ mt: 1 }}>
          Heading 4 — fluid scale
        </Typography>
        <Typography variant="lead" sx={{ mt: 1, color: 'text.secondary' }}>
          Lead paragraph in Inter. Sizes scale smoothly from 320px phones to 4K displays.
        </Typography>
        <PulseLine sx={{ mt: 3 }} />
      </GlassCard>

      {/* Colour */}
      <SectionHeader eyebrow="Colour" title="Brand, factors & risk" as="h3" />
      <Grid container spacing={2} sx={{ mb: 4 }}>
        {[
          ['Primary', 'var(--vb-primary)'],
          ['Accent', 'var(--vb-accent)'],
          ['Surface', 'var(--vb-surface)'],
          ['Sunken', 'var(--vb-surface-sunken)'],
          ['Waist', 'var(--vb-factor-waist)'],
          ['Blood pressure', 'var(--vb-factor-bp)'],
          ['Glucose', 'var(--vb-factor-glucose)'],
          ['Triglycerides', 'var(--vb-factor-tg)'],
          ['HDL', 'var(--vb-factor-hdl)'],
          ['Risk · low', 'var(--vb-risk-low)'],
          ['Risk · medium', 'var(--vb-risk-medium)'],
          ['Risk · high', 'var(--vb-risk-high)'],
        ].map(([n, v]) => (
          <Grid key={n} size={{ xs: 6, sm: 4, md: 2 }}>
            <Swatch name={n} cssVar={v} />
          </Grid>
        ))}
      </Grid>

      {/* Badges + buttons */}
      <GlassCard sx={{ mb: 4 }}>
        <Stack spacing={3}>
          <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1 }}>
            {FACTOR_ORDER.map((k) => (
              <FactorBadge key={k} factor={k} showThreshold />
            ))}
          </Stack>
          <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1 }}>
            <RiskBadge level="Low Severity" />
            <RiskBadge level="Medium Severity" />
            <RiskBadge level="High Severity" />
            <RiskBadge level="high" tone="solid" />
          </Stack>
          <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1.5 }}>
            <Button variant="gradient" size="large" endIcon={<ArrowForwardRoundedIcon />}>
              Start free assessment
            </Button>
            <Button variant="contained">Contained</Button>
            <Button variant="contained" color="accent">
              Accent
            </Button>
            <Button variant="outlined">Outlined</Button>
            <Button variant="text">Text</Button>
          </Stack>
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, sm: 6 }}>
              <TextField fullWidth label="Waist circumference" placeholder="e.g. 92" helperText="Measured at the navel, in cm" />
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <TextField fullWidth label="Fasting glucose" error helperText="Enter a value between 50 and 400" />
            </Grid>
          </Grid>
        </Stack>
      </GlassCard>

      {/* Data viz */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, md: 5 }}>
          <GlassCard sx={{ height: '100%' }}>
            <Typography variant="h5" component="h3" sx={{ mb: 3 }}>
              Risk gauge
            </Typography>
            <RiskGauge value={0.58} />
          </GlassCard>
        </Grid>
        <Grid size={{ xs: 12, md: 7 }}>
          <GlassCard sx={{ height: '100%' }}>
            <Typography variant="h5" component="h3" sx={{ mb: 3 }}>
              Metabolic ring
            </Typography>
            <MetabolicRing
              showLegend
              factors={{ waist: 'elevated', bloodPressure: 'borderline', glucose: 'normal', triglycerides: 'elevated', hdl: 'normal' }}
            />
          </GlassCard>
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <StatTile label="Waist" value={96} unit="cm" factor="waist" trend={{ delta: -3, goodWhen: 'down', suffix: ' cm' }} caption="vs last month" />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <StatTile label="Blood pressure" value="128/82" unit="mmHg" factor="bloodPressure" trend={{ delta: 4, goodWhen: 'down' }} />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <StatTile label="HDL" value={52} unit="mg/dL" factor="hdl" trend={{ delta: 6, goodWhen: 'up' }} />
        </Grid>
        <Grid size={{ xs: 12, sm: 6, lg: 3 }}>
          <StatTile label="Assessments" value={7} icon={<DashboardRoundedIcon />} loading={false} onClick={() => {}} />
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <AnimatedReveal>
            <GlassCard accent="var(--vb-factor-glucose)" interactive>
              <Typography variant="h5" component="h3">
                Interactive card with factor accent
              </Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
                Reveals on scroll; lifts on hover.
              </Typography>
            </GlassCard>
          </AnimatedReveal>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <GlassCard>
            <PulseLoader label="Analysing your results…" />
          </GlassCard>
        </Grid>
      </Grid>

      <EmptyState
        title="No assessments yet"
        description="Take your first 2-minute assessment to see your Metabolic Ring."
        action={<Button variant="gradient">Start assessment</Button>}
      />
    </Box>
  );
};

export default DesignSystem;
