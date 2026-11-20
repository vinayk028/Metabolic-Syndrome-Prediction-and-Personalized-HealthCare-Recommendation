/**
 * MUI Theme — "Vital Balance"
 *
 * Built entirely from ./tokens. CSS variables are enabled, so every palette
 * value is also available in plain CSS as `var(--mui-palette-…)` (and as the
 * shorter `--vb-*` aliases defined in styles/global.css).
 *
 * Colour scheme is applied as <html data-theme="light|dark">.
 */
import { createTheme, type Theme } from '@mui/material/styles';
import {
  brand,
  breakpoints,
  factorColors,
  fluidType,
  fonts,
  motion,
  radius,
  riskColors,
  zIndex,
  gradients,
  type FactorKey,
  type RiskKey,
} from './tokens';

const factorPalette = (mode: 'light' | 'dark') =>
  Object.fromEntries(
    (Object.keys(factorColors) as FactorKey[]).map((k) => [
      k,
      { main: factorColors[k].main, soft: mode === 'light' ? factorColors[k].soft : factorColors[k].softDark },
    ]),
  ) as Record<FactorKey, { main: string; soft: string }>;

const riskPalette = (mode: 'light' | 'dark') =>
  Object.fromEntries(
    (Object.keys(riskColors) as RiskKey[]).map((k) => {
      const c = riskColors[k];
      return [
        k,
        mode === 'light'
          ? { main: c.main, light: c.light, soft: c.soft, contrastText: c.contrastText }
          : { main: c.light, light: c.light, soft: c.softDark, contrastText: brand.slate[950] },
      ];
    }),
  ) as Record<RiskKey, { main: string; light: string; soft: string; contrastText: string }>;

const headingBase = {
  fontFamily: fonts.display,
  fontWeight: 700,
  letterSpacing: '-0.02em',
} as const;

const theme: Theme = createTheme({
  cssVariables: {
    colorSchemeSelector: 'data-theme',
    cssVarPrefix: 'mui',
  },
  defaultColorScheme: 'light',
  colorSchemes: {
    light: {
      palette: {
        primary: { main: brand.teal[700], light: brand.teal[500], dark: brand.teal[800], contrastText: '#FFFFFF' },
        secondary: { main: brand.slate[800], light: brand.slate[600], dark: brand.slate[900], contrastText: '#FFFFFF' },
        accent: { main: brand.coral[600], light: brand.coral[400], dark: brand.coral[700], contrastText: '#FFFFFF' },
        success: { main: riskColors.low.main, light: riskColors.low.light, dark: '#166534', contrastText: '#FFFFFF' },
        warning: { main: riskColors.medium.main, light: riskColors.medium.light, dark: '#92400E', contrastText: '#FFFFFF' },
        error: { main: riskColors.high.main, light: riskColors.high.light, dark: '#991B1B', contrastText: '#FFFFFF' },
        info: { main: factorColors.glucose.main, light: '#38BDF8', dark: '#075985', contrastText: '#FFFFFF' },
        background: { default: brand.slate[50], paper: '#FFFFFF' },
        text: { primary: brand.slate[900], secondary: brand.slate[600], disabled: brand.slate[400] },
        divider: brand.slate[200],
        factor: factorPalette('light'),
        risk: riskPalette('light'),
        surface: {
          raised: '#FFFFFF',
          sunken: brand.slate[100],
          glass: 'rgba(255, 255, 255, 0.72)',
          border: brand.slate[200],
        },
      },
    },
    dark: {
      palette: {
        primary: { main: brand.teal[400], light: brand.teal[300], dark: brand.teal[500], contrastText: brand.slate[950] },
        secondary: { main: brand.slate[200], light: brand.slate[100], dark: brand.slate[300], contrastText: brand.slate[950] },
        accent: { main: brand.coral[400], light: brand.coral[300], dark: brand.coral[500], contrastText: brand.slate[950] },
        success: { main: riskColors.low.light, light: '#86EFAC', dark: riskColors.low.main, contrastText: brand.slate[950] },
        warning: { main: riskColors.medium.light, light: '#FDE68A', dark: riskColors.medium.main, contrastText: brand.slate[950] },
        error: { main: riskColors.high.light, light: '#FCA5A5', dark: riskColors.high.main, contrastText: brand.slate[950] },
        info: { main: '#38BDF8', light: '#7DD3FC', dark: factorColors.glucose.main, contrastText: brand.slate[950] },
        background: { default: brand.slate[950], paper: '#111A2E' },
        text: { primary: brand.slate[100], secondary: brand.slate[400], disabled: brand.slate[600] },
        divider: 'rgba(148, 163, 184, 0.16)',
        factor: factorPalette('dark'),
        risk: riskPalette('dark'),
        surface: {
          raised: '#15203A',
          sunken: '#0E1628',
          glass: 'rgba(17, 26, 46, 0.72)',
          border: 'rgba(148, 163, 184, 0.16)',
        },
      },
    },
  },

  breakpoints: { values: breakpoints },

  // rem-based spacing: theme.spacing(1) = 0.5rem, so every sx margin/padding
  // scales with the fluid root font-size (see styles/global.css).
  spacing: (factor: number) => `${0.5 * factor}rem`,

  shape: { borderRadius: radius.md },

  zIndex: {
    appBar: zIndex.navbar,
    drawer: zIndex.modal - 10,
    modal: zIndex.modal,
    snackbar: zIndex.toast,
    tooltip: zIndex.toast + 10,
    speedDial: zIndex.chat,
    fab: zIndex.chat,
    mobileStepper: zIndex.bottomNav,
  },

  transitions: {
    easing: {
      easeInOut: motion.css.standard,
      easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
      easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
      sharp: motion.css.emphasized,
    },
  },

  typography: {
    fontFamily: fonts.body,
    htmlFontSize: 16,
    display: { ...headingBase, fontSize: fluidType.display, fontWeight: 800, lineHeight: 1.05, letterSpacing: '-0.035em' },
    h1: { ...headingBase, fontSize: fluidType.h1, fontWeight: 800, lineHeight: 1.1, letterSpacing: '-0.03em' },
    h2: { ...headingBase, fontSize: fluidType.h2, lineHeight: 1.15 },
    h3: { ...headingBase, fontSize: fluidType.h3, lineHeight: 1.25 },
    h4: { ...headingBase, fontSize: fluidType.h4, fontWeight: 650, lineHeight: 1.3, letterSpacing: '-0.015em' },
    h5: { ...headingBase, fontSize: fluidType.h5, fontWeight: 650, lineHeight: 1.4, letterSpacing: '-0.01em' },
    h6: { ...headingBase, fontSize: fluidType.h6, fontWeight: 650, lineHeight: 1.5, letterSpacing: '-0.005em' },
    lead: { fontFamily: fonts.body, fontSize: fluidType.lead, lineHeight: 1.65, fontWeight: 400 },
    eyebrow: {
      fontFamily: fonts.display,
      fontSize: '0.75rem',
      fontWeight: 700,
      lineHeight: 1.4,
      letterSpacing: '0.14em',
      textTransform: 'uppercase',
    },
    subtitle1: { fontSize: '1rem', fontWeight: 600, lineHeight: 1.5 },
    subtitle2: { fontSize: '0.875rem', fontWeight: 600, lineHeight: 1.5 },
    body1: { fontSize: '1rem', lineHeight: 1.65 },
    body2: { fontSize: '0.875rem', lineHeight: 1.6 },
    caption: { fontSize: '0.75rem', lineHeight: 1.5 },
    button: { fontFamily: fonts.display, fontWeight: 650, textTransform: 'none', letterSpacing: '0' },
  },

  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { fontFeatureSettings: '"cv11", "ss01"' },
      },
    },

    MuiTypography: {
      defaultProps: {
        variantMapping: { display: 'h1', eyebrow: 'span', lead: 'p' },
      },
    },

    MuiContainer: {
      defaultProps: { maxWidth: 'lg' },
      styleOverrides: {
        // Fluid side gutter (16px on phones → 40px on large screens), no breakpoint jumps
        root: ({ theme }) => ({
          paddingLeft: 'var(--vb-gutter)',
          paddingRight: 'var(--vb-gutter)',
          [theme.breakpoints.up('sm')]: { paddingLeft: 'var(--vb-gutter)', paddingRight: 'var(--vb-gutter)' },
        }),
      },
    },

    MuiButton: {
      defaultProps: { disableElevation: true },
      styleOverrides: {
        root: {
          borderRadius: radius.pill,
          padding: '10px 22px',
          minHeight: 44, // touch target
          transition: `transform ${motion.duration.fast}s ${motion.css.standard}, box-shadow ${motion.duration.base}s ${motion.css.standard}, background-color ${motion.duration.base}s ${motion.css.standard}`,
          '&:active': { transform: 'scale(0.98)' },
        },
        sizeLarge: { padding: '14px 28px', fontSize: '1rem', minHeight: 52 },
        sizeSmall: { padding: '6px 14px', minHeight: 36 },
        outlined: ({ theme }) => ({
          borderColor: theme.vars.palette.divider,
          color: theme.vars.palette.text.primary,
          '&:hover': {
            borderColor: theme.vars.palette.primary.main,
            backgroundColor: `rgba(${theme.vars.palette.primary.mainChannel} / 0.06)`,
          },
        }),
      },
      variants: [
        {
          props: { variant: 'gradient' },
          style: ({ theme }) => ({
            color: '#FFFFFF',
            backgroundImage: gradients.brand,
            boxShadow: 'var(--vb-shadow-brand)',
            '&:hover': {
              backgroundImage: gradients.brand,
              boxShadow: 'var(--vb-shadow-brand-lg)',
              transform: 'translateY(-1px)',
            },
            '&.Mui-disabled': { opacity: 0.5, color: '#FFFFFF' },
            ...theme.applyStyles('dark', { color: brand.slate[950], backgroundImage: `linear-gradient(135deg, ${brand.teal[300]} 0%, ${brand.teal[400]} 100%)` }),
          }),
        },
      ],
    },

    MuiIconButton: {
      styleOverrides: {
        root: { borderRadius: radius.sm },
        sizeMedium: { width: 44, height: 44 },
      },
    },

    MuiPaper: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: { backgroundImage: 'none' },
        rounded: { borderRadius: radius.lg },
        outlined: ({ theme }) => ({ borderColor: theme.vars.palette.surface.border }),
      },
    },

    MuiCard: {
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: radius.lg,
          border: `1px solid ${theme.vars.palette.surface.border}`,
          backgroundColor: theme.vars.palette.surface.raised,
          boxShadow: 'var(--vb-shadow-sm)',
        }),
      },
    },
    MuiCardContent: {
      styleOverrides: {
        // Fluid card padding (≈18px phone → 28px desktop)
        root: {
          padding: 'clamp(1.125rem, 0.8rem + 1.2vw, 1.75rem)',
          '&:last-child': { paddingBottom: 'clamp(1.125rem, 0.8rem + 1.2vw, 1.75rem)' },
        },
      },
    },

    MuiAppBar: {
      defaultProps: { elevation: 0, color: 'transparent' },
    },

    MuiOutlinedInput: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: radius.sm,
          backgroundColor: theme.vars.palette.surface.raised,
          minHeight: 48,
          '& .MuiOutlinedInput-notchedOutline': { borderColor: theme.vars.palette.surface.border },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: theme.vars.palette.text.disabled },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: theme.vars.palette.primary.main,
            borderWidth: 2,
          },
          '&.Mui-focused': { boxShadow: `0 0 0 4px rgba(${theme.vars.palette.primary.mainChannel} / 0.12)` },
        }),
        input: { fontSize: 16 }, // ≥16px prevents iOS zoom-on-focus
      },
    },

    MuiSlider: {
      styleOverrides: {
        root: { height: 6 },
        thumb: ({ theme }) => ({
          width: 22,
          height: 22,
          backgroundColor: theme.vars.palette.background.paper,
          border: `3px solid currentColor`,
          boxShadow: 'var(--vb-shadow-sm)',
          '&:hover, &.Mui-focusVisible': {
            boxShadow: `0 0 0 8px rgba(${theme.vars.palette.primary.mainChannel} / 0.14)`,
          },
        }),
        rail: ({ theme }) => ({ opacity: 1, backgroundColor: theme.vars.palette.surface.sunken }),
      },
    },

    MuiTabs: {
      styleOverrides: {
        indicator: { height: 3, borderRadius: 3 },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: ({ theme }) => ({
          textTransform: 'none',
          fontFamily: fonts.display,
          fontWeight: 650,
          minHeight: 48,
          color: theme.vars.palette.text.secondary,
        }),
      },
    },

    MuiChip: {
      styleOverrides: {
        root: { borderRadius: radius.pill, fontWeight: 600 },
      },
    },

    MuiSwitch: {
      defaultProps: { color: 'primary' },
    },

    MuiAlert: {
      styleOverrides: {
        root: { borderRadius: radius.md, alignItems: 'center' },
      },
    },

    MuiLinearProgress: {
      styleOverrides: {
        root: ({ theme }) => ({ height: 8, borderRadius: radius.pill, backgroundColor: theme.vars.palette.surface.sunken }),
        bar: { borderRadius: radius.pill },
      },
    },

    MuiTooltip: {
      defaultProps: { arrow: true },
      styleOverrides: {
        tooltip: ({ theme }) => ({
          backgroundColor: brand.slate[900],
          fontSize: '0.8125rem',
          padding: '8px 12px',
          borderRadius: radius.sm,
          ...theme.applyStyles('dark', { backgroundColor: brand.slate[100], color: brand.slate[900] }),
        }),
        arrow: ({ theme }) => ({
          color: brand.slate[900],
          ...theme.applyStyles('dark', { color: brand.slate[100] }),
        }),
      },
    },

    MuiDialog: {
      styleOverrides: {
        paper: { borderRadius: radius.xl },
      },
    },

    MuiDrawer: {
      styleOverrides: {
        paper: ({ theme }) => ({ backgroundColor: theme.vars.palette.background.paper }),
      },
    },

    MuiSkeleton: {
      defaultProps: { animation: 'wave' },
      styleOverrides: { rounded: { borderRadius: radius.md } },
    },
  },
});

export default theme;
