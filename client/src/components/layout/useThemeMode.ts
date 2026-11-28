/** Resolved light/dark mode + setters, on top of MUI's colour-scheme manager. */
import { useColorScheme } from '@mui/material';

export type ThemeChoice = 'light' | 'dark';

export const useThemeMode = () => {
  const { mode, systemMode, setMode } = useColorScheme();
  const resolved: ThemeChoice = ((mode === 'system' ? systemMode : mode) ?? 'light') as ThemeChoice;
  const isDark = resolved === 'dark';
  return {
    mode: resolved,
    isDark,
    setTheme: (next: ThemeChoice) => setMode(next),
    toggle: () => setMode(isDark ? 'light' : 'dark'),
  };
};
