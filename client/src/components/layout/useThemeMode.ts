/** Resolved light/dark mode + toggle, on top of MUI's colour-scheme manager. */
import { useColorScheme } from '@mui/material';

export const useThemeMode = () => {
  const { mode, systemMode, setMode } = useColorScheme();
  const resolved = (mode === 'system' ? systemMode : mode) ?? 'light';
  const isDark = resolved === 'dark';
  return { isDark, toggle: () => setMode(isDark ? 'light' : 'dark') };
};
