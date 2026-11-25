/**
 * Theme controls. MUI's colour-scheme manager does the work: it flips
 * <html data-theme>, which every --vb-* token follows, and remembers the
 * choice in localStorage ("mui-mode").
 *
 * ThemeChooser — "Theme" row with Light / Dark options (account menu and the
 *                small-screen menu)
 * ThemeIconButton — compact sun/moon toggle in the bar for guests
 */
import { IconButton, ToggleButton, ToggleButtonGroup, Tooltip } from '@mui/material';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import PaletteOutlinedIcon from '@mui/icons-material/PaletteOutlined';
import { useThemeMode, type ThemeChoice } from './useThemeMode';

export const ThemeChooser = ({ className = '' }: { className?: string }) => {
  const { mode, setTheme } = useThemeMode();
  return (
    <div className={`vb-theme ${className}`} role="group" aria-labelledby="vb-theme-label">
      <span className="vb-theme__label" id="vb-theme-label">
        <PaletteOutlinedIcon aria-hidden />
        Theme
      </span>
      <ToggleButtonGroup
        exclusive
        size="small"
        value={mode}
        onChange={(_, v: ThemeChoice | null) => v && setTheme(v)}
        className="vb-theme__options"
        aria-labelledby="vb-theme-label"
      >
        <ToggleButton value="light" aria-label="Light theme">
          <LightModeRoundedIcon aria-hidden /> Light
        </ToggleButton>
        <ToggleButton value="dark" aria-label="Dark theme">
          <DarkModeRoundedIcon aria-hidden /> Dark
        </ToggleButton>
      </ToggleButtonGroup>
    </div>
  );
};

/** Compact sun/moon button for the bar (guests on wide screens). */
export const ThemeIconButton = ({ className }: { className?: string }) => {
  const { isDark, toggle } = useThemeMode();
  const label = isDark ? 'Switch to light theme' : 'Switch to dark theme';
  return (
    <Tooltip title={label}>
      <IconButton onClick={toggle} aria-label={label} className={className}>
        {isDark ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}
      </IconButton>
    </Tooltip>
  );
};
