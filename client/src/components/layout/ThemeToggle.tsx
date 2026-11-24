/**
 * Light / dark theme controls. MUI's colour-scheme manager does the work:
 * it flips <html data-theme>, which every --vb-* token follows, and remembers
 * the choice in localStorage ("mui-mode").
 */
import { IconButton, MenuItem, Switch, Tooltip } from '@mui/material';
import DarkModeRoundedIcon from '@mui/icons-material/DarkModeRounded';
import LightModeRoundedIcon from '@mui/icons-material/LightModeRounded';
import { useThemeMode } from './useThemeMode';

/** Row for the account dropdown — stays open so the change is visible. */
export const ThemeMenuItem = () => {
  const { isDark, toggle } = useThemeMode();
  return (
    <MenuItem onClick={toggle} className="vb-account-menu__item vb-theme-item" role="menuitemcheckbox" aria-checked={isDark}>
      {isDark ? <DarkModeRoundedIcon /> : <LightModeRoundedIcon />}
      <span className="vb-theme-item__label">Dark mode</span>
      <Switch size="small" checked={isDark} tabIndex={-1} inputProps={{ 'aria-hidden': true }} />
    </MenuItem>
  );
};

/** Compact sun/moon button for the bar (guests on wide screens). */
export const ThemeIconButton = ({ className }: { className?: string }) => {
  const { isDark, toggle } = useThemeMode();
  const label = isDark ? 'Switch to light mode' : 'Switch to dark mode';
  return (
    <Tooltip title={label}>
      <IconButton onClick={toggle} aria-label={label} className={className}>
        {isDark ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}
      </IconButton>
    </Tooltip>
  );
};

/** Labelled switch row for the full-screen menu. */
export const ThemeSwitchRow = () => {
  const { isDark, toggle } = useThemeMode();
  return (
    <label className="vb-theme-row">
      <span className="vb-theme-row__icon" aria-hidden>
        {isDark ? <DarkModeRoundedIcon /> : <LightModeRoundedIcon />}
      </span>
      <span className="vb-theme-row__label">Dark mode</span>
      <Switch checked={isDark} onChange={toggle} />
    </label>
  );
};
