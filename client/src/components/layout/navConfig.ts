/**
 * Navigation config shared by Navbar and Footer.
 */
import type { SvgIconComponent } from '@mui/icons-material';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import SpaceDashboardRoundedIcon from '@mui/icons-material/SpaceDashboardRounded';
import AutoStoriesRoundedIcon from '@mui/icons-material/AutoStoriesRounded';
import PersonRoundedIcon from '@mui/icons-material/PersonRounded';

export interface NavItem {
  label: string;
  path: string;
  icon: SvgIconComponent;
  /** One-line hint shown in the full-screen menu on small screens */
  hint?: string;
  /** Requires login (still shown; ProtectedRoute redirects) */
  auth?: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { label: 'Home', path: '/', icon: HomeRoundedIcon, hint: 'Start here' },
  { label: 'Assessment', path: '/assessment', icon: MonitorHeartRoundedIcon, hint: 'Check your 5 risk factors' },
  { label: 'Dashboard', path: '/dashboard', icon: SpaceDashboardRoundedIcon, hint: 'Your results & trends', auth: true },
  { label: 'Resources', path: '/resources', icon: AutoStoriesRoundedIcon, hint: 'Guides & health news' },
];

export const PROFILE_ITEM: NavItem = {
  label: 'Profile',
  path: '/profile',
  icon: PersonRoundedIcon,
  hint: 'Your details & security',
  auth: true,
};

/**
 * Routes that render edge-to-edge (no page container). Pages listed here own
 * their own width/padding.
 */
export const FULL_BLEED_ROUTES: string[] = ['/'];


/** Browser-tab titles (also what screen readers announce on page change). */
const PAGE_TITLES: Record<string, string> = {
  '/assessment': 'Assessment',
  '/dashboard': 'Dashboard',
  '/resources': 'Resources',
  '/about': 'About & FAQ',
  '/login': 'Sign in',
  '/signup': 'Create account',
  '/profile': 'Profile',
};

/** "Dashboard · MetS Health"; the home page keeps the full site title; unknown paths are "Page not found". */
export const pageTitle = (path: string) => {
  const p = path.length > 1 ? path.replace(/\/+$/, '') : path;
  if (p === '/') return 'MetS Health · Know your metabolic risk';
  return `${PAGE_TITLES[p] ?? 'Page not found'} · MetS Health`;
};

export const isPathActive = (current: string, path: string) =>
  path === '/' ? current === '/' : current === path || current.startsWith(`${path}/`);

export const REPO_URL =
  'https://github.com/vinayk028/Metabolic-Syndrome-Prediction-and-Personalized-HealthCare-Recommendation';
