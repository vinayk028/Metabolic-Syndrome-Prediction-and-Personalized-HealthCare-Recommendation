/**
 * Pages are loaded on demand: each page's code (and libraries only it uses,
 * such as the Dashboard's charts) downloads the first time it is opened, so
 * the first visit only pays for the page being shown.
 *
 * preloadPage() starts that download early — called when a link to the page
 * is hovered, focused or touched — so navigation still feels instant.
 */
import { lazy } from 'react';

const loaders = {
  '/': () => import('./landing/Landing'),
  '/about': () => import('./about/About'),
  '/assessment': () => import('./assessment/Assessment'),
  '/resources': () => import('./resources/Resources'),
  '/login': () => import('./auth/Login'),
  '/signup': () => import('./auth/Signup'),
  '/profile': () => import('./profile/Profile'),
  '/dashboard': () => import('./dashboard/Dashboard'),
};

export const Landing = lazy(loaders['/']);
export const About = lazy(loaders['/about']);
export const Assessment = lazy(loaders['/assessment']);
export const Resources = lazy(loaders['/resources']);
export const Login = lazy(loaders['/login']);
export const Signup = lazy(loaders['/signup']);
export const Profile = lazy(loaders['/profile']);
export const Dashboard = lazy(loaders['/dashboard']);

/** Start downloading a page's code before it is opened (safe to call repeatedly). */
export const preloadPage = (path: string) => {
  const pathname = path.split(/[?#]/)[0] || '/';
  void loaders[pathname as keyof typeof loaders]?.().catch(() => {
    // Offline or a failed fetch: the real navigation will retry and show the error
  });
};
