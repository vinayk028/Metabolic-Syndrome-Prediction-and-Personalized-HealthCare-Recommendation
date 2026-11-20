/**
 * Navbar — one adaptive navigation bar for every screen size.
 *  - Wide screens  : logo · centred links (sliding active pill) · account actions
 *  - Narrow screens: logo · primary CTA / avatar · menu button → full-screen menu
 * Transparent over the page at the top, frosted glass once scrolled.
 */
import { useEffect, useState, type MouseEvent } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { AppBar, Avatar, Box, Button, Container, Divider, Drawer, IconButton, Menu, MenuItem, Toolbar } from '@mui/material';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import LogoutRoundedIcon from '@mui/icons-material/LogoutRounded';
import PersonRoundedIcon from '@mui/icons-material/PersonRounded';
import SpaceDashboardRoundedIcon from '@mui/icons-material/SpaceDashboardRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
import { motion } from 'framer-motion';
import { useAuthStore } from '../../stores';
import BrandLogo from '../ui/BrandLogo';
import { NAV_ITEMS, PROFILE_ITEM, isPathActive } from './navConfig';
import './Navbar.css';

const initials = (first?: string, last?: string) => `${first?.charAt(0) ?? ''}${last?.charAt(0) ?? ''}`.toUpperCase() || '?';

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [accountAnchor, setAccountAnchor] = useState<null | HTMLElement>(null);
  const { pathname } = useLocation();
  const { user, isAuthenticated, logout } = useAuthStore();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Close the full-screen menu if the viewport grows past the breakpoint
  useEffect(() => {
    const mq = window.matchMedia('(min-width: 900px)');
    const onChange = (e: MediaQueryListEvent) => e.matches && setMenuOpen(false);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  // Close overlays on navigation (state reset during render — no effect needed)
  const [lastPath, setLastPath] = useState(pathname);
  if (lastPath !== pathname) {
    setLastPath(pathname);
    setMenuOpen(false);
    setAccountAnchor(null);
  }

  const openAccount = (e: MouseEvent<HTMLElement>) => setAccountAnchor(e.currentTarget);
  const closeAccount = () => setAccountAnchor(null);
  const handleLogout = () => {
    closeAccount();
    setMenuOpen(false);
    logout();
  };

  const loggedIn = isAuthenticated && !!user;
  const userInitials = initials(user?.firstName, user?.lastName);
  const menuItems = [...NAV_ITEMS, ...(loggedIn ? [PROFILE_ITEM] : [])];

  return (
    <>
      <AppBar position="fixed" className={`vb-nav${scrolled || menuOpen ? ' is-scrolled' : ''}`} component="header">
        <Container maxWidth="xl">
          <Toolbar disableGutters className="vb-nav__bar">
            <Link to="/" className="vb-nav__logo" aria-label="MetS Health home">
              <BrandLogo size={36} />
            </Link>

            {/* Inline links (wide screens) */}
            <Box component="nav" aria-label="Primary" className="vb-nav__links">
              {NAV_ITEMS.map(({ label, path }) => {
                const active = isPathActive(pathname, path);
                return (
                  <NavLink key={path} to={path} end={path === '/'} className={`vb-nav__link${active ? ' is-active' : ''}`}>
                    {active && (
                      <motion.span
                        layoutId="vb-nav-pill"
                        className="vb-nav__pill"
                        transition={{ type: 'spring', stiffness: 500, damping: 40 }}
                      />
                    )}
                    <span className="vb-nav__link-label">{label}</span>
                  </NavLink>
                );
              })}
            </Box>

            <Box className="vb-nav__actions">
              {loggedIn ? (
                <IconButton
                  onClick={openAccount}
                  className="vb-nav__avatar-btn"
                  aria-label="Account menu"
                  aria-haspopup="menu"
                  aria-expanded={Boolean(accountAnchor)}
                >
                  <Avatar className="vb-nav__avatar">{userInitials}</Avatar>
                </IconButton>
              ) : (
                <>
                  <Button component={Link} to="/login" variant="text" className="vb-nav__signin">
                    Sign in
                  </Button>
                  <Button component={Link} to="/signup" variant="gradient" className="vb-nav__cta">
                    Get started
                  </Button>
                </>
              )}

              <IconButton
                className="vb-nav__menu-btn"
                onClick={() => setMenuOpen((o) => !o)}
                aria-label={menuOpen ? 'Close menu' : 'Open menu'}
                aria-expanded={menuOpen}
                aria-controls="vb-nav-menu"
              >
                {menuOpen ? <CloseRoundedIcon /> : <MenuRoundedIcon />}
              </IconButton>
            </Box>
          </Toolbar>
        </Container>
      </AppBar>

      {/* Account dropdown */}
      <Menu
        anchorEl={accountAnchor}
        open={Boolean(accountAnchor)}
        onClose={closeAccount}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{ paper: { className: 'vb-account-menu' } }}
      >
        <Box className="vb-account-menu__header">
          <Avatar className="vb-nav__avatar vb-nav__avatar--lg">{userInitials}</Avatar>
          <Box sx={{ minWidth: 0 }}>
            <div className="vb-account-menu__name">{user?.fullName}</div>
            <div className="vb-account-menu__email">{user?.email}</div>
          </Box>
        </Box>
        <Divider />
        <MenuItem component={Link} to="/dashboard" onClick={closeAccount} className="vb-account-menu__item">
          <SpaceDashboardRoundedIcon /> Dashboard
        </MenuItem>
        <MenuItem component={Link} to="/profile" onClick={closeAccount} className="vb-account-menu__item">
          <PersonRoundedIcon /> Profile
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout} className="vb-account-menu__item is-danger">
          <LogoutRoundedIcon /> Sign out
        </MenuItem>
      </Menu>

      {/* Full-screen menu (narrow screens) — slides down beneath the bar */}
      <Drawer
        id="vb-nav-menu"
        anchor="top"
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        // Sit just under the app bar so its menu/close button stays usable
        sx={{ zIndex: 1190 }}
        disableEnforceFocus
        slotProps={{ paper: { className: 'vb-menu' }, backdrop: { className: 'vb-menu__backdrop' } }}
      >
        <Box component="nav" aria-label="Site" className="vb-menu__inner">
          <ul className="vb-menu__list">
            {menuItems.map(({ label, path, hint, icon: Icon }, i) => {
              const active = isPathActive(pathname, path);
              return (
                <motion.li
                  key={path}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.04 * i + 0.05, duration: 0.3, ease: [0.2, 0, 0, 1] }}
                >
                  <NavLink
                    to={path}
                    end={path === '/'}
                    className={`vb-menu__item${active ? ' is-active' : ''}`}
                    aria-current={active ? 'page' : undefined}
                  >
                    <span className="vb-menu__icon">
                      <Icon />
                    </span>
                    <span className="vb-menu__text">
                      <span className="vb-menu__label">{label}</span>
                      {hint && <span className="vb-menu__hint">{hint}</span>}
                    </span>
                    <ChevronRightRoundedIcon className="vb-menu__chevron" />
                  </NavLink>
                </motion.li>
              );
            })}
          </ul>

          <Box className="vb-menu__footer">
            {loggedIn ? (
              <>
                <Box className="vb-menu__user">
                  <Avatar className="vb-nav__avatar vb-nav__avatar--lg">{userInitials}</Avatar>
                  <Box sx={{ minWidth: 0 }}>
                    <div className="vb-account-menu__name">{user?.fullName}</div>
                    <div className="vb-account-menu__email">{user?.email}</div>
                  </Box>
                </Box>
                <Button fullWidth size="large" variant="outlined" startIcon={<LogoutRoundedIcon />} onClick={handleLogout}>
                  Sign out
                </Button>
              </>
            ) : (
              <>
                <Button fullWidth size="large" component={Link} to="/signup" variant="gradient" endIcon={<ArrowForwardRoundedIcon />}>
                  Create free account
                </Button>
                <Button fullWidth size="large" component={Link} to="/login" variant="outlined">
                  Sign in
                </Button>
              </>
            )}
          </Box>
        </Box>
      </Drawer>

      {/* Spacer for the fixed bar */}
      <Box aria-hidden className="vb-nav__spacer" />
    </>
  );
};

export default Navbar;
