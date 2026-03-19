import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  useMediaQuery,
  useTheme,
  Box,
  Button,
  Avatar,
  Menu,
  MenuItem,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Home as HomeIcon,
  Assessment as AssessmentIcon,
  Info as InfoIcon,
  MenuBook as ResourcesIcon,
  Favorite as HeartIcon,
  Person as PersonIcon,
  Login as LoginIcon,
  Logout as LogoutIcon,
  Dashboard as DashboardIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../stores';
import './Navbar.css';

interface NavItem {
  label: string;
  path: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { label: 'Home', path: '/', icon: <HomeIcon /> },
  { label: 'Assessment', path: '/assessment', icon: <AssessmentIcon /> },
  { label: 'Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
  { label: 'Resources', path: '/resources', icon: <ResourcesIcon /> },
  { label: 'About', path: '/about', icon: <InfoIcon /> },
];

const Navbar: React.FC = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const location = useLocation();
  const { user, isAuthenticated, logout } = useAuthStore();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleMenuClose();
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <>
      <AppBar 
        position="fixed" 
        className={`navbar ${scrolled ? 'navbar-scrolled' : ''}`}
        elevation={0}
      >
        <Toolbar className="navbar-toolbar">
          {/* Logo */}
          <Link to="/" className="navbar-logo">
            <div className="logo-icon">
              <HeartIcon />
            </div>
            <div className="logo-text">
              <span className="logo-main">MetS Health</span>
              <span className="logo-sub">Risk Assessment</span>
            </div>
          </Link>

          {/* Desktop Navigation */}
          {!isMobile && (
            <nav className="navbar-links">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span className="nav-label">{item.label}</span>
                  <span className="nav-indicator"></span>
                </Link>
              ))}
            </nav>
          )}

          {/* Auth Section - Desktop */}
          {!isMobile && (
            <Box className="navbar-auth">
              {isAuthenticated && user ? (
                <>
                  <IconButton
                    onClick={handleMenuOpen}
                    className="user-avatar-btn"
                    aria-label="account menu"
                  >
                    <Avatar className="user-avatar">
                      {user.firstName?.charAt(0)}{user.lastName?.charAt(0)}
                    </Avatar>
                  </IconButton>
                  <Menu
                    anchorEl={anchorEl}
                    open={Boolean(anchorEl)}
                    onClose={handleMenuClose}
                    className="user-menu"
                    anchorOrigin={{
                      vertical: 'bottom',
                      horizontal: 'right',
                    }}
                    transformOrigin={{
                      vertical: 'top',
                      horizontal: 'right',
                    }}
                  >
                    <Box className="user-menu-header">
                      <Avatar className="menu-avatar">
                        {user.firstName?.charAt(0)}{user.lastName?.charAt(0)}
                      </Avatar>
                      <Box>
                        <div className="menu-user-name">{user.fullName}</div>
                        <div className="menu-user-email">{user.email}</div>
                      </Box>
                    </Box>
                    <Divider />
                    <MenuItem 
                      component={Link} 
                      to="/profile" 
                      onClick={handleMenuClose}
                      className="user-menu-item"
                    >
                      <PersonIcon /> Profile
                    </MenuItem>
                    <MenuItem 
                      onClick={handleLogout}
                      className="user-menu-item logout-item"
                    >
                      <LogoutIcon /> Logout
                    </MenuItem>
                  </Menu>
                </>
              ) : (
                <Box className="auth-buttons">
                  <Button
                    component={Link}
                    to="/login"
                    variant="text"
                    className="login-btn"
                    startIcon={<LoginIcon />}
                  >
                    Login
                  </Button>
                  <Button
                    component={Link}
                    to="/signup"
                    variant="contained"
                    className="signup-btn"
                  >
                    Sign Up
                  </Button>
                </Box>
              )}
            </Box>
          )}

          {/* Mobile Menu Button */}
          {isMobile && (
            <IconButton
              className="mobile-menu-btn"
              onClick={handleDrawerToggle}
              aria-label="open navigation menu"
            >
              <MenuIcon />
            </IconButton>
          )}
        </Toolbar>
      </AppBar>

      {/* Mobile Drawer */}
      <Drawer
        anchor="right"
        open={mobileOpen}
        onClose={handleDrawerToggle}
        classes={{ paper: 'drawer-paper' }}
      >
        <Box className="drawer-header">
          <div className="drawer-logo">
            <HeartIcon />
            <span>MetS Health</span>
          </div>
          <IconButton className="drawer-close" onClick={handleDrawerToggle}>
            <CloseIcon />
          </IconButton>
        </Box>

        {/* User Info in Drawer */}
        {isAuthenticated && user && (
          <Box className="drawer-user-info">
            <Avatar className="drawer-avatar">
              {user.firstName?.charAt(0)}{user.lastName?.charAt(0)}
            </Avatar>
            <Box>
              <div className="drawer-user-name">{user.fullName}</div>
              <div className="drawer-user-email">{user.email}</div>
            </Box>
          </Box>
        )}

        <List className="drawer-list">
          {navItems.map((item) => (
            <ListItem
              key={item.path}
              component={Link}
              to={item.path}
              className={`drawer-item ${isActive(item.path) ? 'active' : ''}`}
              onClick={handleDrawerToggle}
            >
              <div className="drawer-icon">{item.icon}</div>
              <span>{item.label}</span>
            </ListItem>
          ))}

          <Divider sx={{ my: 1 }} />

          {isAuthenticated && user ? (
            <>
              <ListItem
                component={Link}
                to="/profile"
                className={`drawer-item ${isActive('/profile') ? 'active' : ''}`}
                onClick={handleDrawerToggle}
              >
                <div className="drawer-icon"><PersonIcon /></div>
                <span>Profile</span>
              </ListItem>
              <ListItem
                className="drawer-item drawer-logout"
                onClick={() => {
                  logout();
                  handleDrawerToggle();
                }}
              >
                <div className="drawer-icon"><LogoutIcon /></div>
                <span>Logout</span>
              </ListItem>
            </>
          ) : (
            <>
              <ListItem
                component={Link}
                to="/login"
                className={`drawer-item ${isActive('/login') ? 'active' : ''}`}
                onClick={handleDrawerToggle}
              >
                <div className="drawer-icon"><LoginIcon /></div>
                <span>Login</span>
              </ListItem>
              <ListItem
                component={Link}
                to="/signup"
                className={`drawer-item drawer-signup ${isActive('/signup') ? 'active' : ''}`}
                onClick={handleDrawerToggle}
              >
                <div className="drawer-icon"><PersonIcon /></div>
                <span>Sign Up</span>
              </ListItem>
            </>
          )}
        </List>
      </Drawer>

      {/* Spacer for fixed navbar */}
      <Toolbar sx={{ minHeight: { xs: '60px', md: '70px' } }} />
    </>
  );
};

export default Navbar;
