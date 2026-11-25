/**
 * Layout — app shell. One adaptive layout for every screen size:
 *
 *   skip link → Navbar → <main> (page) → Footer → Chat
 *
 * Pages render inside a centred container by default. Routes listed in
 * FULL_BLEED_ROUTES render edge-to-edge and manage their own width (landing).
 */
import { useEffect, type ReactNode } from 'react';
import { useLocation } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import { motion } from 'framer-motion';
import Navbar from './Navbar';
import Footer from './Footer';
import ChatAssistant from '../chat/ChatAssistant';
import { FULL_BLEED_ROUTES } from './navConfig';
import './Layout.css';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const { pathname, hash } = useLocation();
  const fullBleed = FULL_BLEED_ROUTES.includes(pathname);

  // Route change → top of page; "/#section" links → scroll to that section once rendered
  useEffect(() => {
    if (!hash) {
      window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
      return;
    }
    const id = decodeURIComponent(hash.slice(1));
    const frame = window.requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [pathname, hash]);

  return (
    <Box className="vb-layout">
      <a href="#main-content" className="vb-skip-link">
        Skip to content
      </a>

      <Navbar />

      <Box component="main" id="main-content" tabIndex={-1} className="vb-main">
        {/* Enter-only transition: <Routes> reads the live location, so an exit
            animation would flash the *new* page inside the old wrapper. */}
        <motion.div
          key={pathname}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: [0.2, 0, 0, 1] }}
          className="vb-main__page"
        >
          {fullBleed ? (
            children
          ) : (
            // Legacy pages are designed for `lg`; redesigned pages can widen in Phase 4.
            <Container maxWidth="xl" className="vb-page">
              {children}
            </Container>
          )}
        </motion.div>
      </Box>

      <Footer />
      <ChatAssistant />
    </Box>
  );
};

export default Layout;
