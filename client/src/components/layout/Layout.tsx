/**
 * Layout — app shell. One adaptive layout for every screen size:
 *
 *   skip link → Navbar → <main> (page) → Footer → Chat
 *
 * Pages render inside a centred container by default. Routes listed in
 * FULL_BLEED_ROUTES render edge-to-edge and manage their own width (landing).
 *
 * Pages load on demand: while a page's code downloads, a placeholder keeps the
 * footer in place, and links start that download on hover / focus / touch.
 */
import { Suspense, useEffect, type ReactNode } from 'react';
import { useLocation } from 'react-router-dom';
import { Box, CircularProgress, Container } from '@mui/material';
import { motion } from 'framer-motion';
import Navbar from './Navbar';
import Footer from './Footer';
import ChatAssistant from '../chat/ChatAssistant';
import { FULL_BLEED_ROUTES, pageTitle } from './navConfig';
import { preloadPage } from '../../pages/lazyPages';
import './Layout.css';

/** Shown only while a page's code is still downloading (spinner fades in late, so fast loads never flash it). */
const PageLoading = () => (
  <div className="vb-page-loading" role="status" aria-label="Loading page">
    <CircularProgress size={28} />
  </div>
);

/** Internal link under the pointer / focus → start loading that page's code. */
const preloadFromEvent = (e: Event) => {
  const link = (e.target as Element | null)?.closest?.('a[href^="/"]');
  if (link) preloadPage(link.getAttribute('href')!);
};

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  const { pathname, hash } = useLocation();
  const fullBleed = FULL_BLEED_ROUTES.includes(pathname);

  // Each page gets its own browser-tab title
  useEffect(() => {
    document.title = pageTitle(pathname);
  }, [pathname]);

  // One listener for every internal link (nav, buttons, footer…)
  useEffect(() => {
    const events = ['pointerover', 'focusin', 'touchstart'] as const;
    events.forEach((t) => document.addEventListener(t, preloadFromEvent, { passive: true }));
    return () => events.forEach((t) => document.removeEventListener(t, preloadFromEvent));
  }, []);

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
          <Suspense fallback={<PageLoading />}>
            {fullBleed ? (
              children
            ) : (
              <Container maxWidth="xl" className="vb-page">
                {children}
              </Container>
            )}
          </Suspense>
        </motion.div>
      </Box>

      <Footer />
      <ChatAssistant />
    </Box>
  );
};

export default Layout;
