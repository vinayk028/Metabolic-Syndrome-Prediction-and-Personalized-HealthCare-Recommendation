import type { ReactNode } from 'react';
import { Box, Container, Typography } from '@mui/material';
import { Favorite as HeartIcon } from '@mui/icons-material';
import Navbar from './Navbar';
import './Layout.css';

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <Box className="layout">
      <Navbar />
      <Box className="main-content">
        <Container maxWidth="lg" className="container">
          {children}
        </Container>
      </Box>
      <Box component="footer" className="footer">
        <Container maxWidth="lg">
          <Box className="footer-content">
            <Box className="footer-brand">
              <HeartIcon className="footer-heart" />
              <Typography variant="body1" className="footer-brand-text">
                MetS Health
              </Typography>
            </Box>
            <Typography variant="body2" className="footer-copy">
              © {new Date().getFullYear()} MetS Predictor. All rights reserved.
            </Typography>
            <Typography variant="caption" className="footer-disclaimer">
              For educational purposes only. Not medical advice.
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default Layout;
