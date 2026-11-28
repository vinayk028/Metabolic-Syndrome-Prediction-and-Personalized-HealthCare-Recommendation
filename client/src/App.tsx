/**
 * App Component - Main entry point
 */

import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { MotionConfig } from 'framer-motion';
import theme from './theme/theme';
import { useAuthStore } from './stores';
import Layout from './components/layout/Layout';
import ProtectedRoute from './components/common/ProtectedRoute';

// Pages load on demand (see pages/lazyPages.ts)
import { About, Assessment, Dashboard, Landing, Login, NotFound, Profile, Resources, Signup } from './pages/lazyPages';


function App() {
    const initAuth = useAuthStore((s) => s.initAuth);

    useEffect(() => {
        initAuth();
    }, [initAuth]);

    return (
        <ThemeProvider theme={theme} defaultMode="light" disableTransitionOnChange>
            <MotionConfig reducedMotion="user">
                <CssBaseline />
                <Router>
                    <Layout>
                        <Routes>
                            <Route path="/" element={<Landing />} />
                            <Route path="/about" element={<About />} />
                            <Route path="/assessment" element={<Assessment />} />
                            <Route path="/resources" element={<Resources />} />
                            <Route path="/login" element={<Login />} />
                            <Route path="/signup" element={<Signup />} />
                            <Route path="/profile" element={
                                <ProtectedRoute>
                                    <Profile />
                                </ProtectedRoute>
                            } />
                            <Route path="/dashboard" element={
                                <ProtectedRoute>
                                    <Dashboard />
                                </ProtectedRoute>
                            } />
                            {/* Any other address */}
                            <Route path="*" element={<NotFound />} />
                        </Routes>
                    </Layout>
                </Router>
            </MotionConfig>
        </ThemeProvider>
    );
}

export default App;
