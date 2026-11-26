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

// Pages
import Landing from './pages/landing/Landing';
import About from './pages/about/About';
import Assessment from './pages/assessment/Assessment';
import Resources from './pages/resources/Resources';
import Login from './pages/auth/Login';
import Signup from './pages/auth/Signup';
import Profile from './pages/profile/Profile';
import Dashboard from './pages/dashboard/Dashboard';


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
                        </Routes>
                    </Layout>
                </Router>
            </MotionConfig>
        </ThemeProvider>
    );
}

export default App;
