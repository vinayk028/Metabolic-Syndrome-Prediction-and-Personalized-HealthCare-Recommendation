/**
 * Login Page
 */

import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
    Box,
    Paper,
    Typography,
    TextField,
    Button,
    IconButton,
    InputAdornment,
    Alert,
    CircularProgress,
    Divider,
} from '@mui/material';
import {
    Visibility,
    VisibilityOff,
    Email as EmailIcon,
    Lock as LockIcon,
    Login as LoginIcon,
    Favorite as HeartIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../stores';
import { login as loginApi } from '../data/api';
import './Auth.css';

const Login: React.FC = () => {
    const navigate = useNavigate();
    const login = useAuthStore((s) => s.login);

    const [formData, setFormData] = useState({ email: '', password: '' });
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError('');
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const response = await loginApi(formData);
            if (response.success && response.token && response.user) {
                login(response.token, response.user);
                navigate('/profile');
            } else {
                setError(response.message || 'Login failed');
            }
        } catch (err: any) {
            setError(err.response?.data?.message || 'Login failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-background">
                <div className="auth-shape auth-shape-1"></div>
                <div className="auth-shape auth-shape-2"></div>
                <div className="auth-shape auth-shape-3"></div>
            </div>

            <Paper className="auth-card" elevation={0}>
                <div className="auth-header">
                    <div className="auth-logo">
                        <HeartIcon />
                    </div>
                    <Typography variant="h4" className="auth-title">
                        Welcome Back
                    </Typography>
                    <Typography variant="body1" className="auth-subtitle">
                        Sign in to continue to MetS Health
                    </Typography>
                </div>

                {error && (
                    <Alert severity="error" className="auth-alert">
                        {error}
                    </Alert>
                )}

                <form onSubmit={handleSubmit} className="auth-form">
                    <TextField
                        fullWidth
                        label="Email Address"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        autoComplete="email"
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <EmailIcon className="input-icon" />
                                </InputAdornment>
                            ),
                        }}
                    />

                    <TextField
                        fullWidth
                        label="Password"
                        name="password"
                        type={showPassword ? 'text' : 'password'}
                        value={formData.password}
                        onChange={handleChange}
                        required
                        autoComplete="current-password"
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <LockIcon className="input-icon" />
                                </InputAdornment>
                            ),
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowPassword(!showPassword)}
                                        edge="end"
                                        size="small"
                                    >
                                        {showPassword ? <VisibilityOff /> : <Visibility />}
                                    </IconButton>
                                </InputAdornment>
                            ),
                        }}
                    />

                    <Button
                        type="submit"
                        fullWidth
                        variant="contained"
                        size="large"
                        disabled={loading}
                        className="auth-submit-btn"
                        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <LoginIcon />}
                    >
                        {loading ? 'Signing In...' : 'Sign In'}
                    </Button>
                </form>

                <Divider className="auth-divider">
                    <Typography variant="body2" color="textSecondary">
                        New to MetS Health?
                    </Typography>
                </Divider>

                <Box className="auth-footer">
                    <Typography variant="body2">
                        Don't have an account?{' '}
                        <Link to="/signup" className="auth-link">
                            Create Account
                        </Link>
                    </Typography>
                </Box>
            </Paper>
        </div>
    );
};

export default Login;
