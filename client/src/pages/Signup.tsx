/**
 * Signup Page
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
    Grid,
} from '@mui/material';
import {
    Visibility,
    VisibilityOff,
    Email as EmailIcon,
    Lock as LockIcon,
    Person as PersonIcon,
    PersonAdd as PersonAddIcon,
    Favorite as HeartIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../stores';
import { signup as signupApi } from '../data/api';
import './Auth.css';

const Signup: React.FC = () => {
    const navigate = useNavigate();
    const login = useAuthStore((s) => s.login);

    const [formData, setFormData] = useState({
        firstName: '',
        lastName: '',
        email: '',
        password: '',
        confirmPassword: '',
    });
    const [showPassword, setShowPassword] = useState(false);
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        setError('');
        setFieldErrors(prev => ({ ...prev, [name]: '' }));
    };

    const validateForm = (): boolean => {
        const errors: Record<string, string> = {};

        if (formData.firstName.trim().length < 2) {
            errors.firstName = 'First name must be at least 2 characters';
        }
        if (formData.lastName.trim().length < 2) {
            errors.lastName = 'Last name must be at least 2 characters';
        }
        if (!/\S+@\S+\.\S+/.test(formData.email)) {
            errors.email = 'Please enter a valid email address';
        }
        if (formData.password.length < 6) {
            errors.password = 'Password must be at least 6 characters';
        }
        if (formData.password !== formData.confirmPassword) {
            errors.confirmPassword = 'Passwords do not match';
        }

        setFieldErrors(errors);
        return Object.keys(errors).length === 0;
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!validateForm()) return;

        setLoading(true);
        setError('');

        try {
            const response = await signupApi(formData);
            if (response.success && response.token && response.user) {
                login(response.token, response.user);
                navigate('/profile');
            } else {
                setError(response.message || 'Signup failed');
            }
        } catch (err: any) {
            const errorMessage = err.response?.data?.message || 'Signup failed. Please try again.';
            setError(errorMessage);

            // Handle validation errors from server
            if (err.response?.data?.errors) {
                const serverErrors: Record<string, string> = {};
                err.response.data.errors.forEach((e: { path: string; msg: string }) => {
                    serverErrors[e.path] = e.msg;
                });
                setFieldErrors(serverErrors);
            }
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

            <Paper className="auth-card auth-card-signup" elevation={0}>
                <div className="auth-header">
                    <div className="auth-logo">
                        <HeartIcon />
                    </div>
                    <Typography variant="h4" className="auth-title">
                        Create Account
                    </Typography>
                    <Typography variant="body1" className="auth-subtitle">
                        Join MetS Health to start your wellness journey
                    </Typography>
                </div>

                {error && (
                    <Alert severity="error" className="auth-alert">
                        {error}
                    </Alert>
                )}

                <form onSubmit={handleSubmit} className="auth-form">
                    <Grid container spacing={2}>
                        <Grid size={{ xs: 12, sm: 6 }}>
                            <TextField
                                fullWidth
                                label="First Name"
                                name="firstName"
                                value={formData.firstName}
                                onChange={handleChange}
                                required
                                error={!!fieldErrors.firstName}
                                helperText={fieldErrors.firstName}
                                InputProps={{
                                    startAdornment: (
                                        <InputAdornment position="start">
                                            <PersonIcon className="input-icon" />
                                        </InputAdornment>
                                    ),
                                }}
                            />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 6 }}>
                            <TextField
                                fullWidth
                                label="Last Name"
                                name="lastName"
                                value={formData.lastName}
                                onChange={handleChange}
                                required
                                error={!!fieldErrors.lastName}
                                helperText={fieldErrors.lastName}
                                InputProps={{
                                    startAdornment: (
                                        <InputAdornment position="start">
                                            <PersonIcon className="input-icon" />
                                        </InputAdornment>
                                    ),
                                }}
                            />
                        </Grid>
                    </Grid>

                    <TextField
                        fullWidth
                        label="Email Address"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        error={!!fieldErrors.email}
                        helperText={fieldErrors.email}
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
                        error={!!fieldErrors.password}
                        helperText={fieldErrors.password || 'At least 6 characters'}
                        autoComplete="new-password"
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

                    <TextField
                        fullWidth
                        label="Confirm Password"
                        name="confirmPassword"
                        type={showConfirmPassword ? 'text' : 'password'}
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        required
                        error={!!fieldErrors.confirmPassword}
                        helperText={fieldErrors.confirmPassword}
                        autoComplete="new-password"
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <LockIcon className="input-icon" />
                                </InputAdornment>
                            ),
                            endAdornment: (
                                <InputAdornment position="end">
                                    <IconButton
                                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                        edge="end"
                                        size="small"
                                    >
                                        {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
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
                        startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <PersonAddIcon />}
                    >
                        {loading ? 'Creating Account...' : 'Create Account'}
                    </Button>
                </form>

                <Divider className="auth-divider">
                    <Typography variant="body2" color="textSecondary">
                        Already have an account?
                    </Typography>
                </Divider>

                <Box className="auth-footer">
                    <Typography variant="body2">
                        Already registered?{' '}
                        <Link to="/login" className="auth-link">
                            Sign In
                        </Link>
                    </Typography>
                </Box>
            </Paper>
        </div>
    );
};

export default Signup;
