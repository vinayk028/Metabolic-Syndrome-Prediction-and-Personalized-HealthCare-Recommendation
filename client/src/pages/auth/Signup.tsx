/**
 * Signup — create a MetS Health account. Logic unchanged: client-side
 * validation → signupApi → store login → navigate to /profile; server
 * validation errors are mapped back onto their fields.
 */
import { useState, type ChangeEvent, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Alert, Button, CircularProgress } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import MailOutlineRoundedIcon from '@mui/icons-material/MailOutlineRounded';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import PersonOutlineRoundedIcon from '@mui/icons-material/PersonOutlineRounded';
import { useAuthStore } from '../../stores';
import { signup as signupApi } from '../../services/api';
import { LabeledField, PasswordField } from '../../components/ui';
import AuthShell from './AuthShell';

const Signup = () => {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);

  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    setError('');
    setFieldErrors((prev) => ({ ...prev, [name]: '' }));
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

  const handleSubmit = async (e: FormEvent) => {
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
    } catch (err) {
      const data = axios.isAxiosError(err) ? err.response?.data : undefined;
      setError(data?.message || 'Signup failed. Please try again.');

      // Map server validation errors onto their fields
      if (Array.isArray(data?.errors)) {
        const serverErrors: Record<string, string> = {};
        data.errors.forEach((fe: { path: string; msg: string }) => {
          serverErrors[fe.path] = fe.msg;
        });
        setFieldErrors(serverErrors);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Get started"
      title="Create your account"
      subtitle="Free, and takes less than a minute. Save your results and watch your progress."
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="au-link">
            Sign in
          </Link>
        </>
      }
    >
      {error && (
        <Alert severity="error" className="au-alert">
          {error}
        </Alert>
      )}

      <form onSubmit={handleSubmit} className="au-form">
        <div className="au-row">
          <LabeledField
            label="First name"
            name="firstName"
            value={formData.firstName}
            onChange={handleChange}
            required
            autoComplete="given-name"
            error={!!fieldErrors.firstName}
            helperText={fieldErrors.firstName}
            icon={<PersonOutlineRoundedIcon />}
          />
          <LabeledField
            label="Last name"
            name="lastName"
            value={formData.lastName}
            onChange={handleChange}
            required
            autoComplete="family-name"
            error={!!fieldErrors.lastName}
            helperText={fieldErrors.lastName}
            icon={<PersonOutlineRoundedIcon />}
          />
        </div>

        <LabeledField
          label="Email"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleChange}
          required
          autoComplete="email"
          placeholder="you@example.com"
          error={!!fieldErrors.email}
          helperText={fieldErrors.email}
          icon={<MailOutlineRoundedIcon />}
        />

        <PasswordField
          label="Password"
          name="password"
          value={formData.password}
          onChange={handleChange}
          required
          autoComplete="new-password"
          error={!!fieldErrors.password}
          helperText={fieldErrors.password || 'At least 6 characters'}
          icon={<LockOutlinedIcon />}
        />

        <PasswordField
          label="Confirm password"
          name="confirmPassword"
          value={formData.confirmPassword}
          onChange={handleChange}
          required
          autoComplete="new-password"
          error={!!fieldErrors.confirmPassword}
          helperText={fieldErrors.confirmPassword}
          icon={<LockOutlinedIcon />}
        />

        <Button
          type="submit"
          fullWidth
          variant="gradient"
          size="large"
          disabled={loading}
          endIcon={loading ? undefined : <ArrowForwardRoundedIcon />}
          className="au-submit"
        >
          {loading ? <CircularProgress size={22} color="inherit" /> : 'Create account'}
        </Button>

        <p className="au-fine">
          By creating an account you agree that MetS Health is for information only and is not a substitute for
          professional medical advice.
        </p>
      </form>
    </AuthShell>
  );
};

export default Signup;
