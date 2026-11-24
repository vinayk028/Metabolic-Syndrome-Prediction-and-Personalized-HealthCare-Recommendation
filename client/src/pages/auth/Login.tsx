/**
 * Login — sign in to MetS Health. Logic unchanged: loginApi → store login →
 * navigate to /profile.
 */
import { useState, type ChangeEvent, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Alert, Button, CircularProgress } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import MailOutlineRoundedIcon from '@mui/icons-material/MailOutlineRounded';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import { useAuthStore } from '../../stores';
import { login as loginApi } from '../../services/api';
import { LabeledField, PasswordField } from '../../components/ui';
import AuthShell from './AuthShell';

const Login = () => {
  const navigate = useNavigate();
  const login = useAuthStore((s) => s.login);

  const [formData, setFormData] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError('');
  };

  const handleSubmit = async (e: FormEvent) => {
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
    } catch (err) {
      const message = axios.isAxiosError(err) ? err.response?.data?.message : undefined;
      setError(message || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell
      eyebrow="Welcome back"
      title="Sign in"
      subtitle="Pick up where you left off — your results and progress are waiting."
      footer={
        <>
          New to MetS Health?{' '}
          <Link to="/signup" className="au-link">
            Create a free account
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
        <LabeledField
          label="Email"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleChange}
          required
          autoComplete="email"
          placeholder="you@example.com"
          icon={<MailOutlineRoundedIcon />}
        />
        <PasswordField
          label="Password"
          name="password"
          value={formData.password}
          onChange={handleChange}
          required
          autoComplete="current-password"
          placeholder="Your password"
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
          {loading ? <CircularProgress size={22} color="inherit" /> : 'Sign in'}
        </Button>
      </form>
    </AuthShell>
  );
};

export default Login;
