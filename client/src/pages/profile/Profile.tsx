/**
 * Profile — account details and security.
 *
 * Logic unchanged: same auth redirect, profile/password/delete handlers and
 * API calls. Assessment history lives on the Dashboard.
 */
import { useEffect, useState, type ChangeEvent, type FormEvent, type ReactNode, type SyntheticEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Alert,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  MenuItem,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import PersonOutlineRoundedIcon from '@mui/icons-material/PersonOutlineRounded';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import EditRoundedIcon from '@mui/icons-material/EditRounded';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import LogoutRoundedIcon from '@mui/icons-material/LogoutRounded';
import SpaceDashboardRoundedIcon from '@mui/icons-material/SpaceDashboardRounded';
import DeleteOutlineRoundedIcon from '@mui/icons-material/DeleteOutlineRounded';
import PhoneOutlinedIcon from '@mui/icons-material/PhoneOutlined';
import CakeOutlinedIcon from '@mui/icons-material/CakeOutlined';
import PlaceOutlinedIcon from '@mui/icons-material/PlaceOutlined';
import MailOutlineRoundedIcon from '@mui/icons-material/MailOutlineRounded';
import { useAuthStore } from '../../stores';
import { updateProfile, updatePassword, deleteAccount } from '../../services/api';
import type { UpdateProfileData } from '../../types/types';
import { LabeledField, PasswordField, PulseLoader } from '../../components/ui';
import './Profile.css';


const apiMessage = (err: unknown, fallback: string) =>
  (axios.isAxiosError(err) ? err.response?.data?.message : undefined) || fallback;

const formatDate = (dateString: string) =>
  new Date(dateString).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });

interface TabPanelProps {
  children?: ReactNode;
  index: number;
  value: number;
}

const TabPanel = ({ children, value, index }: TabPanelProps) => (
  <div role="tabpanel" hidden={value !== index} id={`pf-tabpanel-${index}`} aria-labelledby={`pf-tab-${index}`}>
    {value === index && <div className="pf-panel">{children}</div>}
  </div>
);

const Card = ({ title, sub, action, children, tone }: { title: string; sub?: string; action?: ReactNode; children: ReactNode; tone?: 'danger' }) => (
  <section className={`pf-card${tone === 'danger' ? ' pf-card--danger' : ''}`}>
    <header className="pf-card__head">
      <div>
        <Typography component="h2" className="pf-card__title">
          {title}
        </Typography>
        {sub && <p className="pf-card__sub">{sub}</p>}
      </div>
      {action}
    </header>
    {children}
  </section>
);

/** One label/value pair of the details <dl>: the group div holds only <dt> + <dd>. */
const Detail = ({ icon, label, value, wide }: { icon: ReactNode; label: string; value?: string | null; wide?: boolean }) => (
  <div className={`pf-detail${wide ? ' pf-details__wide' : ''}`}>
    <dt>
      <span className="pf-detail__icon" aria-hidden>
        {icon}
      </span>
      {label}
    </dt>
    <dd className={value ? undefined : 'is-empty'}>{value || 'Not added yet'}</dd>
  </div>
);

const Profile = () => {
  const navigate = useNavigate();
  const { user, updateUser, logout, isAuthenticated, isLoading: authLoading } = useAuthStore();

  const [tabValue, setTabValue] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);

  const [profileData, setProfileData] = useState<UpdateProfileData>({
    firstName: '',
    lastName: '',
    phone: '',
    dateOfBirth: '',
    gender: '',
    address: '',
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [authLoading, isAuthenticated, navigate]);

  const resetProfileData = () => {
    if (!user) return;
    setProfileData({
      firstName: user.firstName || '',
      lastName: user.lastName || '',
      phone: user.phone || '',
      dateOfBirth: user.dateOfBirth ? user.dateOfBirth.split('T')[0] : '',
      gender: user.gender || '',
      address: user.address || '',
    });
  };

  // Initialize profile data
  useEffect(() => {
    if (user) {
      setProfileData({
        firstName: user.firstName || '',
        lastName: user.lastName || '',
        phone: user.phone || '',
        dateOfBirth: user.dateOfBirth ? user.dateOfBirth.split('T')[0] : '',
        gender: user.gender || '',
        address: user.address || '',
      });
    }
  }, [user]);

  const handleTabChange = (_event: SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError('');
    setSuccess('');
  };

  const handleProfileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setProfileData((prev) => ({ ...prev, [name]: value }));
  };

  const handlePasswordChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setPasswordData((prev) => ({ ...prev, [name]: value }));
  };

  const handleProfileSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await updateProfile(profileData);
      if (response.success && response.user) {
        updateUser(response.user);
        setSuccess('Profile updated successfully!');
        setIsEditing(false);
      } else {
        setError(response.message || 'Failed to update profile');
      }
    } catch (err) {
      setError(apiMessage(err, 'Failed to update profile'));
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError('New passwords do not match');
      return;
    }

    if (passwordData.newPassword.length < 6) {
      setError('New password must be at least 6 characters');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await updatePassword(passwordData);
      if (response.success) {
        setSuccess('Password updated successfully!');
        setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
      } else {
        setError(response.message || 'Failed to update password');
      }
    } catch (err) {
      setError(apiMessage(err, 'Failed to update password'));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    setLoading(true);
    try {
      const response = await deleteAccount();
      if (response.success) {
        logout();
        navigate('/');
      } else {
        setError(response.message || 'Failed to delete account');
      }
    } catch (err) {
      setError(apiMessage(err, 'Failed to delete account'));
    } finally {
      setLoading(false);
      setDeleteDialogOpen(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  if (authLoading) return <PulseLoader label="Loading your profile…" fullPage />;
  if (!user) return null;

  const initials = `${user.firstName?.charAt(0) ?? ''}${user.lastName?.charAt(0) ?? ''}`.toUpperCase();

  return (
    <div className="pf">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="pf-head">
        <div className="pf-head__id">
          <span className="pf-avatar" aria-hidden>
            {initials}
          </span>
          <div className="pf-head__text">
            <Typography variant="h2" component="h1" className="pf-head__name">
              {user.fullName || `${user.firstName} ${user.lastName}`}
            </Typography>
            <p className="pf-head__email">{user.email}</p>
            <div className="pf-head__chips">
              <span className="pf-chip">{user.role === 'admin' ? 'Administrator' : 'Member'}</span>
              {user.createdAt && (
                <span className="pf-chip pf-chip--muted">
                  Since {new Date(user.createdAt).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })}
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="pf-head__actions">
          <Button variant="outlined" size="large" startIcon={<SpaceDashboardRoundedIcon />} onClick={() => navigate('/dashboard')}>
            Dashboard
          </Button>
          <Button variant="outlined" size="large" color="inherit" startIcon={<LogoutRoundedIcon />} onClick={handleLogout} className="pf-signout">
            Sign out
          </Button>
        </div>
      </header>

      {/* ── Tabs ───────────────────────────────────────────────────────── */}
      <Tabs value={tabValue} onChange={handleTabChange} variant="fullWidth" className="pf-tabs" aria-label="Profile sections">
        <Tab id="pf-tab-0" aria-controls="pf-tabpanel-0" icon={<PersonOutlineRoundedIcon />} iconPosition="start" label="Personal" />
        <Tab id="pf-tab-1" aria-controls="pf-tabpanel-1" icon={<LockOutlinedIcon />} iconPosition="start" label="Security" />
      </Tabs>

      {(error || success) && (
        <div className="pf-alerts">
          {error && (
            <Alert severity="error" onClose={() => setError('')}>
              {error}
            </Alert>
          )}
          {success && (
            <Alert severity="success" onClose={() => setSuccess('')}>
              {success}
            </Alert>
          )}
        </div>
      )}

      {/* ── Personal ───────────────────────────────────────────────────── */}
      <TabPanel value={tabValue} index={0}>
        <Card
          title="Personal information"
          sub={isEditing ? 'Update your details and save.' : 'Used to personalise your reports.'}
          action={
            isEditing ? (
              <Button
                color="inherit"
                startIcon={<CloseRoundedIcon />}
                onClick={() => {
                  setIsEditing(false);
                  resetProfileData();
                }}
              >
                Cancel
              </Button>
            ) : (
              <Button variant="outlined" startIcon={<EditRoundedIcon />} onClick={() => setIsEditing(true)}>
                Edit
              </Button>
            )
          }
        >
          {!isEditing ? (
            <dl className="pf-details">
              <Detail icon={<PersonOutlineRoundedIcon />} label="First name" value={user.firstName} />
              <Detail icon={<PersonOutlineRoundedIcon />} label="Last name" value={user.lastName} />
              <Detail icon={<MailOutlineRoundedIcon />} label="Email" value={user.email} />
              <Detail icon={<PhoneOutlinedIcon />} label="Phone" value={user.phone} />
              <Detail
                icon={<CakeOutlinedIcon />}
                label="Date of birth"
                value={user.dateOfBirth ? formatDate(user.dateOfBirth) : null}
              />
              <Detail icon={<PersonOutlineRoundedIcon />} label="Gender" value={user.gender} />
              <Detail icon={<PlaceOutlinedIcon />} label="Address" value={user.address} wide />
            </dl>
          ) : (
            <form onSubmit={handleProfileSubmit} className="pf-form">
              <div className="pf-grid">
                <LabeledField label="First name" name="firstName" value={profileData.firstName} onChange={handleProfileChange} autoComplete="given-name" />
                <LabeledField label="Last name" name="lastName" value={profileData.lastName} onChange={handleProfileChange} autoComplete="family-name" />
                <LabeledField label="Email" value={user.email} disabled helperText="Email can't be changed" icon={<MailOutlineRoundedIcon />} />
                <LabeledField label="Phone" name="phone" value={profileData.phone} onChange={handleProfileChange} autoComplete="tel" icon={<PhoneOutlinedIcon />} />
                <LabeledField label="Date of birth" name="dateOfBirth" type="date" value={profileData.dateOfBirth} onChange={handleProfileChange} />
                <LabeledField select label="Gender" name="gender" value={profileData.gender || ''} onChange={handleProfileChange}>
                  <MenuItem value="">
                    <em>Prefer not to say</em>
                  </MenuItem>
                  <MenuItem value="Male">Male</MenuItem>
                  <MenuItem value="Female">Female</MenuItem>
                  <MenuItem value="Other">Other</MenuItem>
                </LabeledField>
                <div className="pf-grid__wide">
                  <LabeledField label="Address" name="address" value={profileData.address} onChange={handleProfileChange} multiline minRows={2} autoComplete="street-address" />
                </div>
              </div>
              <div className="pf-actions">
                <Button
                  type="submit"
                  variant="gradient"
                  size="large"
                  disabled={loading}
                  startIcon={loading ? undefined : <CheckRoundedIcon />}
                >
                  {loading ? <CircularProgress size={22} color="inherit" /> : 'Save changes'}
                </Button>
              </div>
            </form>
          )}

          <p className="pf-meta">
            Member since {formatDate(user.createdAt)}
            {user.lastLogin && <> · Last sign-in {formatDate(user.lastLogin)}</>}
          </p>
        </Card>
      </TabPanel>

      {/* ── Security ───────────────────────────────────────────────────── */}
      <TabPanel value={tabValue} index={1}>
        <Card title="Change password" sub="Use at least 6 characters. You'll stay signed in on this device.">
          <form onSubmit={handlePasswordSubmit} className="pf-form">
            <div className="pf-grid">
              <div className="pf-grid__wide">
                <PasswordField
                  label="Current password"
                  name="currentPassword"
                  value={passwordData.currentPassword}
                  onChange={handlePasswordChange}
                  required
                  autoComplete="current-password"
                  icon={<LockOutlinedIcon />}
                />
              </div>
              <PasswordField
                label="New password"
                name="newPassword"
                value={passwordData.newPassword}
                onChange={handlePasswordChange}
                required
                autoComplete="new-password"
                helperText="At least 6 characters"
                icon={<LockOutlinedIcon />}
              />
              <PasswordField
                label="Confirm new password"
                name="confirmPassword"
                value={passwordData.confirmPassword}
                onChange={handlePasswordChange}
                required
                autoComplete="new-password"
                icon={<LockOutlinedIcon />}
              />
            </div>
            <div className="pf-actions">
              <Button type="submit" variant="gradient" size="large" disabled={loading} startIcon={loading ? undefined : <CheckRoundedIcon />}>
                {loading ? <CircularProgress size={22} color="inherit" /> : 'Update password'}
              </Button>
            </div>
          </form>
        </Card>

        <Card title="Delete account" sub="Permanently remove your account and all saved assessments." tone="danger">
          <div className="pf-danger">
            <p>This can't be undone. Download any reports you want to keep first.</p>
            <Button variant="outlined" color="error" startIcon={<DeleteOutlineRoundedIcon />} onClick={() => setDeleteDialogOpen(true)}>
              Delete account
            </Button>
          </div>
        </Card>
      </TabPanel>

      {/* ── Delete confirmation ───────────────────────────────────────── */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        maxWidth="xs"
        fullWidth
        aria-labelledby="pf-del-title"
        slotProps={{ paper: { className: 'pf-dialog' } }}
      >
        <DialogTitle id="pf-del-title" className="pf-dialog__title">
          <span className="pf-dialog__icon" aria-hidden>
            <DeleteOutlineRoundedIcon />
          </span>
          Delete your account?
        </DialogTitle>
        <DialogContent>
          <p className="pf-dialog__text">
            Your account and all saved assessments will be permanently removed. This can't be undone.
          </p>
        </DialogContent>
        <DialogActions className="pf-dialog__actions">
          <Button variant="outlined" color="inherit" onClick={() => setDeleteDialogOpen(false)}>
            Keep my account
          </Button>
          <Button onClick={handleDeleteAccount} color="error" variant="contained" disabled={loading}>
            {loading ? <CircularProgress size={20} color="inherit" /> : 'Delete account'}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default Profile;
