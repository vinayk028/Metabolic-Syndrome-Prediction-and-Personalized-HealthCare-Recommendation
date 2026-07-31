import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Avatar,
  Tabs,
  Tab,
  Grid,
  Divider,
  Alert,
  CircularProgress,
  IconButton,
  InputAdornment,
  Chip,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  MenuItem,
} from '@mui/material';
import {
  Person as PersonIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Lock as LockIcon,
  Visibility,
  VisibilityOff,
  Email as EmailIcon,
  Phone as PhoneIcon,
  Cake as CakeIcon,
  LocationOn as LocationIcon,
  History as HistoryIcon,
  Delete as DeleteIcon,
  Logout as LogoutIcon,
  Assessment as AssessmentIcon,
  CalendarToday as CalendarIcon,
  TrendingUp as TrendingUpIcon,
  Dashboard as DashboardIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  Restaurant as DietIcon,
  FitnessCenter as ExerciseIcon,
  SelfImprovement as YogaIcon,
  Block as AvoidIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { useAuthStore } from '../stores';
import { updateProfile, updatePassword, deleteAccount, getAssessmentHistory } from '../data/api';
import type { UpdateProfileData, AssessmentHistoryItem } from '../data/types';
import './Profile.css';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`profile-tabpanel-${index}`}
      aria-labelledby={`profile-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const Profile: React.FC = () => {
  const navigate = useNavigate();
  const { user, updateUser, logout, isAuthenticated, isLoading: authLoading } = useAuthStore();
  
  const [tabValue, setTabValue] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [error, setError] = useState('');
  const [assessmentHistory, setAssessmentHistory] = useState<AssessmentHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

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
  const [showPasswords, setShowPasswords] = useState({
    current: false,
    new: false,
    confirm: false,
  });

  // Redirect if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [authLoading, isAuthenticated, navigate]);

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

  // Load assessment history
  useEffect(() => {
    const loadHistory = async () => {
      if (user && tabValue === 2) {
        setHistoryLoading(true);
        try {
          const response = await getAssessmentHistory();
          if (response.success) {
            setAssessmentHistory(response.assessmentHistory || []);
          }
        } catch (err) {
          console.error('Error loading assessment history:', err);
        } finally {
          setHistoryLoading(false);
        }
      }
    };
    loadHistory();
  }, [user, tabValue]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError('');
    setSuccess('');
  };

  const handleProfileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setProfileData(prev => ({ ...prev, [name]: value }));
  };

  const handlePasswordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setPasswordData(prev => ({ ...prev, [name]: value }));
  };

  const handleProfileSubmit = async (e: React.FormEvent) => {
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
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
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
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to update password');
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
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to delete account');
    } finally {
      setLoading(false);
      setDeleteDialogOpen(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const toggleExpand = (id: string) => {
    setExpandedCard(prev => prev === id ? null : id);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low Severity':
        return '#10b981';
      case 'Medium Severity':
        return '#f59e0b';
      case 'High Severity':
        return '#ef4444';
      default:
        return '#64748b';
    }
  };

  if (authLoading) {
    return (
      <Box className="profile-loading">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>Loading...</Typography>
      </Box>
    );
  }

  if (!user) return null;

  return (
    <div className="profile-container">
      <div className="profile-header">
        <div className="profile-header-bg"></div>
        <div className="profile-header-content">
          <Avatar className="profile-avatar">
            {user.firstName?.charAt(0)}{user.lastName?.charAt(0)}
          </Avatar>
          <div className="profile-header-info">
            <Typography variant="h4" className="profile-name">
              {user.fullName}
            </Typography>
            <Typography variant="body1" className="profile-email">
              {user.email}
            </Typography>
            <Chip 
              label={user.role === 'admin' ? 'Administrator' : 'Member'} 
              size="small" 
              className="profile-role-chip"
            />
          </div>
          <Button
            variant="outlined"
            startIcon={<LogoutIcon />}
            onClick={handleLogout}
            className="profile-logout-btn"
          >
            Logout
          </Button>
        </div>
      </div>

      <Paper className="profile-content" elevation={0}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          className="profile-tabs"
          variant="fullWidth"
        >
          <Tab icon={<PersonIcon />} label="Profile" />
          <Tab icon={<LockIcon />} label="Security" />
          <Tab icon={<HistoryIcon />} label="History" />
        </Tabs>

        {(error || success) && (
          <Box sx={{ px: 3, pt: 2 }}>
            {error && <Alert severity="error" onClose={() => setError('')}>{error}</Alert>}
            {success && <Alert severity="success" onClose={() => setSuccess('')}>{success}</Alert>}
          </Box>
        )}

        {/* Profile Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box className="profile-tab-content">
            <Box className="profile-section-header">
              <Typography variant="h6">Personal Information</Typography>
              {!isEditing ? (
                <Button
                  startIcon={<EditIcon />}
                  onClick={() => setIsEditing(true)}
                  className="edit-btn"
                >
                  Edit Profile
                </Button>
              ) : (
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    startIcon={<CancelIcon />}
                    onClick={() => {
                      setIsEditing(false);
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
                    }}
                    color="inherit"
                  >
                    Cancel
                  </Button>
                </Box>
              )}
            </Box>

            <form onSubmit={handleProfileSubmit}>
              <Grid container spacing={3}>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="First Name"
                    name="firstName"
                    value={profileData.firstName}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonIcon className="field-icon" />
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
                    value={profileData.lastName}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PersonIcon className="field-icon" />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="Email"
                    value={user.email}
                    disabled
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <EmailIcon className="field-icon" />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="Phone"
                    name="phone"
                    value={profileData.phone}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <PhoneIcon className="field-icon" />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="Date of Birth"
                    name="dateOfBirth"
                    type="date"
                    value={profileData.dateOfBirth}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    InputLabelProps={{ shrink: true }}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <CakeIcon className="field-icon" />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    select
                    label="Gender"
                    name="gender"
                    value={profileData.gender || ''}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    InputLabelProps={{ shrink: true }}
                  >
                    <MenuItem value="">
                      <em>Select Gender</em>
                    </MenuItem>
                    <MenuItem value="Male">Male</MenuItem>
                    <MenuItem value="Female">Female</MenuItem>
                    <MenuItem value="Other">Other</MenuItem>
                  </TextField>
                </Grid>
                <Grid size={{ xs: 12 }}>
                  <TextField
                    fullWidth
                    label="Address"
                    name="address"
                    value={profileData.address}
                    onChange={handleProfileChange}
                    disabled={!isEditing}
                    multiline
                    rows={2}
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LocationIcon className="field-icon" />
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
              </Grid>

              {isEditing && (
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    type="submit"
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
                    disabled={loading}
                    className="save-btn"
                  >
                    {loading ? 'Saving...' : 'Save Changes'}
                  </Button>
                </Box>
              )}
            </form>

            <Divider sx={{ my: 4 }} />

            <Box className="account-info">
              <Typography variant="subtitle2" color="textSecondary">
                <CalendarIcon sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'text-bottom' }} />
                Member since: {formatDate(user.createdAt)}
              </Typography>
              {user.lastLogin && (
                <Typography variant="subtitle2" color="textSecondary">
                  Last login: {formatDate(user.lastLogin)}
                </Typography>
              )}
            </Box>
          </Box>
        </TabPanel>

        {/* Security Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box className="profile-tab-content">
            <Typography variant="h6" sx={{ mb: 3 }}>Change Password</Typography>
            
            <form onSubmit={handlePasswordSubmit}>
              <Grid container spacing={3}>
                <Grid size={{ xs: 12 }}>
                  <TextField
                    fullWidth
                    label="Current Password"
                    name="currentPassword"
                    type={showPasswords.current ? 'text' : 'password'}
                    value={passwordData.currentPassword}
                    onChange={handlePasswordChange}
                    required
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon className="field-icon" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPasswords(p => ({ ...p, current: !p.current }))}
                            edge="end"
                            size="small"
                          >
                            {showPasswords.current ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="New Password"
                    name="newPassword"
                    type={showPasswords.new ? 'text' : 'password'}
                    value={passwordData.newPassword}
                    onChange={handlePasswordChange}
                    required
                    helperText="At least 6 characters"
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon className="field-icon" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPasswords(p => ({ ...p, new: !p.new }))}
                            edge="end"
                            size="small"
                          >
                            {showPasswords.new ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
                <Grid size={{ xs: 12, sm: 6 }}>
                  <TextField
                    fullWidth
                    label="Confirm New Password"
                    name="confirmPassword"
                    type={showPasswords.confirm ? 'text' : 'password'}
                    value={passwordData.confirmPassword}
                    onChange={handlePasswordChange}
                    required
                    InputProps={{
                      startAdornment: (
                        <InputAdornment position="start">
                          <LockIcon className="field-icon" />
                        </InputAdornment>
                      ),
                      endAdornment: (
                        <InputAdornment position="end">
                          <IconButton
                            onClick={() => setShowPasswords(p => ({ ...p, confirm: !p.confirm }))}
                            edge="end"
                            size="small"
                          >
                            {showPasswords.confirm ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        </InputAdornment>
                      ),
                    }}
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  type="submit"
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
                  disabled={loading}
                  className="save-btn"
                >
                  {loading ? 'Updating...' : 'Update Password'}
                </Button>
              </Box>
            </form>

            <Divider sx={{ my: 4 }} />

            <Box className="danger-zone">
              <Typography variant="h6" color="error" sx={{ mb: 2 }}>
                Danger Zone
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                Once you delete your account, there is no going back. Please be certain.
              </Typography>
              <Button
                variant="outlined"
                color="error"
                startIcon={<DeleteIcon />}
                onClick={() => setDeleteDialogOpen(true)}
              >
                Delete Account
              </Button>
            </Box>
          </Box>
        </TabPanel>

        {/* History Tab */}
        <TabPanel value={tabValue} index={2}>
          <Box className="profile-tab-content">
            <Box className="profile-section-header">
              <Typography variant="h6">Assessment History</Typography>
              {assessmentHistory.length > 0 && (
                <Button
                  startIcon={<DashboardIcon />}
                  onClick={() => navigate('/dashboard')}
                  className="edit-btn"
                >
                  View Dashboard
                </Button>
              )}
            </Box>

            {assessmentHistory.length > 0 && (
              <Box className="history-summary-banner">
                <Box className="history-summary-item">
                  <SpeedIcon sx={{ color: '#00b2a7' }} />
                  <Box>
                    <Typography variant="caption" className="history-summary-label">Total Assessments</Typography>
                    <Typography variant="h6" className="history-summary-value">{assessmentHistory.length} / 7</Typography>
                  </Box>
                </Box>
                <Divider orientation="vertical" flexItem />
                <Box className="history-summary-item">
                  <TrendingUpIcon sx={{ color: '#00b2a7' }} />
                  <Box>
                    <Typography variant="caption" className="history-summary-label">Latest Probability</Typography>
                    <Typography variant="h6" className="history-summary-value">
                      {(assessmentHistory[0]?.probability * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                </Box>
                <Divider orientation="vertical" flexItem />
                <Box className="history-summary-item">
                  <AssessmentIcon sx={{ color: getRiskColor(assessmentHistory[0]?.riskLevel) }} />
                  <Box>
                    <Typography variant="caption" className="history-summary-label">Latest Risk</Typography>
                    <Chip
                      label={assessmentHistory[0]?.riskLevel}
                      size="small"
                      sx={{
                        backgroundColor: `${getRiskColor(assessmentHistory[0]?.riskLevel)}18`,
                        color: getRiskColor(assessmentHistory[0]?.riskLevel),
                        fontWeight: 700,
                        fontSize: '0.7rem',
                      }}
                    />
                  </Box>
                </Box>
              </Box>
            )}

            {historyLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : assessmentHistory.length === 0 ? (
              <Box className="empty-history">
                <AssessmentIcon sx={{ fontSize: 60, color: '#cbd5e1', mb: 2 }} />
                <Typography variant="h6" color="textSecondary">
                  No assessments yet
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                  Complete your first health assessment to see your history here.
                </Typography>
                <Button
                  variant="contained"
                  onClick={() => navigate('/assessment')}
                >
                  Start Assessment
                </Button>
              </Box>
            ) : (
              <Grid container spacing={2}>
                {assessmentHistory.slice().reverse().map((assessment, index) => {
                  const cardId = assessment._id || String(index);
                  const isExpanded = expandedCard === cardId;
                  return (
                    <Grid size={{ xs: 12 }} key={cardId}>
                      <Card className={`history-card ${isExpanded ? 'expanded' : ''}`}>
                        <CardContent>
                          <Box className="history-card-header">
                            <Box>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                <Chip
                                  label={`#${assessmentHistory.length - index}`}
                                  size="small"
                                  sx={{ 
                                    fontWeight: 800, 
                                    backgroundColor: '#f1f5f9', 
                                    color: '#475569',
                                    fontSize: '0.7rem',
                                  }}
                                />
                                <Typography variant="subtitle2" color="textSecondary">
                                  {formatDate(assessment.date)}
                                </Typography>
                              </Box>
                              <Chip
                                label={assessment.riskLevel}
                                size="small"
                                sx={{
                                  mt: 0.5,
                                  backgroundColor: `${getRiskColor(assessment.riskLevel)}20`,
                                  color: getRiskColor(assessment.riskLevel),
                                  fontWeight: 600,
                                }}
                              />
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                              <Box className="history-stats">
                                <Box className="history-stat">
                                  <TrendingUpIcon sx={{ color: '#00b2a7' }} />
                                  <Box>
                                    <Typography variant="caption" color="textSecondary">
                                      Probability
                                    </Typography>
                                    <Typography variant="h6">
                                      {(assessment.probability * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                </Box>
                                <Box className="history-stat">
                                  <AssessmentIcon sx={{ color: getRiskColor(assessment.riskLevel) }} />
                                  <Box>
                                    <Typography variant="caption" color="textSecondary">
                                      Severity
                                    </Typography>
                                    <Typography variant="h6">
                                      {(assessment.severity * 100).toFixed(1)}%
                                    </Typography>
                                  </Box>
                                </Box>
                              </Box>
                              <IconButton
                                onClick={() => toggleExpand(cardId)}
                                size="small"
                                sx={{
                                  backgroundColor: '#f1f5f9',
                                  transition: 'all 0.3s ease',
                                  '&:hover': { backgroundColor: '#e2e8f0' },
                                }}
                              >
                                {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                              </IconButton>
                            </Box>
                          </Box>

                          {/* Expanded Details */}
                          {isExpanded && (
                            <Box className="history-expanded-content">
                              <Divider sx={{ my: 2 }} />

                              {/* Input Parameters */}
                              {assessment.inputParameters && (
                                <>
                                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5, color: '#1e293b' }}>
                                    Health Parameters
                                  </Typography>
                                  <Box className="history-params-grid">
                                    <Box className="history-param-item">
                                      <Typography variant="caption">Age</Typography>
                                      <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.age} yrs</Typography>
                                    </Box>
                                    <Box className="history-param-item">
                                      <Typography variant="caption">Gender</Typography>
                                      <Typography variant="body2" fontWeight={700}>
                                        {assessment.inputParameters.gender === 'Men' ? 'Male' : 'Female'}
                                      </Typography>
                                    </Box>
                                    <Box className="history-param-item">
                                      <Typography variant="caption">Systolic BP</Typography>
                                      <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.systolicBP} mmHg</Typography>
                                    </Box>
                                    <Box className="history-param-item">
                                      <Typography variant="caption">Diastolic BP</Typography>
                                      <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.diastolicBP} mmHg</Typography>
                                    </Box>
                                    <Box className="history-param-item">
                                      <Typography variant="caption">Waist</Typography>
                                      <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.waistCircumference} cm</Typography>
                                    </Box>
                                    {assessment.inputParameters.hdlCholesterol && (
                                      <Box className="history-param-item">
                                        <Typography variant="caption">HDL</Typography>
                                        <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.hdlCholesterol} mg/dL</Typography>
                                      </Box>
                                    )}
                                    {assessment.inputParameters.triglyceride && (
                                      <Box className="history-param-item">
                                        <Typography variant="caption">Triglyceride</Typography>
                                        <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.triglyceride} mg/dL</Typography>
                                      </Box>
                                    )}
                                    {assessment.inputParameters.fpg && (
                                      <Box className="history-param-item">
                                        <Typography variant="caption">Glucose</Typography>
                                        <Typography variant="body2" fontWeight={700}>{assessment.inputParameters.fpg} mg/dL</Typography>
                                      </Box>
                                    )}
                                  </Box>

                                  {/* Conditions */}
                                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, mt: 1.5 }}>
                                    <Chip
                                      icon={assessment.inputParameters.fattyLiver ? <WarningIcon /> : <CheckCircleIcon />}
                                      label={`Fatty Liver: ${assessment.inputParameters.fattyLiver ? 'Yes' : 'No'}`}
                                      size="small"
                                      sx={{
                                        fontWeight: 600,
                                        borderRadius: '10px',
                                        backgroundColor: assessment.inputParameters.fattyLiver ? '#fef2f2' : '#ecfdf5',
                                        color: assessment.inputParameters.fattyLiver ? '#dc2626' : '#059669',
                                        border: `1px solid ${assessment.inputParameters.fattyLiver ? '#fecaca' : '#a7f3d0'}`,
                                        '& svg': { color: assessment.inputParameters.fattyLiver ? '#ef4444' : '#10b981' },
                                      }}
                                    />
                                    <Chip
                                      icon={assessment.inputParameters.hypertension ? <WarningIcon /> : <CheckCircleIcon />}
                                      label={`Hypertension: ${assessment.inputParameters.hypertension ? 'Yes' : 'No'}`}
                                      size="small"
                                      sx={{
                                        fontWeight: 600,
                                        borderRadius: '10px',
                                        backgroundColor: assessment.inputParameters.hypertension ? '#fef2f2' : '#ecfdf5',
                                        color: assessment.inputParameters.hypertension ? '#dc2626' : '#059669',
                                        border: `1px solid ${assessment.inputParameters.hypertension ? '#fecaca' : '#a7f3d0'}`,
                                        '& svg': { color: assessment.inputParameters.hypertension ? '#ef4444' : '#10b981' },
                                      }}
                                    />
                                    <Chip
                                      icon={assessment.inputParameters.diabetes ? <WarningIcon /> : <CheckCircleIcon />}
                                      label={`Diabetes: ${assessment.inputParameters.diabetes ? 'Yes' : 'No'}`}
                                      size="small"
                                      sx={{
                                        fontWeight: 600,
                                        borderRadius: '10px',
                                        backgroundColor: assessment.inputParameters.diabetes ? '#fef2f2' : '#ecfdf5',
                                        color: assessment.inputParameters.diabetes ? '#dc2626' : '#059669',
                                        border: `1px solid ${assessment.inputParameters.diabetes ? '#fecaca' : '#a7f3d0'}`,
                                        '& svg': { color: assessment.inputParameters.diabetes ? '#ef4444' : '#10b981' },
                                      }}
                                    />
                                  </Box>
                                </>
                              )}

                              {/* Recommendations Summary */}
                              {assessment.recommendations && (
                                <>
                                  <Divider sx={{ my: 2 }} />
                                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1.5, color: '#1e293b' }}>
                                    Recommendations Summary
                                  </Typography>
                                  <Box className="history-recs-summary">
                                    <Box className="history-rec-badge diet">
                                      <DietIcon fontSize="small" />
                                      <Typography variant="body2" fontWeight={700}>
                                        {assessment.recommendations.dietPlan?.length || 0}
                                      </Typography>
                                      <Typography variant="caption">Diet</Typography>
                                    </Box>
                                    <Box className="history-rec-badge avoid">
                                      <AvoidIcon fontSize="small" />
                                      <Typography variant="body2" fontWeight={700}>
                                        {assessment.recommendations.avoidList?.length || 0}
                                      </Typography>
                                      <Typography variant="caption">Avoid</Typography>
                                    </Box>
                                    <Box className="history-rec-badge exercise">
                                      <ExerciseIcon fontSize="small" />
                                      <Typography variant="body2" fontWeight={700}>
                                        {assessment.recommendations.exercisePlan?.length || 0}
                                      </Typography>
                                      <Typography variant="caption">Exercise</Typography>
                                    </Box>
                                    <Box className="history-rec-badge yoga">
                                      <YogaIcon fontSize="small" />
                                      <Typography variant="body2" fontWeight={700}>
                                        {assessment.recommendations.yogaPoses?.length || 0}
                                      </Typography>
                                      <Typography variant="caption">Yoga</Typography>
                                    </Box>
                                  </Box>
                                </>
                              )}
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            )}
          </Box>
        </TabPanel>
      </Paper>

      {/* Delete Account Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Account</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete your account? This action cannot be undone
            and all your data will be permanently removed.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteAccount} color="error" variant="contained">
            {loading ? <CircularProgress size={20} color="inherit" /> : 'Delete Account'}
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default Profile;
