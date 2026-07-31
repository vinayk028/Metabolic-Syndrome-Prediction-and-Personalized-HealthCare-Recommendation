/**
 * Dashboard Page - Personalized patient dashboard with charts & MetS tracking
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Button,
  Divider,
  LinearProgress,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  Assessment as AssessmentIcon,
  Favorite as HeartIcon,
  Bloodtype as BloodIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Restaurant as DietIcon,
  FitnessCenter as ExerciseIcon,
  SelfImprovement as YogaIcon,
  ArrowForward as ArrowForwardIcon,
  Refresh as RefreshIcon,
  CalendarToday as CalendarIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  LocalHospital as HospitalIcon,
} from '@mui/icons-material';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
  Cell,
} from 'recharts';
import { useAuthStore } from '../stores';
import { getAssessmentHistory } from '../data/api';
import type { AssessmentHistoryItem } from '../data/types';
import './Dashboard.css';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [assessments, setAssessments] = useState<AssessmentHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [authLoading, isAuthenticated, navigate]);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!user) return;
      setLoading(true);
      try {
        const response = await getAssessmentHistory();
        if (response.success) {
          setAssessments(response.assessmentHistory || []);
        }
      } catch (err) {
        console.error('Error fetching assessment history:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [user]);

  const handleRefresh = async () => {
    setLoading(true);
    try {
      const response = await getAssessmentHistory();
      if (response.success) {
        setAssessments(response.assessmentHistory || []);
      }
    } catch (err) {
      console.error('Error refreshing:', err);
    } finally {
      setLoading(false);
    }
  };

  // Sort assessments chronologically (oldest first for charts)
  const sortedAssessments = useMemo(
    () => [...assessments].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [assessments]
  );

  const latestAssessment = sortedAssessments.length > 0 ? sortedAssessments[sortedAssessments.length - 1] : null;
  const previousAssessment = sortedAssessments.length > 1 ? sortedAssessments[sortedAssessments.length - 2] : null;

  // Compute trends
  const probTrend = latestAssessment && previousAssessment
    ? latestAssessment.probability - previousAssessment.probability
    : 0;
  const sevTrend = latestAssessment && previousAssessment
    ? latestAssessment.severity - previousAssessment.severity
    : 0;

  // Chart data
  const trendChartData = useMemo(() =>
    sortedAssessments.map((a, i) => ({
      name: `#${i + 1}`,
      date: new Date(a.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      probability: +(a.probability * 100).toFixed(1),
      severity: +(a.severity * 100).toFixed(1),
    })),
    [sortedAssessments]
  );

  const riskDistribution = useMemo(() => {
    const counts = { 'Low Severity': 0, 'Medium Severity': 0, 'High Severity': 0 };
    sortedAssessments.forEach(a => {
      if (counts[a.riskLevel as keyof typeof counts] !== undefined) {
        counts[a.riskLevel as keyof typeof counts]++;
      }
    });
    return [
      { name: 'Low', value: counts['Low Severity'], fill: '#10b981' },
      { name: 'Medium', value: counts['Medium Severity'], fill: '#f59e0b' },
      { name: 'High', value: counts['High Severity'], fill: '#ef4444' },
    ];
  }, [sortedAssessments]);

  const vitalRadarData = useMemo(() => {
    if (!latestAssessment?.inputParameters) return [];
    const p = latestAssessment.inputParameters;
    return [
      { metric: 'Systolic BP', value: Math.min(((p.systolicBP || 120) / 200) * 100, 100), fullMark: 100 },
      { metric: 'Diastolic BP', value: Math.min(((p.diastolicBP || 80) / 130) * 100, 100), fullMark: 100 },
      { metric: 'Waist', value: Math.min(((p.waistCircumference || 75) / 150) * 100, 100), fullMark: 100 },
      { metric: 'HDL', value: Math.min(((p.hdlCholesterol || 50) / 100) * 100, 100), fullMark: 100 },
      { metric: 'Triglyceride', value: Math.min(((p.triglyceride || 150) / 500) * 100, 100), fullMark: 100 },
      { metric: 'Glucose', value: Math.min(((p.fpg || 90) / 200) * 100, 100), fullMark: 100 },
    ];
  }, [latestAssessment]);

  const formatDate = (dateString: string) =>
    new Date(dateString).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low Severity': return '#10b981';
      case 'Medium Severity': return '#f59e0b';
      case 'High Severity': return '#ef4444';
      default: return '#64748b';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'Low Severity': return <CheckCircleIcon />;
      case 'Medium Severity': return <WarningIcon />;
      case 'High Severity': return <HospitalIcon />;
      default: return <AssessmentIcon />;
    }
  };

  const getTrendIcon = (value: number) => {
    if (value > 0.01) return <TrendingUpIcon sx={{ color: '#ef4444' }} />;
    if (value < -0.01) return <TrendingDownIcon sx={{ color: '#10b981' }} />;
    return <TrendingFlatIcon sx={{ color: '#64748b' }} />;
  };

  const getTrendLabel = (value: number) => {
    const pct = Math.abs(value * 100).toFixed(1);
    if (value > 0.01) return `+${pct}%`;
    if (value < -0.01) return `-${pct}%`;
    return 'No change';
  };

  if (authLoading || !user) {
    return (
      <Box className="dashboard-loading">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>Loading Dashboard...</Typography>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box className="dashboard-loading">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>Fetching your health data...</Typography>
      </Box>
    );
  }

  return (
    <div className="dashboard-container">
      {/* Header */}
      <Box className="dashboard-header">
        <Box className="dashboard-header-left">
          <Typography variant="h3" className="dashboard-title">
            Health Dashboard
          </Typography>
          <Typography variant="body1" className="dashboard-subtitle">
            Welcome back, <strong>{user.firstName}</strong>. Here's your personalized MetS health overview.
          </Typography>
        </Box>
        <Box className="dashboard-header-actions">
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} className="refresh-btn">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<AssessmentIcon />}
            onClick={() => navigate('/assessment')}
            className="new-assessment-btn"
          >
            New Assessment
          </Button>
        </Box>
      </Box>

      {sortedAssessments.length === 0 ? (
        /* Empty State */
        <Paper className="dashboard-empty">
          <Box className="dashboard-empty-content">
            <AssessmentIcon className="empty-icon" />
            <Typography variant="h5">No Assessments Yet</Typography>
            <Typography variant="body1" color="textSecondary" sx={{ mb: 3, maxWidth: 480 }}>
              Take your first health assessment to unlock your personalized dashboard with trend charts,
              risk tracking, and health insights.
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<AssessmentIcon />}
              onClick={() => navigate('/assessment')}
              className="new-assessment-btn"
            >
              Start Your First Assessment
            </Button>
          </Box>
        </Paper>
      ) : (
        <>
          {/* Summary KPI Cards */}
          <Grid container spacing={3} className="kpi-section">
            {/* Latest Risk Level */}
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card className="kpi-card">
                <CardContent className="kpi-content">
                  <Box className="kpi-icon-wrapper" sx={{ background: `${getRiskColor(latestAssessment!.riskLevel)}18` }}>
                    {getRiskIcon(latestAssessment!.riskLevel)}
                  </Box>
                  <Box className="kpi-text">
                    <Typography variant="caption" className="kpi-label">Risk Level</Typography>
                    <Typography variant="h5" className="kpi-value" sx={{ color: getRiskColor(latestAssessment!.riskLevel) }}>
                      {latestAssessment!.riskLevel.replace(' Severity', '')}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Probability */}
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card className="kpi-card">
                <CardContent className="kpi-content">
                  <Box className="kpi-icon-wrapper probability">
                    <SpeedIcon />
                  </Box>
                  <Box className="kpi-text">
                    <Typography variant="caption" className="kpi-label">Probability</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="h5" className="kpi-value">
                        {(latestAssessment!.probability * 100).toFixed(1)}%
                      </Typography>
                      {previousAssessment && (
                        <Chip
                          icon={getTrendIcon(probTrend)}
                          label={getTrendLabel(probTrend)}
                          size="small"
                          className={`trend-chip ${probTrend > 0.01 ? 'up' : probTrend < -0.01 ? 'down' : 'flat'}`}
                        />
                      )}
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Severity */}
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card className="kpi-card">
                <CardContent className="kpi-content">
                  <Box className="kpi-icon-wrapper severity">
                    <HeartIcon />
                  </Box>
                  <Box className="kpi-text">
                    <Typography variant="caption" className="kpi-label">Severity Score</Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="h5" className="kpi-value">
                        {(latestAssessment!.severity * 100).toFixed(1)}%
                      </Typography>
                      {previousAssessment && (
                        <Chip
                          icon={getTrendIcon(sevTrend)}
                          label={getTrendLabel(sevTrend)}
                          size="small"
                          className={`trend-chip ${sevTrend > 0.01 ? 'up' : sevTrend < -0.01 ? 'down' : 'flat'}`}
                        />
                      )}
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Total Assessments */}
            <Grid size={{ xs: 12, sm: 6, md: 3 }}>
              <Card className="kpi-card">
                <CardContent className="kpi-content">
                  <Box className="kpi-icon-wrapper assessments">
                    <TimelineIcon />
                  </Box>
                  <Box className="kpi-text">
                    <Typography variant="caption" className="kpi-label">Assessments</Typography>
                    <Typography variant="h5" className="kpi-value">
                      {sortedAssessments.length} <span className="kpi-unit">/ 7</span>
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Charts Row */}
          <Grid container spacing={3} className="charts-section">
            {/* Trend Line Chart */}
            <Grid size={{ xs: 12, lg: 8 }}>
              <Paper className="chart-card">
                <Box className="chart-header">
                  <Box>
                    <Typography variant="h6" className="chart-title">
                      <TimelineIcon className="chart-title-icon" /> Risk Trend
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Probability & severity over last {sortedAssessments.length} assessments
                    </Typography>
                  </Box>
                </Box>
                <Box className="chart-body">
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={trendChartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="probGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00b2a7" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#00b2a7" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="sevGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} unit="%" />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #e2e8f0',
                          borderRadius: 12,
                          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        }}
                        formatter={(value) => [`${value}%`]}
                      />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="probability"
                        stroke="#00b2a7"
                        strokeWidth={3}
                        fill="url(#probGradient)"
                        dot={{ r: 5, fill: '#00b2a7', strokeWidth: 2, stroke: '#fff' }}
                        activeDot={{ r: 7 }}
                        name="Probability"
                      />
                      <Area
                        type="monotone"
                        dataKey="severity"
                        stroke="#f59e0b"
                        strokeWidth={3}
                        fill="url(#sevGradient)"
                        dot={{ r: 5, fill: '#f59e0b', strokeWidth: 2, stroke: '#fff' }}
                        activeDot={{ r: 7 }}
                        name="Severity"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>

            {/* Risk Distribution */}
            <Grid size={{ xs: 12, lg: 4 }}>
              <Paper className="chart-card">
                <Box className="chart-header">
                  <Box>
                    <Typography variant="h6" className="chart-title">
                      <AssessmentIcon className="chart-title-icon" /> Risk Distribution
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Across all {sortedAssessments.length} assessments
                    </Typography>
                  </Box>
                </Box>
                <Box className="chart-body">
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={riskDistribution} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={13} />
                      <YAxis allowDecimals={false} stroke="#94a3b8" fontSize={12} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #e2e8f0',
                          borderRadius: 12,
                          boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        }}
                        formatter={(value) => [`${value} assessment(s)`]}
                      />
                      <Bar dataKey="value" radius={[10, 10, 0, 0]} barSize={52} name="Count">
                        {riskDistribution.map((entry, index) => (
                          <Cell key={index} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>
          </Grid>

          {/* Vitals & Latest Details Row */}
          <Grid container spacing={3} className="details-section">
            {/* Radar Chart - Vitals */}
            {vitalRadarData.length > 0 && (
              <Grid size={{ xs: 12, md: 5 }}>
                <Paper className="chart-card">
                  <Box className="chart-header">
                    <Box>
                      <Typography variant="h6" className="chart-title">
                        <BloodIcon className="chart-title-icon" /> Latest Vitals Overview
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Normalized health parameters
                      </Typography>
                    </Box>
                  </Box>
                  <Box className="chart-body radar-body">
                    <ResponsiveContainer width="100%" height={340}>
                      <RadarChart outerRadius="70%" data={vitalRadarData}>
                        <PolarGrid stroke="#e2e8f0" />
                        <PolarAngleAxis dataKey="metric" stroke="#64748b" fontSize={12} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                        <Radar
                          name="Vitals"
                          dataKey="value"
                          stroke="#00b2a7"
                          fill="#00b2a7"
                          fillOpacity={0.2}
                          strokeWidth={2}
                          dot={{ r: 4, fill: '#00b2a7' }}
                        />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: 12,
                            boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                          }}
                          formatter={(value) => [`${Number(value).toFixed(0)}%`, 'Normalized']}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </Box>
                </Paper>
              </Grid>
            )}

            {/* Latest Assessment Details */}
            <Grid size={{ xs: 12, md: vitalRadarData.length > 0 ? 7 : 12 }}>
              <Paper className="chart-card latest-details-card">
                <Box className="chart-header">
                  <Box>
                    <Typography variant="h6" className="chart-title">
                      <CalendarIcon className="chart-title-icon" /> Latest Assessment Details
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {latestAssessment && formatDate(latestAssessment.date)}
                    </Typography>
                  </Box>
                  <Chip
                    label={latestAssessment?.riskLevel}
                    sx={{
                      backgroundColor: `${getRiskColor(latestAssessment!.riskLevel)}18`,
                      color: getRiskColor(latestAssessment!.riskLevel),
                      fontWeight: 700,
                      border: `1px solid ${getRiskColor(latestAssessment!.riskLevel)}40`,
                    }}
                  />
                </Box>

                <Box className="latest-details-body">
                  {/* Progress Bars */}
                  <Box className="detail-progress-group">
                    <Box className="detail-progress-item">
                      <Box className="detail-progress-label">
                        <Typography variant="body2" fontWeight={600}>MetS Probability</Typography>
                        <Typography variant="body2" fontWeight={700} color="#00b2a7">
                          {(latestAssessment!.probability * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={latestAssessment!.probability * 100}
                        className="detail-progress-bar probability-bar"
                      />
                    </Box>
                    <Box className="detail-progress-item">
                      <Box className="detail-progress-label">
                        <Typography variant="body2" fontWeight={600}>Severity Score</Typography>
                        <Typography variant="body2" fontWeight={700} sx={{ color: getRiskColor(latestAssessment!.riskLevel) }}>
                          {(latestAssessment!.severity * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={latestAssessment!.severity * 100}
                        className="detail-progress-bar severity-bar"
                        sx={{
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getRiskColor(latestAssessment!.riskLevel),
                          }
                        }}
                      />
                    </Box>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  {/* Input Parameters Grid */}
                  {latestAssessment?.inputParameters && (
                    <Box className="vitals-grid">
                      <Box className="vital-item">
                        <Typography variant="caption">Age</Typography>
                        <Typography variant="h6">{latestAssessment.inputParameters.age} yrs</Typography>
                      </Box>
                      <Box className="vital-item">
                        <Typography variant="caption">Gender</Typography>
                        <Typography variant="h6">{latestAssessment.inputParameters.gender === 'Men' ? 'Male' : 'Female'}</Typography>
                      </Box>
                      <Box className="vital-item">
                        <Typography variant="caption">Systolic BP</Typography>
                        <Typography variant="h6">{latestAssessment.inputParameters.systolicBP} <span className="vital-unit">mmHg</span></Typography>
                      </Box>
                      <Box className="vital-item">
                        <Typography variant="caption">Diastolic BP</Typography>
                        <Typography variant="h6">{latestAssessment.inputParameters.diastolicBP} <span className="vital-unit">mmHg</span></Typography>
                      </Box>
                      <Box className="vital-item">
                        <Typography variant="caption">Waist</Typography>
                        <Typography variant="h6">{latestAssessment.inputParameters.waistCircumference} <span className="vital-unit">cm</span></Typography>
                      </Box>
                      {latestAssessment.inputParameters.hdlCholesterol && (
                        <Box className="vital-item">
                          <Typography variant="caption">HDL</Typography>
                          <Typography variant="h6">{latestAssessment.inputParameters.hdlCholesterol} <span className="vital-unit">mg/dL</span></Typography>
                        </Box>
                      )}
                      {latestAssessment.inputParameters.triglyceride && (
                        <Box className="vital-item">
                          <Typography variant="caption">Triglyceride</Typography>
                          <Typography variant="h6">{latestAssessment.inputParameters.triglyceride} <span className="vital-unit">mg/dL</span></Typography>
                        </Box>
                      )}
                      {latestAssessment.inputParameters.fpg && (
                        <Box className="vital-item">
                          <Typography variant="caption">Glucose</Typography>
                          <Typography variant="h6">{latestAssessment.inputParameters.fpg} <span className="vital-unit">mg/dL</span></Typography>
                        </Box>
                      )}
                    </Box>
                  )}

                  <Divider sx={{ my: 2 }} />

                  {/* Conditions */}
                  {latestAssessment?.inputParameters && (
                    <Box className="conditions-row">
                      <Chip
                        icon={latestAssessment.inputParameters.fattyLiver ? <WarningIcon /> : <CheckCircleIcon />}
                        label={`Fatty Liver: ${latestAssessment.inputParameters.fattyLiver ? 'Yes' : 'No'}`}
                        size="small"
                        className={`condition-chip ${latestAssessment.inputParameters.fattyLiver ? 'positive' : 'negative'}`}
                      />
                      <Chip
                        icon={latestAssessment.inputParameters.hypertension ? <WarningIcon /> : <CheckCircleIcon />}
                        label={`Hypertension: ${latestAssessment.inputParameters.hypertension ? 'Yes' : 'No'}`}
                        size="small"
                        className={`condition-chip ${latestAssessment.inputParameters.hypertension ? 'positive' : 'negative'}`}
                      />
                      <Chip
                        icon={latestAssessment.inputParameters.diabetes ? <WarningIcon /> : <CheckCircleIcon />}
                        label={`Diabetes: ${latestAssessment.inputParameters.diabetes ? 'Yes' : 'No'}`}
                        size="small"
                        className={`condition-chip ${latestAssessment.inputParameters.diabetes ? 'positive' : 'negative'}`}
                      />
                    </Box>
                  )}
                </Box>
              </Paper>
            </Grid>
          </Grid>

          {/* Latest Recommendations */}
          {latestAssessment?.recommendations && (
            <Paper className="recommendations-overview">
              <Box className="chart-header">
                <Box>
                  <Typography variant="h6" className="chart-title">
                    <DietIcon className="chart-title-icon" /> Latest Recommendations Summary
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Based on your most recent assessment
                  </Typography>
                </Box>
                <Button
                  variant="outlined"
                  size="small"
                  endIcon={<ArrowForwardIcon />}
                  onClick={() => navigate('/profile')}
                  className="view-all-btn"
                >
                  View Full History
                </Button>
              </Box>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                  <Box className="rec-summary-card diet">
                    <DietIcon className="rec-icon" />
                    <Typography variant="h4" className="rec-count">
                      {latestAssessment.recommendations.dietPlan?.length || 0}
                    </Typography>
                    <Typography variant="body2">Diet Recommendations</Typography>
                  </Box>
                </Grid>
                <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                  <Box className="rec-summary-card avoid">
                    <WarningIcon className="rec-icon" />
                    <Typography variant="h4" className="rec-count">
                      {latestAssessment.recommendations.avoidList?.length || 0}
                    </Typography>
                    <Typography variant="body2">Foods to Avoid</Typography>
                  </Box>
                </Grid>
                <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                  <Box className="rec-summary-card exercise">
                    <ExerciseIcon className="rec-icon" />
                    <Typography variant="h4" className="rec-count">
                      {latestAssessment.recommendations.exercisePlan?.length || 0}
                    </Typography>
                    <Typography variant="body2">Exercise Plans</Typography>
                  </Box>
                </Grid>
                <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                  <Box className="rec-summary-card yoga">
                    <YogaIcon className="rec-icon" />
                    <Typography variant="h4" className="rec-count">
                      {latestAssessment.recommendations.yogaPoses?.length || 0}
                    </Typography>
                    <Typography variant="body2">Yoga Poses</Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>
          )}

          {/* Assessment Timeline */}
          <Paper className="timeline-card">
            <Box className="chart-header">
              <Box>
                <Typography variant="h6" className="chart-title">
                  <CalendarIcon className="chart-title-icon" /> Assessment Timeline
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Your last {sortedAssessments.length} assessment(s)
                </Typography>
              </Box>
            </Box>
            <Box className="timeline-body">
              {[...sortedAssessments].reverse().map((assessment, index) => (
                <Box className="timeline-item" key={assessment._id || index}>
                  <Box className="timeline-dot" sx={{ borderColor: getRiskColor(assessment.riskLevel) }}>
                    <Box className="timeline-dot-inner" sx={{ backgroundColor: getRiskColor(assessment.riskLevel) }} />
                  </Box>
                  {index < sortedAssessments.length - 1 && <Box className="timeline-line" />}
                  <Box className="timeline-content">
                    <Box className="timeline-content-header">
                      <Typography variant="subtitle2" className="timeline-date">
                        {formatDate(assessment.date)}
                      </Typography>
                      <Chip
                        label={assessment.riskLevel}
                        size="small"
                        sx={{
                          backgroundColor: `${getRiskColor(assessment.riskLevel)}18`,
                          color: getRiskColor(assessment.riskLevel),
                          fontWeight: 700,
                          fontSize: '0.7rem',
                        }}
                      />
                    </Box>
                    <Box className="timeline-metrics">
                      <Typography variant="body2">
                        Probability: <strong>{(assessment.probability * 100).toFixed(1)}%</strong>
                      </Typography>
                      <Typography variant="body2">
                        Severity: <strong>{(assessment.severity * 100).toFixed(1)}%</strong>
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          </Paper>
        </>
      )}
    </div>
  );
};

export default Dashboard;
