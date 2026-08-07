import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Healing as HealingIcon,
  MonitorHeart as HeartMonitorIcon,
  Bloodtype as BloodIcon,
  Scale as ScaleIcon,
  Speed as SpeedIcon,
  LocalDrink as DrinkIcon,
} from '@mui/icons-material';
import ABOUT_HERO_IMAGE from '../../assets/about_hero.jpg';
import DOCTOR_IMAGE from '../../assets/doctor.jpg';
import HEALTH_CHECK_IMAGE from '../../assets/health_check.jpg';
import './About.css';

const conditions = [
  {
    icon: <ScaleIcon />,
    title: 'Increased Waist Circumference',
    description: 'Abdominal obesity - Men: >102 cm, Women: >88 cm',
    colorClass: 'condition-orange',
  },
  {
    icon: <BloodIcon />,
    title: 'High Triglyceride Levels',
    description: 'Triglycerides ≥150 mg/dL or on medication',
    colorClass: 'condition-red',
  },
  {
    icon: <HeartMonitorIcon />,
    title: 'Low HDL Cholesterol',
    description: 'Men: <40 mg/dL, Women: <50 mg/dL',
    colorClass: 'condition-rose',
  },
  {
    icon: <SpeedIcon />,
    title: 'High Blood Pressure',
    description: 'Systolic ≥130 or Diastolic ≥85 mmHg',
    colorClass: 'condition-purple',
  },
  {
    icon: <DrinkIcon />,
    title: 'High Fasting Blood Sugar',
    description: 'Fasting glucose ≥100 mg/dL',
    colorClass: 'condition-blue',
  },
];

const riskFactors = [
  'Age (risk increases with age)',
  'Obesity (particularly abdominal obesity)',
  'Physical inactivity',
  'Insulin resistance',
  'Genetics and family history',
  'Hormonal imbalances',
  'Poor diet high in processed foods',
  'Chronic stress',
];

const preventionTips = [
  'Regular physical activity (30 min/day)',
  'Healthy diet rich in fruits, vegetables, whole grains',
  'Weight loss (if overweight)',
  'Smoking cessation',
  'Limiting alcohol consumption',
  'Regular health check-ups',
  'Stress management',
  'Adequate sleep (7-9 hours)',
];

const About = () => {
  return (
    <Box className="about-page">
      <Box className="page-header">
        <Typography variant="h2" className="page-title">
          About Metabolic Syndrome
        </Typography>
        <Typography variant="h6" className="page-subtitle">
          Understanding the condition that affects millions worldwide
        </Typography>
      </Box>

      <Card className="intro-card">
        <CardContent>
          <Box className="intro-content-wrapper">
            <Box className="intro-image-container">
              <img 
                src={ABOUT_HERO_IMAGE} 
                alt="Medical professionals team" 
                className="intro-image"
              />
            </Box>
            <Box className="intro-text-content">
              <Box className="intro-icon">
                <HealingIcon />
              </Box>
              <Typography variant="h4" className="intro-title">
                What is Metabolic Syndrome?
              </Typography>
              <Typography variant="body1" className="intro-text">
                Metabolic syndrome is a cluster of interconnected metabolic abnormalities that 
                significantly increase your risk of developing cardiovascular disease, type 2 diabetes, 
                and stroke. It's not a disease itself but rather a group of risk factors that often 
                occur together. Having three or more of these conditions qualifies as metabolic syndrome.
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Box className="conditions-section">
        <Typography variant="h4" className="section-title">
          The Five Conditions
        </Typography>
        <Typography variant="body1" className="section-subtitle">
          Having three or more of these conditions indicates metabolic syndrome
        </Typography>
        <Grid container spacing={3}>
          {conditions.map((condition, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={index}>
              <Card className="condition-card">
                <CardContent>
                  <Box className={`condition-icon ${condition.colorClass}`}>{condition.icon}</Box>
                  <Typography variant="h6" className="condition-title">
                    {condition.title}
                  </Typography>
                  <Typography variant="body2" className="condition-description">
                    {condition.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      <Grid container spacing={3} className="lists-section">
        <Grid size={{ xs: 12, md: 6 }}>
          <Card className="list-card risk-card">
            <Box className="list-card-image-container">
              <img 
                src={DOCTOR_IMAGE} 
                alt="Doctor consultation" 
                className="list-card-image"
              />
            </Box>
            <CardContent className="list-card-content">
              <Box className="list-header">
                <WarningIcon className="list-icon warning" />
                <Typography variant="h5" className="list-title">
                  Risk Factors
                </Typography>
              </Box>
              <List>
                {riskFactors.map((factor, index) => (
                  <ListItem key={index} className="list-item">
                    <ListItemIcon>
                      <WarningIcon className="item-icon warning" />
                    </ListItemIcon>
                    <ListItemText primary={factor} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card className="list-card prevention-card">
            <Box className="list-card-image-container">
              <img 
                src={HEALTH_CHECK_IMAGE} 
                alt="Health checkup and prevention" 
                className="list-card-image"
              />
            </Box>
            <CardContent className="list-card-content">
              <Box className="list-header">
                <CheckIcon className="list-icon success" />
                <Typography variant="h5" className="list-title">
                  Prevention & Management
                </Typography>
              </Box>
              <List>
                {preventionTips.map((tip, index) => (
                  <ListItem key={index} className="list-item">
                    <ListItemIcon>
                      <CheckIcon className="item-icon success" />
                    </ListItemIcon>
                    <ListItemText primary={tip} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card className="importance-card">
        <CardContent>
          <Typography variant="h4" className="importance-title">
            Why Early Detection Matters
          </Typography>
          <Grid container spacing={4} className="importance-grid">
            <Grid size={{ xs: 12, md: 4 }}>
              <Box className="importance-item">
                <Typography variant="h2" className="importance-number">5x</Typography>
                <Typography variant="body1" className="importance-text">
                  higher risk of developing type 2 diabetes
                </Typography>
              </Box>
            </Grid>
            <Grid size={{ xs: 12, md: 4 }}>
              <Box className="importance-item">
                <Typography variant="h2" className="importance-number">2x</Typography>
                <Typography variant="body1" className="importance-text">
                  higher risk of cardiovascular disease
                </Typography>
              </Box>
            </Grid>
            <Grid size={{ xs: 12, md: 4 }}>
              <Box className="importance-item">
                <Typography variant="h2" className="importance-number">80%</Typography>
                <Typography variant="body1" className="importance-text">
                  of cases can be managed with lifestyle changes
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default About;
