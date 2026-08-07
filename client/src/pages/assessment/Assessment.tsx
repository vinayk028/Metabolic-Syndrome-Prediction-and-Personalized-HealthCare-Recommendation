import {
  Box,
  Typography,
  Card,
  CardContent,
  Stepper,
  Step,
  StepLabel,
  Button,
  TextField,
  FormControl,
  FormLabel,
  RadioGroup,
  FormControlLabel,
  Radio,
  Slider,
  Switch,
  Grid,
  Alert,
  LinearProgress,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Checkbox,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  Download as DownloadIcon,
  Restaurant as DietIcon,
  FitnessCenter as ExerciseIcon,
  SelfImprovement as YogaIcon,
  Block as AvoidIcon,
  Warning as WarningIcon,
} from "@mui/icons-material";
import { useAssessmentStore } from "../../stores/assessmentStore";
import ASSESSMENT_BANNER from '../../assets/assessment_banner.jpg';
import "./Assessment.css";

// Steps will be dynamic based on whether user has metabolic syndrome
const STEPS_WITH_ADDITIONAL = [
  "Basic Information",
  "Additional Information",
  "Results & Recommendations",
];
const STEPS_WITHOUT_ADDITIONAL = [
  "Basic Information",
  "Results & Recommendations",
];

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box className="tab-panel">{children}</Box>}
    </div>
  );
}

const Assessment = () => {
  const {
    activeStep,
    loading,
    error,
    tabValue,
    termsAccepted,
    termsOpen,
    termsCheckbox,
    patientInfo,
    additionalInfo,
    results,
    recommendations,
    setPatientInfo,
    setAdditionalInfo,
    setTabValue,
    setError,
    setActiveStep,
    setTermsCheckbox,
    acceptTerms,
    predict,
    calculateSeverity: handleSeverityCalculation,
    downloadReport: handleDownloadReport,
    startNewAssessment,
  } = useAssessmentStore();

  // Determine which steps to show based on metabolic syndrome status
  const steps = results.hasMetabolicSyndrome
    ? STEPS_WITH_ADDITIONAL
    : STEPS_WITHOUT_ADDITIONAL;

  const getRiskColor = (probability: number) => {
    if (probability < 0.35) return "success";
    if (probability < 0.65) return "warning";
    return "error";
  };

  const getSeverityColor = (riskLevel?: string) => {
    if (riskLevel === "Low Severity") return "success";
    if (riskLevel === "Medium Severity") return "warning";
    return "error";
  };

  // Determine if we should show results
  const showResults =
    (results.hasMetabolicSyndrome && activeStep === 2) ||
    (!results.hasMetabolicSyndrome &&
      activeStep === 1 &&
      results.probability > 0);

  return (
    <Box className="assessment-page">
      {/* Healthcare Banner */}
      <Box className="assessment-banner">
        <img
          src={ASSESSMENT_BANNER}
          alt="Medical consultation"
          className="assessment-banner-image"
        />
        <Box className="assessment-banner-overlay"></Box>
        <Box className="assessment-banner-content">
          <Typography variant="h2" className="page-title">
            Health Assessment
          </Typography>
          <Typography variant="h6" className="page-subtitle">
            Complete the assessment to get your personalized health
            recommendations
          </Typography>
        </Box>
      </Box>

      <Card className="stepper-card">
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Step 1: Basic Information */}
      {activeStep === 0 && (
        <Card className="form-card">
          <CardContent>
            <Typography variant="h5" className="form-title">
              Patient Information
            </Typography>
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 6 }}>
                <Box className="form-field">
                  <Typography gutterBottom>
                    Age: {patientInfo.age} years
                  </Typography>
                  <Slider
                    value={patientInfo.age}
                    onChange={(_, value) =>
                      setPatientInfo("age", value as number)
                    }
                    min={20}
                    max={80}
                    valueLabelDisplay="auto"
                  />
                </Box>
              </Grid>
              <Grid size={{ xs: 12, md: 6 }}>
                <FormControl component="fieldset">
                  <FormLabel>Gender</FormLabel>
                  <RadioGroup
                    row
                    value={patientInfo.gender}
                    onChange={(e) => setPatientInfo("gender", e.target.value)}
                  >
                    <FormControlLabel
                      value="Men"
                      control={<Radio />}
                      label="Male"
                    />
                    <FormControlLabel
                      value="Women"
                      control={<Radio />}
                      label="Female"
                    />
                  </RadioGroup>
                </FormControl>
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <Box className="switch-field">
                  <Typography>Fatty Liver Diagnosis</Typography>
                  <Switch
                    checked={patientInfo.fattyLiver}
                    onChange={(e) =>
                      setPatientInfo("fattyLiver", e.target.checked)
                    }
                  />
                </Box>
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <Box className="switch-field">
                  <Typography>Hypertension Diagnosis</Typography>
                  <Switch
                    checked={patientInfo.hypertension}
                    onChange={(e) =>
                      setPatientInfo("hypertension", e.target.checked)
                    }
                  />
                </Box>
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <Box className="switch-field">
                  <Typography>Diabetes Diagnosis</Typography>
                  <Switch
                    checked={patientInfo.diabetes}
                    onChange={(e) =>
                      setPatientInfo("diabetes", e.target.checked)
                    }
                  />
                </Box>
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="Systolic Blood Pressure (mmHg)"
                  type="number"
                  value={patientInfo.systolicBP}
                  onChange={(e) =>
                    setPatientInfo("systolicBP", parseInt(e.target.value))
                  }
                  inputProps={{ min: 70, max: 200 }}
                />
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="Diastolic Blood Pressure (mmHg)"
                  type="number"
                  value={patientInfo.diastolicBP}
                  onChange={(e) =>
                    setPatientInfo("diastolicBP", parseInt(e.target.value))
                  }
                  inputProps={{ min: 40, max: 130 }}
                />
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="Waist Circumference (cm)"
                  type="number"
                  value={patientInfo.waistCircumference}
                  onChange={(e) =>
                    setPatientInfo(
                      "waistCircumference",
                      parseInt(e.target.value)
                    )
                  }
                  inputProps={{ min: 50, max: 150 }}
                />
              </Grid>
            </Grid>
            <Box className="form-actions">
              <Button
                variant="contained"
                size="large"
                onClick={predict}
                disabled={loading}
                className="submit-button"
              >
                {loading ? <CircularProgress size={24} /> : "Predict Risk"}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Step 2: Additional Information - Only shown if user has metabolic syndrome */}
      {results.hasMetabolicSyndrome && activeStep === 1 && (
        <Card className="form-card">
          <CardContent>
            <Alert severity="warning" className="step-alert">
              You have a high probability of Metabolic Syndrome. Please provide
              additional information for severity assessment.
            </Alert>
            <Typography variant="h5" className="form-title">
              Additional Information
            </Typography>
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="HDL Cholesterol (mg/dL)"
                  type="number"
                  value={additionalInfo.hdlCholesterol}
                  onChange={(e) =>
                    setAdditionalInfo(
                      "hdlCholesterol",
                      parseInt(e.target.value)
                    )
                  }
                  inputProps={{ min: 20, max: 100 }}
                  helperText="Good cholesterol level"
                />
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="Triglyceride (mg/dL)"
                  type="number"
                  value={additionalInfo.triglyceride}
                  onChange={(e) =>
                    setAdditionalInfo("triglyceride", parseInt(e.target.value))
                  }
                  inputProps={{ min: 50, max: 500 }}
                  helperText="Fat in your blood"
                />
              </Grid>
              <Grid size={{ xs: 12, md: 4 }}>
                <TextField
                  fullWidth
                  label="Fasting Plasma Glucose (mg/dL)"
                  type="number"
                  value={additionalInfo.fpg}
                  onChange={(e) =>
                    setAdditionalInfo("fpg", parseInt(e.target.value))
                  }
                  inputProps={{ min: 70, max: 200 }}
                  helperText="Blood sugar when fasting"
                />
              </Grid>
            </Grid>
            <Box className="form-actions">
              <Button variant="outlined" onClick={() => setActiveStep(0)}>
                Back
              </Button>
              <Button
                variant="contained"
                size="large"
                onClick={handleSeverityCalculation}
                disabled={loading}
                className="submit-button"
              >
                {loading ? (
                  <CircularProgress size={24} />
                ) : (
                  "Calculate Severity"
                )}
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Results & Recommendations - Shown at the final step */}
      {showResults && (
        <Box className="results-section">
          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 6 }}>
              <Card className="result-card">
                <CardContent>
                  <Typography variant="h5" className="result-title">
                    Risk Assessment
                  </Typography>
                  <Box className="progress-container">
                    <Typography variant="body2" className="progress-label">
                      Probability: {(results.probability * 100).toFixed(1)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={results.probability * 100}
                      color={getRiskColor(results.probability)}
                      className="risk-progress"
                    />
                  </Box>
                  {results.severity !== undefined && (
                    <Box className="progress-container">
                      <Typography variant="body2" className="progress-label">
                        Severity: {(results.severity * 100).toFixed(1)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={results.severity * 100}
                        color={getSeverityColor(results.riskLevel)}
                        className="risk-progress"
                      />
                    </Box>
                  )}
                  <Alert
                    severity={
                      results.hasMetabolicSyndrome
                        ? getSeverityColor(results.riskLevel)
                        : "success"
                    }
                    className="result-alert"
                  >
                    {results.riskLevel ||
                      (results.hasMetabolicSyndrome ? "High Risk" : "Low Risk")}
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <Card className="metrics-card">
                <CardContent>
                  <Typography variant="h5" className="result-title">
                    Your Health Metrics
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid size={{ xs: 6 }}>
                      <Box className="metric-item">
                        <Typography variant="body2">Age</Typography>
                        <Typography variant="h6">
                          {patientInfo.age} years
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 6 }}>
                      <Box className="metric-item">
                        <Typography variant="body2">Gender</Typography>
                        <Typography variant="h6">
                          {patientInfo.gender}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 6 }}>
                      <Box className="metric-item">
                        <Typography variant="body2">Blood Pressure</Typography>
                        <Typography variant="h6">
                          {patientInfo.systolicBP}/{patientInfo.diastolicBP}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid size={{ xs: 6 }}>
                      <Box className="metric-item">
                        <Typography variant="body2">Waist</Typography>
                        <Typography variant="h6">
                          {patientInfo.waistCircumference} cm
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Card className="recommendations-card">
            <CardContent>
              <Typography variant="h5" className="recommendations-title">
                Health Recommendations
              </Typography>
              <Tabs
                value={tabValue}
                onChange={(_, newValue) => setTabValue(newValue)}
                className="recommendations-tabs"
                variant="scrollable"
                scrollButtons="auto"
              >
                <Tab icon={<DietIcon />} label="Diet Plan" />
                <Tab icon={<AvoidIcon />} label="Foods to Avoid" />
                <Tab icon={<ExerciseIcon />} label="Exercise Plan" />
                <Tab icon={<YogaIcon />} label="Yoga Poses" />
              </Tabs>

              <TabPanel value={tabValue} index={0}>
                <List>
                  {recommendations.dietPlan.length > 0 ? (
                    recommendations.dietPlan.map((item, index) => (
                      <ListItem key={index} className="recommendation-item">
                        <ListItemIcon>
                          <CheckIcon className="check-icon" />
                        </ListItemIcon>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))
                  ) : (
                    <Typography className="no-data">
                      No diet recommendations available
                    </Typography>
                  )}
                </List>
              </TabPanel>

              <TabPanel value={tabValue} index={1}>
                <List>
                  {recommendations.avoidList.length > 0 ? (
                    recommendations.avoidList.map((item, index) => (
                      <ListItem
                        key={index}
                        className="recommendation-item avoid"
                      >
                        <ListItemIcon>
                          <CancelIcon className="cancel-icon" />
                        </ListItemIcon>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))
                  ) : (
                    <Typography className="no-data">
                      No items to avoid listed
                    </Typography>
                  )}
                </List>
              </TabPanel>

              <TabPanel value={tabValue} index={2}>
                <List>
                  {recommendations.exercisePlan.length > 0 ? (
                    recommendations.exercisePlan.map((item, index) => (
                      <ListItem key={index} className="recommendation-item">
                        <ListItemIcon>
                          <CheckIcon className="check-icon" />
                        </ListItemIcon>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))
                  ) : (
                    <Typography className="no-data">
                      No exercise recommendations available
                    </Typography>
                  )}
                </List>
              </TabPanel>

              <TabPanel value={tabValue} index={3}>
                <List>
                  {recommendations.yogaPoses.length > 0 ? (
                    recommendations.yogaPoses.map((item, index) => (
                      <ListItem key={index} className="recommendation-item">
                        <ListItemIcon>
                          <CheckIcon className="check-icon" />
                        </ListItemIcon>
                        <ListItemText primary={item} />
                      </ListItem>
                    ))
                  ) : (
                    <Typography className="no-data">
                      No yoga recommendations available
                    </Typography>
                  )}
                </List>
              </TabPanel>

              <Box className="download-section">
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<DownloadIcon />}
                  onClick={handleDownloadReport}
                  className="download-button"
                >
                  Download Health Plan
                </Button>
              </Box>
            </CardContent>
          </Card>

          <Box className="restart-section">
            <Button variant="outlined" onClick={startNewAssessment}>
              Start New Assessment
            </Button>
          </Box>
        </Box>
      )}

      {/* Terms and Conditions Dialog */}
      <Dialog
        open={termsOpen && !termsAccepted}
        onClose={() => {}}
        maxWidth="md"
        fullWidth
        className="terms-dialog"
        disableEscapeKeyDown
      >
        <DialogTitle className="terms-dialog-title">
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <WarningIcon sx={{ color: "#f59e0b" }} />
            <Typography variant="h5" component="span">
              Important Disclaimer & Terms of Use
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent className="terms-dialog-content">
          <Alert severity="warning" sx={{ mb: 2 }}>
            Please read these terms carefully before using the Health Assessment
            Tool.
          </Alert>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>
            1. Medical Disclaimer
          </Typography>
          <Typography paragraph>
            This tool is designed for{" "}
            <strong>educational and informational purposes only</strong>. It is
            NOT intended to be a substitute for professional medical advice,
            diagnosis, or treatment. Always seek the advice of your physician or
            other qualified health provider with any questions you may have
            regarding a medical condition.
          </Typography>

          <Typography variant="h6" gutterBottom>
            2. Accuracy of Results
          </Typography>
          <Typography paragraph>
            The predictions and recommendations provided by this tool are based
            on statistical models and general health guidelines. Individual
            results may vary, and the tool cannot account for all personal
            health factors. The accuracy of results depends on the accuracy of
            the information you provide.
          </Typography>

          <Typography variant="h6" gutterBottom>
            3. No Doctor-Patient Relationship
          </Typography>
          <Typography paragraph>
            Using this assessment tool does not create a doctor-patient
            relationship. The recommendations provided are general in nature and
            should be discussed with your healthcare provider before making any
            changes to your diet, exercise, or medication regimen.
          </Typography>

          <Typography variant="h6" gutterBottom>
            4. Emergency Situations
          </Typography>
          <Typography paragraph>
            If you are experiencing a medical emergency, please call your local
            emergency services immediately. Do not rely on this tool for
            emergency medical situations.
          </Typography>

          <Typography variant="h6" gutterBottom>
            5. Data Privacy
          </Typography>
          <Typography paragraph>
            The health information you enter is processed locally and is not
            stored on our servers permanently. However, we recommend not
            entering any personally identifiable information beyond what is
            necessary for the assessment.
          </Typography>

          <Typography variant="h6" gutterBottom>
            6. Limitation of Liability
          </Typography>
          <Typography paragraph>
            The creators and operators of this tool shall not be liable for any
            damages arising from the use of this assessment tool or the
            recommendations provided. Use this tool at your own risk.
          </Typography>
        </DialogContent>
        <DialogActions className="terms-dialog-actions">
          <Box
            sx={{
              width: "100%",
              display: "flex",
              flexDirection: "column",
              gap: 2,
            }}
          >
            <FormControlLabel
              control={
                <Checkbox
                  checked={termsCheckbox}
                  onChange={(e) => setTermsCheckbox(e.target.checked)}
                  sx={{
                    color: "#cbd5e1",
                    "&.Mui-checked": {
                      color: "#10b981",
                    },
                    "& .MuiSvgIcon-root": {
                      fontSize: "1.5rem",
                    },
                    "&.Mui-checked .MuiSvgIcon-root": {
                      backgroundColor: "#10b981",
                      borderRadius: "4px",
                      color: "#ffffff",
                    },
                  }}
                />
              }
              label={
                <Typography>
                  I have read, understood, and agree to these terms and
                  conditions. I understand that this tool is for informational
                  purposes only and is not a substitute for professional medical
                  advice.
                </Typography>
              }
            />
            <Button
              variant="contained"
              onClick={acceptTerms}
              disabled={!termsCheckbox}
              fullWidth
              size="large"
              className="accept-terms-button"
            >
              Accept & Continue to Assessment
            </Button>
          </Box>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Assessment;
