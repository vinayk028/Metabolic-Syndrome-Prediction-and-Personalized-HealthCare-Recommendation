/**
 * Assessment — guided check of the five metabolic-syndrome markers.
 *
 * Flow (unchanged, driven by assessmentStore):
 *   0 Basic information → predict
 *   1 Additional information (only when MetS is likely) → calculate severity
 *   last  Results & recommendations
 */
import type { ReactNode } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Alert,
  Button,
  CircularProgress,
  InputAdornment,
  Slider,
  Switch,
  Tab,
  Tabs,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import CloseRoundedIcon from '@mui/icons-material/CloseRounded';
import FileDownloadRoundedIcon from '@mui/icons-material/FileDownloadRounded';
import RestartAltRoundedIcon from '@mui/icons-material/RestartAltRounded';
import RestaurantRoundedIcon from '@mui/icons-material/RestaurantRounded';
import BlockRoundedIcon from '@mui/icons-material/BlockRounded';
import DirectionsRunRoundedIcon from '@mui/icons-material/DirectionsRunRounded';
import SelfImprovementRoundedIcon from '@mui/icons-material/SelfImprovementRounded';
import GppMaybeRoundedIcon from '@mui/icons-material/GppMaybeRounded';
import PersonRoundedIcon from '@mui/icons-material/PersonRounded';
import MedicalInformationRoundedIcon from '@mui/icons-material/MedicalInformationRounded';
import ScienceRoundedIcon from '@mui/icons-material/ScienceRounded';
import { useAssessmentStore, useAuthStore } from '../../stores';
import { RangeBar, RiskBadge, RiskGauge } from '../../components/ui';
import { FACTORS, toRiskKey } from '../../theme/factors';
import { factorVar } from '../../theme/cssVars';
import type { FactorKey } from '../../theme/tokens';
import { STATUS_LABEL, buildMarkerRows } from '../../theme/markers';
import TermsDialog from './TermsDialog';
import './Assessment.css';

const STEPS_WITH_ADDITIONAL = ['Basic information', 'Blood tests', 'Your results'];
const STEPS_WITHOUT_ADDITIONAL = ['Basic information', 'Your results'];

const SEVERITY_LABEL: Record<string, string> = { low: 'Low', medium: 'Moderate', high: 'High' };

/* ── Small building blocks ─────────────────────────────────────────────── */

interface TabPanelProps {
  children?: ReactNode;
  index: number;
  value: number;
}

const TabPanel = ({ children, value, index }: TabPanelProps) => (
  <div role="tabpanel" hidden={value !== index} id={`as-plan-${index}`} aria-labelledby={`as-plan-tab-${index}`}>
    {value === index && <div className="as-plan__panel">{children}</div>}
  </div>
);

const Section = ({ icon, title, hint, children }: { icon: ReactNode; title: string; hint?: string; children: ReactNode }) => (
  <section className="as-section">
    <header className="as-section__head">
      <span className="as-section__icon" aria-hidden>
        {icon}
      </span>
      <div>
        <Typography component="h3" className="as-section__title">
          {title}
        </Typography>
        {hint && <p className="as-section__hint">{hint}</p>}
      </div>
    </header>
    {children}
  </section>
);

interface MeasureFieldProps {
  factor: FactorKey;
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}

/** Keys that would let a non-integer into a number field (e, +, -, .) */
const BLOCKED_KEYS = new Set(['e', 'E', '+', '-', '.', ',']);

/**
 * Whole-number field restricted to [min, max]: only digits can be typed,
 * length is capped to max's digit count, and the value is clamped on blur.
 */
const MeasureField = ({ factor, label, unit, value, min, max, onChange }: MeasureFieldProps) => {
  const Icon = FACTORS[factor].icon;
  const maxDigits = String(max).length;
  const id = `as-field-${label.toLowerCase().replace(/[^a-z]+/g, '-')}`;
  return (
    <div className="as-measure" style={{ ['--f' as string]: factorVar(factor) }}>
      <label htmlFor={id} className="as-label as-measure__label">
        {label}
      </label>
      <TextField
        fullWidth
        id={id}
        type="number"
        value={Number.isFinite(value) ? value : ''}
        onKeyDown={(e) => {
          if (BLOCKED_KEYS.has(e.key)) e.preventDefault();
        }}
        onChange={(e) => onChange(parseInt(e.target.value.replace(/\D/g, '').slice(0, maxDigits)))}
        onBlur={() => {
          if (Number.isFinite(value)) onChange(Math.min(max, Math.max(min, value)));
        }}
        helperText={`Value range: ${min}–${max} ${unit}`}
        slotProps={{
          htmlInput: { min, max, step: 1, inputMode: 'numeric', pattern: '[0-9]*' },
          input: {
            startAdornment: (
              <InputAdornment position="start">
                <span className="as-measure__icon" aria-hidden>
                  <Icon />
                </span>
              </InputAdornment>
            ),
            endAdornment: <InputAdornment position="end">{unit}</InputAdornment>,
          },
        }}
      />
    </div>
  );
};

const DiagnosisToggle = ({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) => (
  <label className={`as-toggle${checked ? ' is-on' : ''}`}>
    <span className="as-toggle__label">{label}</span>
    <Switch checked={checked} onChange={(e) => onChange(e.target.checked)} />
  </label>
);

/* ── Page ───────────────────────────────────────────────────────────────── */

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
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const navigate = useNavigate();
  const location = useLocation();

  /** Leave without accepting: back to the previous in-app page, or Home if
   *  the assessment was opened directly (no in-app history to return to). */
  const exitAssessment = () => {
    if (location.key === 'default') navigate('/', { replace: true });
    else navigate(-1);
  };

  const steps = results.hasMetabolicSyndrome ? STEPS_WITH_ADDITIONAL : STEPS_WITHOUT_ADDITIONAL;

  const showResults =
    (results.hasMetabolicSyndrome && activeStep === 2) ||
    (!results.hasMetabolicSyndrome && activeStep === 1 && results.probability > 0);

  const gender = patientInfo.gender;

  const severityKey = toRiskKey(results.riskLevel);
  const markerRows = buildMarkerRows({
    gender,
    systolicBP: patientInfo.systolicBP,
    diastolicBP: patientInfo.diastolicBP,
    waistCircumference: patientInfo.waistCircumference,
    ...(results.hasMetabolicSyndrome
      ? { hdlCholesterol: additionalInfo.hdlCholesterol, triglyceride: additionalInfo.triglyceride, fpg: additionalInfo.fpg }
      : {}),
  });

  const plan = [
    { label: 'Diet plan', icon: <RestaurantRoundedIcon />, items: recommendations.dietPlan, empty: 'No diet recommendations available.' },
    { label: 'Foods to limit', icon: <BlockRoundedIcon />, items: recommendations.avoidList, empty: 'No foods to limit listed.', avoid: true },
    { label: 'Exercise', icon: <DirectionsRunRoundedIcon />, items: recommendations.exercisePlan, empty: 'No exercise recommendations available.' },
    { label: 'Yoga', icon: <SelfImprovementRoundedIcon />, items: recommendations.yogaPoses, empty: 'No yoga recommendations available.' },
  ];

  return (
    <div className="as">
      {/* ── Header + progress ─────────────────────────────────────────── */}
      <header className="as-head">
        <div className="as-head__copy">
          <span className="as-eyebrow">Health assessment</span>
          <Typography variant="h2" component="h1" className="as-head__title">
            Check your <em className="as-serif">metabolic risk</em>
          </Typography>
          <Typography className="as-head__sub">
            Takes about two minutes. Keep your latest check-up or blood-test numbers handy.
          </Typography>
        </div>

        <ol className="as-steps" aria-label="Progress">
          {steps.map((label, i) => {
            const state = i < activeStep || (showResults && i === steps.length - 1) ? 'done' : i === activeStep ? 'current' : 'todo';
            return (
              <li key={label} className={`as-steps__item is-${state}`} aria-current={state === 'current' ? 'step' : undefined}>
                <span className="as-steps__dot" aria-hidden>
                  {state === 'done' ? <CheckRoundedIcon /> : i + 1}
                </span>
                <span className="as-steps__label">{label}</span>
              </li>
            );
          })}
        </ol>
      </header>

      {error && (
        <Alert severity="error" className="as-alert" onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* ── Step 1: basic information ─────────────────────────────────── */}
      {activeStep === 0 && (
        <div className="as-card">
          <Section icon={<PersonRoundedIcon />} title="About you">
            <div className="as-grid as-grid--2">
              <div className="as-age">
                <div className="as-age__head">
                  <span className="as-label" id="as-age-label">
                    Age
                  </span>
                  <span className="as-age__value">
                    {patientInfo.age}
                    <small> years</small>
                  </span>
                </div>
                <Slider
                  aria-labelledby="as-age-label"
                  value={patientInfo.age}
                  onChange={(_, value) => setPatientInfo('age', value as number)}
                  min={20}
                  max={80}
                  valueLabelDisplay="auto"
                />
                <div className="as-age__scale" aria-hidden>
                  <span>20</span>
                  <span>80</span>
                </div>
              </div>

              <div>
                <span className="as-label" id="as-gender-label">
                  Gender
                </span>
                <ToggleButtonGroup
                  exclusive
                  fullWidth
                  aria-labelledby="as-gender-label"
                  value={patientInfo.gender}
                  onChange={(_, v) => v && setPatientInfo('gender', v)}
                  className="as-segment"
                >
                  <ToggleButton value="Men">Male</ToggleButton>
                  <ToggleButton value="Women">Female</ToggleButton>
                </ToggleButtonGroup>
                <p className="as-note">Thresholds for waist and HDL differ by gender.</p>
              </div>
            </div>
          </Section>

          <Section icon={<MedicalInformationRoundedIcon />} title="Existing diagnoses" hint="Turn on anything a doctor has diagnosed.">
            <div className="as-grid as-grid--3">
              <DiagnosisToggle label="Fatty liver" checked={patientInfo.fattyLiver} onChange={(v) => setPatientInfo('fattyLiver', v)} />
              <DiagnosisToggle label="Hypertension" checked={patientInfo.hypertension} onChange={(v) => setPatientInfo('hypertension', v)} />
              <DiagnosisToggle label="Diabetes" checked={patientInfo.diabetes} onChange={(v) => setPatientInfo('diabetes', v)} />
            </div>
          </Section>

          <Section icon={<ScienceRoundedIcon />} title="Measurements" hint="From your most recent check-up.">
            <div className="as-grid as-grid--3">
              <MeasureField
                factor="bloodPressure"
                label="Systolic blood pressure"
                unit="mmHg"
                value={patientInfo.systolicBP}
                min={70}
                max={200}
                onChange={(v) => setPatientInfo('systolicBP', v)}
              />
              <MeasureField
                factor="bloodPressure"
                label="Diastolic blood pressure"
                unit="mmHg"
                value={patientInfo.diastolicBP}
                min={40}
                max={130}
                onChange={(v) => setPatientInfo('diastolicBP', v)}
              />
              <MeasureField
                factor="waist"
                label="Waist circumference"
                unit="cm"
                value={patientInfo.waistCircumference}
                min={50}
                max={150}
                onChange={(v) => setPatientInfo('waistCircumference', v)}
              />
            </div>
          </Section>

          <div className="as-actions">
            <Button
              variant="gradient"
              size="large"
              onClick={predict}
              disabled={loading}
              endIcon={loading ? undefined : <ArrowForwardRoundedIcon />}
              className="as-primary"
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Predict my risk'}
            </Button>
          </div>
        </div>
      )}

      {/* ── Step 2: blood tests (only when MetS is likely) ────────────── */}
      {results.hasMetabolicSyndrome && activeStep === 1 && (
        <div className="as-card">
          <div className="as-callout" role="status">
            <GppMaybeRoundedIcon aria-hidden />
            <div>
              <strong>Your result suggests a higher chance of metabolic syndrome.</strong>
              <p>Add three values from a fasting blood test so we can work out how severe it is and tailor your plan.</p>
            </div>
          </div>

          <Section icon={<ScienceRoundedIcon />} title="Blood tests" hint="Use fasting results if you have them.">
            <div className="as-grid as-grid--3">
              <MeasureField
                factor="hdl"
                label="HDL cholesterol"
                unit="mg/dL"
                value={additionalInfo.hdlCholesterol}
                min={20}
                max={100}
                onChange={(v) => setAdditionalInfo('hdlCholesterol', v)}
              />
              <MeasureField
                factor="triglycerides"
                label="Triglycerides"
                unit="mg/dL"
                value={additionalInfo.triglyceride}
                min={50}
                max={500}
                onChange={(v) => setAdditionalInfo('triglyceride', v)}
              />
              <MeasureField
                factor="glucose"
                label="Fasting glucose"
                unit="mg/dL"
                value={additionalInfo.fpg}
                min={70}
                max={200}
                onChange={(v) => setAdditionalInfo('fpg', v)}
              />
            </div>
          </Section>

          <div className="as-actions">
            <Button variant="outlined" size="large" onClick={() => setActiveStep(0)} startIcon={<ArrowBackRoundedIcon />}>
              Back
            </Button>
            <Button
              variant="gradient"
              size="large"
              onClick={handleSeverityCalculation}
              disabled={loading}
              endIcon={loading ? undefined : <ArrowForwardRoundedIcon />}
              className="as-primary"
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Calculate severity'}
            </Button>
          </div>
        </div>
      )}

      {/* ── Results & recommendations ─────────────────────────────────── */}
      {showResults && (
        <div className="as-results">
          {!isAuthenticated && (
            <div className="as-save">
              <p>
                <strong>Want to keep this result?</strong> Create a free account to save assessments and track your
                progress on your dashboard.
              </p>
              <div className="as-save__actions">
                <Button component={Link} to="/signup" variant="contained" size="small">
                  Create free account
                </Button>
                <Button component={Link} to="/login" size="small">
                  Sign in
                </Button>
              </div>
            </div>
          )}

          <div className="as-results__top">
            <section className="as-card as-score" aria-labelledby="as-score-title">
              <Typography id="as-score-title" component="h2" className="as-card__title">
                Your result
              </Typography>
              <RiskGauge
                value={results.probability}
                risk={results.hasMetabolicSyndrome ? results.riskLevel : 'low'}
                label="Chance of metabolic syndrome"
                showBadge={false}
              />
              <dl className="as-score__facts">
                <div>
                  <dt>Outlook</dt>
                  <dd>
                    <RiskBadge
                      level={results.hasMetabolicSyndrome ? 'high' : 'low'}
                      label={results.hasMetabolicSyndrome ? 'Metabolic syndrome likely' : 'Unlikely'}
                      size="small"
                    />
                  </dd>
                </div>
                {severityKey && (
                  <div>
                    <dt>Severity</dt>
                    <dd>
                      <RiskBadge level={severityKey} label={SEVERITY_LABEL[severityKey]} size="small" />
                    </dd>
                  </div>
                )}
              </dl>
              <p className="as-score__text">
                {results.hasMetabolicSyndrome
                  ? 'Your markers point to metabolic syndrome. The plan below is matched to your result — share the report with your doctor.'
                  : 'Good news — your markers point to a low chance of metabolic syndrome. The plan below helps you keep it that way.'}
              </p>
            </section>

            <section className="as-card as-markers" aria-labelledby="as-markers-title">
              <div className="as-card__head">
                <Typography id="as-markers-title" component="h2" className="as-card__title">
                  Your markers
                </Typography>
                <span className="as-card__meta">
                  {gender === 'Women' ? 'Female' : 'Male'} · {patientInfo.age} years
                </span>
              </div>
              <ul className="as-markers__list">
                {markerRows.map((m) => {
                  const F = FACTORS[m.key];
                  const Icon = F.icon;
                  return (
                    <li key={m.key} className="as-marker" style={{ ['--f' as string]: factorVar(m.key) }}>
                      <span className="as-marker__icon" aria-hidden>
                        <Icon />
                      </span>
                      <div className="as-marker__body">
                        <div className="as-marker__row">
                          <span className="as-marker__name">{F.label}</span>
                          <span className="as-marker__value">
                            {m.display} <small>{m.unit}</small>
                          </span>
                        </div>
                        <RangeBar
                          min={m.min}
                          max={m.max}
                          zones={m.zones}
                          value={m.value}
                          thickness={6}
                          label={`${F.label} ${m.display} ${m.unit}, ${STATUS_LABEL[m.status]}`}
                        />
                      </div>
                      <RiskBadge level={m.risk} label={STATUS_LABEL[m.status]} size="small" />
                    </li>
                  );
                })}
              </ul>
              {!results.hasMetabolicSyndrome && (
                <p className="as-note">Blood-test markers are only needed when metabolic syndrome is likely.</p>
              )}
            </section>
          </div>

          <section className="as-card as-plan" aria-labelledby="as-plan-title">
            <div className="as-card__head">
              <div>
                <Typography id="as-plan-title" component="h2" className="as-card__title">
                  Your personal plan
                </Typography>
                <p className="as-card__sub">Matched to your result. Small, steady changes make the biggest difference.</p>
              </div>
            </div>

            <Tabs
              value={tabValue}
              onChange={(_, v) => setTabValue(v)}
              variant="scrollable"
              scrollButtons="auto"
              allowScrollButtonsMobile
              className="as-plan__tabs"
              aria-label="Plan sections"
            >
              {plan.map((p, i) => (
                <Tab
                  key={p.label}
                  icon={p.icon}
                  iconPosition="start"
                  label={p.label}
                  id={`as-plan-tab-${i}`}
                  aria-controls={`as-plan-${i}`}
                />
              ))}
            </Tabs>

            {plan.map((p, i) => (
              <TabPanel key={p.label} value={tabValue} index={i}>
                {p.items.length > 0 ? (
                  <ul className={`as-plan__list${p.avoid ? ' is-avoid' : ''}`}>
                    {p.items.map((item, j) => (
                      <li key={j}>
                        <span className="as-plan__mark" aria-hidden>
                          {p.avoid ? <CloseRoundedIcon /> : <CheckRoundedIcon />}
                        </span>
                        {item}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="as-empty">{p.empty}</p>
                )}
              </TabPanel>
            ))}
          </section>

          <div className="as-actions as-actions--center">
            <Button variant="outlined" size="large" onClick={startNewAssessment} startIcon={<RestartAltRoundedIcon />}>
              Start new assessment
            </Button>
            <Button
              variant="gradient"
              size="large"
              onClick={handleDownloadReport}
              startIcon={<FileDownloadRoundedIcon />}
              className="as-primary"
            >
              Download report (PDF)
            </Button>
          </div>
        </div>
      )}

      {/* ── Terms (with a way out) ───────────────────────────────── */}
      <TermsDialog
        open={termsOpen && !termsAccepted}
        checked={termsCheckbox}
        onCheck={setTermsCheckbox}
        onAccept={acceptTerms}
        onExit={exitAssessment}
      />
    </div>
  );
};

export default Assessment;
