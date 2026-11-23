/**
 * TermsDialog — the disclaimer gate shown before an assessment.
 *
 * Three key points up front, the full terms one click away, and two clear
 * ways out: accept (after ticking the box) or go back to where you came from.
 * Escape also goes back; clicking the backdrop does nothing (no accidental
 * dismissal). Full-screen on phones.
 */
import { useId, useState } from 'react';
import {
  Button,
  Checkbox,
  Collapse,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import ArrowBackRoundedIcon from '@mui/icons-material/ArrowBackRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import ExpandMoreRoundedIcon from '@mui/icons-material/ExpandMoreRounded';
import GppMaybeRoundedIcon from '@mui/icons-material/GppMaybeRounded';
import FactCheckRoundedIcon from '@mui/icons-material/FactCheckRounded';
import EmergencyRoundedIcon from '@mui/icons-material/EmergencyRounded';

interface TermsDialogProps {
  open: boolean;
  checked: boolean;
  onCheck: (v: boolean) => void;
  onAccept: () => void;
  /** Leave the assessment without accepting */
  onExit: () => void;
}

const KEY_POINTS = [
  {
    icon: <FactCheckRoundedIcon />,
    title: 'Results depend on your numbers',
    text: 'Estimates are based on statistical models and the values you enter — use recent, accurate readings.',
  },
  {
    icon: <EmergencyRoundedIcon />,
    title: 'Not for emergencies',
    text: 'If you feel unwell or think it is an emergency, contact your local emergency services now.',
  },
];

const TermsDialog = ({ open, checked, onCheck, onAccept, onExit }: TermsDialogProps) => {
  const theme = useTheme();
  const fullScreen = useMediaQuery(theme.breakpoints.down('sm'));
  const [showFull, setShowFull] = useState(false);
  const fullId = useId();

  return (
    <Dialog
      open={open}
      onClose={(_, reason) => {
        // Escape = "take me back"; a stray backdrop click should not exit
        if (reason === 'escapeKeyDown') onExit();
      }}
      maxWidth="md"
      fullWidth
      fullScreen={fullScreen}
      scroll="paper"
      aria-labelledby="as-terms-title"
      aria-describedby="as-terms-intro"
      slotProps={{ paper: { className: 'as-terms' } }}
    >
      <DialogTitle id="as-terms-title" className="as-terms__title">
        <span className="as-terms__icon" aria-hidden>
          <GppMaybeRoundedIcon />
        </span>
        <span>
          Before you start
          <span className="as-terms__sub" id="as-terms-intro">
            A quick note on what this assessment can and can't do.
          </span>
        </span>
      </DialogTitle>

      <DialogContent dividers className="as-terms__content">
        <ul className="as-terms__points">
          {KEY_POINTS.map((p) => (
            <li key={p.title}>
              <span className="as-terms__point-icon" aria-hidden>
                {p.icon}
              </span>
              <div>
                <strong>{p.title}</strong>
                <p>{p.text}</p>
              </div>
            </li>
          ))}
        </ul>

        <Button
          size="small"
          onClick={() => setShowFull((v) => !v)}
          aria-expanded={showFull}
          aria-controls={fullId}
          endIcon={<ExpandMoreRoundedIcon className={`as-terms__chev${showFull ? ' is-open' : ''}`} />}
          className="as-terms__toggle"
        >
          {showFull ? 'Hide full terms' : 'Read full terms'}
        </Button>

        <Collapse in={showFull} id={fullId}>
          <div className="as-terms__full">
            <h3>1. Medical disclaimer</h3>
            <p>
              This tool is designed for <strong>educational and informational purposes only</strong>. It is not a
              substitute for professional medical advice, diagnosis or treatment. Always seek the advice of your physician
              or another qualified health provider with any questions you may have about a medical condition.
            </p>
            <h3>2. Accuracy of results</h3>
            <p>
              Predictions and recommendations are based on statistical models and general health guidelines. Individual
              results may vary, and the tool cannot account for every personal health factor. Results are only as
              accurate as the information you provide.
            </p>
            <h3>3. No doctor–patient relationship</h3>
            <p>
              Using this tool does not create a doctor–patient relationship. Recommendations are general and should be
              discussed with your healthcare provider before changing your diet, exercise or medication.
            </p>
            <h3>4. Emergencies</h3>
            <p>
              If you are experiencing a medical emergency, call your local emergency services immediately. Do not rely
              on this tool in an emergency.
            </p>
            <h3>5. Data privacy</h3>
            <p>
              The health information you enter is processed locally and is not stored on our servers permanently. We
              recommend not entering personally identifiable information beyond what the assessment needs.
            </p>
            <h3>6. Limitation of liability</h3>
            <p>
              The creators and operators of this tool are not liable for any damages arising from its use or from the
              recommendations provided. Use this tool at your own risk.
            </p>
          </div>
        </Collapse>
      </DialogContent>

      <DialogActions className="as-terms__actions">
        <FormControlLabel
          className="as-terms__agree"
          control={<Checkbox checked={checked} onChange={(e) => onCheck(e.target.checked)} />}
          label="I understand this tool is for information only and agree to the terms."
        />
        <div className="as-terms__buttons">
          <Button variant="outlined" size="large" startIcon={<ArrowBackRoundedIcon />} onClick={onExit}>
            Go back
          </Button>
          <Button
            variant="gradient"
            size="large"
            endIcon={<ArrowForwardRoundedIcon />}
            onClick={onAccept}
            disabled={!checked}
          >
            Accept and continue
          </Button>
        </div>
      </DialogActions>
    </Dialog>
  );
};

export default TermsDialog;
