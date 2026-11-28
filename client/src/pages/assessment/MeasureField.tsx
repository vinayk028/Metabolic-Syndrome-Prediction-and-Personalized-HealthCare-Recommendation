/**
 * MeasureField — whole-number input for one marker on the assessment form.
 * Bold label above, marker icon + unit inside, and the accepted range below.
 */
import { InputAdornment, TextField } from '@mui/material';
import { FACTORS } from '../../theme/factors';
import { factorVar } from '../../theme/cssVars';
import type { FactorKey } from '../../theme/tokens';
import './MeasureField.css';

export interface MeasureFieldProps {
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

export default MeasureField;
