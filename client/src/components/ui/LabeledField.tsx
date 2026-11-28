/**
 * LabeledField — a TextField with a bold label *above* the box (no floating
 * label), matching the assessment form. Pass every normal TextField prop.
 *
 * PasswordField — the same, with a show/hide toggle.
 */
import { useId, useState, type ReactNode } from 'react';
import { Box, IconButton, InputAdornment, TextField, type TextFieldProps } from '@mui/material';
import VisibilityRoundedIcon from '@mui/icons-material/VisibilityRounded';
import VisibilityOffRoundedIcon from '@mui/icons-material/VisibilityOffRounded';

type LabeledFieldProps = Omit<TextFieldProps, 'label'> & {
  label: ReactNode;
  /** Optional icon shown at the start of the input */
  icon?: ReactNode;
  /** Optional right-aligned element in the label row (e.g. "Forgot?") */
  labelAction?: ReactNode;
};

export const LabeledField = ({ label, icon, labelAction, id, slotProps, ...rest }: LabeledFieldProps) => {
  const autoId = useId();
  const fieldId = id ?? `f${autoId.replace(/:/g, '')}`;
  const inputSlot = (slotProps?.input ?? {}) as Record<string, unknown>;

  return (
    <Box sx={{ minWidth: 0 }}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 1, mb: 0.75 }}>
        <Box
          component="label"
          htmlFor={fieldId}
          sx={{
            fontFamily: 'var(--vb-font-display)',
            fontSize: '0.9375rem',
            fontWeight: 700,
            color: 'text.primary',
          }}
        >
          {label}
          {rest.required && (
            <Box component="span" aria-hidden sx={{ color: 'text.disabled', fontWeight: 500, ml: 0.5 }}>
              *
            </Box>
          )}
        </Box>
        {labelAction}
      </Box>
      <TextField
        id={fieldId}
        fullWidth
        {...rest}
        slotProps={{
          ...slotProps,
          input: {
            ...(icon
              ? {
                  startAdornment: (
                    <InputAdornment position="start" sx={{ color: 'text.secondary', '& svg': { fontSize: 20 } }}>
                      {icon}
                    </InputAdornment>
                  ),
                }
              : {}),
            ...inputSlot,
          },
        }}
      />
    </Box>
  );
};

type PasswordFieldProps = Omit<LabeledFieldProps, 'type'>;

export const PasswordField = ({ slotProps, ...rest }: PasswordFieldProps) => {
  const [show, setShow] = useState(false);
  const inputSlot = (slotProps?.input ?? {}) as Record<string, unknown>;
  return (
    <LabeledField
      {...rest}
      type={show ? 'text' : 'password'}
      slotProps={{
        ...slotProps,
        input: {
          ...inputSlot,
          endAdornment: (
            <InputAdornment position="end">
              <IconButton
                onClick={() => setShow((v) => !v)}
                edge="end"
                size="small"
                aria-label={show ? 'Hide password' : 'Show password'}
              >
                {show ? <VisibilityOffRoundedIcon fontSize="small" /> : <VisibilityRoundedIcon fontSize="small" />}
              </IconButton>
            </InputAdornment>
          ),
        },
      }}
    />
  );
};

export default LabeledField;
