import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../utils/render';
import MeasureField from '@/pages/assessment/MeasureField';

/** Controlled wrapper, as the assessment page uses it. */
const Harness = ({ initial = 90, onValue }: { initial?: number; onValue?: (v: number) => void }) => {
  const [value, setValue] = useState(initial);
  return (
    <MeasureField
      factor="waist"
      label="Waist circumference"
      unit="cm"
      min={40}
      max={200}
      value={value}
      onChange={(v) => {
        setValue(v);
        onValue?.(v);
      }}
    />
  );
};

const input = () => screen.getByLabelText('Waist circumference') as HTMLInputElement;

describe('MeasureField', () => {
  it('labels the input and shows the accepted range', () => {
    renderWithProviders(<Harness />);
    expect(input()).toHaveValue(90);
    expect(screen.getByText('Value range: 40–200 cm')).toBeInTheDocument();
  });

  it('blocks keys that would allow decimals, signs or exponents', async () => {
    renderWithProviders(<Harness initial={NaN} />);
    await userEvent.type(input(), '-1.5e');
    expect(input()).toHaveValue(15);
  });

  it('caps the number of digits to the maximum’s length', () => {
    renderWithProviders(<Harness initial={NaN} />);
    fireEvent.change(input(), { target: { value: '12345' } });
    expect(input()).toHaveValue(123);
  });

  it('clamps out-of-range values when the field loses focus', () => {
    const onValue = vi.fn();
    renderWithProviders(<Harness initial={NaN} onValue={onValue} />);
    fireEvent.change(input(), { target: { value: '999' } });
    fireEvent.blur(input());
    expect(input()).toHaveValue(200);

    fireEvent.change(input(), { target: { value: '5' } });
    fireEvent.blur(input());
    expect(input()).toHaveValue(40);
    expect(onValue).toHaveBeenLastCalledWith(40);
  });

  it('leaves an empty field empty on blur', () => {
    renderWithProviders(<Harness />);
    fireEvent.change(input(), { target: { value: '' } });
    fireEvent.blur(input());
    expect(input()).toHaveValue(null);
  });
});
