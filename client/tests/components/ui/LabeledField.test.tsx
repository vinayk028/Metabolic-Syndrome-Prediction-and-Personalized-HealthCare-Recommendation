import { describe, expect, it } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../utils/render';
import { LabeledField, PasswordField } from '@/components/ui/LabeledField';

describe('LabeledField', () => {
  it('links the label above the box to the input', async () => {
    renderWithProviders(<LabeledField label="Email" name="email" />);
    const input = screen.getByLabelText('Email');
    await userEvent.type(input, 'a@b.co');
    expect(input).toHaveValue('a@b.co');
  });

  it('marks required fields and keeps a label action beside the label', () => {
    renderWithProviders(<LabeledField label="Password" required labelAction={<a href="/forgot">Forgot?</a>} />);
    expect(screen.getByLabelText(/password/i)).toBeRequired();
    expect(screen.getByRole('link', { name: 'Forgot?' })).toBeInTheDocument();
  });
});

describe('PasswordField', () => {
  it('hides the password until "Show password" is pressed', async () => {
    renderWithProviders(<PasswordField label="Password" />);
    const input = screen.getByLabelText('Password');
    expect(input).toHaveAttribute('type', 'password');

    await userEvent.click(screen.getByRole('button', { name: 'Show password' }));
    expect(input).toHaveAttribute('type', 'text');

    await userEvent.click(screen.getByRole('button', { name: 'Hide password' }));
    expect(input).toHaveAttribute('type', 'password');
  });
});
