import { describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../utils/render';
import TermsDialog from '@/pages/assessment/TermsDialog';

const setup = (checked = false) => {
  const props = { open: true, checked, onCheck: vi.fn(), onAccept: vi.fn(), onExit: vi.fn() };
  renderWithProviders(<TermsDialog {...props} />);
  return props;
};

describe('TermsDialog', () => {
  it('keeps "Accept and continue" disabled until the box is ticked', async () => {
    const props = setup(false);
    expect(screen.getByRole('button', { name: /accept and continue/i })).toBeDisabled();
    await userEvent.click(screen.getByRole('checkbox'));
    expect(props.onCheck).toHaveBeenCalledWith(true);
  });

  it('accepts once ticked', async () => {
    const props = setup(true);
    await userEvent.click(screen.getByRole('button', { name: /accept and continue/i }));
    expect(props.onAccept).toHaveBeenCalledTimes(1);
  });

  it('offers a way out with "Go back" and with Escape', async () => {
    const props = setup();
    await userEvent.click(screen.getByRole('button', { name: /go back/i }));
    await userEvent.keyboard('{Escape}');
    expect(props.onExit).toHaveBeenCalledTimes(2);
  });

  it('reveals the full terms on request', async () => {
    setup();
    const toggle = screen.getByRole('button', { name: /read full terms/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(toggle);
    expect(screen.getByRole('button', { name: /hide full terms/i })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('1. Medical disclaimer')).toBeVisible();
  });
});
