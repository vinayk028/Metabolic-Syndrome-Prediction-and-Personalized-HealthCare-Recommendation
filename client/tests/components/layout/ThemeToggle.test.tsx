import { describe, expect, it } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../utils/render';
import { ThemeChooser, ThemeIconButton } from '@/components/layout/ThemeToggle';

const html = document.documentElement;

describe('ThemeChooser', () => {
  it('starts on Light and switches the page to Dark', async () => {
    renderWithProviders(<ThemeChooser />);
    const light = screen.getByRole('button', { name: 'Light theme' });
    const dark = screen.getByRole('button', { name: 'Dark theme' });
    expect(light).toHaveAttribute('aria-pressed', 'true');

    await userEvent.click(dark);
    expect(dark).toHaveAttribute('aria-pressed', 'true');
    await waitFor(() => expect(html).toHaveAttribute('data-theme', 'dark'));
    expect(localStorage.getItem('mui-mode')).toBe('dark');

    // clicking the active option again keeps it selected
    await userEvent.click(dark);
    expect(dark).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('ThemeIconButton', () => {
  it('toggles between light and dark', async () => {
    renderWithProviders(<ThemeIconButton />);
    await userEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }));
    await waitFor(() => expect(html).toHaveAttribute('data-theme', 'dark'));
    await userEvent.click(screen.getByRole('button', { name: 'Switch to light theme' }));
    await waitFor(() => expect(html).toHaveAttribute('data-theme', 'light'));
  });
});
