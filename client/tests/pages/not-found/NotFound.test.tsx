import { describe, expect, it } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../utils/render';
import { axeViolations } from '../../utils/axe';
import NotFound from '@/pages/not-found/NotFound';

describe('NotFound', () => {
  it('says the page was not found and shows the address', () => {
    renderWithProviders(<NotFound />, { route: '/no-such-page' });
    expect(screen.getByRole('heading', { level: 1, name: 'Page not found' })).toBeInTheDocument();
    expect(screen.getByText('/no-such-page')).toBeInTheDocument();
  });

  it('offers ways back', () => {
    renderWithProviders(<NotFound />, { route: '/x' });
    expect(screen.getByRole('link', { name: /go to home/i })).toHaveAttribute('href', '/');
    expect(screen.getByRole('link', { name: /start free assessment/i })).toHaveAttribute('href', '/assessment');
  });

  it('shows the address as text, never as markup', () => {
    renderWithProviders(<NotFound />, { route: '/<img src=x onerror=alert(1)>' });
    expect(document.querySelector('img')).toBeNull();
  });

  it('is accessible', async () => {
    renderWithProviders(<NotFound />, { route: '/x' });
    expect(await axeViolations()).toEqual([]);
  });
});
