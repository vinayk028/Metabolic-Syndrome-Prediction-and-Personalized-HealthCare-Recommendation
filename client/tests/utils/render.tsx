/** render() with the app's theme and a router, like the real app tree. */
import type { ReactElement } from 'react';
import { render } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material';
import theme from '@/theme/theme';

export const renderWithProviders = (ui: ReactElement, { route = '/' } = {}) =>
  render(
    <ThemeProvider theme={theme} defaultMode="light" disableTransitionOnChange>
      <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
    </ThemeProvider>,
  );
