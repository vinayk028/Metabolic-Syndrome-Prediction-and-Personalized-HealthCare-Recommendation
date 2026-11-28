import { describe, expect, it } from 'vitest';
import { isPathActive, pageTitle } from '@/components/layout/navConfig';

describe('pageTitle', () => {
  it('keeps the full site title on the home page', () => {
    expect(pageTitle('/')).toBe('MetS Health · Know your metabolic risk');
  });

  it('names each page', () => {
    expect(pageTitle('/dashboard')).toBe('Dashboard · MetS Health');
    expect(pageTitle('/about')).toBe('About & FAQ · MetS Health');
    expect(pageTitle('/login')).toBe('Sign in · MetS Health');
  });

  it('ignores a trailing slash', () => {
    expect(pageTitle('/resources/')).toBe('Resources · MetS Health');
  });

  it('marks unknown addresses as not found', () => {
    expect(pageTitle('/no-such-page')).toBe('Page not found · MetS Health');
  });
});

describe('isPathActive', () => {
  it('matches home only exactly', () => {
    expect(isPathActive('/', '/')).toBe(true);
    expect(isPathActive('/about', '/')).toBe(false);
  });

  it('matches a section and its sub-pages', () => {
    expect(isPathActive('/dashboard', '/dashboard')).toBe(true);
    expect(isPathActive('/dashboard/history', '/dashboard')).toBe(true);
    expect(isPathActive('/dashboards', '/dashboard')).toBe(false);
  });
});
