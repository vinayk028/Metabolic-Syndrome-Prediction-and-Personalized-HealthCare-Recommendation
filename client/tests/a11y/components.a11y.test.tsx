/**
 * Accessibility guards (axe-core) for the parts fixed in the accessibility
 * pass, plus the chat's HTML escaping. Colour contrast is checked separately
 * in a real browser — jsdom cannot compute colours.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../utils/render';
import { axeViolations } from '../utils/axe';
import { useAuthStore, useChatStore } from '@/stores';
import type { User } from '@/types/types';
import Navbar from '@/components/layout/Navbar';
import ChatAssistant from '@/components/chat/ChatAssistant';
import TermsDialog from '@/pages/assessment/TermsDialog';
import MeasureField from '@/pages/assessment/MeasureField';
import Profile from '@/pages/profile/Profile';
import { LabeledField, PasswordField } from '@/components/ui/LabeledField';

vi.mock('@/services/api', () => ({
  getCurrentUser: vi.fn(),
  sendChatMessage: vi.fn(),
  updateProfile: vi.fn(),
  updatePassword: vi.fn(),
  deleteAccount: vi.fn(),
}));

const user: User = {
  id: 'u1',
  firstName: 'Asha',
  lastName: 'Rao',
  fullName: 'Asha Rao',
  email: 'asha@example.org',
  role: 'user',
  createdAt: '2025-01-10T00:00:00Z',
};

const signIn = () => useAuthStore.setState({ user, token: 't', isAuthenticated: true, isLoading: false });

beforeEach(() => {
  useAuthStore.setState({ user: null, token: null, isAuthenticated: false, isLoading: false });
  useChatStore.setState({ isOpen: false });
  Element.prototype.scrollIntoView ??= () => {}; // jsdom has no scrolling
});

describe('navigation', () => {
  it('account menu: valid menu, theme choices are reachable radio items', async () => {
    signIn();
    renderWithProviders(<Navbar />);
    const avatar = screen.getByRole('button', { name: 'AR, account menu' }); // name starts with the visible initials
    await userEvent.click(avatar);

    const menu = screen.getByRole('menu');
    const light = within(menu).getByRole('menuitemradio', { name: 'Light theme' });
    const dark = within(menu).getByRole('menuitemradio', { name: 'Dark theme' });
    expect(light).toHaveAttribute('aria-checked', 'true');
    expect(dark).toHaveAttribute('aria-checked', 'false');

    await userEvent.click(dark);
    expect(dark).toHaveAttribute('aria-checked', 'true');
    expect(await axeViolations()).toEqual([]);
  });

  it('small-screen menu: named dialog with valid content', async () => {
    renderWithProviders(<Navbar />);
    await userEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    expect(screen.getByRole('dialog', { name: 'Menu' })).toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });
});

describe('assessment', () => {
  it('terms dialog passes', async () => {
    renderWithProviders(<TermsDialog open checked={false} onCheck={() => {}} onAccept={() => {}} onExit={() => {}} />);
    expect(await axeViolations()).toEqual([]);
  });

  it('measurement field passes', async () => {
    renderWithProviders(
      <MeasureField factor="glucose" label="Fasting glucose" unit="mg/dL" value={95} min={50} max={400} onChange={() => {}} />,
    );
    expect(await axeViolations()).toEqual([]);
  });
});

describe('forms', () => {
  it('labelled fields and password field pass', async () => {
    renderWithProviders(
      <form>
        <LabeledField label="Email" name="email" required />
        <PasswordField label="Password" name="password" />
      </form>,
    );
    expect(await axeViolations()).toEqual([]);
  });
});

describe('profile', () => {
  it('details are a valid description list', async () => {
    signIn();
    const { container } = renderWithProviders(<Profile />, { route: '/profile' });
    const dl = container.querySelector('dl')!;
    expect(dl).toBeInTheDocument();
    // every group holds exactly one term and one value
    for (const group of dl.children) {
      expect([...group.children].map((c) => c.tagName)).toEqual(['DT', 'DD']);
    }
    expect(within(dl).getByText('Asha')).toBeInTheDocument();
    expect(await axeViolations()).toEqual([]);
  });
});

describe('chat assistant', () => {
  const openWith = (messages: { role: 'user' | 'assistant'; content: string }[]) =>
    useChatStore.setState({
      isOpen: true,
      messages: messages.map((m, i) => ({ id: `m${i}`, timestamp: new Date('2026-01-01T10:00:00Z'), ...m })),
    });

  it('renders replies with lists and headings as valid, accessible markup', async () => {
    openWith([{ role: 'assistant', content: '## Tips\n- Walk daily\n- Sleep well\n\nAsk me more.' }]);
    renderWithProviders(<ChatAssistant />);
    const log = screen.getByRole('log', { name: 'Conversation' });
    expect(within(log).getAllByRole('listitem')).toHaveLength(2);
    expect(within(log).queryByRole('heading')).toBeNull(); // reply headings are bold lines, not page headings
    expect(await axeViolations()).toEqual([]);
  });

  it('never turns message text into HTML', () => {
    openWith([
      { role: 'user', content: '<img src=x onerror="window.__pwned=1"><b>bold?</b>' },
      { role: 'assistant', content: 'Reply <script>window.__pwned=1</script> with **bold** & more' },
    ]);
    const { container } = renderWithProviders(<ChatAssistant />);
    const log = container.querySelector('.chat-messages')!;
    expect(log.querySelector('img, b, script')).toBeNull();
    expect(log).toHaveTextContent('<img src=x onerror="window.__pwned=1"><b>bold?</b>');
    expect(log).toHaveTextContent('Reply <script>window.__pwned=1</script> with bold & more');
    expect(log.querySelector('strong')).toHaveTextContent('bold'); // markdown still works
  });
});
