import { beforeEach, describe, expect, it, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../../utils/render';
import { useNewsStore } from '@/stores';
import NewsCarousel from '@/pages/landing/sections/NewsCarousel';

const article = (n: number) => ({
  title: `Article ${n}`,
  description: 'About metabolic health',
  content: '',
  url: `https://example.org/${n}`,
  image: null,
  source: 'Example News',
  publishedAt: '2026-05-01T00:00:00Z',
  author: null,
});

// Keep the component from calling the real news service.
const noFetch = vi.fn().mockResolvedValue(undefined);

describe('NewsCarousel', () => {
  beforeEach(() => {
    noFetch.mockClear();
  });

  it('renders nothing when there is no news', () => {
    useNewsStore.setState({ slideshowArticles: [], slideshowLoading: false, fetchSlideshowNews: noFetch });
    const { container } = renderWithProviders(<NewsCarousel />);
    expect(container).toBeEmptyDOMElement();
    expect(noFetch).toHaveBeenCalled();
  });

  it('shows one link per article plus an "All news" link', () => {
    useNewsStore.setState({
      slideshowArticles: [article(1), article(2)],
      slideshowLoading: false,
      fetchSlideshowNews: noFetch,
    });
    renderWithProviders(<NewsCarousel />);
    expect(screen.getByRole('heading', { name: /latest on metabolic health/i })).toBeInTheDocument();
    expect(screen.getByText('Article 1')).toBeInTheDocument();
    expect(screen.getByText('Article 2')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /all news/i })).toHaveAttribute('href', '/resources');
  });
});
