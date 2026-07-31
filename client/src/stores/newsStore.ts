/**
 * News Store (Zustand)
 * Manages news articles for Home (slideshow) and Resources pages
 */

import { create } from 'zustand';
import type { NewsArticle } from '../data/types';
import { getMetabolicSyndromeNews } from '../data/api';

const DEFAULT_NEWS_IMAGE = 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?auto=format&fit=crop&q=100';

interface NewsState {
    // Home slideshow (articles 16–20 from DB)
    slideshowArticles: NewsArticle[];
    slideshowLoading: boolean;

    // Resources page (articles 1–15 from DB)
    resourceArticles: NewsArticle[];
    resourceLoading: boolean;
    resourceError: string | null;

    // Actions
    fetchSlideshowNews: () => Promise<void>;
    fetchResourceNews: () => Promise<void>;
    getHighQualityImage: (url: string | null) => string;
}

export const useNewsStore = create<NewsState>((set) => ({
    slideshowArticles: [],
    slideshowLoading: true,

    resourceArticles: [],
    resourceLoading: true,
    resourceError: null,

    fetchSlideshowNews: async () => {
        set({ slideshowLoading: true });
        try {
            const response = await getMetabolicSyndromeNews(5, 15);
            if (response.success && response.articles.length > 0) {
                set({ slideshowArticles: response.articles.slice(0, 5) });
            }
        } catch (error) {
            console.error('Error fetching slideshow news:', error);
        } finally {
            set({ slideshowLoading: false });
        }
    },

    fetchResourceNews: async () => {
        set({ resourceLoading: true, resourceError: null });
        try {
            const response = await getMetabolicSyndromeNews(15);
            if (response.success) {
                set({ resourceArticles: response.articles });
                if (response.articles.length === 0 && response.message) {
                    set({ resourceError: response.message });
                }
            } else {
                set({ resourceError: 'Failed to fetch news' });
            }
        } catch (error) {
            console.error('Error fetching resource news:', error);
            set({ resourceError: 'Unable to load news. Please try again later.' });
        } finally {
            set({ resourceLoading: false });
        }
    },

    getHighQualityImage: (url) => {
        if (!url) return DEFAULT_NEWS_IMAGE;
        if (url.includes('guim.co.uk')) {
            return url.replace(/\/\d+\.jpg/, '/1000.jpg');
        }
        if (url.includes('unsplash.com')) {
            return url.replace(/[?&](w|h)=\d+/g, '').replace(/[?&]fit=crop/, '') + '?auto=format&q=100';
        }
        return url;
    },
}));
