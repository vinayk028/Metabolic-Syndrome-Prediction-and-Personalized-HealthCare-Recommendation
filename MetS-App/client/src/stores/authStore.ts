/**
 * Auth Store (Zustand)
 * Manages user authentication state
 */

import { create } from 'zustand';
import type { User } from '../data/types';
import { getCurrentUser } from '../data/api';

interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    isLoading: boolean;

    // Actions
    login: (token: string, user: User) => void;
    logout: () => void;
    updateUser: (user: User) => void;
    initAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: true,

    login: (token, user) => {
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify(user));
        set({ token, user, isAuthenticated: true });
    },

    logout: () => {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        set({ token: null, user: null, isAuthenticated: false });
    },

    updateUser: (user) => {
        localStorage.setItem('user', JSON.stringify(user));
        set({ user });
    },

    initAuth: async () => {
        const storedToken = localStorage.getItem('token');
        const storedUser = localStorage.getItem('user');

        if (storedToken && storedUser) {
            set({
                token: storedToken,
                user: JSON.parse(storedUser),
                isAuthenticated: true,
            });

            // Verify token is still valid
            try {
                const response = await getCurrentUser();
                if (response.success && response.user) {
                    localStorage.setItem('user', JSON.stringify(response.user));
                    set({ user: response.user }); 
                }
            } catch {
                localStorage.removeItem('token');
                localStorage.removeItem('user');
                set({ token: null, user: null, isAuthenticated: false });
            }
        }

        set({ isLoading: false });
    },
}));
