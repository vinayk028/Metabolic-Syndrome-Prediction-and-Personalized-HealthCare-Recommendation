/**
 * Chat Store (Zustand)
 * Manages chat assistant state: messages, open/close, loading, and API calls.
 */

import { create } from 'zustand';
import type { ChatMessage } from '../types/types';
import { sendChatMessage } from '../services/api';

// ============ Quick Actions ============

export const QUICK_ACTIONS = [
    { label: '🔍 Start Assessment', message: 'How do I start a health assessment?' },
    { label: '📊 My Results', message: 'Can you explain my latest assessment results?' },
    { label: '📰 Latest News', message: 'What are the latest health news articles?' },
    { label: '❓ What is MetS?', message: 'What is Metabolic Syndrome?' },
];

// ============ Helpers ============

const generateId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

const WELCOME_MESSAGE: ChatMessage = {
    id: 'welcome',
    role: 'assistant',
    content: `👋 Hi! I'm your **MetS Health Assistant**. I can help you with:\n\n- **Assessment** — Guide you through the health check\n- **Results** — Explain your risk scores and what they mean\n- **Recommendations** — Diet, exercise, and yoga suggestions\n- **News** — Latest metabolic syndrome research & articles\n- **General Health** — Answer questions about MetS\n\nHow can I help you today?`,
    timestamp: new Date(),
};

// ============ Store Interface ============

interface ChatState {
    isOpen: boolean;
    messages: ChatMessage[];
    loading: boolean;
    error: string | null;

    // Actions
    toggleChat: () => void;
    sendMessage: (content: string) => Promise<void>;
    clearChat: () => void;
}

// ============ Store ============

export const useChatStore = create<ChatState>((set, get) => ({
    isOpen: false,
    messages: [WELCOME_MESSAGE],
    loading: false,
    error: null,

    toggleChat: () => set((s) => ({ isOpen: !s.isOpen })),

    sendMessage: async (content) => {
        const { messages } = get();

        // Add user message
        const userMessage: ChatMessage = {
            id: generateId(),
            role: 'user',
            content,
            timestamp: new Date(),
        };
        set({ messages: [...messages, userMessage], loading: true, error: null });

        try {
            // Build history for API (exclude welcome message, only role + content)
            const history = get()
                .messages.filter((m) => m.id !== 'welcome')
                .slice(0, -1) // exclude the user message we just added (API receives it as 'message')
                .map(({ role, content }) => ({ role, content }));

            const { success, response, message } = await sendChatMessage(content, history);

            if (!success) {
                throw new Error(message || 'Failed to get response');
            }

            const assistantMessage: ChatMessage = {
                id: generateId(),
                role: 'assistant',
                content: response,
                timestamp: new Date(),
            };
            set((s) => ({ messages: [...s.messages, assistantMessage] }));
        } catch (err) {
            const errorMsg = err instanceof Error ? err.message : 'Something went wrong. Please try again.';
            set({ error: errorMsg });

            // Add error as assistant message so user sees it in the chat
            const errorMessage: ChatMessage = {
                id: generateId(),
                role: 'assistant',
                content: `⚠️ Sorry, I couldn't process that. ${errorMsg}`,
                timestamp: new Date(),
            };
            set((s) => ({ messages: [...s.messages, errorMessage] }));
        } finally {
            set({ loading: false });
        }
    },

    clearChat: () => set({ messages: [WELCOME_MESSAGE], error: null }),
}));
