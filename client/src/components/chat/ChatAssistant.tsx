/**
 * ChatAssistant Component
 * Floating chat widget with FAB button and expandable chat panel.
 * Matches MetS Health app design system — teal theme, glassmorphism, smooth animations.
 * Available on every page via Layout.
 */

import { useState, useRef, useEffect, type KeyboardEvent } from 'react';
import {
    Fab,
    Paper,
    Box,
    Typography,
    TextField,
    IconButton,
    Avatar,
    Chip,
    Button,
    Tooltip,
} from '@mui/material';
// Path imports (as elsewhere in the app): the package barrel pulls in every icon in dev and tests
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import ClearIcon from '@mui/icons-material/DeleteOutline';
import AssistantIcon from '@mui/icons-material/HealthAndSafety';
import { useChatStore, QUICK_ACTIONS } from '../../stores/chatStore';
import './ChatAssistant.css';

// ============ Markdown Renderer ============

/** Escape HTML so reply text can never inject markup or script */
const escapeHtml = (text: string): string =>
    text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');

/** Simple markdown → HTML converter for assistant replies (input is escaped first) */
const renderMarkdown = (text: string): string => {
    return escapeHtml(text)
        // Bold: **text**
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic: *text* (not preceded/followed by *)
        .replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
        // Inline code: `code`
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Headings (## / ###): a bold line, not a page heading
        .replace(/^#{2,3} (.*)$/gm, '<span class="chat-heading">$1</span>')
        // Unordered list items: - text or • text
        .replace(/^[-•] (.*$)/gm, '<li>$1</li>')
        // Wrap consecutive <li> tags in <ul> (newlines between items dropped: a <ul> may only hold <li>)
        .replace(/((?:<li>.*<\/li>\n?)+)/g, (items) => `<ul>${items.replace(/\n/g, '')}</ul>`)
        // Paragraphs: double newline
        .replace(/\n\n/g, '</p><p>')
        // Single newline → <br> (a heading is a block, so it needs no break after it)
        .replace(/(<\/span>)\n/g, '$1')
        .replace(/\n/g, '<br>')
        // Wrap in paragraph
        .replace(/^(.*)$/, '<p>$1</p>')
        // Clean up empty paragraphs
        .replace(/<p><\/p>/g, '')
        // Clean up paragraphs wrapping lists
        .replace(/<p>(<ul>)/g, '$1')
        .replace(/(<\/ul>)<\/p>/g, '$1');
};

// ============ Sub-Components ============

const TypingIndicator = () => (
    <div className="chat-typing">
        <Avatar className="chat-typing-avatar">
            <AssistantIcon sx={{ fontSize: 16 }} />
        </Avatar>
        <div className="chat-typing-bubble">
            <div className="chat-typing-dot" />
            <div className="chat-typing-dot" />
            <div className="chat-typing-dot" />
        </div>
    </div>
);

// ============ Main Component ============

const ChatAssistant = () => {
    const { isOpen, messages, loading, toggleChat, sendMessage, clearChat } = useChatStore();
    const [input, setInput] = useState('');
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Auto-scroll to bottom on new messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    // Focus input when chat opens
    useEffect(() => {
        if (isOpen) {
            setTimeout(() => inputRef.current?.focus(), 250);
        }
    }, [isOpen]);

    // Escape closes the panel; lock page scroll while the full-screen sheet is open on phones
    useEffect(() => {
        if (!isOpen) return;
        const onKey = (e: globalThis.KeyboardEvent) => {
            if (e.key === 'Escape') toggleChat();
        };
        const isPhone = window.matchMedia('(max-width: 599.95px)').matches;
        const prevOverflow = document.body.style.overflow;
        if (isPhone) document.body.style.overflow = 'hidden';
        window.addEventListener('keydown', onKey);
        return () => {
            window.removeEventListener('keydown', onKey);
            document.body.style.overflow = prevOverflow;
        };
    }, [isOpen, toggleChat]);

    const handleSend = () => {
        const trimmed = input.trim();
        if (!trimmed || loading) return;
        setInput('');
        sendMessage(trimmed);
    };

    const handleKeyDown = (e: KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleQuickAction = (message: string) => {
        if (loading) return;
        sendMessage(message);
    };

    // Show quick actions only when there's just the welcome message
    const showQuickActions = messages.length <= 1;

    return (
        <>
            {/* ===== Chat Panel ===== */}
            {isOpen && (
                <Paper
                    className="chat-panel"
                    elevation={0}
                    id="vb-chat-panel"
                    role="dialog"
                    aria-label="MetS Health Assistant"
                >
                    {/* Header */}
                    <Box className="chat-header">
                        <Box className="chat-header-info">
                            <Avatar className="chat-header-avatar">
                                <AssistantIcon />
                            </Avatar>
                            <Box className="chat-header-text">
                                <Typography variant="subtitle2" component="h2">
                                    MetS Health Assistant
                                </Typography>
                                <Box className="chat-header-status">
                                    <span className="chat-status-dot" />
                                    <Typography variant="caption" component="span">
                                        Online — Ready to help
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                        <Box className="chat-header-actions">
                            <Tooltip title="Clear conversation" arrow>
                                <IconButton size="small" onClick={clearChat} aria-label="Clear conversation">
                                    <ClearIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Close" arrow>
                                <IconButton size="small" onClick={toggleChat} aria-label="Close chat">
                                    <CloseIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>

                    {/* Messages */}
                    <Box className="chat-messages" role="log" aria-live="polite" aria-label="Conversation" tabIndex={0}>
                        {messages.map((msg) => (
                            <Box key={msg.id} className={`chat-message ${msg.role}`}>
                                <Box>
                                    {msg.role === 'assistant' ? (
                                        <Box
                                            className="chat-bubble"
                                            dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                                        />
                                    ) : (
                                        // The user's own text is shown as plain text, never as HTML
                                        <Box className="chat-bubble">{msg.content}</Box>
                                    )}
                                    <Typography className="chat-time">
                                        {new Date(msg.timestamp).toLocaleTimeString([], {
                                            hour: '2-digit',
                                            minute: '2-digit',
                                        })}
                                    </Typography>
                                </Box>
                            </Box>
                        ))}
                        {loading && <TypingIndicator />}
                        <div ref={messagesEndRef} />
                    </Box>

                    {/* Quick Actions */}
                    {showQuickActions && (
                        <Box className="chat-quick-actions">
                            {QUICK_ACTIONS.map((action) => (
                                <Chip
                                    key={action.label}
                                    label={action.label}
                                    variant="outlined"
                                    size="small"
                                    onClick={() => handleQuickAction(action.message)}
                                />
                            ))}
                        </Box>
                    )}

                    {/* Input */}
                    <Box className="chat-input-area">
                        <TextField
                            inputRef={inputRef}
                            fullWidth
                            size="small"
                            placeholder="Ask me anything about MetS..."
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            disabled={loading}
                            slotProps={{
                                input: { sx: { pr: 1 } },
                            }}
                        />
                        <Button
                            className="chat-send-btn"
                            onClick={handleSend}
                            disabled={!input.trim() || loading}
                            aria-label="Send message"
                        >
                            <SendIcon fontSize="small" />
                        </Button>
                    </Box>
                </Paper>
            )}

            {/* ===== FAB Button ===== */}
            <Fab
                className={`chat-fab ${isOpen ? 'chat-fab-open' : ''}`}
                onClick={toggleChat}
                aria-label={isOpen ? 'Close chat assistant' : 'Open chat assistant'}
                aria-expanded={isOpen}
                aria-controls={isOpen ? 'vb-chat-panel' : undefined}
            >
                {isOpen ? (
                    <CloseIcon className="chat-fab-icon" />
                ) : (
                    <ChatIcon className="chat-fab-icon" />
                )}
            </Fab>
        </>
    );
};

export default ChatAssistant;
