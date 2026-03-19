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
import {
    Chat as ChatIcon,
    Close as CloseIcon,
    Send as SendIcon,
    DeleteOutline as ClearIcon,
    HealthAndSafety as AssistantIcon,
} from '@mui/icons-material';
import { useChatStore, QUICK_ACTIONS } from '../stores/chatStore';
import './ChatAssistant.css';

// ============ Markdown Renderer ============

/** Simple markdown → HTML converter for chat bubbles */
const renderMarkdown = (text: string): string => {
    return text
        // Bold: **text**
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // Italic: *text* (not preceded/followed by *)
        .replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
        // Inline code: `code`
        .replace(/`(.*?)`/g, '<code>$1</code>')
        // Headers: ### text
        .replace(/^### (.*$)/gm, '<h4>$1</h4>')
        .replace(/^## (.*$)/gm, '<h3>$1</h3>')
        // Unordered list items: - text or • text
        .replace(/^[-•] (.*$)/gm, '<li>$1</li>')
        // Wrap consecutive <li> tags in <ul>
        .replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
        // Paragraphs: double newline
        .replace(/\n\n/g, '</p><p>')
        // Single newline → <br>
        .replace(/\n/g, '<br>')
        // Wrap in paragraph
        .replace(/^(.*)$/, '<p>$1</p>')
        // Clean up empty paragraphs
        .replace(/<p><\/p>/g, '')
        // Clean up paragraphs wrapping block elements
        .replace(/<p>(<h[34]>)/g, '$1')
        .replace(/(<\/h[34]>)<\/p>/g, '$1')
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
                <Paper className="chat-panel" elevation={0}>
                    {/* Header */}
                    <Box className="chat-header">
                        <Box className="chat-header-info">
                            <Avatar className="chat-header-avatar">
                                <AssistantIcon />
                            </Avatar>
                            <Box className="chat-header-text">
                                <Typography variant="subtitle2" component="h6">
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
                                <IconButton size="small" onClick={clearChat}>
                                    <ClearIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Close" arrow>
                                <IconButton size="small" onClick={toggleChat}>
                                    <CloseIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>

                    {/* Messages */}
                    <Box className="chat-messages">
                        {messages.map((msg) => (
                            <Box key={msg.id} className={`chat-message ${msg.role}`}>
                                <Box>
                                    <Box
                                        className="chat-bubble"
                                        dangerouslySetInnerHTML={{
                                            __html: msg.role === 'assistant'
                                                ? renderMarkdown(msg.content)
                                                : msg.content,
                                        }}
                                    />
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
                aria-label="Chat assistant"
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
