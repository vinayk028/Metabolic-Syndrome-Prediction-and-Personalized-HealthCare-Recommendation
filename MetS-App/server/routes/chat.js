/**
 * Chat Routes
 * POST /api/chat — Send a message to the AI assistant
 */

const express = require('express');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { config } = require('../config');
const { asyncHandler } = require('../middleware');
const chatService = require('../services/chatService');

const router = express.Router();

// ==================== Rate Limiting (In-Memory) ====================

const rateLimitMap = new Map();
const RATE_LIMIT = { maxRequests: 30, windowMs: 60 * 60 * 1000 }; // 30 msgs/hour

const checkRateLimit = (key) => {
    const now = Date.now();
    const entry = rateLimitMap.get(key);

    if (!entry || now - entry.windowStart > RATE_LIMIT.windowMs) {
        rateLimitMap.set(key, { windowStart: now, count: 1 });
        return true;
    }

    if (entry.count >= RATE_LIMIT.maxRequests) return false;

    entry.count++;
    return true;
};

// Cleanup stale entries every 30 minutes
setInterval(() => {
    const now = Date.now();
    for (const [key, entry] of rateLimitMap) {
        if (now - entry.windowStart > RATE_LIMIT.windowMs) rateLimitMap.delete(key);
    }
}, 30 * 60 * 1000);

// ==================== Auth (Optional) ====================

/**
 * Try to extract user from token, but don't require it.
 * Logged-in users get personalized responses; guests get general answers.
 */
const optionalAuth = async (req, _res, next) => {
    try {
        const authHeader = req.headers.authorization;
        const token = authHeader?.startsWith('Bearer') ? authHeader.split(' ')[1] : null;

        if (token) {
            const decoded = jwt.verify(token, config.jwtSecret);
            req.user = await User.findById(decoded.id);
        }
    } catch {
        // Token invalid or expired — continue as guest
        req.user = null;
    }
    next();
};

// ==================== Route ====================

router.post('/', optionalAuth, asyncHandler(async (req, res) => {
    const { message, history } = req.body;

    // Validate input
    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ success: false, message: 'Message is required.' });
    }

    if (message.trim().length > 1000) {
        return res.status(400).json({ success: false, message: 'Message too long. Max 1000 characters.' });
    }

    // Rate limit by user ID or IP
    const rateLimitKey = req.user?._id?.toString() || req.ip;
    if (!checkRateLimit(rateLimitKey)) {
        return res.status(429).json({ success: false, message: 'Too many messages. Please try again later.' });
    }

    // Validate history format
    const validHistory = Array.isArray(history)
        ? history.filter((m) => m.role && m.content && ['user', 'assistant'].includes(m.role))
        : [];

    try {
        const response = await chatService.chat(message.trim(), validHistory, req.user || null);
        res.json({ success: true, response });
    } catch (error) {
        console.error('Chat error:', error.message);

        // Return user-friendly error based on type
        const status = error.response?.status;
        if (status === 400) {
            return res.status(400).json({ success: false, message: 'Invalid API key. Check your LLM_API_KEY in .env.' });
        }
        if (status === 429) {
            return res.status(429).json({ success: false, message: 'API rate limit reached. Please try again in a minute.' });
        }

        res.status(500).json({ success: false, message: 'Assistant is temporarily unavailable. Please try again.' });
    }
}));

module.exports = router;
