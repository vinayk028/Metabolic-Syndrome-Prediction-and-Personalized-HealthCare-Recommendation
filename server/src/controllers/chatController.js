const chatService = require('../services/chatService');

const RATE_LIMIT = { maxRequests: 30, windowMs: 60 * 60 * 1000 }; // 30 msgs/hour
const rateLimitMap = new Map();

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

setInterval(() => {
    const now = Date.now();
    for (const [key, entry] of rateLimitMap) {
        if (now - entry.windowStart > RATE_LIMIT.windowMs) rateLimitMap.delete(key);
    }
}, 30 * 60 * 1000);

const sendMessage = async (req, res) => {
    const { message, history } = req.body;

    if (!message || typeof message !== 'string' || !message.trim()) {
        return res.status(400).json({ success: false, message: 'Message is required.' });
    }

    if (message.trim().length > 1000) {
        return res.status(400).json({ success: false, message: 'Message too long. Max 1000 characters.' });
    }

    const rateLimitKey = req.user?._id?.toString() || req.ip;
    if (!checkRateLimit(rateLimitKey)) {
        return res.status(429).json({ success: false, message: 'Too many messages. Please try again later.' });
    }

    const validHistory = Array.isArray(history)
        ? history.filter((m) => m.role && m.content && ['user', 'assistant'].includes(m.role))
        : [];

    try {
        const response = await chatService.chat(message.trim(), validHistory, req.user || null);
        res.json({ success: true, response });
    } catch (error) {
        console.error('Chat error:', error.message);

        const status = error.response?.status;
        if (status === 400) {
            return res.status(400).json({ success: false, message: 'Invalid API key. Check your LLM_API_KEY in .env.' });
        }
        if (status === 429) {
            return res.status(429).json({ success: false, message: 'API rate limit reached. Please try again in a minute.' });
        }

        res.status(500).json({ success: false, message: 'Assistant is temporarily unavailable. Please try again.' });
    }
};

module.exports = { sendMessage };