/**
 * Chat Routes
 * POST /api/chat — Send a message to the AI assistant
 */

const express = require('express');
const { asyncHandler } = require('../middlewares/middleware');
const chatController = require('../controllers/chatController');

const router = express.Router();

const optionalAuth = async (req, _res, next) => {
    try {
        const authHeader = req.headers.authorization;
        const token = authHeader?.startsWith('Bearer') ? authHeader.split(' ')[1] : null;

        if (token) {
            const jwt = require('jsonwebtoken');
            const { config } = require('../config/config');
            const decoded = jwt.verify(token, config.jwtSecret);
            const User = require('../models/User');
            req.user = await User.findById(decoded.id);
        }
    } catch {
        req.user = null;
    }
    next();
};

router.post('/', optionalAuth, asyncHandler(chatController.sendMessage));

module.exports = router;
