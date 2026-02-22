/**
 * Middleware - Auth, Error Handling, Utilities
 */

const jwt = require('jsonwebtoken');
const User = require('./models/User');
const { config } = require('./config');

// ==================== Auth Middleware ====================

const protect = async (req, res, next) => {
    try {
        const authHeader = req.headers.authorization;
        const token = authHeader?.startsWith('Bearer') ? authHeader.split(' ')[1] : null;

        if (!token) {
            return res.status(401).json({ success: false, message: 'Not authorized. Please login.' });
        }

        const decoded = jwt.verify(token, config.jwtSecret);
        const user = await User.findById(decoded.id);

        if (!user) {
            return res.status(401).json({ success: false, message: 'User not found. Please login again.' });
        }

        if (!user.isActive) {
            return res.status(401).json({ success: false, message: 'Account deactivated. Contact support.' });
        }

        req.user = user;
        next();
    } catch (error) {
        console.error('Auth error:', error.message);

        if (error.name === 'JsonWebTokenError') {
            return res.status(401).json({ success: false, message: 'Invalid token. Please login again.' });
        }
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({ success: false, message: 'Token expired. Please login again.' });
        }
        return res.status(401).json({ success: false, message: 'Not authorized.' });
    }
};

const generateToken = (userId) => {
    return jwt.sign({ id: userId }, config.jwtSecret, { expiresIn: config.jwtExpiresIn });
};

// ==================== Error Handling ====================

const notFoundHandler = (req, res) => {
    res.status(404).json({ success: false, message: `Route ${req.originalUrl} not found` });
};

// Note: `next` is required by Express to identify this as an error-handling middleware
const errorHandler = (err, _req, res, _next) => {
    console.error('Error:', config.isDev ? err : err.message);

    let statusCode = err.statusCode || 500;
    let message = err.message || 'Internal Server Error';

    if (err.name === 'CastError') {
        statusCode = 400;
        message = 'Invalid resource ID';
    }

    if (err.code === 11000) {
        statusCode = 400;
        message = `${Object.keys(err.keyValue)[0]} already exists`;
    }

    if (err.name === 'ValidationError') {
        statusCode = 400;
        message = Object.values(err.errors).map(val => val.message).join(', ');
    }

    res.status(statusCode).json({
        success: false,
        message,
        ...(config.isDev && { stack: err.stack }),
    });
};

// ==================== Utilities ====================

const asyncHandler = (fn) => (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
};

const formatUserResponse = (user) => ({
    id: user._id,
    firstName: user.firstName,
    lastName: user.lastName,
    email: user.email,
    fullName: user.fullName,
    phone: user.phone,
    dateOfBirth: user.dateOfBirth,
    gender: user.gender,
    address: user.address,
    profileImage: user.profileImage,
    role: user.role,
    lastLogin: user.lastLogin,
    assessmentHistory: user.assessmentHistory,
    createdAt: user.createdAt,
});

module.exports = {
    protect,
    generateToken,
    notFoundHandler,
    errorHandler,
    asyncHandler,
    formatUserResponse,
};
