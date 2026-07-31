/**
 * Authentication Routes
 */

const express = require('express');
const router = express.Router();
const { body, validationResult } = require('express-validator');

const User = require('../models/User');
const { protect, generateToken, asyncHandler, formatUserResponse } = require('../middleware');

// ==================== Validation Rules ====================

const signupValidation = [
    body('firstName').trim().notEmpty().withMessage('First name is required')
        .isLength({ min: 2, max: 50 }).withMessage('First name must be 2-50 characters'),
    body('lastName').trim().notEmpty().withMessage('Last name is required')
        .isLength({ min: 2, max: 50 }).withMessage('Last name must be 2-50 characters'),
    body('email').trim().notEmpty().withMessage('Email is required')
        .isEmail().withMessage('Please enter a valid email').normalizeEmail(),
    body('password').notEmpty().withMessage('Password is required')
        .isLength({ min: 6 }).withMessage('Password must be at least 6 characters'),
    body('confirmPassword').notEmpty().withMessage('Please confirm your password')
        .custom((value, { req }) => {
            if (value !== req.body.password) throw new Error('Passwords do not match');
            return true;
        }),
];

const loginValidation = [
    body('email').trim().notEmpty().withMessage('Email is required')
        .isEmail().withMessage('Please enter a valid email').normalizeEmail(),
    body('password').notEmpty().withMessage('Password is required'),
];

const profileValidation = [
    body('firstName').optional().trim().isLength({ min: 2, max: 50 }).withMessage('First name must be 2-50 characters'),
    body('lastName').optional().trim().isLength({ min: 2, max: 50 }).withMessage('Last name must be 2-50 characters'),
    body('phone').optional().trim(),
    body('gender').optional().isIn(['Male', 'Female', 'Other', '']).withMessage('Invalid gender value'),
    body('address').optional().trim(),
];

const passwordValidation = [
    body('currentPassword').notEmpty().withMessage('Current password is required'),
    body('newPassword').notEmpty().withMessage('New password is required')
        .isLength({ min: 6 }).withMessage('New password must be at least 6 characters'),
    body('confirmPassword').notEmpty().withMessage('Please confirm your new password')
        .custom((value, { req }) => {
            if (value !== req.body.newPassword) throw new Error('Passwords do not match');
            return true;
        }),
];

// ==================== Helper ====================

const checkValidation = (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        res.status(400).json({ success: false, message: 'Validation failed', errors: errors.array() });
        return false;
    }
    return true;
};

// ==================== Routes ====================

// POST /api/auth/signup
router.post('/signup', signupValidation, asyncHandler(async (req, res) => {
    if (!checkValidation(req, res)) return;

    const { firstName, lastName, email, password } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
        return res.status(400).json({ success: false, message: 'An account with this email already exists' });
    }

    const user = await User.create({ firstName, lastName, email, password });

    res.status(201).json({
        success: true,
        message: 'Account created successfully',
        token: generateToken(user._id),
        user: formatUserResponse(user),
    });
}));

// POST /api/auth/login
router.post('/login', loginValidation, asyncHandler(async (req, res) => {
    if (!checkValidation(req, res)) return;

    const { email, password } = req.body;
    const user = await User.findOne({ email }).select('+password');

    if (!user || !(await user.comparePassword(password))) {
        return res.status(401).json({ success: false, message: 'Invalid email or password' });
    }

    if (!user.isActive) {
        return res.status(401).json({ success: false, message: 'Account deactivated. Contact support.' });
    }

    user.lastLogin = new Date();
    await user.save({ validateBeforeSave: false });

    res.json({
        success: true,
        message: 'Login successful',
        token: generateToken(user._id),
        user: formatUserResponse(user),
    });
}));

// GET /api/auth/me
router.get('/me', protect, asyncHandler(async (req, res) => {
    const user = await User.findById(req.user.id);
    res.json({ success: true, user: formatUserResponse(user) });
}));

// PUT /api/auth/profile
router.put('/profile', protect, profileValidation, asyncHandler(async (req, res) => {
    if (!checkValidation(req, res)) return;

    const allowedFields = ['firstName', 'lastName', 'phone', 'dateOfBirth', 'gender', 'address', 'profileImage'];
    const updates = {};
    for (const field of allowedFields) {
        if (req.body[field] !== undefined) updates[field] = req.body[field];
    }

    const user = await User.findByIdAndUpdate(req.user.id, { $set: updates }, { new: true, runValidators: true });

    res.json({ success: true, message: 'Profile updated successfully', user: formatUserResponse(user) });
}));

// PUT /api/auth/password
router.put('/password', protect, passwordValidation, asyncHandler(async (req, res) => {
    if (!checkValidation(req, res)) return;

    const user = await User.findById(req.user.id).select('+password');

    if (!(await user.comparePassword(req.body.currentPassword))) {
        return res.status(401).json({ success: false, message: 'Current password is incorrect' });
    }

    user.password = req.body.newPassword;
    await user.save();

    res.json({ success: true, message: 'Password updated successfully', token: generateToken(user._id) });
}));

// POST /api/auth/assessment
router.post('/assessment', protect, asyncHandler(async (req, res) => {
    const { probability, severity, riskLevel, recommendations, inputParameters } = req.body;

    const user = await User.findById(req.user.id);
    
    // Add new assessment
    user.assessmentHistory.push({
        date: new Date(),
        probability,
        severity,
        riskLevel,
        inputParameters: inputParameters || {},
        recommendations
    });
    
    // Sort by date descending and keep only the latest 7
    user.assessmentHistory.sort((a, b) => new Date(b.date) - new Date(a.date));
    if (user.assessmentHistory.length > 7) {
        user.assessmentHistory = user.assessmentHistory.slice(0, 7);
    }
    
    await user.save({ validateBeforeSave: false });

    res.json({ success: true, message: 'Assessment saved', assessmentHistory: user.assessmentHistory });
}));

// GET /api/auth/assessments
router.get('/assessments', protect, asyncHandler(async (req, res) => {
    const user = await User.findById(req.user.id);
    // Return only latest 7, sorted by date descending
    const history = (user.assessmentHistory || [])
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .slice(0, 7);
    res.json({ success: true, assessmentHistory: history });
}));

// DELETE /api/auth/account
router.delete('/account', protect, asyncHandler(async (req, res) => {
    await User.findByIdAndDelete(req.user.id);
    res.json({ success: true, message: 'Account deleted successfully' });
}));

module.exports = router;
