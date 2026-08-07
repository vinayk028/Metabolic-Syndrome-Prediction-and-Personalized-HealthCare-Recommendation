/**
 * Authentication Routes
 */

const express = require('express');
const { body } = require('express-validator');
const router = express.Router();
const { protect, asyncHandler } = require('../middlewares/middleware');
const authController = require('../controllers/authController');

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

router.post('/signup', signupValidation, asyncHandler(authController.signup));
router.post('/login', loginValidation, asyncHandler(authController.login));
router.get('/me', protect, asyncHandler(authController.getCurrentUser));
router.put('/profile', protect, profileValidation, asyncHandler(authController.updateProfile));
router.put('/password', protect, passwordValidation, asyncHandler(authController.updatePassword));
router.post('/assessment', protect, asyncHandler(authController.saveAssessment));
router.get('/assessments', protect, asyncHandler(authController.getAssessmentHistory));
router.delete('/account', protect, asyncHandler(authController.deleteAccount));

module.exports = router;
