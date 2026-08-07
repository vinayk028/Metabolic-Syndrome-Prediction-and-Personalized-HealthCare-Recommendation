const { validationResult } = require('express-validator');
const User = require('../models/User');
const { generateToken, formatUserResponse } = require('../middlewares/middleware');

const validateRequest = (req, res) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        res.status(400).json({ success: false, message: 'Validation failed', errors: errors.array() });
        return false;
    }
    return true;
};

const signup = async (req, res) => {
    if (!validateRequest(req, res)) return;

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
};

const login = async (req, res) => {
    if (!validateRequest(req, res)) return;

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
};

const getCurrentUser = async (req, res) => {
    const user = await User.findById(req.user.id);
    res.json({ success: true, user: formatUserResponse(user) });
};

const updateProfile = async (req, res) => {
    if (!validateRequest(req, res)) return;

    const allowedFields = ['firstName', 'lastName', 'phone', 'dateOfBirth', 'gender', 'address', 'profileImage'];
    const updates = {};
    for (const field of allowedFields) {
        if (req.body[field] !== undefined) updates[field] = req.body[field];
    }

    const user = await User.findByIdAndUpdate(req.user.id, { $set: updates }, { new: true, runValidators: true });

    res.json({ success: true, message: 'Profile updated successfully', user: formatUserResponse(user) });
};

const updatePassword = async (req, res) => {
    if (!validateRequest(req, res)) return;

    const user = await User.findById(req.user.id).select('+password');

    if (!(await user.comparePassword(req.body.currentPassword))) {
        return res.status(401).json({ success: false, message: 'Current password is incorrect' });
    }

    user.password = req.body.newPassword;
    await user.save();

    res.json({ success: true, message: 'Password updated successfully', token: generateToken(user._id) });
};

const saveAssessment = async (req, res) => {
    const { probability, severity, riskLevel, recommendations, inputParameters } = req.body;
    const user = await User.findById(req.user.id);

    user.assessmentHistory.push({
        date: new Date(),
        probability,
        severity,
        riskLevel,
        inputParameters: inputParameters || {},
        recommendations,
    });

    user.assessmentHistory.sort((a, b) => new Date(b.date) - new Date(a.date));
    if (user.assessmentHistory.length > 7) {
        user.assessmentHistory = user.assessmentHistory.slice(0, 7);
    }

    await user.save({ validateBeforeSave: false });

    res.json({ success: true, message: 'Assessment saved', assessmentHistory: user.assessmentHistory });
};

const getAssessmentHistory = async (req, res) => {
    const user = await User.findById(req.user.id);
    const history = (user.assessmentHistory || [])
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .slice(0, 7);

    res.json({ success: true, assessmentHistory: history });
};

const deleteAccount = async (req, res) => {
    await User.findByIdAndDelete(req.user.id);
    res.json({ success: true, message: 'Account deleted successfully' });
};

module.exports = {
    signup,
    login,
    getCurrentUser,
    updateProfile,
    updatePassword,
    saveAssessment,
    getAssessmentHistory,
    deleteAccount,
};
