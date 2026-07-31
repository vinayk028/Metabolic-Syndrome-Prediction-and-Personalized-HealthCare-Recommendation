/**
 * Metabolic Syndrome Routes
 * 
 * Flow:
 *   POST /predict          → Bayesian Network probability + hasMetabolicSyndrome (yes/no)
 *   POST /severity         → cMetS_S severity score + Low/Medium/High classification
 *   POST /recommendations  → diet, exercise, yoga recommendations
 *   POST /report           → downloadable health report
 */

const express = require('express');
const router = express.Router();

const metsService = require('../services/metsService');
const recommendationsService = require('../services/recommendationsService');
const reportService = require('../services/reportService');
const { asyncHandler } = require('../middleware');

// ==================== Routes ====================

// Step 1: Get probability from Bayesian Network
router.post('/predict', asyncHandler(async (req, res) => {
    const { fattyLiver, hypertension, diabetes, waistCircumference, systolicBP, diastolicBP, age, gender } = req.body;

    if (age === undefined || !gender) {
        return res.status(400).json({ error: 'age and gender are required' });
    }

    const evidence = {
        fattyLiver: fattyLiver ? 1 : 0,
        hypertension: hypertension ? 1 : 0,
        diabetes: diabetes ? 1 : 0,
        waistCircumference: parseInt(waistCircumference, 10),
        systolicBP: parseInt(systolicBP, 10),
        diastolicBP: parseInt(diastolicBP, 10),
    };

    // Returns { probability, hasMetabolicSyndrome }
    const result = await metsService.predictWithBayesianNetwork(evidence);

    res.json({
        probability: result.probability,
        hasMetabolicSyndrome: result.hasMetabolicSyndrome,
    });
}));

// Step 2: Calculate severity (only called if hasMetabolicSyndrome = true)
router.post('/severity', asyncHandler(async (req, res) => {
    const { gender, age, systolicBP, waistCircumference, fpg, triglyceride, hdlCholesterol, probability } = req.body;

    if (!gender || age === undefined || !systolicBP || !waistCircumference || !fpg || !triglyceride || !hdlCholesterol || probability === undefined) {
        return res.status(400).json({ error: 'All severity fields are required' });
    }

    const B = metsService.calculateSeverityScore(
        gender, parseInt(age), parseInt(systolicBP), parseInt(waistCircumference),
        parseInt(fpg), parseInt(triglyceride), parseInt(hdlCholesterol)
    );
    const severity = metsService.calculateFinalSeverity(probability, B);
    const riskLevel = metsService.classifySeverity(severity);

    res.json({ severity, riskLevel });
}));

// Step 3: Get recommendations based on severity
router.post('/recommendations', asyncHandler(async (req, res) => {
    const { gender, riskLevel, age } = req.body;

    if (!gender || !riskLevel || age === undefined) {
        return res.status(400).json({ error: 'gender, riskLevel, and age are required' });
    }

    const recommendations = recommendationsService.getRecommendations(gender, riskLevel, parseInt(age));
    res.json(recommendations);
}));

// Download report as PDF (returns base64-encoded PDF in JSON)
router.post('/report', asyncHandler(async (req, res) => {
    const { userInfo, results, recommendations } = req.body;

    if (!userInfo || !results) {
        return res.status(400).json({ error: 'userInfo and results are required' });
    }

    try {
        const base64Pdf = await reportService.generatePdfReport(userInfo, results, recommendations || {});
        // Return JSON with base64-encoded PDF (client will decode and download)
        res.json({ report: base64Pdf });
    } catch (error) {
        console.error('Report generation failed:', error);
        res.status(500).json({ error: 'Failed to generate report. Please ensure the Spring Boot report service is running on port 8081.' });
    }
}));

module.exports = router;
