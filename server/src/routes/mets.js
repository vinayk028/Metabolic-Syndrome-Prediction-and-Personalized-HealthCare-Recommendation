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
const { asyncHandler } = require('../middlewares/middleware');
const metsController = require('../controllers/metsController');

router.post('/predict', asyncHandler(metsController.predict));
router.post('/severity', asyncHandler(metsController.severity));
router.post('/recommendations', asyncHandler(metsController.recommendations));
router.post('/report', asyncHandler(metsController.report));

module.exports = router;
