/**
 * Metabolic Syndrome Service
 * 
 * Flow:
 *   1. predict()  → calls Python service → gets raw probability from Bayesian Network
 *                  → probability > 0.5 means has MetS
 *   2. severity() → uses cMetS_S formula → combines probability + clinical markers
 *                  → classifies into Low / Medium / High Severity
 */

const { config } = require('../config/config');

// ==================== Constants ====================

const METS_THRESHOLD = 0.65; // probability > 65% = has metabolic syndrome

// cMetS_S Coefficients by gender and age group (from research paper)
const CMETS_COEFFICIENTS = {
    Men: {
        '20-39': { intercept: -1.79, sbp: 0.0016, wc: 0.0045, fpg: 0.0017, logTg: 0.24, hdlC: -0.0042 },
        '40-60': { intercept: -1.67, sbp: 0.0007, wc: 0.0034, fpg: 0.0014, logTg: 0.25, hdlC: -0.0042 },
        'default': { intercept: -2.28, sbp: 0.0019, wc: 0.0067, fpg: 0.0027, logTg: 0.28, hdlC: -0.0054 },
    },
    Women: {
        '20-39': { intercept: -2.43, sbp: 0.0039, wc: 0.0066, fpg: 0.004, logTg: 0.28, hdlC: -0.0052 },
        '40-60': { intercept: -2.37, sbp: 0.001, wc: 0.0021, fpg: 0.0015, logTg: 0.41, hdlC: -0.004 },
        'default': { intercept: -4.13, sbp: 0.0065, wc: 0.012, fpg: 0.007, logTg: 0.39, hdlC: -0.006 },
    },
    Other: {
        '20-39': { intercept: -2.34, sbp: 0.003, wc: 0.0061, fpg: 0.0032, logTg: 0.29, hdlC: -0.0055 },
        '40-60': { intercept: -1.94, sbp: 0.0006, wc: 0.0019, fpg: 0.0011, logTg: 0.33, hdlC: -0.003 },
        'default': { intercept: -3.39, sbp: 0.0044, wc: 0.0099, fpg: 0.0054, logTg: 0.36, hdlC: -0.0063 },
    },
};

// ==================== Helper Functions ====================

const getAgeGroupKey = (age) => {
    if (age >= 20 && age <= 39) return '20-39';
    if (age >= 40 && age <= 60) return '40-60';
    return 'default';
};

// ==================== Core Functions ====================

/**
 * Call Python service to get raw probability from the Bayesian Network.
 * Returns: { probability: 0.xxxx, hasMetabolicSyndrome: true/false }
 */
const predictWithBayesianNetwork = async (evidence) => {
    const response = await fetch(`${config.pythonServiceUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(evidence),
    });

    if (!response.ok) {
        throw new Error(`Python service error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
        throw new Error(data.error);
    }

    // Python service returns just { probability: 0.xxxx }
    // We decide the threshold here
    const probability = data.probability;
    const hasMetabolicSyndrome = probability > METS_THRESHOLD;

    return { probability, hasMetabolicSyndrome };
};

/**
 * Calculate cMetS_S severity score from clinical markers.
 * Uses published coefficients based on gender and age group.
 */
const calculateSeverityScore = (gender, age, sbp, wc, fpg, tg, hdlC) => {
    const logTg = Math.log(tg);
    const genderKey = gender === 'Men' || gender === 'Women' ? gender : 'Other';
    const c = CMETS_COEFFICIENTS[genderKey][getAgeGroupKey(age)];

    return c.intercept + c.sbp * sbp + c.wc * wc + c.fpg * fpg + c.logTg * logTg + c.hdlC * hdlC;
};

/**
 * Combine Bayesian probability + cMetS_S score into final severity (0 to 0.99).
 */
const calculateFinalSeverity = (probability, B) => {
    return Math.min(0.99, Math.max(0, probability + B));
};

/**
 * Classify severity into Low / Medium / High.
 */
const classifySeverity = (severity) => {
    if (severity <= 0.30) return 'Low Severity';
    if (severity <= 0.60) return 'Medium Severity';
    return 'High Severity';
};

/**
 * Health check for the Python prediction service.
 */
const checkPythonServiceHealth = async () => {
    try {
        const response = await fetch(`${config.pythonServiceUrl}/health`);
        const data = await response.json();
        return data.status === 'ok' ? 'running' : 'error';
    } catch {
        return 'not running';
    }
};

module.exports = {
    predictWithBayesianNetwork,
    calculateSeverityScore,
    calculateFinalSeverity,
    classifySeverity,
    checkPythonServiceHealth,
};
