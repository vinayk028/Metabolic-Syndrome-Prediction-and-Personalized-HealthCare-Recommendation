const metsService = require('../services/metsService');
const recommendationsService = require('../services/recommendationsService');
const reportService = require('../services/reportService');

const predict = async (req, res) => {
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

    const result = await metsService.predictWithBayesianNetwork(evidence);

    res.json({
        probability: result.probability,
        hasMetabolicSyndrome: result.hasMetabolicSyndrome,
    });
};

const severity = async (req, res) => {
    const { gender, age, systolicBP, waistCircumference, fpg, triglyceride, hdlCholesterol, probability } = req.body;

    if (!gender || age === undefined || !systolicBP || !waistCircumference || !fpg || !triglyceride || !hdlCholesterol || probability === undefined) {
        return res.status(400).json({ error: 'All severity fields are required' });
    }

    const B = metsService.calculateSeverityScore(
        gender,
        parseInt(age),
        parseInt(systolicBP),
        parseInt(waistCircumference),
        parseInt(fpg),
        parseInt(triglyceride),
        parseInt(hdlCholesterol)
    );
    const severityValue = metsService.calculateFinalSeverity(probability, B);
    const riskLevel = metsService.classifySeverity(severityValue);

    res.json({ severity: severityValue, riskLevel });
};

const recommendations = async (req, res) => {
    const { gender, riskLevel, age } = req.body;

    if (!gender || !riskLevel || age === undefined) {
        return res.status(400).json({ error: 'gender, riskLevel, and age are required' });
    }

    const result = recommendationsService.getRecommendations(gender, riskLevel, parseInt(age));
    res.json(result);
};

const report = async (req, res) => {
    const { userInfo, results, recommendations } = req.body;

    if (!userInfo || !results) {
        return res.status(400).json({ error: 'userInfo and results are required' });
    }

    try {
        const base64Pdf = await reportService.generatePdfReport(userInfo, results, recommendations || {});
        res.json({ report: base64Pdf });
    } catch (error) {
        console.error('Report generation failed:', error);
        res.status(500).json({ error: 'Failed to generate report. Please ensure the Spring Boot report service is running on port 8081.' });
    }
};

module.exports = { predict, severity, recommendations, report };