/**
 * Report Service
 * Generates a downloadable PDF health report.
 */

const PDFDocument = require('pdfkit');

/**
 * Generate health report as PDF Buffer
 * @param {Object} userInfo - Patient information
 * @param {Object} results - Assessment results (probability, severity, riskLevel)
 * @param {Object} recommendations - Health recommendations
 * @returns {Promise<Buffer>} PDF buffer
 */
const generateHealthReportPDF = (userInfo, results, recommendations) => {
    return new Promise((resolve, reject) => {
        try {
            const doc = new PDFDocument({
                size: 'A4',
                margin: 50,
            });

            const chunks = [];
            doc.on('data', (chunk) => chunks.push(chunk));
            doc.on('end', () => resolve(Buffer.concat(chunks)));
            doc.on('error', reject);

            const date = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

            // Title
            doc.fontSize(24).font('Helvetica-Bold').text('METABOLIC SYNDROME HEALTH REPORT', { align: 'center' });
            doc.fontSize(10).font('Helvetica').text(`Generated on: ${date}`, { align: 'center' });
            doc.moveDown();

            // Patient Information Section
            doc.fontSize(14).font('Helvetica-Bold').text('PATIENT INFORMATION', { underline: true });
            doc.fontSize(11).font('Helvetica');
            Object.entries(userInfo).forEach(([key, value]) => {
                doc.text(`${key}: ${value}`);
            });
            doc.moveDown();

            // Assessment Results Section
            doc.fontSize(14).font('Helvetica-Bold').text('ASSESSMENT RESULTS', { underline: true });
            doc.fontSize(11).font('Helvetica');
            doc.text(`Probability of Metabolic Syndrome: ${(results.probability * 100).toFixed(1)}%`);
            if (results.severity !== undefined) {
                doc.text(`Severity Score: ${results.severity.toFixed(2)}`);
                doc.text(`Risk Level: ${results.riskLevel}`);
            }
            doc.moveDown();

            // Health Recommendations Section
            doc.fontSize(14).font('Helvetica-Bold').text('HEALTH RECOMMENDATIONS', { underline: true });
            doc.fontSize(11).font('Helvetica');

            if (recommendations.dietPlan?.length) {
                doc.fontSize(12).font('Helvetica-Bold').text('Diet Plan Recommendations:');
                doc.fontSize(11).font('Helvetica');
                recommendations.dietPlan.forEach((item) => {
                    doc.text(`• ${item}`);
                });
                doc.moveDown(0.5);
            }

            if (recommendations.avoidList?.length) {
                doc.fontSize(12).font('Helvetica-Bold').text('Foods to Avoid:');
                doc.fontSize(11).font('Helvetica');
                recommendations.avoidList.forEach((item) => {
                    doc.text(`• ${item}`);
                });
                doc.moveDown(0.5);
            }

            if (recommendations.exercisePlan?.length) {
                doc.fontSize(12).font('Helvetica-Bold').text('Exercise Plan Recommendations:');
                doc.fontSize(11).font('Helvetica');
                recommendations.exercisePlan.forEach((item) => {
                    doc.text(`• ${item}`);
                });
                doc.moveDown(0.5);
            }

            if (recommendations.yogaPoses?.length) {
                doc.fontSize(12).font('Helvetica-Bold').text('Yoga Poses Recommendations:');
                doc.fontSize(11).font('Helvetica');
                recommendations.yogaPoses.forEach((item) => {
                    doc.text(`• ${item}`);
                });
                doc.moveDown(0.5);
            }

            // Disclaimer Section
            doc.moveDown();
            doc.fontSize(10).font('Helvetica-Bold').text('DISCLAIMER', { underline: true });
            doc.fontSize(9).font('Helvetica');
            const disclaimerText =
                'This health plan is generated based on the information you provided and is for informational purposes only. ' +
                'It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. ' +
                'Always seek the advice of your physician or other qualified health provider with any questions you may have regarding your health.';
            doc.text(disclaimerText, { align: 'left' });

            doc.end();
        } catch (err) {
            reject(err);
        }
    });
};

module.exports = { generateHealthReportPDF };
