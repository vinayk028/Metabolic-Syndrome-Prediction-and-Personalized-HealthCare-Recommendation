/**
 * Report Service
 * Calls the Spring Boot report-service microservice for PDF generation.
 */

const REPORT_SERVICE_URL = process.env.REPORT_SERVICE_URL || 'http://localhost:8081';

const generatePdfReport = async (userInfo, results, recommendations) => {
    try {
        const response = await fetch(REPORT_SERVICE_URL + '/api/reports/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patientInfo: userInfo,
                results,
                recommendations,
            }),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Report service error:', response.status, errorText);
            throw new Error(`Report service error: ${response.status} - ${errorText}`);
        }

        // Get PDF as buffer
        const pdfBuffer = Buffer.from(await response.arrayBuffer());
        
        // Convert to base64 for transmission to client
        const base64Pdf = pdfBuffer.toString('base64');
        
        return base64Pdf;
    } catch (error) {
        console.error('Failed to generate PDF report:', error.message);
        throw error;
    }
};

module.exports = { generatePdfReport };
