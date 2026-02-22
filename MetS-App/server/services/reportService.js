/**
 * Report Service
 * Generates a downloadable markdown health report.
 */

const generateHealthReport = (userInfo, results, recommendations) => {
    const date = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });

    const sections = [
        `# METABOLIC SYNDROME HEALTH PLAN`,
        `Generated on: ${date}\n`,

        `## PATIENT INFORMATION`,
        ...Object.entries(userInfo).map(([key, value]) => `- ${key}: ${value}`),
        '',

        `## ASSESSMENT RESULTS`,
        `- Probability of Metabolic Syndrome: ${(results.probability * 100).toFixed(1)}%`,
        ...(results.severity !== undefined
            ? [`- Severity Score: ${results.severity.toFixed(2)}`, `- Risk Level: ${results.riskLevel}`]
            : []),
        '',

        `## HEALTH RECOMMENDATIONS`,
        ...formatList('Diet Plan Recommendations', recommendations.dietPlan),
        ...formatList('Foods to Avoid', recommendations.avoidList),
        ...formatList('Exercise Plan Recommendations', recommendations.exercisePlan),
        ...formatList('Yoga Poses Recommendations', recommendations.yogaPoses),

        `## DISCLAIMER`,
        `This health plan is generated based on the information you provided and is for informational purposes only. ` +
        `It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. ` +
        `Always seek the advice of your physician or other qualified health provider with any questions you may have regarding your health.`,
    ];

    return sections.join('\n');
};

const formatList = (title, items) => {
    if (!items?.length) return [];
    return [`### ${title}`, ...items.map(item => `- ${item}`), ''];
};

module.exports = { generateHealthReport };
