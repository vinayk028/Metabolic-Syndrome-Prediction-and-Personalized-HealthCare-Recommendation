/**
 * Recommendations Service
 * Loads recommendations from JSON and returns them based on gender, risk level, and age.
 */

const fs = require('fs');
const { config } = require('../config');

// ==================== Data ====================

let recommendations = {};

// ==================== Functions ====================

const loadRecommendations = () => {
    try {
        recommendations = JSON.parse(fs.readFileSync(config.recommendationsPath, 'utf8'));
        console.log('✅ Recommendations loaded successfully');
    } catch (error) {
        console.error('❌ Error loading recommendations:', error.message);
    }
};

const getAgeGroup = (age) => (age < 40 ? '20-40' : '40-60');

const getRecommendations = (gender, riskLevel, age) => {
    const result = { dietPlan: [], avoidList: [], exercisePlan: [], yogaPoses: [] };

    try {
        const genderData = recommendations[gender]?.[riskLevel]?.[getAgeGroup(age)];

        if (genderData) {
            if (genderData['Diet Plan']) {
                result.dietPlan = genderData['Diet Plan']['Recommended'] || [];
                result.avoidList = genderData['Diet Plan']['Avoid'] || [];
            }

            if (genderData['Exercise Plan']) {
                const exercise = genderData['Exercise Plan'];
                result.exercisePlan = Array.isArray(exercise)
                    ? exercise
                    : Object.entries(exercise).map(([key, value]) => `${key}: ${value}`);
            }
        }

        const yogaPoses = recommendations['Yoga Poses for Metabolic Syndrome']?.[riskLevel];
        if (yogaPoses) result.yogaPoses = yogaPoses;
    } catch (error) {
        console.error('Error getting recommendations:', error.message);
    }

    return result;
};

// Load on startup
loadRecommendations();

module.exports = { getRecommendations };
