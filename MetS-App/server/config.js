/**
 * Configuration & Database Connection
 */

require('dotenv').config();
const path = require('path');
const mongoose = require('mongoose');

// ==================== Configuration ====================

const config = {
    // Server
    port: parseInt(process.env.PORT, 10) || 5000,
    nodeEnv: process.env.NODE_ENV || 'development',
    isDev: process.env.NODE_ENV !== 'production',

    // Database
    mongoUri: process.env.MONGODB_URI || 'mongodb://localhost:27017/mets-app',

    // JWT
    jwtSecret: process.env.JWT_SECRET || 'your-super-secret-jwt-key',
    jwtExpiresIn: process.env.JWT_EXPIRES_IN || '7d',

    // CORS
    corsOrigins: ['http://localhost:5173', 'http://localhost:3000'],

    // Python Service
    pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:5001',

    // News
    guardianApiKey: process.env.GUARDIAN_API_KEY,
    newsRefreshMinutes: 5,
    maxNewsArticles: 25,

    // Paths (resolved from project root)
    recommendationsPath: path.resolve(__dirname, '..', '..', 'MetS-App', 'plugins', 'HealthcareRecommendations', 'Healthcare_Recommendations.json'),

    // LLM Chat Assistant
    llmProvider: process.env.LLM_PROVIDER || 'gemini',          // 'gemini' or 'claude'
    llmApiKey: process.env.LLM_API_KEY || '',
    llmModel: process.env.LLM_MODEL || '',                      // optional: override default model
};

// ==================== Database Connection ====================

const connectDB = async () => {
    try {
        const conn = await mongoose.connect(config.mongoUri);
        console.log(`✅ MongoDB Connected: ${conn.connection.host}`);
        return conn;
    } catch (error) {
        console.error(`❌ MongoDB Connection Error: ${error.message}`);
        process.exit(1);
    }
};

// ==================== Validation ====================

const validateConfig = () => {
    if (config.jwtSecret === 'your-super-secret-jwt-key') {
        console.warn('⚠️  Warning: Using default JWT secret. Set JWT_SECRET in production.');
    }
    if (!config.guardianApiKey) {
        console.warn('⚠️  Warning: GUARDIAN_API_KEY not set. News features will not work.');
    }
};

module.exports = { config, connectDB, validateConfig };
