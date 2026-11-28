/**
 * Configuration & Database Connection
 */

const fs = require('fs');
const path = require('path');
const mongoose = require('mongoose');

const envCandidates = [
    path.resolve(__dirname, '..', '.env'),
    path.resolve(__dirname, '..', '..', '.env'),
    path.resolve(process.cwd(), '.env')
];
const resolvedEnvPath = envCandidates.find(candidate => fs.existsSync(candidate)) || envCandidates[envCandidates.length - 1];
require('dotenv').config({ path: resolvedEnvPath });

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
    // Comma-separated list in CORS_ORIGINS, e.g. "https://mets-client-production.up.railway.app".
    // Falls back to CLIENT_URL, then to the local dev origins.
    corsOrigins: (process.env.CORS_ORIGINS || process.env.CLIENT_URL)
        ? (process.env.CORS_ORIGINS || process.env.CLIENT_URL).split(',').map(origin => origin.trim()).filter(Boolean)
        : ['http://localhost:5173', 'http://localhost:3000'],

    // Python Service
    pythonServiceUrl: process.env.PYTHON_SERVICE_URL || 'http://localhost:5001',

    // News
    guardianApiKey: process.env.GUARDIAN_API_KEY,
    newsRefreshMinutes: 5,
    maxNewsArticles: 25,

    // Paths (resolved from project root)
    recommendationsPath: path.resolve(__dirname, '..', 'utils', 'plugins', 'HealthcareRecommendations', 'Healthcare_Recommendations.json'),

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
        if (!config.isDev) {
            // Refuse to start: tokens signed with a public default secret can be forged
            throw new Error('JWT_SECRET must be set in production.');
        }
        console.warn('⚠️  Warning: Using default JWT secret. Set JWT_SECRET in production.');
    }
    if (!config.guardianApiKey) {
        console.warn('⚠️  Warning: GUARDIAN_API_KEY not set. News features will not work.');
    }
};

module.exports = { config, connectDB, validateConfig };
