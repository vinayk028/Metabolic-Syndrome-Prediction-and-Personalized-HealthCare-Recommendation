/**
 * MetS Predictor API Server
 */

const express = require('express');
const cors = require('cors');
const cron = require('node-cron');

const { config, connectDB, validateConfig } = require('./config');
const { notFoundHandler, errorHandler } = require('./middleware');
const metsService = require('./services/metsService');
const newsService = require('./services/newsService');

// Routes
const authRoutes = require('./routes/auth');
const metsRoutes = require('./routes/mets');
const newsRoutes = require('./routes/news');

// ==================== App Setup ====================

const app = express();

validateConfig();

app.use(cors({ origin: config.corsOrigins, credentials: true }));
app.use(express.json({ limit: '10mb' }));

if (config.isDev) {
    app.use((req, _res, next) => {
        console.log(`${new Date().toISOString()} | ${req.method} ${req.path}`);
        next();
    });
}

// ==================== Routes ====================

app.get('/api/health', async (_req, res) => {
    const pythonService = await metsService.checkPythonServiceHealth();
    res.json({ status: 'ok', environment: config.nodeEnv, pythonService });
});

app.use('/api/auth', authRoutes);
app.use('/api/mets', metsRoutes);
app.use('/api/news', newsRoutes);

app.use(notFoundHandler);
app.use(errorHandler);

// ==================== News Cron ====================

const startNewsCron = () => {
    newsService.fetchAndStoreNews()
        .then(r => console.log('📰 Initial news:', r.message))
        .catch(e => console.error('📰 Initial news error:', e.message));

    cron.schedule(`*/${config.newsRefreshMinutes} * * * *`, async () => {
        try {
            const result = await newsService.fetchAndStoreNews();
            console.log('📰 Cron:', result.message);
        } catch (error) {
            console.error('📰 Cron error:', error.message);
        }
    });
};

// ==================== Startup ====================

const startServer = async () => {
    try {
        await connectDB();

        app.listen(config.port, () => {
            console.log('\n========================================');
            console.log('🚀 MetS Predictor API Server');
            console.log(`📍 http://localhost:${config.port}`);
            console.log(`🌍 ${config.nodeEnv}`);
            console.log(`🐍 Python: ${config.pythonServiceUrl}`);
            console.log('========================================\n');
        });

        startNewsCron();
    } catch (error) {
        console.error('❌ Failed to start server:', error.message);
        process.exit(1);
    }
};

// ==================== Graceful Shutdown ====================

process.on('SIGTERM', () => { console.log('\nShutting down...'); process.exit(0); });
process.on('SIGINT', () => { console.log('\nShutting down...'); process.exit(0); });
process.on('unhandledRejection', (err) => { console.error('❌ Unhandled Rejection:', err.message); });
process.on('uncaughtException', (err) => { console.error('❌ Uncaught Exception:', err.message); process.exit(1); });

startServer();

module.exports = app;
