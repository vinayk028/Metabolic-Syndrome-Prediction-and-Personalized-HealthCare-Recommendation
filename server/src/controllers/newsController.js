const newsService = require('../services/newsService');

const getLatestNews = async (req, res) => {
    const limit = Math.min(parseInt(req.query.limit, 10) || 15, 25);
    const skip = Math.max(parseInt(req.query.skip, 10) || 0, 0);
    const result = await newsService.getLatestNews(limit, skip);

    res.json({
        success: true,
        count: result.count || 0,
        articles: result.articles || [],
        ...(result.message && { message: result.message }),
    });
};

const refreshNews = async (req, res) => {
    const result = await newsService.fetchAndStoreNews();

    res.json({
        success: result.success,
        message: result.message,
        savedCount: result.savedCount || 0,
        updatedCount: result.updatedCount || 0,
    });
};

const getNewsStatus = async (req, res) => {
    const count = await newsService.getNewsCount();

    res.json({
        success: true,
        totalArticles: count,
        refreshInterval: '5 minutes',
    });
};

module.exports = { getLatestNews, refreshNews, getNewsStatus };