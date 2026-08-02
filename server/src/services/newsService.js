/**
 * News Service
 * Fetches metabolic syndrome related news from The Guardian API,
 * stores in MongoDB, and serves to the frontend.
 */

const axios = require('axios');
const https = require('https');
const News = require('../models/News');
const { config } = require('../config/config');

// Disable SSL verification for Guardian API (some environments have cert issues)
const httpsAgent = new https.Agent({ rejectUnauthorized: false });

// ==================== Constants ====================

const HEALTH_KEYWORDS = [
    'metabolic syndrome', 'diabetes', 'obesity', 'blood sugar', 'insulin resistance',
    'hypertension', 'high blood pressure', 'cholesterol', 'triglycerides', 'heart disease',
    'cardiovascular', 'weight loss', 'fatty liver', 'prediabetes', 'glucose', 'BMI', 'waist circumference',
];

// ==================== Helpers ====================

/**
 * Extract matching health keywords from article text.
 * Also doubles as a relevance check — if length > 0, article is relevant.
 */
const extractKeywords = (article) => {
    const text = `${article.webTitle || ''} ${article.fields?.body || ''}`.toLowerCase();
    return HEALTH_KEYWORDS.filter(kw => text.includes(kw.toLowerCase()));
};

const transformArticle = (article, keywords) => ({
    title: article.webTitle,
    description: article.fields?.trailText
        || article.fields?.body?.substring(0, 200)?.replace(/<[^>]*>/g, '')
        || article.webTitle,
    content: article.fields?.body?.replace(/<[^>]*>/g, '') || '',
    url: article.webUrl,
    image: article.fields?.thumbnail || null,
    source: 'The Guardian',
    author: article.fields?.byline || null,
    publishedAt: new Date(article.webPublicationDate),
    keywords,
});

// ==================== Core Functions ====================

const fetchAndStoreNews = async () => {
    if (!config.guardianApiKey) {
        return { success: false, message: 'GUARDIAN_API_KEY not configured' };
    }

    try {
        const response = await axios.get('https://content.guardianapis.com/search', {
            params: {
                q: 'diabetes OR "metabolic syndrome" OR obesity OR "heart disease" OR hypertension',
                section: 'science|lifeandstyle|society',
                'api-key': config.guardianApiKey,
                'show-fields': 'body,thumbnail,byline,trailText',
                'page-size': 50,
                'order-by': 'newest',
            },
            timeout: 10000,
            httpsAgent,
        });

        const articles = response.data?.response?.results;
        if (!articles?.length) {
            return { success: false, message: 'No results from Guardian API' };
        }

        // Filter to relevant articles and cap at max
        const relevant = [];
        for (const article of articles) {
            if (relevant.length >= config.maxNewsArticles) break;
            const keywords = extractKeywords(article);
            if (keywords.length > 0) {
                relevant.push({ article, keywords });
            }
        }

        let savedCount = 0;
        let updatedCount = 0;

        for (const { article, keywords } of relevant) {
            try {
                const data = transformArticle(article, keywords);
                const existing = await News.findOne({ url: data.url });
                if (existing) {
                    await News.updateOne({ url: data.url }, data);
                    updatedCount++;
                } else {
                    await News.create(data);
                    savedCount++;
                }
            } catch (err) {
                if (err.code !== 11000) console.error('Error saving article:', err.message);
            }
        }

        await cleanupOldNews();

        const message = `Synced ${savedCount} new, ${updatedCount} updated`;
        console.log(`✅ ${message}`);
        return { success: true, message, savedCount, updatedCount };
    } catch (error) {
        console.error('❌ Guardian API error:', error.response?.data || error.message);
        return { success: false, message: error.message };
    }
};

const cleanupOldNews = async () => {
    try {
        const total = await News.countDocuments();
        if (total <= config.maxNewsArticles) return;

        const stale = await News.find()
            .sort({ publishedAt: -1 })
            .skip(config.maxNewsArticles)
            .select('_id');

        if (stale.length > 0) {
            await News.deleteMany({ _id: { $in: stale.map(a => a._id) } });
            console.log(`🗑️ Cleaned up ${stale.length} old articles`);
        }
    } catch (error) {
        console.error('Error cleaning up old news:', error.message);
    }
};

const getLatestNews = async (limit = 15, skip = 0) => {
    try {
        const articles = await News.find()
            .sort({ publishedAt: -1 })
            .skip(skip)
            .limit(limit)
            .select('title description content url image source author publishedAt keywords')
            .lean();

        return { success: true, count: articles.length, articles };
    } catch (error) {
        console.error('Error fetching news from DB:', error.message);
        return { success: false, message: error.message, articles: [] };
    }
};

const getNewsCount = () => News.countDocuments();

module.exports = {
    fetchAndStoreNews,
    getLatestNews,
    getNewsCount,
};
