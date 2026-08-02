/**
 * News Routes
 * Handles metabolic syndrome related news
 */

const express = require('express');
const router = express.Router();
const { asyncHandler } = require('../middlewares/middleware');
const newsController = require('../controllers/newsController');

router.get('/', asyncHandler(newsController.getLatestNews));
router.get('/refresh', asyncHandler(newsController.refreshNews));
router.get('/status', asyncHandler(newsController.getNewsStatus));

module.exports = router;
