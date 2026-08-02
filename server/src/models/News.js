const mongoose = require('mongoose');

const newsSchema = new mongoose.Schema({
    title: {
        type: String,
        required: true
    },
    description: {
        type: String,
        default: ''
    },
    content: {
        type: String,
        default: ''
    },
    url: {
        type: String,
        required: true,
        unique: true
    },
    image: {
        type: String,
        default: null
    },
    source: {
        type: String,
        default: 'Unknown'
    },
    author: {
        type: String,
        default: null
    },
    publishedAt: {
        type: Date,
        required: true
    },
    keywords: [{
        type: String
    }],
    createdAt: {
        type: Date,
        default: Date.now
    }
});

// Index for faster queries
newsSchema.index({ publishedAt: -1 });

module.exports = mongoose.model('News', newsSchema);
