/**
 * Chat Service
 * LLM-powered chat assistant with context injection from existing services.
 * Supports Gemini and Claude via a provider abstraction.
 * Uses axios for API calls (same SSL workaround as newsService).
 */

const https = require('https');
const axios = require('axios');
const { config } = require('../config/config');
const newsService = require('./newsService');
const recommendationsService = require('./recommendationsService');

// SSL workaround — same as newsService (fixes "fetch failed" on some networks)
const httpsAgent = new https.Agent({ rejectUnauthorized: false });

// ==================== LLM Providers ====================

/**
 * Call Google Gemini API via REST (avoids SDK fetch issues with SSL)
 * Docs: https://ai.google.dev/gemini-api/docs/text-generation#generate-text-from-text
 */
const callGemini = async (systemPrompt, messages) => {
    const model = config.llmModel || 'gemini-2.0-flash';
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${config.llmApiKey}`;

    // Build Gemini request body
    const contents = messages.map((msg) => ({
        role: msg.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: msg.content }],
    }));

    const body = {
        system_instruction: {
            parts: [{ text: systemPrompt }],
        },
        contents,
        generationConfig: {
            maxOutputTokens: 1024,
            temperature: 0.7,
        },
    };

    const response = await axios.post(url, body, {
        httpsAgent,
        timeout: 30000,
        headers: { 'Content-Type': 'application/json' },
    });

    // Extract text from Gemini response
    const candidate = response.data?.candidates?.[0];
    if (!candidate?.content?.parts?.[0]?.text) {
        throw new Error('Empty response from Gemini API');
    }

    return candidate.content.parts[0].text;
};

/**
 * Call Anthropic Claude API
 */
const callClaude = async (systemPrompt, messages) => {
    const model = config.llmModel || 'claude-3-haiku-20240307';
    const url = 'https://api.anthropic.com/v1/messages';

    const body = {
        model,
        max_tokens: 1024,
        system: systemPrompt,
        messages: messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
        })),
    };

    const response = await axios.post(url, body, {
        httpsAgent,
        timeout: 30000,
        headers: {
            'Content-Type': 'application/json',
            'x-api-key': config.llmApiKey,
            'anthropic-version': '2023-06-01',
        },
    });

    return response.data?.content?.[0]?.text;
};

/** Provider registry — add new providers here */
const LLM_PROVIDERS = {
    gemini: callGemini,
    claude: callClaude,
};

// ==================== Context Builders ====================

/**
 * Build user context string from user document
 */
const buildUserContext = (user) => {
    if (!user) return 'User is not logged in. They can only ask general questions.';

    const lines = [
        `Name: ${user.fullName}`,
        user.gender ? `Gender: ${user.gender}` : null,
        user.dateOfBirth ? `Date of Birth: ${new Date(user.dateOfBirth).toLocaleDateString()}` : null,
    ].filter(Boolean);

    // Latest assessment
    const history = user.assessmentHistory || [];
    if (history.length > 0) {
        const sorted = [...history].sort((a, b) => new Date(b.date) - new Date(a.date));
        const latest = sorted[0];

        lines.push('');
        lines.push('LATEST ASSESSMENT:');
        lines.push(`  Date: ${new Date(latest.date).toLocaleDateString()}`);
        lines.push(`  Probability: ${(latest.probability * 100).toFixed(1)}%`);
        lines.push(`  Severity: ${latest.severity?.toFixed(2) || 'N/A'}`);
        lines.push(`  Risk Level: ${latest.riskLevel || 'N/A'}`);

        if (latest.inputParameters) {
            const p = latest.inputParameters;
            lines.push(`  Input: Age=${p.age}, Gender=${p.gender}, SystolicBP=${p.systolicBP}, DiastolicBP=${p.diastolicBP}, Waist=${p.waistCircumference}cm`);
            if (p.hdlCholesterol) {
                lines.push(`  Labs: HDL=${p.hdlCholesterol}, Triglyceride=${p.triglyceride}, FPG=${p.fpg}`);
            }
        }

        // Trend (if multiple assessments)
        if (sorted.length > 1) {
            const trend = sorted
                .slice(0, 5)
                .reverse()
                .map((a) => `${(a.probability * 100).toFixed(0)}%`)
                .join(' → ');
            lines.push(`  Trend (oldest→newest): ${trend}`);
        }

        // Latest recommendations
        if (latest.recommendations) {
            const r = latest.recommendations;
            if (r.dietPlan?.length) lines.push(`  Diet Plan: ${r.dietPlan.slice(0, 5).join(', ')}`);
            if (r.avoidList?.length) lines.push(`  Avoid: ${r.avoidList.slice(0, 5).join(', ')}`);
            if (r.exercisePlan?.length) lines.push(`  Exercise: ${r.exercisePlan.slice(0, 5).join(', ')}`);
            if (r.yogaPoses?.length) lines.push(`  Yoga: ${r.yogaPoses.slice(0, 5).join(', ')}`);
        }
    } else {
        lines.push('No assessments taken yet.');
    }

    return lines.join('\n');
};

/**
 * Build news context string from latest articles
 */
const buildNewsContext = async () => {
    try {
        const { articles } = await newsService.getLatestNews(5, 0);
        if (!articles.length) return 'No recent news available.';

        return articles
            .map((a, i) => `${i + 1}. "${a.title}" (${new Date(a.publishedAt).toLocaleDateString()}) — ${a.source}`)
            .join('\n');
    } catch {
        return 'News unavailable.';
    }
};

/**
 * Build the full system prompt with all context
 */
const buildSystemPrompt = async (user) => {
    const userContext = buildUserContext(user);
    const newsContext = await buildNewsContext();

    return `You are MetS Health Assistant — a friendly, knowledgeable assistant for the MetS Health web application that predicts Metabolic Syndrome risk and provides personalized health recommendations.

ABOUT THE APP:
- Predicts Metabolic Syndrome (MetS) risk using a Bayesian Network ML model optimized with Genetic Algorithms
- MetS is a cluster of conditions: high blood pressure, high blood sugar, excess body fat, abnormal cholesterol
- Assessment flow: Basic Info (age, gender, BP, waist, medical history) → Additional Labs (HDL, triglyceride, FPG) if at risk → Results & Recommendations
- Severity scoring uses clinical cMetS_S formula with gender & age-specific coefficients
- Risk levels: Low (0-0.30), Medium (0.31-0.60), High (0.61-0.99)
- App pages: Home, About, Assessment (/assessment), Dashboard (/dashboard - assessment history), Resources (/resources - health news), Profile (/profile)

CURRENT USER:
${userContext}

LATEST HEALTH NEWS:
${newsContext}

YOUR BEHAVIOR:
- Be warm, empathetic, and encouraging — many users may be anxious about their health
- Give clear, concise answers (2-3 short paragraphs max unless user asks for detail)
- Use simple language, explain medical terms when you use them
- When explaining results, relate numbers to what they mean practically
- If user hasn't done an assessment, gently guide them to the Assessment page
- For navigation questions, mention the exact page name and path
- You can reference the user's actual assessment data to give personalized answers
- Format responses with markdown for readability (bold, bullet points, headers)
- If asked about news, summarize from the headlines provided

IMPORTANT RULES:
- ALWAYS include a brief disclaimer when giving health-related advice: "This is informational only — please consult your healthcare provider for medical advice."
- NEVER diagnose conditions or prescribe medications
- NEVER make up assessment data — only reference what's in the user context above
- If you don't know something, say so honestly
- Stay focused on metabolic syndrome, health, and the app's features`;
};

// ==================== Main Chat Function ====================

/**
 * Process a chat message and return the LLM response.
 *
 * @param {string} message - The user's message
 * @param {Array} history - Conversation history [{role, content}, ...]
 * @param {Object|null} user - Mongoose user document (null if not logged in)
 * @returns {Promise<string>} - The assistant's response
 */
const chat = async (message, history = [], user = null) => {
    const provider = config.llmProvider;
    const callLLM = LLM_PROVIDERS[provider];

    if (!callLLM) {
        throw new Error(`Unknown LLM provider: "${provider}". Supported: ${Object.keys(LLM_PROVIDERS).join(', ')}`);
    }

    if (!config.llmApiKey) {
        throw new Error('LLM API key not configured. Set LLM_API_KEY in your .env file.');
    }

    // Build system prompt with real-time context
    const systemPrompt = await buildSystemPrompt(user);

    // Prepare messages: keep last 10 exchanges to stay within token limits
    const recentHistory = history.slice(-20);
    const messages = [
        ...recentHistory,
        { role: 'user', content: message },
    ];

    const response = await callLLM(systemPrompt, messages);
    return response;
};

module.exports = { chat };
