# 🏗️ MetS Health Application - System Architecture

> **Complete Technical Architecture Documentation**
> Version 3.0 | February 2026

---

## 📋 Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Technology Stack](#3-technology-stack)
4. [Frontend Architecture (UI)](#4-frontend-architecture-ui)
5. [Backend Architecture (API)](#5-backend-architecture-api)
6. [Python ML Service](#6-python-ml-service)
7. [Database Design](#7-database-design)
8. [Data Flow & Formats](#8-data-flow--formats)
9. [API Endpoints Reference](#9-api-endpoints-reference)
10. [Core Algorithm - The Heart of the System](#10-core-algorithm---the-heart-of-the-system)
11. [Security Architecture](#11-security-architecture)
12. [UI/UX Design System](#12-uiux-design-system)
13. [Deployment Architecture](#13-deployment-architecture)

---

## 1. System Overview

The **MetS Health Application** is a full-stack web application for predicting and managing Metabolic Syndrome risk. It uses a **Bayesian Network** machine learning model for probabilistic predictions and provides personalized health recommendations with a modern, intuitive UI.

### Key Features
- 🔮 **ML-Powered Prediction** — Bayesian Network inference for MetS probability
- 📊 **Severity Assessment** — Clinical cMetS_S scoring formula
- 🥗 **Personalized Recommendations** — Diet, exercise, and yoga plans
- 📰 **Health News** — Real-time metabolic syndrome news from The Guardian
- 👤 **User Management** — Authentication, profiles, and assessment history
- 📄 **Report Generation** — Downloadable health plans in Markdown
- 🎨 **Modern UI** — Material-UI v7 with semantic color system
- ⚡ **Zustand State Management** — Lightweight, scalable global state

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER (Browser)                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │              React 19 + TypeScript + Material-UI v7                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │    │
│  │  │   Home   │ │  About   │ │Assessment│ │ Profile  │ │Resources │      │    │
│  │  │  (Hero)  │ │(Semantic)│ │ (Wizard) │ │(History) │ │  (News)  │      │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │    │
│  │                              ↓                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │           Zustand Stores (Global State Management)              │   │    │
│  │  │  ┌──────────────┐ ┌──────────────────┐ ┌─────────────┐        │   │    │
│  │  │  │  authStore   │ │ assessmentStore   │ │  newsStore  │        │   │    │
│  │  │  └──────────────┘ └──────────────────┘ └─────────────┘        │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  │                              ↓                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │    │
│  │  │      API Service Layer (Axios + Interceptors + JWT)             │   │    │
│  │  └─────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ HTTP/REST (JSON)
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER (Node.js)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     Express.js Server (Port 5000)                        │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │    │
│  │  │ Auth Routes │  │ MetS Routes │  │ News Routes │  │  Middleware │    │    │
│  │  │  /api/auth  │  │  /api/mets  │  │  /api/news  │  │  (JWT/CORS) │    │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘    │    │
│  │         │                │                │                             │    │
│  │  ┌──────▼────────────────▼────────────────▼──────┐                     │    │
│  │  │                 SERVICE LAYER                  │                     │    │
│  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ │                     │    │
│  │  │  │metsService │ │newsService │ │reportService│ │                     │    │
│  │  │  │recommendationsService       │              │ │                     │    │
│  │  │  └────────────┘ └────────────┘ └────────────┘ │                     │    │
│  │  └────────────────────────────────────────────────┘                     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
           │                    │                              │
           │                    │ HTTP (JSON)                  │
           ▼                    ▼                              ▼
┌──────────────────┐  ┌──────────────────┐           ┌──────────────────┐
│    MongoDB       │  │ Prediction Svc   │           │  Guardian API    │
│   (Port 27017)   │  │   (Port 5001)    │           │   (External)     │
│  ┌────────────┐  │  │  ┌────────────┐  │           │                  │
│  │   Users    │  │  │  │  Bayesian  │  │           │  Health News     │
│  │   News     │  │  │  │  Network   │  │           │  Articles        │
│  └────────────┘  │  │  │   Model    │  │           │                  │
└──────────────────┘  │  └────────────┘  │           └──────────────────┘
                      │  pgmpy + Flask   │
                      └──────────────────┘
```

---

## 3. Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 19.2.0 | UI Framework with React Compiler |
| TypeScript | 5.9.3 | Type Safety & Developer Experience |
| Vite | 7.2.4 | Build Tool & HMR Dev Server |
| Material-UI (MUI) | 7.3.7 | Component Library & Design System |
| @mui/icons-material | 7.3.7 | Icon Library (2000+ icons) |
| Zustand | 5.x | Lightweight Global State Management |
| Axios | 1.13.3 | HTTP Client with Interceptors |
| React Router | 7.13.0 | Client-side Routing & Navigation |
| @emotion/react | 11.14.0 | CSS-in-JS Styling |
| @emotion/styled | 11.14.1 | Styled Components |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Node.js | 18+ | Runtime Environment |
| Express.js | 4.x | Web Framework & REST API |
| MongoDB | 7.x | NoSQL Database |
| Mongoose | 8.x | ODM for MongoDB with Schemas |
| jsonwebtoken | 9.x | JWT Authentication |
| bcryptjs | 2.x | Password Hashing (12 rounds) |
| axios | 1.x | HTTP Client for Python Service |
| express-validator | 7.x | Input Validation |
| cors | 2.x | Cross-Origin Resource Sharing |
| dotenv | 16.x | Environment Configuration |
| node-cron | 3.x | Scheduled News Refresh |

### ML Service
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Runtime |
| Flask | 2.0+ | Lightweight API Framework |
| Flask-CORS | 3.0+ | CORS for API |
| pgmpy | 0.1.20+ | Bayesian Network Inference |
| NumPy | 1.21+ | Numerical Computing |

---

## 4. Frontend Architecture (UI)

### 4.1 Component Structure

```
client/src/
├── App.tsx                 # Root component — routing, theme, auth init
├── main.tsx                # Entry point with React 19 features
│
├── components/             # Reusable UI components
│   ├── index.ts            # Barrel exports
│   ├── Layout.tsx          # Main layout with red heart footer
│   ├── Navbar.tsx          # Navigation with red heart logo
│   └── ProtectedRoute.tsx  # Auth guard HOC
│
├── pages/                  # Page components
│   ├── Home.tsx            # Hero with red heart, green CTAs, news slideshow
│   ├── About.tsx           # Semantic colored condition cards
│   ├── Assessment.tsx      # Multi-step wizard (Zustand-driven)
│   ├── Profile.tsx         # User dashboard with history
│   ├── Resources.tsx       # News grid with colored sections
│   ├── Login.tsx           # Auth page (teal logo, green submit)
│   └── Signup.tsx          # Auth page (teal logo, green submit)
│
├── stores/                 # Zustand global state stores
│   ├── index.ts            # Barrel exports
│   ├── authStore.ts        # Auth state (user, token, login/logout)
│   ├── assessmentStore.ts  # Assessment flow (forms, results, recommendations)
│   └── newsStore.ts        # News state (slideshow + resources articles)
│
├── data/
│   ├── api.ts              # All API calls with Axios
│   └── types.ts            # TypeScript interfaces
│
├── theme/
│   └── theme.ts            # MUI theme config (teal primary)
│
└── styles/                 # CSS files for each component/page
    ├── global.css
    └── *.css               # Per-page and per-component styles
```

### 4.2 State Management (Zustand)

The application uses **Zustand** for all global state management — no React Context is used.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZUSTAND STORES                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  useAuthStore                                                            │
│  ─────────────                                                           │
│  State:   user, token, isAuthenticated, isLoading                        │
│  Actions: login(), logout(), updateUser(), initAuth()                    │
│  Used by: App, Navbar, ProtectedRoute, Login, Signup, Profile            │
│                                                                          │
│  useAssessmentStore                                                      │
│  ──────────────────                                                      │
│  State:   patientInfo, additionalInfo, results, recommendations,         │
│           activeStep, loading, error, tabValue, termsAccepted            │
│  Actions: setPatientInfo(), predict(), calculateSeverity(),              │
│           downloadReport(), startNewAssessment()                         │
│  Used by: Assessment                                                     │
│                                                                          │
│  useNewsStore                                                            │
│  ─────────────                                                           │
│  State:   slideshowArticles, resourceArticles, loading, error            │
│  Actions: fetchSlideshowNews(), fetchResourceNews(),                     │
│           getHighQualityImage()                                          │
│  Used by: Home (slideshow), Resources (grid)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **No providers needed** — Zustand stores are imported directly as hooks
- **`startNewAssessment()`** resets all form data back to true defaults (age: 30, BP: 120/80, etc.)
- **`initAuth()`** called once in `App.tsx` via `useEffect` on mount
- **News store is shared** — Home uses `slideshowArticles`, Resources uses `resourceArticles`

### 4.3 Page Flow

```
┌─────────┐    ┌─────────┐    ┌──────────────┐    ┌─────────────┐
│  Home   │───▶│  About  │───▶│  Assessment  │───▶│   Results   │
│ (Hero)  │    │(5 Cards)│    │  (3 Steps)   │    │(Prediction) │
└─────────┘    └─────────┘    └──────────────┘    └─────────────┘
                                     │                    │
                              Step 1: Basic Info    Download Report
                              Step 2: Additional    Save to Profile
                              Step 3: Results       View History
```

### 4.4 Responsive Design

```
Desktop (>900px):  Full navbar, side-by-side layouts, large cards
Tablet (600-900px): Responsive grid, stacked layouts, medium cards
Mobile (<600px):    Drawer menu, single column, compact cards
```

---

## 5. Backend Architecture (API)

### 5.1 Server Structure

```
server/
├── index.js                # Express app entry point
├── config.js               # Configuration & DB connection
├── middleware.js            # Auth, error handling, utilities
│
├── routes/
│   ├── auth.js             # Authentication endpoints
│   ├── mets.js             # MetS prediction endpoints
│   └── news.js             # News endpoints
│
├── services/
│   ├── metsService.js      # Bayesian Network proxy + severity calculations
│   ├── newsService.js      # Guardian API integration & caching
│   ├── recommendationsService.js  # Health recommendations from JSON
│   ├── reportService.js    # Markdown report generation
│   └── prediction/         # Python ML microservice
│       ├── prediction_service.py   # Flask API (Bayesian Network)
│       ├── bayesian_network_model.pkl  # Pre-trained model
│       └── requirements.txt  # Python dependencies
│
└── models/
    ├── User.js             # User schema (profile + assessment history)
    └── News.js             # News article schema
```

### 5.2 Middleware Pipeline

```
Request → CORS → JSON Parser → Auth (JWT) → Route Handler → Error Handler → Response
```

### 5.3 Service Layer Pattern

```javascript
// Routes handle HTTP concerns
router.post('/severity', asyncHandler(async (req, res) => {
    const result = metsService.calculateSeverity(...);
    res.json({ success: true, ...result });
}));

// Services handle business logic
const calculateSeverityScore = (gender, age, sbp, wc, fpg, tg, hdlC) => {
    const logTg = Math.log(tg);
    const coefficients = CMETS_COEFFICIENTS[gender][getAgeGroupKey(age)];
    return coefficients.intercept + /* formula */ ;
};
```

---

## 6. Python ML Service

### 6.1 Location

The Python prediction microservice lives inside the server services folder:

```
server/services/prediction/
├── prediction_service.py       # Flask API — Bayesian Network inference
├── bayesian_network_model.pkl  # Pre-trained GA-optimized model
└── requirements.txt            # Python dependencies (Flask, pgmpy, numpy)
```

### 6.2 Bayesian Network Model

The model uses **Genetic Algorithm optimization** to learn the optimal network structure from clinical data.

```python
# Model Training (Final_Code.ipynb)
class GeneticAlgorithmBayesianNetwork:
    def __init__(self, data, population_size=20, generations=50):
        self.data = data
        self.nodes = list(data.columns)

    def fitness(self, network):
        return BicScore(self.data).score(network)

    def run(self):
        # Evolve network structure using GA
        # Return best network based on BIC score
```

### 6.3 Inference Service

```python
# prediction_service.py
def predict_metabolic_syndrome(evidence):
    inference = VariableElimination(model)
    query_result = inference.query(
        variables=['Metabolic syndrome(0=no, 1=yes)'],
        evidence=evidence
    )
    return float(query_result.values[1])  # P(MetS = Yes)
```

### 6.4 Model Variables

| Variable | Type | Description |
|----------|------|-------------|
| Previous fatty liver | Binary (0/1) | History of fatty liver |
| Previous hypertension | Binary (0/1) | History of hypertension |
| Previous diabetes | Binary (0/1) | History of diabetes |
| Waist circumference | Continuous (cm) | Abdominal obesity measure |
| Systolic BP | Continuous (mmHg) | Blood pressure (systolic) |
| Diastolic BP | Continuous (mmHg) | Blood pressure (diastolic) |
| Metabolic syndrome | Binary (0/1) | Target variable |

---

## 7. Database Design

### 7.1 User Schema

```javascript
const userSchema = {
    // Identity
    firstName: String,
    lastName: String,
    email: { type: String, unique: true },
    password: { type: String, select: false },

    // Profile
    phone: String,
    dateOfBirth: Date,
    gender: ['Male', 'Female', 'Other'],
    address: String,
    profileImage: String,

    // System
    role: ['user', 'admin'],
    isActive: Boolean,
    lastLogin: Date,

    // Health Data
    assessmentHistory: [{
        date: Date,
        probability: Number,
        severity: Number,
        riskLevel: String,
        recommendations: {
            dietPlan: [String],
            avoidList: [String],
            exercisePlan: [String],
            yogaPoses: [String]
        }
    }],

    timestamps: true  // createdAt, updatedAt
};
```

### 7.2 News Schema

```javascript
const newsSchema = {
    title: String,
    description: String,
    content: String,
    url: { type: String, unique: true },
    image: String,
    source: String,
    author: String,
    publishedAt: Date,
    keywords: [String],
    createdAt: Date
};
```

---

## 8. Data Flow & Formats

### 8.1 Prediction Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PREDICTION FLOW                                     │
└──────────────────────────────────────────────────────────────────────────────┘

Step 1: User Input (React — useAssessmentStore)
───────────────────────────────────────────────
{
    age: 45, gender: "Men",
    fattyLiver: true, hypertension: false, diabetes: false,
    systolicBP: 130, diastolicBP: 85, waistCircumference: 95
}
            │
            ▼
Step 2: API Call to Node.js
───────────────────────────
POST /api/mets/predict
{ fattyLiver: 1, hypertension: 0, diabetes: 0,
  waistCircumference: 95, systolicBP: 130, diastolicBP: 85 }
            │
            ▼
Step 3: Forward to Python Prediction Service
─────────────────────────────────────────────
POST http://localhost:5001/predict  (Same payload)
            │
            ▼
Step 4: Bayesian Network Inference
──────────────────────────────────
Evidence → VariableElimination → P(MetS=1|Evidence)
            │
            ▼
Step 5: Response Back
─────────────────────
{ "probability": 0.72, "hasMetabolicSyndrome": true }
```

### 8.2 Severity Calculation Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SEVERITY FLOW                                       │
└──────────────────────────────────────────────────────────────────────────────┘

Input (Additional Info):
{ gender: "Men", age: 45, systolicBP: 130, waistCircumference: 95,
  fpg: 110, triglyceride: 200, hdlCholesterol: 40, probability: 0.72 }
            │
            ▼
cMetS_S Score Calculation (B):
B = intercept + (sbp_coef × SBP) + (wc_coef × WC) +
    (fpg_coef × FPG) + (logTg_coef × log(TG)) + (hdlC_coef × HDL-C)
            │
            ▼
Final Severity = min(0.99, max(0, probability + B))
            │
            ▼
Classification:
  0.00 - 0.30 → "Low Severity"
  0.31 - 0.60 → "Medium Severity"
  0.61 - 0.99 → "High Severity"

Response: { "severity": 0.87, "riskLevel": "High Severity" }
```

### 8.3 Request/Response Formats

#### Authentication

```typescript
// Signup Request
POST /api/auth/signup
{ "firstName": "John", "lastName": "Doe",
  "email": "john@example.com", "password": "secure123", "confirmPassword": "secure123" }

// Response
{ "success": true, "message": "Account created successfully",
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": { "id": "507f...", "firstName": "John", "lastName": "Doe",
            "email": "john@example.com", "fullName": "John Doe",
            "role": "user", "createdAt": "2026-01-31T10:00:00.000Z" } }
```

#### Recommendations

```typescript
// Request
POST /api/mets/recommendations
{ "gender": "Men", "riskLevel": "High Severity", "age": 45 }

// Response
{ "dietPlan": ["Increase fiber intake with whole grains", ...],
  "avoidList": ["Processed foods high in sodium", ...],
  "exercisePlan": ["Cardio: 30-45 minutes, 5 days/week", ...],
  "yogaPoses": ["Surya Namaskar (Sun Salutation)", ...] }
```

---

## 9. API Endpoints Reference

### Authentication (`/api/auth`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/signup` | ❌ | Create new account |
| POST | `/login` | ❌ | User login |
| GET | `/me` | ✅ | Get current user |
| PUT | `/profile` | ✅ | Update profile |
| PUT | `/password` | ✅ | Change password |
| POST | `/assessment` | ✅ | Save assessment |
| GET | `/assessments` | ✅ | Get assessment history |
| DELETE | `/account` | ✅ | Delete account |

### MetS Prediction (`/api/mets`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/predict` | ❌ | Predict MetS probability |
| POST | `/severity` | ❌ | Calculate severity score |
| POST | `/recommendations` | ❌ | Get health recommendations |
| POST | `/report` | ❌ | Generate health report |

### News (`/api/news`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/` | ❌ | Get latest news articles |
| GET | `/refresh` | ❌ | Manually refresh news |
| GET | `/status` | ❌ | Get news service status |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health status |

---

## 10. Core Algorithm - The Heart of the System

### 10.1 Two-Stage Prediction Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE HEART OF THE SYSTEM                                   │
│                                                                              │
│   Stage 1: Bayesian Network          Stage 2: cMetS_S Formula               │
│   ════════════════════════           ═══════════════════════                │
│                                                                              │
│   Input: 6 clinical variables        Input: 6 biomarkers + probability      │
│   Model: Genetic Algorithm BN        Model: Linear regression coefficients  │
│   Output: P(MetS)                    Output: Severity score                 │
│                                                                              │
│   ┌─────────────────────┐            ┌─────────────────────┐                │
│   │  Fatty Liver ──────┐│            │  SBP ─────────────┐ │                │
│   │  Hypertension ─────┼┼──▶ MetS    │  WC ──────────────┼─┼──▶ Severity   │
│   │  Diabetes ─────────┤│    Prob    │  FPG ─────────────┤ │    Score      │
│   │  Waist Circ ───────┤│            │  log(TG) ─────────┤ │                │
│   │  Systolic BP ──────┤│            │  HDL-C ───────────┤ │                │
│   │  Diastolic BP ─────┘│            │  + Probability ───┘ │                │
│   └─────────────────────┘            └─────────────────────┘                │
│                                                                              │
│   Threshold: > 0.65 = High Risk      Classification:                        │
│                                       0.00-0.30 = Low                       │
│                                       0.31-0.60 = Medium                    │
│                                       0.61-0.99 = High                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 cMetS_S Coefficients Table

| Gender | Age Group | Intercept | SBP | WC | FPG | log(TG) | HDL-C |
|--------|-----------|-----------|-----|-----|-----|---------|-------|
| Men | 20-39 | -1.79 | 0.0016 | 0.0045 | 0.0017 | 0.24 | -0.0042 |
| Men | 40-60 | -1.67 | 0.0007 | 0.0034 | 0.0014 | 0.25 | -0.0042 |
| Men | Other | -2.28 | 0.0019 | 0.0067 | 0.0027 | 0.28 | -0.0054 |
| Women | 20-39 | -2.43 | 0.0039 | 0.0066 | 0.0040 | 0.28 | -0.0052 |
| Women | 40-60 | -2.37 | 0.0010 | 0.0021 | 0.0015 | 0.41 | -0.0040 |
| Women | Other | -4.13 | 0.0065 | 0.0120 | 0.0070 | 0.39 | -0.0060 |

---

## 11. Security Architecture

### 11.1 Authentication Flow

```
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐
│  User   │─────▶│  Login  │─────▶│  Server │─────▶│ MongoDB │
└─────────┘      └─────────┘      └─────────┘      └─────────┘
                      │                │
                      │                │ Verify password (bcrypt)
                      │                │ Generate JWT
                      │                ▼
                      │         ┌─────────────┐
                      │◀────────│    JWT      │
                      │         │   Token     │
                      ▼         └─────────────┘
               ┌─────────────┐
               │ Zustand     │
               │ authStore   │  → also persists to localStorage
               └─────────────┘
                      │
                      │ All subsequent requests
                      ▼
               Authorization: Bearer <token>
```

### 11.2 Security Measures

| Layer | Measure | Implementation |
|-------|---------|----------------|
| Password | Hashing | bcrypt with salt rounds = 12 |
| Authentication | JWT | HS256, 7-day expiry |
| Authorization | Middleware | Token verification on protected routes |
| Password Field | Protection | `select: false` in Mongoose |
| Input Validation | express-validator | All auth routes |
| CORS | Whitelist | Only allowed origins |
| Error Handling | Sanitized | No stack traces in production |

---

## 12. UI/UX Design System

### 12.1 Color Palette

#### Primary Colors
| Color | Hex | Usage |
|-------|-----|-------|
| **Teal Light** | `#00b2a7` | Primary brand color, buttons, links |
| **Teal Dark** | `#009188` | Hover states, gradients |
| **Teal Deep** | `#00897b` | Active states, text |

#### Semantic Colors
| Color | Hex | Usage |
|-------|-----|-------|
| **Rose/Red** | `#e11d48` | Heart icons, critical warnings |
| **Green** | `#10b981` | CTA buttons, success states |
| **Amber** | `#f59e0b` | Warnings, medium risk |
| **Blue** | `#3b82f6` | Information, health insights |
| **Purple** | `#8b5cf6` | AI features, news, science |
| **Orange** | `#f59e0b` | Exercise, waist circumference |

### 12.2 Typography

| Element | Font Size | Weight | Usage |
|---------|-----------|--------|-------|
| Hero Title | 3.5rem | 800 | Landing page |
| Page Title | 2.5rem | 700 | Section titles |
| Card Title | 1.5rem | 600 | Card headings |
| Body Text | 1.05rem | 400 | Paragraph text |
| Caption | 0.8rem | 500 | Labels, metadata |

### 12.3 Animations

- **Heartbeat** — Red heart icon on hero and footer
- **Hover Lift** — Cards lift 8px with enhanced shadow
- **Pulse Circle** — Background pulsing on hero section
- **Smooth Transitions** — 0.3s ease on all interactive elements

### 12.4 Accessibility

- WCAG AA contrast ratios (4.5:1 minimum)
- Visible focus states with 2px outline
- ARIA labels on icon-only buttons
- Semantic heading hierarchy
- Full keyboard navigation support

---

## 13. Deployment Architecture

### 13.1 Development Setup

```
┌─────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT                              │
├─────────────────────────────────────────────────────────────┤
│  Terminal 1: Python Prediction Service                       │
│  $ cd server/services/prediction                             │
│  $ python prediction_service.py                              │
│  → http://localhost:5001                                     │
│                                                              │
│  Terminal 2: Node.js Server                                  │
│  $ cd server && npm start                                    │
│  → http://localhost:5000                                     │
│                                                              │
│  Terminal 3: React Client (Vite HMR)                         │
│  $ cd client && npm run dev                                  │
│  → http://localhost:5173                                     │
│                                                              │
│  Background: MongoDB                                         │
│  → mongodb://localhost:27017                                 │
└─────────────────────────────────────────────────────────────┘
```

### 13.2 Build Commands

```bash
# Frontend Production Build
cd client && npm run build     # Output: dist/ (static files)

# Backend (No build needed)
cd server && npm install --production

# Python Service
cd server/services/prediction && pip install -r requirements.txt
```

---

## 📊 Summary

The MetS Health Application is a **modern, scalable, microservices-based** health prediction system that combines:

1. **Machine Learning** — Bayesian Network for probabilistic inference
2. **Clinical Algorithms** — cMetS_S formula for severity assessment
3. **Personalized Health** — Recommendations based on gender, age, and risk
4. **Full-Stack Web** — React 19 + Node.js + MongoDB
5. **Zustand State Management** — Lightweight global stores (auth, assessment, news)
6. **Real-time News** — Guardian API integration
7. **Modern UI/UX** — Material-UI v7 with semantic color system

### Architecture Highlights
- **Zustand** replaces React Context for all global state (zero providers, zero boilerplate)
- **`startNewAssessment()`** guarantees a clean reset to true default values
- **Prediction service** lives inside `server/services/prediction/` for cohesive structure
- **Separation of concerns** — Routes → Services → Models
- **Type safety** — TypeScript throughout the frontend
- **Security** — JWT, bcrypt, input validation, CORS whitelist

---

*Document Version: 3.0*
*Last Updated: February 13, 2026*
