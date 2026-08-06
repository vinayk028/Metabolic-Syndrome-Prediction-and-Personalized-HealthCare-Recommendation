# Metabolic-Syndrome-Prediction-Personalized-HealthCare-Recommendation

## Design (Architecture)

<img width="768" height="846" alt="Screenshot 2025-07-03 215445" src="https://github.com/user-attachments/assets/65471888-1dd8-468e-85c4-a82319055a02" />

## Features
 1. Accurate MetS prediction using evolutionary computing
 2. Early detection and personalized intervention
 3. Personalized healthcare and lifestyle recommendations
 4. User-friendly web application for patients and doctors



<h1 align="center">🏥 MetS Health - Metabolic Syndrome Predictor</h1>

<p align="center">
  <strong>AI-Powered Health Risk Assessment & Personalized Recommendations</strong>
</p>

<p align="center">
  <a href="#-about">About</a> •
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-services--ports">Services</a> •
  <a href="#-api-endpoints">API</a> •
  <a href="#-troubleshooting">Troubleshooting</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React-19.x-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/TypeScript-5.9-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=node.js&logoColor=white" alt="Node.js"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Spring%20Boot-3.2-6DB33F?style=for-the-badge&logo=spring-boot&logoColor=white" alt="Spring Boot"/>
  <img src="https://img.shields.io/badge/MongoDB-7.x-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/>
  <img src="https://img.shields.io/badge/Zustand-5.x-443E38?style=for-the-badge" alt="Zustand"/>
  <img src="https://img.shields.io/badge/MUI-Material%20UI-007FFF?style=for-the-badge&logo=mui&logoColor=white" alt="MUI"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ML-Bayesian%20Network-FF6F00?style=for-the-badge" alt="ML"/>
  <img src="https://img.shields.io/badge/PDF-iText7-DD0000?style=for-the-badge" alt="PDF"/>
  <img src="https://img.shields.io/badge/AI-Gemini%2FClaude-4285F4?style=for-the-badge" alt="AI"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

## 📖 About

**MetS Health** is a full-stack healthcare web application engineered to **predict Metabolic Syndrome risk** using advanced **Bayesian Network machine learning** and provide **personalized health recommendations** with **professional PDF report generation**.

### 🎓 What is Metabolic Syndrome?

Metabolic Syndrome is a cluster of conditions that dramatically increase the risk of serious health issues:
- 🩹 High blood pressure (hypertension)
- 📈 High blood sugar (impaired glucose tolerance)
- ⚖️ Excess body fat (abdominal obesity)  
- ❌ Abnormal cholesterol/triglyceride levels

Together, these conditions increase risk of **heart disease**, **stroke**, and **type 2 diabetes** by up to **5x**.

### 🎯 Our Solution

**MetS Health** enables:
- ✅ **Accurate Risk Prediction** — Bayesian Network + Genetic Algorithm optimization
- ✅ **Clinical Severity Scoring** — Evidence-based cMetS_S formula  
- ✅ **Personalized Health Plans** — Diet, exercise, yoga recommendations
- ✅ **Professional PDF Reports** — Downloadable health assessments with one click
- ✅ **Assessment Tracking** — Monitor progress over time with trend analysis
- ✅ **Expert AI Chat** — Real-time health guidance powered by Gemini/Claude
- ✅ **Curated Health News** — Latest metabolic syndrome news from trusted sources

---

## ✨ Features

### 🔮 Dual-Stage Prediction Engine
- **Stage 1:** Bayesian Network inference → Probability estimation
- **Stage 2:** Clinical cMetS_S scoring → Severity classification (Low/Medium/High)
- **Optimization:** Genetic Algorithm for optimal network structure learning
- **Input Variables:** Age, gender, BP, waist, BMI, cholesterol levels, glucose

### 📋 Assessment Management
- ✅ Multi-step guided assessment form (2-3 steps based on initial results)
- ✅ Real-time form validation & error handling
- ✅ Assessment history with complete tracking
- ✅ Trend analysis comparing previous assessments
- ✅ Risk progression visualization over time

### 📄 PDF Report Generation ⭐ NEW
- **Spring Boot Microservice** — Dedicated report generation service (Port 8081)
- **Professional Format** — Patient info, assessment results, personalized recommendations
- **One-Click Download** — Instant PDF generation and download  
- **Metadata** — Generated date, risk level, severity score, health parameters
- **Secure Transmission** — Base64 encoded for safe transmission

### 🥗 Personalized Recommendations Engine
- **Gender & Age-Specific** diet plans with meal guidelines
- **Foods to Avoid** list with health reasoning
- **Exercise Routines** tailored to severity level
- **Yoga Poses** for stress reduction & cardiovascular health
- **Dynamic Recommendations** based on risk classification

### 👤 User Management & Authentication
- **JWT Authentication** — Secure token-based login/signup
- **Profile Management** — Update personal & medical information
- **Password Security** — Bcrypt hashing with salt rounds
- **Account Controls** — Delete account & associated data
- **Session Management** — Persistent authentication

### 📊 Analytics Dashboard
- **Risk Trend Charts** — Area chart showing probability & severity over time
- **Risk Distribution Analysis** — Bar chart of Low/Medium/High breakdowns
- **Vital Metrics Radar** — Normalized health parameters visualization
- **Timeline View** — Chronological assessment history with status chips
- **KPI Cards** — Current risk level, probability, severity, assessment count

### 💬 AI Chat Assistant
- **Real-time Chat** — Floating widget available on every page
- **AI-Powered Responses** — Google Gemini / Anthropic Claude backend
- **Context Aware** — Health-related queries with domain knowledge
- **Multi-turn Conversations** — Full chat history per session
- **Responsive UI** — Mobile-friendly chat interface

### 📰 Health News Feed
- **Auto-updated** — Automatically syncs latest metabolic syndrome news
- **Curated Sources** — From trusted healthcare outlets & The Guardian
- **Categorized** — Diet, exercise, research, prevention news
- **Background Sync** — Updates every 30 minutes via cron job

---

## 💻 Tech Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 19.x | UI framework with hooks |
| **TypeScript** | 5.9 | Type-safe JavaScript |
| **Vite** | 7.x | Ultra-fast build tool & dev server |
| **Material-UI (MUI)** | 7.x | Professional component library |
| **Recharts** | 3.7 | Data visualization & charts |
| **Axios** | 1.13 | HTTP client with interceptors |
| **Zustand** | 5.x | Lightweight state management |
| **React Router** | 7.x | Client-side routing |

### Backend Services
| Technology | Version | Purpose |
|------------|---------|---------|
| **Node.js** | 18+ | JavaScript runtime |
| **Express.js** | 4.x | Web framework |
| **MongoDB** | 7.x | NoSQL database |
| **Mongoose** | 8.x | ODM (Object Data Modeling) |
| **JWT** | 9.x | Authentication tokens |
| **Bcrypt.js** | 2.4 | Password hashing |
| **CORS** | 2.8 | Cross-origin resource sharing |
| **node-cron** | 3.0 | Task scheduling |

### ML & Report Generation
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | ML environment |
| **pgmpy** | Latest | Probabilistic Graphical Models |
| **NumPy** | 1.48+ | Numerical computing |
| **Spring Boot** | 3.2 | Report microservice |
| **iText7** | 7.2 | PDF generation library |
| **Maven** | 3.8+ | Build tool |
| **Lombok** | Latest | Java boilerplate reduction |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       FRONTEND LAYER                              │
│              React 19 + TypeScript + Vite + MUI                   │
│                    http://localhost:5173                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Pages: Login, Assessment, Dashboard, Profile, Resources    │   │
│  │ State: Zustand (auth, assessment, chat, news, admin)       │   │
│  │ Components: Charts, Forms, Chat Widget, News Feed          │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                         ↓ REST API ↓
┌──────────────────────────────────────────────────────────────────┐
│                    BACKEND API LAYER                              │
│                  Node.js + Express + JWT                          │
│                    http://localhost:5000                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ Routes: /auth, /mets, /chat, /news, /health               │   │
│  │ Middleware: CORS, auth, error handling, validation        │   │
│  │ Database: MongoDB connection with Mongoose ODM            │   │
│  └────────────────────────────────────────────────────────────┘   │
│         ↓                              ↓                          │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Python Service (5001)    │  │ Spring Boot (8081)          │   │
│  │ Bayesian Network ML      │  │ PDF Report Generator        │   │
│  │ ├─ predict()             │  │ ├─ /api/reports/health      │   │
│  │ ├─ severity()            │  │ ├─ generates PDF            │   │
│  │ └─ recommendations()     │  │ ├─ iText7 formatting        │   │
│  │                          │  │ └─ base64 encoding          │   │
│  └──────────────────────────┘  └─────────────────────────────┘   │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ MongoDB Database (27017)                                    │   │
│  │ Collections:                                                │   │
│  │ ├─ users (auth profiles)                                   │   │
│  │ ├─ assessments (health records)                            │   │
│  │ ├─ news (cached articles)                                  │   │
│  │ └─ chat_history (conversations)                            │   │
│  └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Assessment Flow

```
┌─────────────────────────┐
│   START ASSESSMENT      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   STEP 1: BASIC INFORMATION             │
│   ┌─────────────────────────────────┐   │
│   │ • Age (20-80 years)             │   │
│   │ • Gender (Male / Female)        │   │
│   │ • Systolic BP (mmHg)            │   │
│   │ • Diastolic BP (mmHg)           │   │
│   │ • Waist Circumference (cm)      │   │
│   │ • Fatty Liver (Yes/No)          │   │
│   │ • Hypertension (Yes/No)         │   │
│   │ • Diabetes (Yes/No)             │   │
│   └─────────────────────────────────┘   │
│   → Bayesian Network Prediction        │
│   ← Probability Score                   │
└────────────┬────────────────────────────┘
             │
             ▼
      ┌──────────────────┐
      │ Has MetS Risk?   │
      └────┬───────┬─────┘
         YES      NO
         │        │
         ▼        ▼
    ┌────────┐  ┌──────────────────────┐
    │STEP 2  │  │ STEP 2 (Alt)         │
    │ADD'L   │  │ RESULTS              │
    │INFO    │  │ • Low Risk           │
    │        │  │ • Recommendations    │
    │        │  │ • Download Report    │
    │        │  │ • Save Assessment    │
    └────┬───┘  └──────────────────────┘
         │
         ▼
    ┌───────────────┐
    │ STEP 2 CONT'D │
    │ • HDL(mg/dL)  │
    │ • TGL(mg/dL)  │
    │ • FPG(mg/dL)  │
    └────┬──────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │ Clinical Severity Calculation        │
    │ (cMetS_S Formula)                    │
    │ ← Severity Score (0-1)              │
    │ ← Risk Level (Low/Med/High)         │
    └────┬────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────┐
    │ STEP 3: RESULTS            │
    │ • Probability: X.X%        │
    │ • Severity: X.X%           │
    │ • Risk Level: HIGH         │
    │ • Personalized Recs        │
    │ • Download PDF Report ✓    │
    │ • Save to Dashboard        │
    └────────────────────────────┘
```

### Risk Classification

| Severity Level | Score Range | Profile | Color |
|---|---|---|---|
| 🟢 **Low** | 0.00 - 0.30 | Minimal MetS risk, maintain healthy lifestyle | Green |
| 🟡 **Medium** | 0.31 - 0.60 | Moderate risk, lifestyle changes recommended | Yellow |
| 🔴 **High** | 0.61 - 1.00 | High risk, medical consultation advised | Red |

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **Python** 3.8+ ([Download](https://www.python.org/))
- **Java 17+** ([Download](https://www.oracle.com/java/technologies/downloads/))
- **MongoDB** 7.x ([Local](https://www.mongodb.com/try/download/community) or [Atlas Cloud](https://www.mongodb.com/cloud/atlas))
- **Maven** 3.8+ ([Download](https://maven.apache.org/download.cgi))
- **Git** ([Download](https://git-scm.com/))

### 📦 Installation & Setup

#### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/metabolic-syndrome-prediction.git
cd Metabolic-Syndrome-Prediction-and-Personalized-HealthCare-Recommendation
```

#### Step 2: Python Prediction Service

```bash
cd MetS-App/server/services/prediction

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Or (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start service
python prediction_service.py
# ✅ Runs on http://localhost:5001
```

#### Step 3: Spring Boot Report Service

```bash
cd MetS-App/server/services/report-service

# Build project
mvn clean install

# Run service
mvn spring-boot:run
# ✅ Runs on http://localhost:8081
```

#### Step 4: Node.js Backend

```bash
cd MetS-App/server

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/mets-app
JWT_SECRET=your_jwt_secret_key_here_min_32_chars
REPORT_SERVICE_URL=http://localhost:8081
PYTHON_SERVICE_URL=http://localhost:5001
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
EOF

# Start backend
npm start
# ✅ Runs on http://localhost:5000
```

#### Step 5: React Frontend

```bash
cd MetS-App/client

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_API_URL=http://localhost:5000/api
EOF

# Start dev server
npm run dev
# ✅ Runs on http://localhost:5173
```

---

## 🎯 Services & Ports

| Service | Port | Command | Health Check |
|---------|------|---------|--------------|
| **Frontend** | 5173 | `npm run dev` | http://localhost:5173 |
| **Backend API** | 5000 | `npm start` | http://localhost:5000/api/health |
| **Python ML** | 5001 | `python prediction_service.py` | http://localhost:5001/health |
| **Report Service** | 8081 | `mvn spring-boot:run` | http://localhost:8081/api/reports/health |
| **MongoDB** | 27017 | `mongod` | Local database |

### ✅ Startup Order (Recommended)

Open **5 Terminal Windows**:

```bash
# Terminal 1: Python ML Service
cd MetS-App/server/services/prediction
.venv\Scripts\activate
python prediction_service.py
# Wait for: "Serving on port 5001"

# Terminal 2: Spring Boot Report Service
cd MetS-App/server/services/report-service
mvn spring-boot:run
# Wait for: "Tomcat started on port 8081"

# Terminal 3: Node.js Backend
cd MetS-App/server
npm start
# Wait for: "🚀 MetS Predictor API Server"

# Terminal 4: React Frontend
cd MetS-App/client
npm run dev
# Wait for: "VITE v7.2.4 ready in"

# Terminal 5: Verify MongoDB
# Ensure MongoDB is running locally or Atlas connection is active
mongosh  # or connect to Atlas
```

### 🌐 Access Application

Open browser: **http://localhost:5173**

---

## 📡 API Endpoints

### Authentication (`/api/auth`)
```
POST   /signup              Create new account
POST   /login               User login
GET    /me                  Get current user
PUT    /profile             Update profile
PUT    /password            Change password
DELETE /account             Delete account
```

### Metabolic Syndrome (`/api/mets`)
```
POST   /predict             Get MetS probability (needs Python service)
POST   /severity            Calculate severity score
POST   /recommendations     Get personalized recommendations
POST   /report              Generate & download PDF report
```

### Chat (`/api/chat`)
```
POST   /                    Send chat message (AI-powered)
```

### News (`/api/news`)
```
GET    /                    Get metabolic syndrome news
```

### Assessment (`/api/auth/assessments`)
```
POST   /assessments         Save assessment
GET    /assessments         Retrieve assessment history
```

### Health (`/api/health`)
```
GET    /                    Server health status
```

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Python service"
```bash
# Check if Python service is running
curl http://localhost:5001/health

# If not, ensure:
1. Virtual environment is activated
2. Dependencies are installed (pip install -r requirements.txt)
3. No firewall blocking port 5001
```

### Issue: "Cannot connect to Spring Boot"
```bash
# Check if Spring Boot is running
curl http://localhost:8081/api/reports/health

# If not, ensure:
1. Java 17+ is installed (java -version)
2. Maven is installed (mvn -version)
3. No firewall blocking port 8081
4. Build completed: mvn clean install
```

### Issue: "Report download not working"
```bash
# Verify all 3 services are running:
curl http://localhost:5000/api/health
curl http://localhost:5001/health
curl http://localhost:8081/api/reports/health

# Check Node.js server logs for errors like:
# "Report service error: 500"
# "Failed to generate PDF report"

# Solution: Ensure Spring Boot service is running on 8081
```

### Issue: "MongoDB connection failed"
```bash
# Check MongoDB status
# For local MongoDB:
mongosh  # Should connect successfully

# For MongoDB Atlas:
# Verify connection string in .env
# MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/mets-app

# Whitelist your IP in MongoDB Atlas Network Access
```

### Issue: "CORS errors in browser console"
```bash
# Check .env CORS origins
CORS_ORIGINS=http://localhost:5173,http://localhost:3000

# Restart Node.js backend after changing .env
```

### Issue: "AI Chat not responding"
```bash
# Check API keys in .env
echo $GEMINI_API_KEY
echo $ANTHROPIC_API_KEY

# Get free API keys:
# Gemini: https://makersuite.google.com/app/apikey
# Claude: https://console.anthropic.com/
```
---

## 🔧 Environment Variables

### Frontend (`.env` in `MetS-App/client/`)
```env
VITE_API_URL=http://localhost:5000/api
```

### Backend (`.env` in `MetS-App/server/`)
```env
NODE_ENV=development
MONGODB_URI=mongodb://localhost:27017/mets-app
JWT_SECRET=your_jwt_secret_key_here_min_32_characters
REPORT_SERVICE_URL=http://localhost:8081
PYTHON_SERVICE_URL=http://localhost:5001
GEMINI_API_KEY=your_gemini_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
PORT=5000
```

---

## 🚀 Deployment

### Frontend (Vercel / Netlify)
```bash
cd MetS-App/client
npm run build
# Deploy 'dist' folder
```

### Backend (Railway / Render / Heroku)
```bash
cd MetS-App/server
# Set environment variables on platform
npm install
npm start
```

### Python Service (Railway / Render)
```bash
cd MetS-App/server/services/prediction
pip install -r requirements.txt
python prediction_service.py
```

### Spring Boot (Railway / AWS / Azure)
```bash
cd MetS-App/server/services/report-service
mvn clean install
java -jar target/report-service-1.0.0.jar
```

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
---

<p align="center">
  Made with ❤️ for better health outcomes
</p>

<p align="center">
  <a href="#metabolic-syndrome-prediction-personalized-healthcare-recommendation">Back to Top ⬆️</a>
</p>
