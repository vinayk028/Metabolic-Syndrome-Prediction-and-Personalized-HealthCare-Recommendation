# MetS Health Application - Complete Setup Guide

This guide will help you run the MetS (Metabolic Syndrome) Health application after extracting the zip file on your laptop.

---

## 📋 Prerequisites

Install these **before** starting:

### 1. Node.js (v18 or higher)
- Download: https://nodejs.org/
- Choose **LTS version** (recommended)
- ✅ Run installer with default options
- Verify installation:
  ```cmd
  node --version
  npm --version
  ```

### 2. Python 3.8+ (Required for ML Prediction)
- Download: https://www.python.org/downloads/
- ⚠️ **IMPORTANT:** Check ✅ "Add Python to PATH" during installation
- Verify installation:
  ```cmd
  python --version
  pip --version
  ```

### 3. MongoDB (Database)
- Download: https://www.mongodb.com/try/download/community
- Choose **MongoDB Community Server**
- During installation:
  - Select "Complete" installation
  - ✅ Check "Install MongoDB as a Service" (auto-starts on boot)
- Verify it's running:
  ```cmd
  mongosh --version
  ```

---

## 🚀 Quick Start (5 Steps)

### Step 1: Extract the Zip File
Extract to a simple path like:
```
C:\MetS_App\Mets_Final_Code
```

### Step 2: Install Python Dependencies
Open **Command Prompt** and run:
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\server\services\prediction
pip install -r requirements.txt
```
⏱️ Wait 1-2 minutes for installation.

**Python Service Dependencies:**
- Flask (Web framework for the prediction API)
- Flask-CORS (Cross-Origin Resource Sharing)
- pgmpy (Bayesian Network inference engine)
- NumPy (Numerical computing)

### Step 3: Install Node.js Server Dependencies
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\server
npm install
```
⏱️ Wait 1-2 minutes for installation.

**Server Dependencies:**
- Express.js (Web framework)
- MongoDB/Mongoose (Database & ODM)
- JWT & bcryptjs (Authentication)
- Axios (HTTP client for Python service)
- node-cron (Scheduled news refresh)

### Step 4: Install React Client Dependencies
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\client
npm install
```
⏱️ Wait 2-3 minutes for installation.

**Client Dependencies:**
- React 19 & TypeScript (UI framework)
- Material-UI v7 (Component library with modern design system)
- Zustand (Lightweight global state management)
- Vite 7 (Lightning-fast build tool)
- React Router v7 (Client-side routing)
- Axios (API communication)

### Step 5: Start All Services (3 Terminals)

You need **3 separate Command Prompt windows** running simultaneously:

---

#### 🐍 Terminal 1: Python Prediction Service (START FIRST!)
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\server\services\prediction
python prediction_service.py
```

✅ **Success output:**
```
============================================================
🐍 MetS Bayesian Network Prediction Service
============================================================
✅ Bayesian Network model loaded successfully
📊 Model nodes: [...]
📍 Starting service on http://0.0.0.0:5001
============================================================
```

---

#### 🟢 Terminal 2: Node.js Backend Server
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\server
npm start
```

✅ **Success output:**
```
========================================
🚀 MetS Predictor API Server
========================================
📍 URL: http://localhost:5000
🌍 Environment: development
🐍 Python Service: http://localhost:5001
========================================
✅ MongoDB Connected: localhost
```

---

#### ⚛️ Terminal 3: React Frontend Client
```cmd
cd C:\MetS_App\Mets_Final_Code\mets-app\client
npm run dev
```

✅ **Success output:**
```
  VITE v7.2.4  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

---

### Step 6: Open the Application 🎉
Open your browser and go to:
```
http://localhost:5173
```

**You should see:**
- Modern teal-themed interface with gradient backgrounds
- Navigation bar with Home, About, Assessment, Resources
- Hero section with red heart icon and green "Start Assessment" button
- Feature cards with varied colorful icons (teal, purple, green, orange, blue, rose)

---

## 📁 Project Structure

```
Mets_Final_Code/
├── bayesian_network_model.pkl    ← Pre-trained ML model (DO NOT DELETE!)
├── recommendations.json          ← Health recommendations data
├── requirements.txt              ← Python deps for Streamlit app
├── app.py                        ← Standalone Streamlit app
├── ARCHITECTURE.md               ← System architecture documentation
├── README.md                     ← Project overview & quick start
│
└── mets-app/
    ├── SETUP.md                  ← This file
    │
    ├── client/                   ← React Frontend (Vite + TypeScript)
    │   ├── src/
    │   │   ├── pages/            ← Enhanced UI pages
    │   │   │   ├── Home.tsx      ← Hero with red heart, news slideshow
    │   │   │   ├── About.tsx     ← Semantic colored condition icons
    │   │   │   ├── Assessment.tsx ← Multi-step wizard (Zustand-driven)
    │   │   │   ├── Profile.tsx   ← User dashboard
    │   │   │   ├── Resources.tsx ← News grid with colored sections
    │   │   │   ├── Login.tsx     ← Auth page
    │   │   │   └── Signup.tsx    ← Auth page
    │   │   ├── components/
    │   │   │   ├── Layout.tsx    ← Red heart footer with pulsing animation
    │   │   │   ├── Navbar.tsx    ← Red heart logo branding
    │   │   │   └── ProtectedRoute.tsx ← Auth guard
    │   │   ├── stores/           ← Zustand global state management
    │   │   │   ├── authStore.ts  ← Auth state (user, token, login/logout)
    │   │   │   ├── assessmentStore.ts ← Assessment flow state
    │   │   │   └── newsStore.ts  ← News state (slideshow + resources)
    │   │   ├── data/
    │   │   │   ├── api.ts        ← All API calls with Axios
    │   │   │   └── types.ts      ← TypeScript interfaces
    │   │   └── theme/
    │   │       └── theme.ts      ← Material-UI theme (teal primary)
    │   └── package.json
    │
    └── server/                   ← Node.js Backend (Express)
        ├── index.js              ← Main server entry
        ├── config.js             ← Configuration
        ├── middleware.js          ← Auth, error handling, utilities
        ├── .env                  ← Environment variables (API keys)
        ├── package.json
        ├── models/               ← MongoDB schemas
        │   ├── User.js           ← User schema (profile + history)
        │   └── News.js           ← News article schema
        ├── routes/               ← API endpoints
        │   ├── auth.js           ← Authentication routes
        │   ├── mets.js           ← Prediction & severity routes
        │   └── news.js           ← Guardian news integration
        └── services/             ← Business logic & ML service
            ├── metsService.js    ← Bayesian Network proxy + cMetS_S
            ├── newsService.js    ← News caching from Guardian API
            ├── recommendationsService.js ← Health plans from JSON
            ├── reportService.js  ← Markdown report generation
            └── prediction/       ← Python ML microservice
                ├── prediction_service.py  ← Flask API
                ├── bayesian_network_model.pkl ← Pre-trained model
                └── requirements.txt ← Python dependencies
```

---

## 🎨 UI Features & Design System

### Color Palette
- **Primary (Teal):** `#00b2a7` → `#009188` — Main brand color
- **Heart/Love (Rose):** `#e11d48` → `#be123c` — Heart icons, warnings
- **Success (Green):** `#10b981` → `#059669` — CTA buttons, success states
- **Warning (Amber):** `#f59e0b` → `#d97706` — Risk indicators
- **Info (Blue):** `#3b82f6` → `#2563eb` — Information sections
- **Purple:** `#8b5cf6` → `#7c3aed` — News, science sections

### State Management (Zustand)
The app uses **Zustand** for all global state — no React Context providers needed:
- **`useAuthStore`** — User authentication (login, logout, token, profile)
- **`useAssessmentStore`** — Assessment wizard (forms, results, recommendations, reset)
- **`useNewsStore`** — News articles (Home slideshow + Resources grid)

### Enhanced Responsiveness
- Mobile-optimized layout with collapsible drawer navigation
- Touch-friendly button sizes and spacing
- Adaptive grid layouts for all screen sizes
- Smooth transitions and hover effects

---

## 🌐 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | React web application |
| **Backend API** | http://localhost:5000 | Node.js REST API |
| **Python ML Service** | http://localhost:5001 | Bayesian Network predictions |
| **MongoDB** | mongodb://localhost:27017 | Database |

---

## ⚙️ Configuration (.env file)

The `.env` file is located at: `mets-app/server/.env`

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/mets_health
JWT_SECRET=mets_health_jwt_secret_key_2026_secure
JWT_EXPIRES_IN=7d
GUARDIAN_API_KEY=your_guardian_api_key_here
```

### Getting a Guardian API Key (for News Feature)
1. Go to: https://open-platform.theguardian.com/access/
2. Register for a free developer key
3. Replace `your_guardian_api_key_here` with your key
4. Restart the server

> ⚠️ **Note:** The app works without the Guardian API key — only the news feature will be disabled.

---

## 🔧 Troubleshooting

### ❌ "MongoDB connection failed"
**Solution:** MongoDB service is not running
```cmd
# Check if MongoDB is running
net start MongoDB

# Or start it manually via Services
# Press Win+R → services.msc → Find "MongoDB Server" → Start
```

### ❌ "'node' is not recognized"
**Solution:** Node.js not installed or not in PATH
- Reinstall Node.js from https://nodejs.org/
- Restart Command Prompt after installation

### ❌ "'python' is not recognized"
**Solution:** Python not installed or not in PATH
- Reinstall Python from https://www.python.org/
- ⚠️ Check "Add Python to PATH" during installation
- Restart Command Prompt

### ❌ "Model file not found" or "Prediction error"
**Solution:** The ML model file is missing
- Ensure `bayesian_network_model.pkl` exists in `server/services/prediction/`
- A backup copy also exists in the project root: `Mets_Final_Code/bayesian_network_model.pkl`

### ❌ "Port 5000/5001/5173 already in use"
**Solution:** Another app is using the port
```cmd
# Find and kill the process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# Or change the port in .env file
```

### ❌ "Cannot find module" error
**Solution:** Dependencies not installed
```cmd
# Reinstall all dependencies
cd mets-app\server
rmdir /s /q node_modules
npm install

cd ..\client
rmdir /s /q node_modules
npm install
```

### ❌ Python service starts but predictions fail
**Solution:** pgmpy version mismatch
```cmd
cd mets-app\server\services\prediction
pip install --upgrade pgmpy numpy
```

---

## 📝 Quick Commands Reference

| Action | Command |
|--------|---------|
| Start Python Service | `cd mets-app\server\services\prediction && python prediction_service.py` |
| Start Backend Server | `cd mets-app\server && npm start` |
| Start Frontend Client | `cd mets-app\client && npm run dev` |
| Build Production Client | `cd mets-app\client && npm run build` |
| Install Python Deps | `cd mets-app\server\services\prediction && pip install -r requirements.txt` |
| Install Server Deps | `cd mets-app\server && npm install` |
| Install Client Deps | `cd mets-app\client && npm install` |
| Check MongoDB Status | `net start MongoDB` |

---

## ✅ Pre-Flight Checklist

Before running, verify:

- [ ] Node.js installed (`node --version`)
- [ ] Python 3.8+ installed (`python --version`)
- [ ] MongoDB installed and running
- [ ] `bayesian_network_model.pkl` exists in `server/services/prediction/`
- [ ] `recommendations.json` exists in root folder
- [ ] Python dependencies installed (Flask, pgmpy, numpy)
- [ ] Server dependencies installed (Express, Mongoose, JWT)
- [ ] Client dependencies installed (React 19, MUI v7, Zustand, Vite 7)

---

## 🚦 Startup Order (Important!)

Always start services in this order:

```
1️⃣ Python Prediction Service (Terminal 1)  ← FIRST
       ⬇️
2️⃣ Node.js Backend Server (Terminal 2)     ← SECOND
       ⬇️
3️⃣ React Frontend Client (Terminal 3)      ← THIRD
       ⬇️
4️⃣ Open http://localhost:5173 in browser   ← DONE!
```

---

## 🛑 Stopping the Application

Press `Ctrl + C` in each terminal window to stop the services.

---

## 🔄 Running the Standalone Streamlit App (Optional)

If you want to run the original Streamlit version:

```cmd
cd C:\MetS_App\Mets_Final_Code
pip install -r requirements.txt
streamlit run app.py
```

This opens at: http://localhost:8501

**Note:** The Streamlit app uses a different UI but the same Bayesian Network model.

---

## 💡 Tips

1. **Keep all 3 terminals open** while using the app
2. **Python service must start first** — the Node.js server checks its health
3. **Changes to React code** auto-refresh in browser (Hot Module Replacement)
4. **Changes to .env** require server restart
5. **MongoDB data persists** even after stopping the app
6. **Browser caching** — hard refresh (Ctrl+F5) if UI doesn't update
7. **Modern browsers only** — Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

---

## 🎯 Key Features Available

### User Features
- ✅ User registration and authentication with JWT
- ✅ Profile management with assessment history
- ✅ Multi-step assessment wizard with real-time validation
- ✅ MetS probability prediction using Bayesian Network
- ✅ Severity scoring with cMetS_S formula
- ✅ Personalized recommendations (diet, exercise, yoga)
- ✅ Downloadable health reports in Markdown
- ✅ Real-time health news from The Guardian API
- ✅ Responsive design for mobile and desktop

### Technical Features
- ✅ React 19 with TypeScript for type safety
- ✅ Zustand for lightweight global state management (auth, assessment, news)
- ✅ Material-UI v7 with modern design system
- ✅ Vite 7 for lightning-fast development
- ✅ Express.js REST API with JWT authentication
- ✅ MongoDB for data persistence
- ✅ Python Flask microservice for ML predictions (inside `services/prediction/`)
- ✅ Bayesian Network model with 6 clinical variables
- ✅ cMetS_S severity scoring with gender/age coefficients

---

**🎉 Happy Health Tracking!**

*Last Updated: February 2026 | Version 3.0*
