# Metabolic-Syndrome-Prediction-Personalized-HealthCare-Recommendation

## Overview
Metabolic Syndrome (MetS) is a cluster of conditions including high blood pressure, elevated blood sugar, excess body fat, and abnormal cholesterol levels, significantly increasing the risk of heart disease, stroke, and type 2 diabetes. Early diagnosis of MetS remains challenging due to its subtle and overlapping symptoms, leading to delayed intervention, suboptimal treatment outcomes, and increased healthcare costs. With the rising prevalence of MetS driven by sedentary lifestyles and poor diets, there is a critical need for early prediction and personalized preventive strategies to reduce the burden of severe health complications and healthcare expenses.

Accurate early detection of MetS allows for timely interventions such as lifestyle changes and medication, which can reverse or slow the progression of the syndrome. However, existing approaches often lack personalization and fail to integrate diverse patient data from genetics, lifestyle, and medical records, limiting predictive accuracy and care effectiveness. Utilizing advanced techniques like evolutionary computing can enhance prediction models, enabling personalized lifestyle recommendations tailored to individual patient profiles, thereby transforming healthcare from reactive to proactive care.

To address these challenges, the proposed methodology introduces a four-phase system for accurate MetS prediction and personalized care. Phase I involves optimal feature selection to identify the most critical variables contributing to MetS. Phase II predicts the presence of MetS using a genetically optimized Bayesian Network. Phase III calculates a MetS severity score and classifies patients into low, medium, and high-risk categories based on this score. Finally, Phase IV generates personalized healthcare and dietary plans considering patient-specific factors such as age, gender, and blood rate, ensuring targeted interventions for effective management of MetS.


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
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/React-19.x-61DAFB?style=for-the-badge&logo=react&logoColor=white" alt="React"/>
  <img src="https://img.shields.io/badge/TypeScript-5.9-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/Zustand-5.x-443E38?style=for-the-badge" alt="Zustand"/>
  <img src="https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=node.js&logoColor=white" alt="Node.js"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MongoDB-7.x-47A248?style=for-the-badge&logo=mongodb&logoColor=white" alt="MongoDB"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Bayesian%20Network-FF6F00?style=for-the-badge" alt="ML"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

## 📖 About

**MetS Health** is a full-stack web application that uses **Bayesian Network machine learning** to predict Metabolic Syndrome risk and provide personalized health recommendations.

Metabolic Syndrome is a cluster of conditions (high blood pressure, high blood sugar, excess body fat, abnormal cholesterol levels) that increase the risk of heart disease, stroke, and type 2 diabetes.

### 🎯 What Makes This Special?

- **Two-Stage ML Model** — Combines Bayesian Network inference with clinical cMetS_S scoring
- **Genetic Algorithm Optimization** — Network structure learned through evolutionary algorithms
- **Personalized Health Plans** — Diet, exercise, and yoga recommendations based on your risk profile
- **Real-time Health News** — Curated metabolic syndrome news from The Guardian
- **Zustand State Management** — Lightweight, scalable global state with zero boilerplate

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔮 Prediction Engine
- Bayesian Network with Variable Elimination
- Genetic Algorithm optimized structure
- 6 clinical input variables
- Probabilistic risk assessment

### 📊 Severity Assessment
- Clinical cMetS_S formula
- Gender & age-specific coefficients
- 3-tier risk classification
- Evidence-based scoring

</td>
<td width="50%">

### 🥗 Health Recommendations
- Personalized diet plans
- Foods to avoid list
- Exercise routines
- Yoga pose suggestions

### 👤 User Management
- Secure authentication (JWT)
- Profile management
- Assessment history tracking
- Downloadable health reports

</td>
</tr>
</table>

---

## 🖥️ Demo

### Assessment Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Step 1        │     │   Step 2        │     │   Step 3        │
│   Basic Info    │ ──▶ │   Additional    │ ──▶ │   Results &     │
│   • Age/Gender  │     │   • HDL/LDL     │     │   Recommendations│
│   • BP/Waist    │     │   • Triglyceride│     │   • Diet Plan   │
│   • History     │     │   • Glucose     │     │   • Exercise    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Risk Classification

| Severity | Score Range | Color |
|----------|-------------|-------|
| 🟢 Low | 0.00 - 0.30 | Green |
| 🟡 Medium | 0.31 - 0.60 | Yellow |
| 🔴 High | 0.61 - 0.99 | Red |

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org/))
- **Python** 3.8+ ([Download](https://www.python.org/))
- **MongoDB** 7.x ([Download](https://www.mongodb.com/try/download/community))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mets-health.git
cd mets-health

# Install Python dependencies (prediction service)
cd mets-app/server/services/prediction
pip install -r requirements.txt

# Install Node.js server dependencies
cd ../../
npm install

# Install React client dependencies
cd ../client
npm install
```

### Running the Application

You need **3 terminal windows**:

```bash
# Terminal 1: Python ML Service (START FIRST!)
cd mets-app/server/services/prediction
python prediction_service.py
# ✅ Running on http://localhost:5001

# Terminal 2: Node.js Backend
cd mets-app/server
npm start
# ✅ Running on http://localhost:5000

# Terminal 3: React Frontend
cd mets-app/client
npm run dev
# ✅ Running on http://localhost:5173
```

### Open in Browser

```
http://localhost:5173
```

🎉 **That's it!** The application is now running.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                   │
│            React 19 + TypeScript + MUI v7 + Zustand             │
│                   http://localhost:5173                          │
│                                                                  │
│  ┌──────────────┐ ┌──────────────────┐ ┌─────────────┐         │
│  │  authStore   │ │ assessmentStore   │ │  newsStore  │         │
│  └──────────────┘ └──────────────────┘ └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API SERVER                                  │
│                Node.js + Express + JWT                           │
│                  http://localhost:5000                           │
└─────────────────────────────────────────────────────────────────┘
              │                              │
              ▼                              ▼
┌──────────────────────┐        ┌──────────────────────┐
│      MongoDB         │        │  Prediction Service  │
│   User & News Data   │        │   Bayesian Network   │
│ mongodb://localhost  │        │ http://localhost:5001│
└──────────────────────┘        └──────────────────────┘
```

### Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 19, TypeScript, Material-UI 7, Zustand, Vite 7, Axios |
| **Backend** | Node.js, Express.js, MongoDB, Mongoose, JWT, node-cron |
| **ML Service** | Python, Flask, pgmpy, NumPy |
| **Database** | MongoDB 7.x |

### State Management (Zustand)

| Store | Purpose | Used By |
|-------|---------|---------|
| `useAuthStore` | User auth, token, login/logout | App, Navbar, Login, Signup, Profile |
| `useAssessmentStore` | Assessment forms, results, recommendations | Assessment |
| `useNewsStore` | News articles (slideshow + grid) | Home, Resources |

📚 **[Full Architecture Documentation](ARCHITECTURE.md)**

---

## 📁 Project Structure

```
Mets_Final_Code/
├── 📄 README.md                    # This file
├── 📄 ARCHITECTURE.md              # Detailed architecture docs
├── 🧠 bayesian_network_model.pkl   # Pre-trained ML model
├── 📋 recommendations.json         # Health recommendations data
├── 🐍 app.py                       # Standalone Streamlit app
├── 📓 Final_Code.ipynb             # Model training notebook
│
└── 📁 mets-app/
    ├── 📄 SETUP.md                 # Setup instructions
    │
    ├── 📁 client/                  # React Frontend
    │   ├── 📁 src/
    │   │   ├── 📁 components/      # Layout, Navbar, ProtectedRoute
    │   │   ├── 📁 pages/           # Home, About, Assessment, Profile, Resources, Login, Signup
    │   │   ├── 📁 stores/          # Zustand stores (auth, assessment, news)
    │   │   ├── 📁 data/            # API service & TypeScript types
    │   │   └── 📁 theme/           # MUI theme configuration
    │   └── 📄 package.json
    │
    └── 📁 server/                  # Node.js Backend
        ├── 📄 index.js             # Express entry point
        ├── 📄 config.js            # Configuration & DB connection
        ├── 📄 middleware.js         # Auth, error handling, utilities
        ├── 📁 routes/              # API endpoints (auth, mets, news)
        ├── 📁 models/              # MongoDB schemas (User, News)
        └── 📁 services/            # Business logic
            ├── 📄 metsService.js          # Severity calculations
            ├── 📄 newsService.js          # Guardian API integration
            ├── 📄 recommendationsService.js # Health plans
            ├── 📄 reportService.js        # Report generation
            └── 📁 prediction/             # Python ML microservice
                ├── 🐍 prediction_service.py   # Flask API
                ├── 🧠 bayesian_network_model.pkl
                └── 📄 requirements.txt
```

---

## 🔌 API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/signup` | Create account |
| `POST` | `/api/auth/login` | User login |
| `GET` | `/api/auth/me` | Get current user |
| `PUT` | `/api/auth/profile` | Update profile |
| `PUT` | `/api/auth/password` | Change password |
| `POST` | `/api/auth/assessment` | Save assessment |
| `GET` | `/api/auth/assessments` | Get assessment history |
| `DELETE` | `/api/auth/account` | Delete account |

### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/mets/predict` | Predict MetS probability |
| `POST` | `/api/mets/severity` | Calculate severity score |
| `POST` | `/api/mets/recommendations` | Get health recommendations |
| `POST` | `/api/mets/report` | Generate health report |

### News

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/news` | Get health news articles |
| `GET` | `/api/news/refresh` | Refresh news from API |
| `GET` | `/api/news/status` | News service status |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | System health status |

📚 **[Full API Documentation](ARCHITECTURE.md#9-api-endpoints-reference)**

---

## 🧠 The ML Model

### Bayesian Network

The core prediction engine uses a **Bayesian Network** trained with **Genetic Algorithm optimization**:

```python
# Model learns optimal structure from clinical data
class GeneticAlgorithmBayesianNetwork:
    def __init__(self, data, population_size=20, generations=50):
        self.mutation_rate = 0.1

    def fitness(self, network):
        return BicScore(self.data).score(network)

    def run(self):
        # Evolve network structure
        # Return best network based on BIC score
```

### Input Variables

| Variable | Type | Range |
|----------|------|-------|
| Age | Integer | 20-60 |
| Gender | Categorical | Men/Women |
| Fatty Liver History | Binary | Yes/No |
| Hypertension History | Binary | Yes/No |
| Diabetes History | Binary | Yes/No |
| Systolic BP | Integer | 71-185 mmHg |
| Diastolic BP | Integer | 34-150 mmHg |
| Waist Circumference | Integer | 18-142 cm |

### Severity Formula (cMetS_S)

```
B = intercept + (β₁ × SBP) + (β₂ × WC) + (β₃ × FPG) + (β₄ × log(TG)) + (β₅ × HDL-C)

Final Severity = min(0.99, max(0, Probability + B))
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` in `mets-app/server/`:

```env
PORT=5000
MONGODB_URI=mongodb://localhost:27017/mets_health
JWT_SECRET=your_secure_secret_key
JWT_EXPIRES_IN=7d
GUARDIAN_API_KEY=your_guardian_api_key  # Optional, for news
```

### Getting Guardian API Key (Optional)

1. Go to [The Guardian Open Platform](https://open-platform.theguardian.com/access/)
2. Register for a free developer key
3. Add to `.env` file

> **Note:** The app works without the Guardian API key — only the news feature will be disabled.

---

## 📱 Standalone Streamlit App

A standalone version using Streamlit is also available:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Opens at http://localhost:8501
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **pgmpy** — Bayesian Network library for Python
- **Material-UI** — React component library
- **Zustand** — Lightweight state management
- **The Guardian** — News API provider
- Clinical research on cMetS_S scoring methodology

---

## 📞 Support

If you have any questions or need help, please:

1. Check the [SETUP.md](mets-app/SETUP.md) for detailed setup instructions
2. Review the [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Open an issue on GitHub

---

<p align="center">
  Made with ❤️ for better health outcomes
</p>

<p align="center">
  <a href="#-mets-health---metabolic-syndrome-predictor">Back to Top ⬆️</a>
</p>
