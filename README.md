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
 4. User-friendly dashboard for patients and doctors

## Try It Out

Experience the **Metabolic Syndrome Prediction & Personalized Healthcare Recommendation System** right from your browser.  
Click on Launch Demo and explore how it predicts MetS risk and provides tailored healthcare advice.
<p>
  <a href="https://metabolic-syndrome.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Launch%20Demo-brightgreen?style=for-the-badge" alt="Launch Demo"/>
  </a>
</p>




## Run it Locally

### 📋 Prerequisites

- Python 3.8+
- Git
- (Optional) Virtual environment tool (`venv`) / VSCode would be sufficient


#### 1. Clone this Repo into you local and make it as current working directory

```bash
git clone https://github.com/yourusername/Metabolic-Syndrome-Prediction-Personalized-HealthCare-Recommendation.git

cd Metabolic-Syndrome-Prediction-Personalized-HealthCare-Recommendation
```

#### 2. Create a Virtual Environment and Activate the Virtual Environment (Windows)

```bash
python -m venv venv

venv\Scripts\activate
```

#### 3. Install the Requirements

```bash
pip install -r requirements.txt
```

#### 4. Run the App locally
```bash
streamlit run Main.py --server.port <portnumber> (Ex : 8502)
```


## 🤝 Contributing
#### We welcome contributions!
#### Fork and Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```
#### Stage, Commit and Push Chnages

```bash
git add .

git commit -m "Add <description>"

git push origin feature/your-feature-name

```

#### After pushing your branch, create a Pull Request from your forked repo.


