
import os
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random
import json
import io
from datetime import datetime
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BicScore, MaximumLikelihoodEstimator
from sklearn.preprocessing import MinMaxScaler

# Set page configuration
st.set_page_config(
    page_title="MetS Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .high-risk .stProgress > div > div {
        background-color: #f44336;
    }
    .medium-risk .stProgress > div > div {
        background-color: #ff9800;
    }
    .low-risk .stProgress > div > div {
        background-color: #4CAF50;
    }
    .recommendation-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .avoid-card {
        border-left: 5px solid #f44336;
    }
    .metric-card {
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def load_recommendations():
    with open("recommendations.json", "r") as file:
        return json.load(file)
    
def calculate_cMetS_S(gender, age, sbp, wc, fpg, tg, hdl_c):
    log_tg = np.log(tg)
    
    if gender == "Men":
        if 20 <= age <= 39:
            B = -1.79 + 0.0016 * sbp + 0.0045 * wc + 0.0017 * fpg + 0.24 * log_tg - 0.0042 * hdl_c
        elif 40 <= age <= 60:
            B = -1.67 + 0.0007 * sbp + 0.0034 * wc + 0.0014 * fpg + 0.25 * log_tg - 0.0042 * hdl_c
        else:
            B = -2.28 + 0.0019 * sbp + 0.0067 * wc + 0.0027 * fpg + 0.28 * log_tg - 0.0054 * hdl_c
    elif gender == "Women":
        if 20 <= age <= 39:
            B = -2.43 + 0.0039 * sbp + 0.0066 * wc + 0.004 * fpg + 0.28 * log_tg - 0.0052 * hdl_c
        elif 40 <= age <= 60:
            B = -2.37 + 0.001 * sbp + 0.0021 * wc + 0.0015 * fpg + 0.41 * log_tg - 0.004 * hdl_c
        else:
            B = -4.13 + 0.0065 * sbp + 0.012 * wc + 0.007 * fpg + 0.39 * log_tg - 0.006 * hdl_c
    else:
        if 20 <= age <= 39:
            B = -2.34 + 0.003 * sbp + 0.0061 * wc + 0.0032 * fpg + 0.29 * log_tg - 0.0055 * hdl_c
        elif 40 <= age <= 60:
            B = -1.94 + 0.0006 * sbp + 0.0019 * wc + 0.0011 * fpg + 0.33 * log_tg - 0.003 * hdl_c
        else:
            B = -3.39 + 0.0044 * sbp + 0.0099 * wc + 0.0054 * fpg + 0.36 * log_tg - 0.0063 * hdl_c
    
    return B

def classify_severity(severity):
    if 0 <= severity <= 0.30:
        return 'Low Severity'
    elif 0.31 <= severity <= 0.60:
        return 'Medium Severity'
    elif 0.61 <= severity <= 0.99:
        return 'High Severity'
    

class GeneticAlgorithmBayesianNetwork:
    def __init__(self, data, population_size=20, generations=50, mutation_rate=0.1):
        self.data = data
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.nodes = list(data.columns)
    
    def _create_valid_network(self):
        network = BayesianNetwork()
        for node in self.nodes:
            network.add_node(node)
        
        for i in range(len(self.nodes)-1):
            for j in range(i+1, len(self.nodes)):
                if random.random() > 0.7:
                    try:
                        network.add_edge(self.nodes[i], self.nodes[j])
                    except:
                        continue
        return network
    
    def initialize_population(self):
        return [self._create_valid_network() for _ in range(self.population_size)]
    
    def fitness(self, network):
        try:
            return BicScore(self.data).score(network)
        except:
            return float('-inf')
    
    def crossover(self, parent1, parent2):
        child = BayesianNetwork()
        for node in self.nodes:
            child.add_node(node)
        for edge in set(list(parent1.edges()) + list(parent2.edges())):
            if random.random() < 0.5:
                try:
                    child.add_edge(*edge)
                except:
                    continue
        return child
    
    def mutate(self, network):
        if random.random() < self.mutation_rate:
            mutated = network.copy()
            if random.random() < 0.5 and mutated.edges():
                edge = random.choice(list(mutated.edges()))
                mutated.remove_edge(*edge)
            else:
                node1, node2 = random.sample(self.nodes, 2)
                try:
                    mutated.add_edge(node1, node2)
                except:
                    pass
            return mutated
        return network
    
    def run(self):
        population = self.initialize_population()
        best_network, best_score = None, float('-inf')
        
        for gen in range(self.generations):
            fitness_scores = [self.fitness(network) for network in population]
            
            current_best = max(fitness_scores)
            if current_best > best_score:
                best_score = current_best
                best_network = population[fitness_scores.index(current_best)]
            
            selected = [population[max(random.sample(range(len(fitness_scores)), 3), key=lambda i: fitness_scores[i])] for _ in range(self.population_size)]
            
            next_population = []
            for i in range(0, len(selected), 2):
                if i+1 < len(selected):
                    child = self.crossover(selected[i], selected[i+1])
                    child = self.mutate(child)
                    next_population.append(child)
            
            population = next_population
        
        return best_network

def train_and_save_model(data, model_filename='bayesian_network_model.pkl'):
    ga_bn = GeneticAlgorithmBayesianNetwork(data)
    best_model = ga_bn.run()
    best_model.fit(data, estimator=MaximumLikelihoodEstimator)
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)

def load_model(model_filename='bayesian_network_model.pkl'):
    try:
        with open(model_filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file '{model_filename}' not found. Please train the model first.")
        return None

def predict_metabolic_syndrome(model, evidence):
    if model is None:
        return 0.5  # Default probability if model is not available
    
    inference = VariableElimination(model)
    query_result = inference.query(variables=['Metabolic syndrome(0=no, 1=yes)'], evidence=evidence)
    prob_metabolic_syndrome = query_result.values[1]  # Probability of having MetS (value at index 1)
    
    return prob_metabolic_syndrome

def get_recommendations(gender, risk_level, age):
    recommendations = load_recommendations()
    age_group = "20-40" if age < 40 else "40-60"
    
    diet_plan, avoid_list, exercise_plan, yoga_poses_plan = [], [], [], []
    
    if gender in recommendations:
        if risk_level in recommendations[gender]:
            try:
                diet_plan = recommendations[gender][risk_level][age_group]["Diet Plan"]["Recommended"]
                avoid_list = recommendations[gender][risk_level][age_group]["Diet Plan"]["Avoid"]
                exercise_plan = recommendations[gender][risk_level][age_group]["Exercise Plan"]
                yoga_poses_plan = recommendations["Yoga Poses for Metabolic Syndrome"][risk_level]
            except KeyError:
                st.warning("No recommendations available for the given risk level, age group, or gender.")
        else:
            st.warning(f"No recommendations available for risk level: {risk_level}")
    else:
        st.warning(f"No recommendations available for gender: {gender}")
    
    return diet_plan, avoid_list, exercise_plan, yoga_poses_plan

def create_downloadable_report(user_info, results, recommendations):
    """Create a formatted report for download"""
    buffer = io.StringIO()
    
    # Add header and date
    buffer.write(f"# METABOLIC SYNDROME HEALTH PLAN\n")
    buffer.write(f"Generated on: {datetime.now().strftime('%B %d, %Y')}\n\n")
    
    # Add user information
    buffer.write(f"## PATIENT INFORMATION\n")
    for key, value in user_info.items():
        buffer.write(f"- {key}: {value}\n")
    buffer.write("\n")
    
    # Add assessment results
    buffer.write(f"## ASSESSMENT RESULTS\n")
    buffer.write(f"- Probability of Metabolic Syndrome: {results['probability']:.2f} ({results['probability']*100:.1f}%)\n")
    if 'severity' in results:
        buffer.write(f"- Severity Score: {results['severity']:.2f}\n")
        buffer.write(f"- Risk Level: {results['risk_level']}\n")
    buffer.write("\n")
    
    # Add recommendations
    buffer.write(f"## HEALTH RECOMMENDATIONS\n")
    
    if recommendations['diet_plan']:
        buffer.write(f"### Diet Plan Recommendations\n")
        for item in recommendations['diet_plan']:
            buffer.write(f"- {item}\n")
        buffer.write("\n")
    
    if recommendations['avoid_list']:
        buffer.write(f"### Foods to Avoid\n")
        for item in recommendations['avoid_list']:
            buffer.write(f"- {item}\n")
        buffer.write("\n")
    
    if recommendations['exercise_plan']:
        buffer.write(f"### Exercise Plan Recommendations\n")
        for item in recommendations['exercise_plan']:
            buffer.write(f"- {item}\n")
        buffer.write("\n")
    
    if recommendations['yoga_plan']:
        buffer.write(f"### Yoga Poses Recommendations\n")
        for item in recommendations['yoga_plan']:
            buffer.write(f"- {item}\n")
        buffer.write("\n")
    
    # Add disclaimer
    buffer.write(f"## DISCLAIMER\n")
    buffer.write("This health plan is generated based on the information you provided and is for informational purposes only. ")
    buffer.write("It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. ")
    buffer.write("Always seek the advice of your physician or other qualified health provider with any questions you may have regarding your health.\n")
    
    return buffer.getvalue()

def display_metric_card(title, value, unit="", help_text=""):
    """Display a metric in a styled card"""
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h4>{title}</h4>
            <h2>{value} {unit}</h2>
            <p>{help_text}</p>
        </div>
        """, unsafe_allow_html=True)

def app():
    apply_custom_css()
    
    # Create a sidebar for navigation
    with st.sidebar:
        st.image('background.png', width=300)
        st.title("Navigation")
        pages = ["Home", "About Metabolic Syndrome", "Assessment Tool", "Resources"]
        page = st.radio("Go to", pages)
    
    if page == "Home":
        st.title("🏥 Metabolic Syndrome Prediction & Recommendations")
        
        st.markdown("""
        ### Welcome to the MetS Predictor
        
        This tool helps you assess your risk of metabolic syndrome and provides personalized health recommendations.
        
        **What is Metabolic Syndrome?**
        Metabolic syndrome is a cluster of conditions that occur together, increasing your risk of heart disease, 
        stroke, and type 2 diabetes. These conditions include increased blood pressure, high blood sugar, 
        excess body fat around the waist, and abnormal cholesterol or triglyceride levels.
        
        **How to use this app:**
        1. Navigate to the **Assessment Tool** page
        2. Fill in your health information
        3. Get your risk assessment and personalized recommendations
        4. Download your personalized health plan
        
        **Privacy Notice**: All data is processed locally on your device and is not stored or shared.
        """)
        
        st.button("Go to Assessment Tool", on_click=lambda: st.session_state.update({"page": "Assessment Tool"}))
    
    elif page == "About Metabolic Syndrome":
        st.title("About Metabolic Syndrome")
        
        st.markdown("""
        ### What is Metabolic Syndrome?
        
        Metabolic syndrome is a group of five conditions that can increase your risk for heart disease, diabetes, and stroke.
        
        ### The five conditions are:
        
        1. **Increased waist circumference** (abdominal obesity)
        2. **High triglyceride levels** in the blood
        3. **Low HDL cholesterol** levels ("good" cholesterol)
        4. **High blood pressure**
        5. **High fasting blood sugar** levels
        
        Having three or more of these conditions qualifies as metabolic syndrome.
        
        ### Risk Factors:
        
        - Age (risk increases with age)
        - Obesity (particularly abdominal obesity)
        - Physical inactivity
        - Insulin resistance
        - Genetics and family history
        - Hormonal imbalances
        
        ### Prevention and Management:
        
        - Regular physical activity
        - Healthy diet rich in fruits, vegetables, whole grains
        - Weight loss (if overweight)
        - Smoking cessation
        - Limiting alcohol consumption
        - Regular health check-ups
        """)
    
    elif page == "Assessment Tool":
        # Initialize session state variables if they don't exist
        if 'show_additional_inputs' not in st.session_state:
            st.session_state.show_additional_inputs = False
        if 'prob_metabolic_syndrome' not in st.session_state:
            st.session_state.prob_metabolic_syndrome = 0
        if 'has_metabolic_syndrome' not in st.session_state:
            st.session_state.has_metabolic_syndrome = False
        if 'severity_calculated' not in st.session_state:
            st.session_state.severity_calculated = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = {}
        if 'results' not in st.session_state:
            st.session_state.results = {}
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = {
                'diet_plan': [],
                'avoid_list': [],
                'exercise_plan': [],
                'yoga_plan': []
            }
            
        st.title("Metabolic Syndrome Assessment Tool")
        
        # Create tabs for the assessment process
        tabs = st.tabs(["Step 1: Basic Information", "Step 2: Additional Information", "Step 3: Results & Recommendations"])
        
        with tabs[0]:
            st.subheader("Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", min_value=20, max_value=60, value=30, help="Your current age in years")
                gender = st.radio("Select Gender", ["Men", "Women"], help="Select your biological gender")
                fatty_liver = st.selectbox("Previous fatty liver diagnosis", ["No", "Yes"], help="Have you been diagnosed with fatty liver?")
                fatty_liver_val = 1 if fatty_liver == "Yes" else 0
            
            with col2:
                hypertension = st.selectbox("Previous hypertension diagnosis", ["No", "Yes"], help="Have you been diagnosed with hypertension?")
                hypertension_val = 1 if hypertension == "Yes" else 0
                diabetes = st.selectbox("Previous diabetes diagnosis", ["No", "Yes"], help="Have you been diagnosed with diabetes?")
                diabetes_val = 1 if diabetes == "Yes" else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sbp = st.number_input("Systolic blood pressure (mmHg)", min_value=71, max_value=185, value=131, 
                                     help="The top number in your blood pressure reading")
            
            with col2:
                dbp = st.number_input("Diastolic blood pressure (mmHg)", min_value=34, max_value=150, value=85, 
                                     help="The bottom number in your blood pressure reading")
            
            with col3:
                wc = st.number_input("Waist circumference (cm)", min_value=18, max_value=142, value=75, 
                                    help="Measure around your waist at the level of your navel")
            
            # Store user information
            st.session_state.user_data = {
                "Age": age,
                "Gender": gender,
                "Fatty Liver": fatty_liver,
                "Hypertension": hypertension,
                "Diabetes": diabetes,
                "Systolic BP": f"{sbp} mmHg",
                "Diastolic BP": f"{dbp} mmHg",
                "Waist Circumference": f"{wc} cm"
            }
            
            evidence = {
                'Previous fatty liver (0=no, 1=yes)': fatty_liver_val,
                'Previous hypertension(0=no, 1=yes)': hypertension_val,
                'Previous diabetes(0=no, 1=yes)': diabetes_val,
                'Waist circumference(cm)': wc,
                'Systolic blood pressure(mmHg)': sbp,
                'Diastolic blood pressure(mmHg)': dbp
            }
            
            model = load_model()
            
            # First button - Predict Metabolic Syndrome
            if st.button("Predict Metabolic Syndrome", use_container_width=True):
                with st.spinner("Calculating risk..."):
                    # Make the initial prediction
                    prob_metabolic_syndrome = predict_metabolic_syndrome(model, evidence)
                    st.session_state.prob_metabolic_syndrome = prob_metabolic_syndrome
                    
                    # Store results
                    st.session_state.results = {
                        'probability': prob_metabolic_syndrome
                    }
                    
                    # Determine if the user has metabolic syndrome based on threshold
                    if prob_metabolic_syndrome > 0.65:
                        st.warning("⚠️ You have a high probability of Metabolic Syndrome.")
                        st.session_state.has_metabolic_syndrome = True
                        st.session_state.show_additional_inputs = True
                    else:
                        st.success("✅ You have a low probability of Metabolic Syndrome.")
                        st.session_state.has_metabolic_syndrome = False
                        st.session_state.show_additional_inputs = False
                    
                    # Switch to appropriate tab based on result
                    if st.session_state.has_metabolic_syndrome:
                        st.info("Please proceed to Step 2 to provide additional information.")
                    else:
                        st.info("You can view your results and recommendations in Step 3.")
        
        with tabs[1]:
            if not st.session_state.has_metabolic_syndrome and st.session_state.prob_metabolic_syndrome > 0:
                st.success("Based on your information, you have a low probability of metabolic syndrome. " 
                         "You can skip this step and proceed to view your recommendations.")
            else:
                st.subheader("Additional Information for Severity Assessment")
                
                if st.session_state.prob_metabolic_syndrome == 0:
                    st.info("Please complete Step 1 first to check your risk of metabolic syndrome.")
                else:
                    st.info("Please provide additional information to determine the severity of your metabolic syndrome.")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        hdl_c = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=80, value=50, 
                                               help="High-density lipoprotein cholesterol, often called 'good' cholesterol")
                    
                    with col2:
                        tg = st.number_input("Triglyceride (mg/dL)", min_value=50, max_value=500, value=175, 
                                            help="A type of fat in your blood")
                    
                    with col3:
                        fpg = st.number_input("Fasting Plasma Glucose (mg/dL)", min_value=70, max_value=126, value=75, 
                                             help="Your blood sugar level when you haven't eaten for at least 8 hours")
                    
                    # Update user data
                    st.session_state.user_data.update({
                        "HDL Cholesterol": f"{hdl_c} mg/dL",
                        "Triglyceride": f"{tg} mg/dL",
                        "Fasting Glucose": f"{fpg} mg/dL"
                    })
                    
                    # Second button - Check Severity
                    if st.button("Check Severity", use_container_width=True):
                        with st.spinner("Calculating severity..."):
                            # Calculate B value using the cMetS_S formula
                            B = calculate_cMetS_S(gender, age, sbp, wc, fpg, tg, hdl_c)
                            
                            # Calculate severity score (simplified version)
                            severity = min(0.99, max(0, st.session_state.prob_metabolic_syndrome + B))
                            
                            # Classify severity
                            risk_level = classify_severity(severity)
                            
                            # Update results
                            st.session_state.results.update({
                                'severity': severity,
                                'risk_level': risk_level
                            })
                            
                            # Get recommendations
                            diet_plan, avoid_list, exercise_plan, yoga_plan = get_recommendations(gender, risk_level, age)
                            
                            # Store recommendations
                            st.session_state.recommendations = {
                                'diet_plan': diet_plan,
                                'avoid_list': avoid_list,
                                'exercise_plan': exercise_plan,
                                'yoga_plan': yoga_plan
                            }
                            
                            st.session_state.severity_calculated = True
                            
                            # Provide feedback
                            st.info("Severity calculated! Please proceed to Step 3 to view your results and recommendations.")
        
        with tabs[2]:
            st.subheader("Results & Recommendations")
            
            if st.session_state.prob_metabolic_syndrome == 0:
                st.info("Please complete Step 1 first to check your risk of metabolic syndrome.")
            else:
                # Display the initial result
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Risk Assessment")
                    
                    # Show probability with a progress bar
                    prob_percentage = st.session_state.prob_metabolic_syndrome * 100
                    risk_class = ""
                    if prob_percentage > 65:
                        risk_class = "high-risk"
                    elif prob_percentage > 35:
                        risk_class = "medium-risk"
                    else:
                        risk_class = "low-risk"
                    
                    st.markdown(f'<div class="{risk_class}">', unsafe_allow_html=True)
                    st.progress(st.session_state.prob_metabolic_syndrome)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.write(f"Probability of Metabolic Syndrome: {st.session_state.prob_metabolic_syndrome:.2f} or {prob_percentage:.1f}%")
                    
                    if 'severity' in st.session_state.results:
                        severity = st.session_state.results['severity']
                        risk_level = st.session_state.results['risk_level']
                        
                        st.subheader("Severity Assessment")
                        
                        severity_class = ""
                        if risk_level == "High Severity":
                            severity_class = "high-risk"
                        elif risk_level == "Medium Severity":
                            severity_class = "medium-risk"
                        else:
                            severity_class = "low-risk"
                        
                        st.markdown(f'<div class="{severity_class}">', unsafe_allow_html=True)
                        st.progress(severity)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.write(f"Severity Score: {severity:.2f}")
                        
                        if risk_level == "Low Severity":
                            st.success(f"Prediction: {risk_level}")
                        elif risk_level == "Medium Severity":
                            st.warning(f"Prediction: {risk_level}")
                        else:
                            st.error(f"Prediction: {risk_level}")
                
                with col2:
                    st.subheader("Health Metrics")
                    metrics = st.session_state.user_data
                    
                    for key, value in metrics.items():
                        st.text(f"{key}: {value}")
                
                # Display recommendations
                if st.session_state.has_metabolic_syndrome or st.session_state.severity_calculated:
                    st.subheader("Health Recommendations")
                    
                    tabs_recommendations = st.tabs(["Diet Plan", "Foods to Avoid", "Exercise Plan", "Yoga Poses"])
                    
                    with tabs_recommendations[0]:
                        diet_plan = st.session_state.recommendations['diet_plan']
                        if not diet_plan:
                            st.warning("No diet plan available for this risk level.")
                        else:
                            for item in diet_plan:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>✅ {item}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with tabs_recommendations[1]:
                        avoid_list = st.session_state.recommendations['avoid_list']
                        if not avoid_list:
                            st.warning("No foods to avoid for this risk level.")
                        else:
                            for item in avoid_list:
                                st.markdown(f"""
                                <div class="recommendation-card avoid-card">
                                    <h4>❌ {item}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with tabs_recommendations[2]:
                        exercise_plan = st.session_state.recommendations['exercise_plan']
                        if not exercise_plan:
                            st.warning("No exercise plan available for this risk level.")
                        else:
                            for item in exercise_plan:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>✅ {item}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with tabs_recommendations[3]:
                        yoga_plan = st.session_state.recommendations['yoga_plan']
                        if not yoga_plan:
                            st.warning("No yoga poses plan available for this risk level.")
                        else:
                            for item in yoga_plan:
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>✅ {item}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Create downloadable report
                    report = create_downloadable_report(
                        st.session_state.user_data,
                        st.session_state.results,
                        st.session_state.recommendations
                    )
                    
                    st.download_button(
                        label="📥 Download Your Health Plan",
                        data=report,
                        file_name=f"metabolic_syndrome_health_plan_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
    
    elif page == "Resources":
        st.title("Resources and References")
        
        st.markdown("""
        ### Metabolic Syndrome Resources
        
        **Official Medical Organizations:**
        - [American Heart Association - Metabolic Syndrome](https://www.heart.org/en/health-topics/metabolic-syndrome)
        - [Mayo Clinic - Metabolic Syndrome](https://www.mayoclinic.org/diseases-conditions/metabolic-syndrome/symptoms-causes/syc-20351916)
        - [National Heart, Lung, and Blood Institute](https://www.nhlbi.nih.gov/health-topics/metabolic-syndrome)
        - [World Health Organization - Noncommunicable Diseases](https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases)
        
        **Research and Articles:**
        - [PubMed - Recent Research on Metabolic Syndrome](https://pubmed.ncbi.nlm.nih.gov/?term=metabolic+syndrome)
        - [American Diabetes Association - Metabolic Syndrome](https://diabetes.org/diabetes-risk/prediabetes/metabolic-syndrome)
        
        **Lifestyle Management Resources:**
        - [DASH Diet for Metabolic Syndrome](https://www.nhlbi.nih.gov/health-topics/dash-eating-plan)
        - [Physical Activity Guidelines - CDC](https://www.cdc.gov/physicalactivity/basics/index.htm)
        - [Stress Management Techniques - Mayo Clinic](https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-management/art-20044289)
        
        **Monitoring Tools:**
        - [Blood Pressure Monitoring Guide - American Heart Association](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings)
        - [Blood Glucose Monitoring - CDC](https://www.cdc.gov/diabetes/managing/managing-blood-sugar/bloodglucosemonitoring.html)
        
        ### References for this Application:
        
        This application uses a Bayesian Network model trained on anonymized clinical data to predict the risk of metabolic syndrome.
        
        
        The recommendations provided are based on clinical guidelines from multiple international health organizations and are tailored based on gender, age, and risk level.
        
        **Disclaimer:** This application is for informational purposes only and is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
        """)

if __name__ == "__main__":
    app()