"""
Metabolic Syndrome Prediction Service
=====================================
Loads a pre-trained Genetically Optimized Bayesian Network model
and returns the raw probability of metabolic syndrome.

Flow: User inputs → Evidence → Bayesian Network Inference → Raw Probability
That's it. No thresholds, no classification — just the probability.
The Node.js backend decides what to do with it.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import traceback

# pgmpy imports
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

try:
    from pgmpy.models import DiscreteBayesianNetwork
except ImportError:
    DiscreteBayesianNetwork = None

# ==================== App Setup ====================

def create_app():
    app = Flask(__name__)
    CORS(app)
    return app


app = create_app()

# ==================== Globals ====================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)

DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "PredictionModel",
    "src",
    "Bayesian_Model",
    "Bayesian_Prediction_Model.pkl"
)

MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

model = None
valid_states = {}  # valid CPD states per variable (for mapping user input)

# ==================== Model Loading ====================

def load_model():
    """Load the pre-trained Bayesian Network and convert for pgmpy >= 1.0.0 compatibility."""
    global model, valid_states

    try:
        # 1. Load pickled model
        with open(MODEL_PATH, 'rb') as f:
            loaded_model = pickle.load(f)

        print(f"✅ Model loaded from {MODEL_PATH}")
        print(f"   Type: {type(loaded_model).__name__}")
        print(f"   Nodes: {list(loaded_model.nodes())}")

        # 2. Extract valid states from CPDs (needed to map user input to closest valid value)
        for cpd in loaded_model.cpds:
            var = cpd.variable
            valid_states[var] = [int(s) for s in cpd.state_names.get(var, [])]
            print(f"   {var}: {len(valid_states[var])} states")

        # 3. Convert old BayesianNetwork → DiscreteBayesianNetwork (pgmpy >= 1.0.0)
        #    Without this, VariableElimination crashes with 'no attribute factors'
        if DiscreteBayesianNetwork and not isinstance(loaded_model, DiscreteBayesianNetwork):
            new_model = DiscreteBayesianNetwork(list(loaded_model.edges()))
            for cpd in loaded_model.cpds:
                new_model.add_cpds(cpd)
            model = new_model
            print(f"✅ Converted to DiscreteBayesianNetwork")
        else:
            model = loaded_model

        return True

    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

# Load the model when the module is imported
if not load_model():
    print("❌ Failed to load Bayesian model.")
else:
    print("✅ Bayesian model loaded successfully.")

# ==================== Helpers ====================

def closest_valid(value, valid_values):
    """Map a user value to the closest value the model was trained on."""
    value = int(value)
    if not valid_values or value in valid_values:
        return value
    return min(valid_values, key=lambda x: abs(x - value))


def build_evidence(data):
    """Convert request JSON into Bayesian Network evidence dict."""
    return {
        'Previous fatty liver (0=no, 1=yes)': closest_valid(
            data.get('fattyLiver', 0),
            valid_states.get('Previous fatty liver (0=no, 1=yes)', [0, 1])
        ),
        'Previous hypertension(0=no, 1=yes)': closest_valid(
            data.get('hypertension', 0),
            valid_states.get('Previous hypertension(0=no, 1=yes)', [0, 1])
        ),
        'Previous diabetes(0=no, 1=yes)': closest_valid(
            data.get('diabetes', 0),
            valid_states.get('Previous diabetes(0=no, 1=yes)', [0, 1])
        ),
        'Waist circumference(cm)': closest_valid(
            data.get('waistCircumference', 75),
            valid_states.get('Waist circumference(cm)', [])
        ),
        'Systolic blood pressure(mmHg)': closest_valid(
            data.get('systolicBP', 120),
            valid_states.get('Systolic blood pressure(mmHg)', [])
        ),
        'Diastolic blood pressure(mmHg)': closest_valid(
            data.get('diastolicBP', 80),
            valid_states.get('Diastolic blood pressure(mmHg)', [])
        ),
    }

# ==================== Prediction ====================

def get_probability(evidence):
    """
    Run Variable Elimination on the Bayesian Network.
    Returns the raw probability of MetS = 1 (yes).
    """
    if model is None:
        raise RuntimeError("Model not loaded")

    inference = VariableElimination(model)
    result = inference.query(
        variables=['Metabolic syndrome(0=no, 1=yes)'],
        evidence=evidence,
    )

    # index 0 = P(no), index 1 = P(yes)
    probability = float(result.values[1])
    return probability

# ==================== Routes ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Input:  { fattyLiver, hypertension, diabetes, waistCircumference, systolicBP, diastolicBP }
    Output: { probability: 0.xxxx }   ← raw float, no thresholds applied here
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        print(f"\n📥 Predict request: {data}")

        evidence = build_evidence(data)
        probability = get_probability(evidence)

        print(f"✅ Probability: {probability:.4f} ({probability * 100:.2f}%)\n")

        return jsonify({'probability': round(probability, 4)})

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== Startup ====================

# if __name__ == '__main__':
#     print('=' * 50)
#     print('🐍 MetS Bayesian Network Prediction Service')
#     print('=' * 50)

#     load_model()

#     print(f'   GET  /health')
#     print(f'   POST /predict')
#     print('=' * 50)

#     port = int(os.environ.get('PORT', 5001))

#     print(f"🚀 Starting server on port {port}...")

#     app.run(host='0.0.0.0', port=port)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.getenv("FLASK_ENV", "development").lower() == "development"
    print(f"🚀 Starting Prediction Service on port {port} (debug={debug})...")
    app.run(host="0.0.0.0", port=port, debug=debug)