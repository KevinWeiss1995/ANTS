import os
import subprocess
import pickle
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

def get_git_repo_root():
    """Dynamically determine the Git repository root."""
    try:
        repo_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], stderr=subprocess.STDOUT)
        return repo_root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        print("Error: Not a git repository.")
        return None

# Get the repository root
repo_root = get_git_repo_root()

# Define paths to the models and scalers
xgboost_model_path = os.path.join(repo_root, 'results', 'models', 'phishing_xgboost.pkl')
random_forest_model_path = os.path.join(repo_root, 'results', 'models', 'DDoS_RandomForest.pkl')
rf_scaler_path = os.path.join(repo_root, 'results', 'models', 'DDoS_RandomForest_scaler.pkl')
phishing_scaler_path = os.path.join(repo_root, 'results', 'models', 'phish_scalar.pkl')  # Path for phishing scaler

# Load the XGBoost model
try:
    with open(xgboost_model_path, 'rb') as model_file:
        phish_model = pickle.load(model_file)  # Changed variable name to phish_model
except FileNotFoundError:
    print(f"Error: XGBoost model file not found at {xgboost_model_path}")
    phish_model = None

# Load the DDoS Random Forest model
try:
    with open(random_forest_model_path, 'rb') as model_file:
        DDoS_model = pickle.load(model_file)  # Keeping the existing variable name
except FileNotFoundError:
    print(f"Error: DDoS Random Forest model file not found at {random_forest_model_path}")
    DDoS_model = None

# Load the scalers
try:
    with open(rf_scaler_path, 'rb') as scaler_file:
        rf_scaler = pickle.load(scaler_file)  # Keeping the existing variable name
except FileNotFoundError:
    print(f"Error: Scaler file not found at {rf_scaler_path}")
    rf_scaler = None

try:
    with open(phishing_scaler_path, 'rb') as scaler_file:  # Load phishing scaler
        phish_scaler = pickle.load(scaler_file)  # Changed variable name to phish_scaler
except FileNotFoundError:
    print(f"Error: Phishing scaler file not found at {phishing_scaler_path}")
    phish_scaler = None

@app.route('/predict/phishing', methods=['POST'])
def predict_xgboost():
    if phish_model is None:
        return jsonify({"error": "Phishing model not loaded"}), 500

    input_data = request.json
    input_df = pd.DataFrame(input_data)

    # Scale the input data using the loaded phishing scaler
    if phish_scaler is not None:
        scaled_data = phish_scaler.transform(input_df)
    else:
        return jsonify({"error": "Phishing scaler not loaded"}), 500

    # Make predictions using the XGBoost model
    predictions = phish_model.predict(scaled_data)

    return jsonify({'predictions': predictions.tolist()})

@app.route('/predict/DDoS', methods=['POST'])
def predict_random_forest():
    if DDoS_model is None:
        return jsonify({"error": "DDoS model not loaded"}), 500

    input_data = request.json
    input_df = pd.DataFrame(input_data)

    # Scale the input data using the loaded scaler
    if rf_scaler is not None:
        scaled_data = rf_scaler.transform(input_df)
    else:
        return jsonify({"error": "Scaler not loaded"}), 500

    # Make predictions using the Random Forest model
    predictions = DDoS_model.predict(scaled_data)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)