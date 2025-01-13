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
scaler_path = os.path.join(repo_root, 'results', 'models', 'DDoS_RandomForest_scaler.pkl')

# Load the XGBoost model
try:
    with open(xgboost_model_path, 'rb') as model_file:
        xgb_model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: XGBoost model file not found at {xgboost_model_path}")
    xgb_model = None

# Load the DDoS Random Forest model
try:
    with open(random_forest_model_path, 'rb') as model_file:
        rf_model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: DDoS Random Forest model file not found at {random_forest_model_path}")
    rf_model = None

# Load the scaler used during training for DDoS Random Forest
try:
    with open(scaler_path, 'rb') as scaler_file:
        rf_scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}")
    rf_scaler = None

@app.route('/predict/phishing', methods=['POST'])
def predict_xgboost():
    if xgb_model is None:
        return jsonify({"error": "Phishing model not loaded"}), 500

    input_data = request.json
    input_df = pd.DataFrame(input_data)

    # Make predictions using the XGBoost model
    predictions = xgb_model.predict(input_df)

    return jsonify({'predictions': predictions.tolist()})

@app.route('/predict/DDoS', methods=['POST'])
def predict_random_forest():
    if rf_model is None:
        return jsonify({"error": "DDoS model not loaded"}), 500

    input_data = request.json
    input_df = pd.DataFrame(input_data)

    # Scale the input data using the loaded scaler
    if rf_scaler is not None:
        scaled_data = rf_scaler.transform(input_df)
    else:
        return jsonify({"error": "Scaler not loaded"}), 500

    # Make predictions using the Random Forest model
    predictions = rf_model.predict(scaled_data)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 