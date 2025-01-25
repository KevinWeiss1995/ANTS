import os
import subprocess
import pickle
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import requests
import datetime

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
        phish_model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: XGBoost model file not found at {xgboost_model_path}")
    phish_model = None

# Load the DDoS Random Forest model
try:
    with open(random_forest_model_path, 'rb') as model_file:
        DDoS_model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: DDoS Random Forest model file not found at {random_forest_model_path}")
    DDoS_model = None

# Load the scaler used during training for DDoS Random Forest
try:
    with open(scaler_path, 'rb') as scaler_file:
        rf_scaler = pickle.load(scaler_file)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}")
    rf_scaler = None

@app.route('/predict/phishing', methods=['POST'])
def predict_xgboost():
    if phish_model is None:
        return jsonify({"error": "Phishing model not loaded"}), 500

    input_data = request.json
    input_df = pd.DataFrame(input_data)

    # Make predictions using the XGBoost model
    predictions = phish_model.predict(input_df)

    return jsonify({'predictions': predictions.tolist()})

@app.route('/predict/DDoS', methods=['POST'])
def predict_random_forest():
    if DDoS_model is None:
        return jsonify({"error": "DDoS model not loaded"}), 500

    input_data = request.json
    # Create a DataFrame with the correct feature names
    feature_names = [
        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow IAT Min', 
        'Flow IAT Std', 'Fwd IAT Total',  # Add all expected feature names here
        # ... other feature names
    ]
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Scale the input data using the loaded scaler
    if rf_scaler is not None:
        scaled_data = rf_scaler.transform(input_df)
    else:
        return jsonify({"error": "Scaler not loaded"}), 500

    # Make predictions using the Random Forest model
    predictions = DDoS_model.predict(scaled_data)

    return jsonify({'predictions': predictions.tolist()})

def test_prediction(features, expected_type, results_dict):
    """Test prediction and update results dictionary"""
    data = {"features": features}
    response = requests.post(api_url, json=data)
    
    # Create test result dictionary
    test_result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "test_type": expected_type,
        "features": features,
        "expected": expected_type.upper(),
        "api_response": None,
        "model_response": None
    }
    
    print("\n" + "="*50)
    print(f"Testing {expected_type.upper()} pattern...")
    print("-"*50)
    
    # Track predictions
    api_prediction = None
    model_prediction = None
    
    # Test API
    if response.status_code == 200:
        api_result = response.json()
        raw_prediction = api_result.get('prediction', 'Unknown')
        api_prediction = "BENIGN" if str(raw_prediction) == "1" else "ATTACK"
        probability = api_result.get('probability')
        
        test_result["api_response"] = {
            "status_code": response.status_code,
            "prediction": api_prediction,
            "probability": probability if probability is not None else "N/A"
        }
        
        print(f"API Prediction: {api_prediction}")
        if probability is not None:
            print(f"Confidence: {float(probability):.2%}")
        else:
            print("Confidence: N/A")
    else:
        print(f"❌ API Error {response.status_code}: {response.text}")
        test_result["api_response"] = {
            "status_code": response.status_code,
            "error": response.text
        }
    
    try:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Ensure the input features are in the correct shape
        X_test = np.array(features).reshape(1, -1)  # Reshape to 2D array
        X_test_scaled = scaler.transform(X_test)
        
        prediction = clf.predict(X_test_scaled)
        model_prediction = "BENIGN" if prediction[0] == 1 else "ATTACK"
        
        if hasattr(clf, 'predict_proba'):
            probas = clf.predict_proba(X_test_scaled)
            attack_prob = probas[0][0]  # Probability of ATTACK
            benign_prob = probas[0][1]  # Probability of BENIGN
            print(f"Probability of ATTACK: {attack_prob:.2%}")
            print(f"Probability of BENIGN: {benign_prob:.2%}")
            
            test_result["model_response"] = {
                "prediction": model_prediction,
                "probability_attack": float(attack_prob),
                "probability_benign": float(benign_prob)
            }
        
        print(f"\nDirect Model Prediction: {model_prediction}")
            
    except Exception as e:
        print(f"❌ Model Error: {str(e)}")
        test_result["model_response"] = {
            "error": str(e)
        }
    
    # Update results
    expected_prediction = "BENIGN" if expected_type.lower() == "benign" else "ATTACK"
    if api_prediction:
        results_dict['total_api_predictions'] += 1
        if api_prediction == expected_prediction:
            results_dict['correct_api_predictions'] += 1
    
    if model_prediction:
        results_dict['total_model_predictions'] += 1
        if model_prediction == expected_prediction:
            results_dict['correct_model_predictions'] += 1
    
    return test_result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 