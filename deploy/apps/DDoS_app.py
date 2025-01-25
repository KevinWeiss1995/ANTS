import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import StandardScaler
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

repo = get_git_repo_root()

# Load the trained model
model_path = "/Users/kweiss/git/cyber/ANTS/results/models/DDoS_RandomForest.pkl"
print("MODEL PATH: ", model_path)
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}") 
    model = None

# Load the scaler used during training
scaler_path = "/Users/kweiss/git/cyber/ANTS/results/models/DDoS_RandomForest_scaler.pkl"
print("SCALER PATH: ", scaler_path)
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}")
    scaler = None

# Add this after loading the scaler but before the preprocess_input function
# print("Scaler feature names:", scaler.feature_names_in_)

# Preprocessing function adapted from preprocessing script
def preprocess_input(input_data):
    try:
        # Parse input JSON data
        data = json.loads(input_data)
        features = data['features']
        print("Raw Input Features:", features)  # Log raw features

        # Updated columns to match the scaler's exact feature order
        columns = [
            "Bwd Packet Length Std",    # Changed order
            "Flow Bytes/s",
            "Total Length of Fwd Packets",
            "Fwd Packet Length Std",    # Changed order
            "Flow IAT Std",             # Changed order
            "Flow IAT Min",             # Changed order
            "Fwd IAT Total"
        ]
        df = pd.DataFrame([features], columns=columns)

        # Handle missing or invalid values
        df = df.fillna(0)
        for col in ["Flow Bytes/s"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Check if the scaler is loaded
        if scaler is not None:
            scaled_features = scaler.transform(df)
            print("Preprocessed Features:", scaled_features)  # Log preprocessed features
            return scaled_features
        else:
            print("Warning: Scaler not found, returning raw features.")
            return df.to_numpy()
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

# Endpoint to predict using the model
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    input_data = request.data.decode("utf-8")
    
    # Preprocess the input data
    feature_array = preprocess_input(input_data)
    if feature_array is None:
        return jsonify({"error": "Invalid input data format or preprocessing error"}), 400

    # Make prediction
    try:
        prediction = model.predict(feature_array)
        prediction_result = int(prediction[0])  # Assuming single value prediction
        return jsonify({"prediction": prediction_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)