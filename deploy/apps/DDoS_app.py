import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

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

model_path = "/Users/kweiss/git/ANTS/results/models/DDoS_RandomForest.pkl"
print("MODEL PATH: ", model_path)
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None

scaler_path = "/Users/kweiss/git/ANTS/results/models/DDoS_RandomForest_scaler.pkl"
print("SCALER PATH: ", scaler_path)
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}")
    scaler = None

print("Scaler feature names:", scaler.feature_names_in_)

def preprocess_input(input_data):
    try:
        data = json.loads(input_data)
        features = data['features']
        print("Raw Input Features:", features)  # Log raw features

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


        df = df.fillna(0)
        for col in ["Flow Bytes/s"]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if scaler is not None:
            scaled_features = scaler.transform(df)
            print("Preprocessed Features:", scaled_features)  
            return scaled_features
        else:
            print("Warning: Scaler not found, returning raw features.")
            return df.to_numpy()
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    input_data = request.data.decode("utf-8")
    
    feature_array = preprocess_input(input_data)
    if feature_array is None:
        return jsonify({"error": "Invalid input data format or preprocessing error"}), 400

    try:
        prediction = model.predict(feature_array)
        prediction_result = int(prediction[0]) 
        return jsonify({"prediction": prediction_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
