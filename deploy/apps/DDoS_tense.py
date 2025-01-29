import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import torch
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

# Add the neural network class definition - must match training exactly
class DDoSNet(torch.nn.Module):
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
        
    def forward(self, x):
        return self.layers(x)

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
model_path = os.path.join(repo, "results", "models", "DDoS_NeuralNetwork_v2.pt")
print("MODEL PATH: ", model_path)
try:
    # Initialize model architecture
    input_size = 7  # Number of features
    model = DDoSNet(input_size)
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}") 
    model = None

# Load the scaler used during training
scaler_path = os.path.join(repo, "results", "models", "DDoS_NeuralNetwork_scaler_v2.pkl")
print("SCALER PATH: ", scaler_path)
try:
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}")
    scaler = None

def preprocess_input(input_data):
    try:
        # Parse input JSON data
        data = json.loads(input_data)
        features = data['features']
        print("Raw Input Features:", features)

        # Updated columns to match the scaler's exact feature order
        columns = [
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Total Length of Fwd Packets",
            "Fwd Packet Length Std",
            "Flow IAT Std",
            "Flow IAT Min",
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
            print("Preprocessed Features:", scaled_features)
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
        # Convert to PyTorch tensor
        input_tensor = torch.FloatTensor(feature_array)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction_result = int(predicted.item())  # Convert to Python int
            
        return jsonify({"prediction": prediction_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)