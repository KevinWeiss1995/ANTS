import os
import json
import joblib
import numpy as np
import pandas as pd
import subprocess
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

class SelfAttentionWithGate(nn.Module):
    def __init__(self, c_qkv, c_hidden, num_heads, inf, chunk_size=None):
        super(SelfAttentionWithGate, self).__init__()
        self.c_qkv = c_qkv
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.inf = inf
        self.chunk_size = chunk_size
        self.linear_q = nn.Linear(c_qkv, c_hidden * num_heads, bias=False)
        self.linear_k = nn.Linear(c_qkv, c_hidden * num_heads, bias=False)
        self.linear_v = nn.Linear(c_qkv, c_hidden * num_heads, bias=False)
        self.linear_o = nn.Linear(c_hidden * num_heads, c_qkv, bias=True)
        self.linear_g = nn.Linear(c_qkv, c_hidden * num_heads, bias=True)

    def forward(self, input_qkv, mask, bias=None):
        q = self.linear_q(input_qkv)
        k = self.linear_k(input_qkv)
        v = self.linear_v(input_qkv)
        
        q = q.view(q.shape[:-1] + (self.num_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.num_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.num_heads, self.c_hidden))
        
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        q /= (self.c_hidden ** 0.5)
        
        key = torch.swapdims(k, -2, -1)
        a = torch.matmul(q, key)
        mask = mask.expand_as(a)
        a += (mask - 1.0) * self.inf
        
        if bias is not None:
            a += bias
        
        a = torch.softmax(a, dim=-1)
        a = torch.matmul(a, v)
        
        a = a.transpose(-2, -3)
        gate = torch.sigmoid(self.linear_g(input_qkv))
        gate = gate.view(gate.shape[:-1] + (self.num_heads, self.c_hidden))
        a = a * gate
        a = a.reshape(a.shape[:-2] + (self.num_heads * self.c_hidden,))
        return self.linear_o(a)

class DropoutRowwise(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutRowwise, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = list(x.shape)
        device = x.device
        mask_shape = [shape[0], 1]
        if len(shape) > 2:
            mask_shape.extend([1] * (len(shape) - 2))
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=device))
        mask = mask.expand_as(x)
        return x * mask / (1 - self.p)

class DropoutColumnwise(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutColumnwise, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = list(x.shape)
        device = x.device
        mask_shape = [1, shape[1]]
        if len(shape) > 2:
            mask_shape.extend([1] * (len(shape) - 2))
        mask = torch.bernoulli(torch.full(mask_shape, 1 - self.p, device=device))
        mask = mask.expand_as(x)
        return x * mask / (1 - self.p)

class DDoSNet(nn.Module):
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        
        self.hidden_dim = 64
        self.num_heads = 4
        self.attention_dim = 32
        self.inf = 1e9
        
        self.input_projection = nn.Linear(input_size, self.hidden_dim)
        self.self_attention = SelfAttentionWithGate(
            c_qkv=self.hidden_dim,
            c_hidden=self.attention_dim,
            num_heads=self.num_heads,
            inf=self.inf,
            chunk_size=None
        )
        
        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            DropoutRowwise(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            DropoutColumnwise(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        attention_mask = torch.ones(
            (batch_size, self.num_heads, 1, 1),
            device=x.device
        )
        
        x = self.self_attention(
            input_qkv=x,
            mask=attention_mask,
            bias=None
        )
        
        x = x.squeeze(1)
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