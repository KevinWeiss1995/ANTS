import sys
import os
import json
import requests
import pickle
import numpy as np
import torch
from pathlib import Path
import warnings

# Get the absolute path to the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    from models.DDoS.DDoS_attention import SelfAttentionWithGate
    from models.DDoS.DDoS_dropout import DropoutRowwise, DropoutColumnwise
except ImportError as e:
    print(f"Error importing modules. Project root: {project_root}")
    print(f"Python path: {sys.path}")
    raise e

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

def list_models(base_path):
    """List all model files in the models directory and let user select one."""
    models_dir = os.path.join(base_path, "results", "models")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pkl', '.pt', '.pth'))]
    
    if not model_files:
        print("No model files found!")
        return None
    
    print("\nAvailable models:")
    for idx, model in enumerate(model_files, 1):
        print(f"{idx}. {model}")
    
    while True:
        try:
            choice = int(input("\nSelect model number: "))
            if 1 <= choice <= len(model_files):
                return os.path.join(models_dir, model_files[choice - 1])
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def is_torch_model(model_path):
    """Check if the selected model is a PyTorch model."""
    return model_path.endswith(('.pt', '.pth'))

def get_corresponding_scaler(model_path, base_path):
    """List scalers and let user select one."""
    models_dir = os.path.join(base_path, "results", "models")
    scaler_files = [f for f in os.listdir(models_dir) if 'scaler' in f.lower() and f.endswith('.pkl')]
    
    if not scaler_files:
        print("No scaler files found!")
        return None
    
    print("\nAvailable scalers:")
    for idx, scaler in enumerate(scaler_files, 1):
        print(f"{idx}. {scaler}")
    
    while True:
        try:
            choice = int(input("\nSelect scaler number: "))
            if 1 <= choice <= len(scaler_files):
                return os.path.join(models_dir, scaler_files[choice - 1])
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# Define the neural network class for PyTorch models
class DDoSNet(torch.nn.Module):
    """Neural Network for DDoS detection with attention and custom dropout."""
    def __init__(self, input_size):
        super(DDoSNet, self).__init__()
        
        # Attention configuration
        self.hidden_dim = 64
        self.num_heads = 4
        self.attention_dim = 32
        self.inf = 1e9
        
        self.input_projection = torch.nn.Linear(input_size, self.hidden_dim)
        self.self_attention = SelfAttentionWithGate(
            c_qkv=self.hidden_dim,
            c_hidden=self.attention_dim,
            num_heads=self.num_heads,
            inf=self.inf,
            chunk_size=None
        )
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 64),
            torch.nn.ReLU(),
            DropoutRowwise(p=0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            DropoutColumnwise(p=0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        
        # Create attention mask
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
        
        x = x.squeeze(1)  # Remove sequence dimension
        return self.layers(x)

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Let user select model and scaler
model_path = list_models(base_path)
if model_path is None:
    print("No models available. Exiting.")
    exit(1)

scaler_path = get_corresponding_scaler(model_path, base_path)
if scaler_path is None:
    print("No scalers available. Exiting.")
    exit(1)

print(f"\nSelected model: {os.path.basename(model_path)}")
print(f"Selected scaler: {os.path.basename(scaler_path)}")

api_url = "http://127.0.0.1:5002/predict"  

attack_data = {
    "features": [
        2500.0,         # Bwd Packet Length Std (high variance)
        10000.0,        # Flow Bytes/s (moderate)
        26.0,           # Total Length of Fwd Packets (small)
        10.0,           # Fwd Packet Length Std (some variance)
        400000.0,       # Flow IAT Std (high variance)
        2.0,            # Flow IAT Min (small)
        1000.0          # Fwd IAT Total (moderate)
    ]
}

print("\nTesting API endpoint...")
response = requests.post(api_url, json=attack_data)

if response.status_code == 200:
    print("API Prediction Response:", response.json())
else:
    print(f"API Error {response.status_code}: {response.text}")

print("\nTesting model directly...")
try:
    # Load model - handle both PyTorch and sklearn models
    is_torch = is_torch_model(model_path)
    if is_torch:
        model = DDoSNet(input_size=7)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Prepare input data
    X_test = np.array(attack_data['features']).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    
    # Make prediction based on model type
    if is_torch:
        with torch.no_grad():
            input_tensor = torch.FloatTensor(X_test_scaled)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction = predicted.item()
            
            # Get prediction probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print("Model prediction:", "BENIGN" if prediction == 1 else "ATTACK")
            print("Prediction probabilities:", probabilities.numpy()[0])
    else:
        prediction = model.predict(X_test_scaled)
        print("Model prediction:", "BENIGN" if prediction[0] == 1 else "ATTACK")
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test_scaled)
            print("Prediction probabilities:", probas[0])
        
except FileNotFoundError:
    print(f"Error: Could not find model files. Please ensure the model has been trained and saved.")
except Exception as e:
    print(f"Error during model testing: {str(e)}")