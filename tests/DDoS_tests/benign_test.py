import json
import requests
import pickle
import os
import numpy as np

# Get the path to the saved model
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest.pkl")
scaler_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest_scaler.pkl")

api_url = "http://127.0.0.1:5001/predict"  

# Updated benign features to match real benign patterns
benign_data = {
    "features": [
        0.0,            # Bwd Packet Length Std (no variance)
        3000000.0,      # Flow Bytes/s (very high)
        12.0,           # Total Length of Fwd Packets (consistent)
        0.0,            # Fwd Packet Length Std (no variance)
        0.0,            # Flow IAT Std (consistent timing)
        4.0,            # Flow IAT Min (small)
        4.0            # Fwd IAT Total (small)
    ]
}

# First, test the API endpoint
print("Testing API endpoint...")
response = requests.post(api_url, json=benign_data)

# Check the response status
if response.status_code == 200:
    print("API Prediction Response:", response.json())
else:
    print(f"API Error {response.status_code}: {response.text}")

# Then, test the model directly
print("\nTesting model directly...")
try:
    # Load the model and scaler
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Prepare the features
    X_test = np.array(benign_data['features']).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    
    # Get prediction
    prediction = clf.predict(X_test_scaled)
    print("Model prediction:", "BENIGN" if prediction[0] == 1 else "ATTACK")
    
    # Get prediction probabilities if available
    if hasattr(clf, 'predict_proba'):
        probas = clf.predict_proba(X_test_scaled)
        print("Prediction probabilities:", probas[0])
        
except FileNotFoundError:
    print(f"Error: Could not find model files. Please ensure the model has been trained and saved.")
except Exception as e:
    print(f"Error during model testing: {str(e)}")
