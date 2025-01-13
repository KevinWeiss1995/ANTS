import json
import requests
import pickle
import os
import numpy as np

# Get the path to the saved model
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest.pkl")
scaler_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest_scaler.pkl")

api_url = "http://127.0.0.1:5001/predict/DDoS"  

# Updated attack features to match real attack patterns
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

# First, test the API endpoint
print("Testing API endpoint...")
response = requests.post(api_url, json=attack_data)

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
    X_test = np.array(attack_data['features']).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    
    # Get prediction with custom threshold
    if hasattr(clf, 'custom_predict'):
        prediction = clf.custom_predict(X_test_scaled)
    else:
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
