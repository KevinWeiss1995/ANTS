import json
import requests
import pickle
import os
import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest.pkl")
scaler_path = os.path.join(base_path, "results", "models", "DDoS_RandomForest_scaler.pkl")

api_url = "http://127.0.0.1:5000/predict/DDoS"  

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

print("Testing API endpoint...")
response = requests.post(api_url, json=attack_data)

if response.status_code == 200:
    print("API Prediction Response:", response.json())
else:
    print(f"API Error {response.status_code}: {response.text}")

print("\nTesting model directly...")
try:
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    X_test = np.array(attack_data['features']).reshape(1, -1)
    X_test_scaled = scaler.transform(X_test)
    
    if hasattr(clf, 'custom_predict'):
        prediction = clf.custom_predict(X_test_scaled)
    else:
        prediction = clf.predict(X_test_scaled)
    print("Model prediction:", "BENIGN" if prediction[0] == 1 else "ATTACK")

    if hasattr(clf, 'predict_proba'):
        probas = clf.predict_proba(X_test_scaled)
        print("Prediction probabilities:", probas[0])
        
except FileNotFoundError:
    print(f"Error: Could not find model files. Please ensure the model has been trained and saved.")
except Exception as e:
    print(f"Error during model testing: {str(e)}")
