import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load the saved Keras model
model_path = os.path.join(project_root, 'results', 'models', 'network', 'network_binary_classifier.keras')
model = keras.models.load_model(model_path)

# Real network traffic data
sample_data = {
    "Fwd Packet Length Max": 48.0,
    "Fwd Packet Length Min": 44.0,
    "Bwd Packet Length Min": 0.0,
    "Flow Bytes/s": 10754.081866869268,
    "Flow IAT Mean": 0.004091866862033478,
    "Flow IAT Min": 7.867813110351562e-06,
    "Fwd IAT Total": 45.98439979553223,
    "Fwd IAT Mean": 0.004091866862033478,
    "Fwd IAT Min": 7.867813110351562e-06,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0,
    "Bwd Header Length": 0,
    "Fwd Packets/s": 244.40897456471671,
    "Bwd Packets/s": 0.0,
    "Min Packet Length": 44,
    "Max Packet Length": 48,
    "Packet Length Mean": 44.00035590355014,
    "Packet Length Std": 0.037729133745881965,
    "Packet Length Variance": 0.001423487533214649,
    "FIN Flag Count": 0,
    "SYN Flag Count": 5620,
    "RST Flag Count": 5619,
    "PSH Flag Count": 0,
    "ACK Flag Count": 5619,
    "URG Flag Count": 0,
    "CWE Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.0,
    "Average Packet Size": 44.00035590355014,
    "Avg Fwd Segment Size": 44.00035590355014,
    "Avg Bwd Segment Size": 0,
    "Fwd Header Length": 44956
}

# Convert to numpy array
input_array = np.array([list(sample_data.values())])

# Make prediction
probability = model.predict(input_array)[0][0]
is_malicious = probability > 0.5

print("\nAnalyzing Network Traffic Sample:")
print(f"Probability of malicious traffic: {probability:.2%}")
print(f"Classification: {'Malicious' if is_malicious else 'Benign'}")

# Highlight suspicious indicators
print("\nSuspicious Indicators:")
if sample_data["SYN Flag Count"] > 100:
    print(f"- High SYN Count: {sample_data['SYN Flag Count']} (Possible SYN Flood)")
if sample_data["RST Flag Count"] > 100:
    print(f"- High RST Count: {sample_data['RST Flag Count']} (Possible RST Flood)")
if sample_data["Flow Bytes/s"] > 10000:
    print(f"- High Flow Rate: {sample_data['Flow Bytes/s']:.2f} bytes/s")
if sample_data["Flow IAT Min"] < 0.0001:
    print(f"- Very small Inter-Arrival Time: {sample_data['Flow IAT Min']}") 