import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

# Get project root (adjusted for new location)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load the saved Keras model
model_path = os.path.join(project_root, 'results', 'models', 'network', 'network_binary_classifier.keras')
print(f"Looking for model at: {model_path}")  # Debug print
model = keras.models.load_model(model_path)

# Sample 1: Normal Traffic (revised to be more realistic)
normal_traffic = {
    "Fwd Packet Length Max": 1280.0,
    "Fwd Packet Length Min": 40.0,
    "Bwd Packet Length Min": 40.0,
    "Flow Bytes/s": 800.0,
    "Flow IAT Mean": 0.05,
    "Flow IAT Min": 0.001,
    "Fwd IAT Total": 10.0,
    "Fwd IAT Mean": 0.05,
    "Fwd IAT Min": 0.001,
    "Bwd IAT Total": 10.0,
    "Bwd IAT Mean": 0.05,
    "Bwd IAT Std": 0.01,
    "Bwd IAT Max": 0.1,
    "Bwd IAT Min": 0.001,
    "Fwd PSH Flags": 1,
    "Bwd Header Length": 320,
    "Fwd Packets/s": 20.0,
    "Bwd Packets/s": 15.0,
    "Min Packet Length": 40,
    "Max Packet Length": 1280,
    "Packet Length Mean": 500.0,
    "Packet Length Std": 300.0,
    "Packet Length Variance": 90000.0,
    "FIN Flag Count": 1,
    "SYN Flag Count": 1,
    "RST Flag Count": 0,
    "PSH Flag Count": 1,
    "ACK Flag Count": 10,
    "URG Flag Count": 0,
    "CWE Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.75,
    "Average Packet Size": 500.0,
    "Avg Fwd Segment Size": 600.0,
    "Avg Bwd Segment Size": 400.0,
    "Fwd Header Length": 320
}

# Sample 2: Attack Traffic (high SYN count indicating potential SYN flood)
attack_traffic = {
    "Fwd Packet Length Max": 50.0,
    "Fwd Packet Length Min": 44.0,
    "Bwd Packet Length Min": 0.0,
    "Flow Bytes/s": 12022.688875459973,
    "Flow IAT Mean": 0.0039095304696718835,
    "Flow IAT Min": 7.867813110351562e-06,
    "Fwd IAT Total": 59.88227820396423,
    "Fwd IAT Mean": 0.0039095304696718835,
    "Fwd IAT Min": 7.867813110351562e-06,
    "Bwd IAT Total": 0.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Bwd IAT Min": 0.0,
    "Fwd PSH Flags": 0,
    "Bwd Header Length": 0,
    "Fwd Packets/s": 255.80189096723348,
    "Bwd Packets/s": 0.0,
    "Min Packet Length": 44,
    "Max Packet Length": 50,
    "Packet Length Mean": 47.0,
    "Packet Length Std": 3.0,
    "Packet Length Variance": 9.0,
    "FIN Flag Count": 0,
    "SYN Flag Count": 7659,
    "RST Flag Count": 7659,
    "PSH Flag Count": 0,
    "ACK Flag Count": 7659,
    "URG Flag Count": 0,
    "CWE Flag Count": 0,
    "ECE Flag Count": 0,
    "Down/Up Ratio": 0.0,
    "Average Packet Size": 47.0,
    "Avg Fwd Segment Size": 47.0,
    "Avg Bwd Segment Size": 0.0,
    "Fwd Header Length": 61272
}

# Convert both samples to numpy arrays
normal_array = np.array([list(normal_traffic.values())])
attack_array = np.array([list(attack_traffic.values())])

# Make predictions
print("\nAnalyzing Normal Traffic Sample:")
normal_prob = model.predict(normal_array)[0][0]
print(f"Probability of malicious traffic: {normal_prob:.2%}")
print(f"Classification: {'Malicious' if normal_prob > 0.5 else 'Benign'}")

print("\nAnalyzing Attack Traffic Sample:")
attack_prob = model.predict(attack_array)[0][0]
print(f"Probability of malicious traffic: {attack_prob:.2%}")
print(f"Classification: {'Malicious' if attack_prob > 0.5 else 'Benign'}")

# Print suspicious indicators for attack traffic
print("\nSuspicious Indicators in Attack Traffic:")
if attack_traffic["SYN Flag Count"] > 100:
    print(f"- High SYN Count: {attack_traffic['SYN Flag Count']} (Possible SYN Flood)")
if attack_traffic["RST Flag Count"] > 100:
    print(f"- High RST Count: {attack_traffic['RST Flag Count']} (Possible RST Flood)")
if attack_traffic["Flow Bytes/s"] > 10000:
    print(f"- High Flow Rate: {attack_traffic['Flow Bytes/s']:.2f} bytes/s")
if attack_traffic["Flow IAT Min"] < 0.0001:
    print(f"- Very small Inter-Arrival Time: {attack_traffic['Flow IAT Min']}") 