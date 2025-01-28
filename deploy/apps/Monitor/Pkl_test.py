import os
import pickle
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(f"### BASE PATH ####:{base_path}")
model_path = os.path.join(base_path, "../", "results", "models", "DDoS_RandomForest.pkl")
scaler_path = os.path.join(base_path, "../", "results", "models", "DDoS_RandomForest_scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

app = Flask(__name__)

def process_packet(packet):
    try:
        length = int(packet.length)  # Extract length from the packet

        features = {
            'Bwd Packet Length Std': length,         # Replace with actual calculation
            'Flow Bytes/s': 10000.0,                # Replace with computed value
            'Total Length of Fwd Packets': 26.0,    # Replace with computed value
            'Fwd Packet Length Std': 10.0,          # Replace with computed value
            'Flow IAT Std': 400000.0,               # Replace with computed value
            'Flow IAT Min': 2.0,                    # Replace with computed value
            'Fwd IAT Total': 1000.0                 # Replace with computed value
        }

        return {
            'timestamp': float(packet.sniff_timestamp),
            'src_ip': packet.ip.src,
            'dst_ip': packet.ip.dst,
            'protocol': packet.transport_layer,
            'length': length,
            'features': features
        }
    except AttributeError:
        return None

def process_packet_data(packets):
    """
    Converts raw packet data into a NumPy array of features for the model.
    """
    try:
        # Extract features for each packet
        processed_data = []
        for packet in packets:
            processed_data.append([
                packet['Bwd Packet Length Std'],
                packet['Flow Bytes/s'],
                packet['Total Length of Fwd Packets'],
                packet['Fwd Packet Length Std'],
                packet['Flow IAT Std'],
                packet['Flow IAT Min'],
                packet['Fwd IAT Total']
            ])

        # Convert to NumPy array
        return np.array(processed_data)
    except Exception as e:
        print(f"Error in process_packet_data: {e}")
        raise


@app.route("/analyze_packets", methods=["POST"])
def analyze_packets():
    try:
        # Extract the features key from the payload
        if 'features' in request.json:
            packets = request.json['features']  # Extract the list of features
        else:
            return jsonify({"error": "No valid 'features' key found in the payload"}), 400

        # Process and validate features
        features = process_packet_data(packets)
        print("Features shape:", features.shape)  # Debugging feature shape
        
        scaled_features = scaler.transform(features)
        print("Scaled features shape:", scaled_features.shape)  # Debugging scaled shape
        
        predictions = model.predict(scaled_features)
        results = ["BENIGN" if pred == 1 else "ATTACK" for pred in predictions]

        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
