import pyshark
import pickle
import time
import os
import csv

# Function to extract packet details and compute model features
def process_packet(packet):
    try:
        length = int(packet.length)  # Extract length from the packet

        # Compute features (adjust these calculations as necessary)
        features = {
            'Bwd Packet Length Std': length,         # Use actual computation
            'Flow Bytes/s': 10000.0,                # Replace with computed value
            'Total Length of Fwd Packets': 26.0,    # Replace with computed value
            'Fwd Packet Length Std': 10.0,          # Replace with computed value
            'Flow IAT Std': 400000.0,               # Replace with computed value
            'Flow IAT Min': 2.0,                    # Replace with computed value
            'Fwd IAT Total': 1000.0                 # Replace with computed value
        }

        # Include 'length' as part of the features
        features['length'] = length

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

# Function to save data to CSV
def save_to_csv(data, csv_file):
    with open(csv_file, 'w', newline='') as file:
        fieldnames = ['timestamp', 'src_ip', 'dst_ip', 'protocol', 'length']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Filter out the 'features' key
        filtered_data = [
            {key: packet[key] for key in fieldnames if key in packet}
            for packet in data
        ]
        writer.writerows(filtered_data)
    print(f"Saved packet data to {csv_file}")

# Function to save features for the model
def save_features_to_pickle(data, pkl_file):
    # Ensure each feature dictionary includes 'length'
    features = [packet['features'] for packet in data if 'features' in packet and 'length' in packet['features']]
    with open(pkl_file, 'wb') as file:
        pickle.dump(features, file)
    print(f"Saved model features to {pkl_file}")

# Function to capture and save packets
def capture_packets(interface, duration):
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file paths
    pkl_file = os.path.join(output_dir, 'packets.pkl')
    csv_file = os.path.join(output_dir, 'packets.csv')

    print(f"Starting capture on interface {interface} for {duration} seconds...")
    capture = pyshark.LiveCapture(interface=interface)
    packet_list = []

    start_time = time.time()
    for packet in capture.sniff_continuously():
        
        if time.time() - start_time > duration:
            break

        processed = process_packet(packet)
        if processed:
            packet_list.append(processed)

    # Save packets to a pickle file (features only)
    save_features_to_pickle(packet_list, pkl_file)

    # Save full packet data to a CSV file
    save_to_csv(packet_list, csv_file)

# Usage example
if __name__ == "__main__":
    capture_packets(interface='en0', duration=10)
