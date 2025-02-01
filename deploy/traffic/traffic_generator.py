import requests
import json
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def generate_normal_traffic_features():
    """Generate features that simulate normal traffic patterns"""
    return {
        "Bwd Packet Length Std": random.uniform(5, 15),         # Very consistent packet lengths
        "Flow Bytes/s": random.uniform(100, 500),               # Moderate flow rate
        "Total Length of Fwd Packets": random.uniform(200, 800), # Medium-sized packets
        "Fwd Packet Length Std": random.uniform(5, 15),         # Consistent packet lengths
        "Flow IAT Std": random.uniform(1000, 3000),            # Variable timing (human-like)
        "Flow IAT Min": random.uniform(200, 800),              # Regular gaps
        "Fwd IAT Total": random.uniform(5000, 10000)          # Longer sessions
    }

def generate_ddos_traffic_features():
    """Generate features that simulate DDoS traffic patterns"""
    return {
        "Bwd Packet Length Std": random.uniform(50, 200),       # Highly variable
        "Flow Bytes/s": random.uniform(8000, 20000),           # Very high throughput
        "Total Length of Fwd Packets": random.uniform(2000, 5000), # Large packets
        "Fwd Packet Length Std": random.uniform(40, 100),       # Variable sizes
        "Flow IAT Std": random.uniform(1, 50),                 # Very consistent timing
        "Flow IAT Min": random.uniform(1, 20),                 # Minimal gaps
        "Fwd IAT Total": random.uniform(50, 300)              # Short, intense bursts
    }

def send_traffic(features):
    """Send traffic features to the model endpoint"""
    url = "http://localhost:5001/predict"
    payload = json.dumps({"features": features})
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            result = response.json()
            # Print raw features and prediction for debugging
            print("\nRaw Features Sent:")
            for k, v in features.items():
                print(f"  {k}: {v:.4f}")
            print(f"Prediction: {result}")
            return result["prediction"]
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return None

def simulate_traffic(traffic_type="normal", num_requests=1):
    """Simulate either normal or DDoS traffic"""
    generator = generate_normal_traffic_features if traffic_type == "normal" else generate_ddos_traffic_features
    features = generator()
    result = send_traffic(features)
    
    if result is not None:
        print("\n" + "="*50)
        print(f"Traffic Type Sent: {traffic_type.upper()}")
        print(f"Features:")
        for k, v in features.items():
            print(f"  {k}: {v:.2f}")
        print(f"Model Prediction: {'DDoS' if result == 0 else 'Normal'} ({result})")
        print("="*50)
    
    return result

def run_traffic_simulation(duration=60, max_workers=5):
    """Run a mixed traffic simulation for a specified duration"""
    print(f"Starting traffic simulation for {duration} seconds...")
    
    start_time = time.time()
    ddos_detections = 0
    normal_detections = 0
    total_requests = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while time.time() - start_time < duration:
            # Randomly choose between normal and DDoS traffic
            traffic_type = random.choice(["normal", "ddos"])
            
            # Submit the traffic simulation to the thread pool
            future = executor.submit(simulate_traffic, traffic_type)
            result = future.result()
            
            if result is not None:
                total_requests += 1
                if result == 1:
                    ddos_detections += 1
                else:
                    normal_detections += 1
            
            # Random delay between requests (0.5 to 2 seconds)
            time.sleep(random.uniform(0.5, 2))
    
    print("\nSimulation Complete!")
    print(f"Total Requests: {total_requests}")
    print(f"DDoS Detections: {ddos_detections}")
    print(f"Normal Traffic Detections: {normal_detections}")
    print(f"DDoS Detection Rate: {(ddos_detections/total_requests)*100:.2f}%")

if __name__ == "__main__":
    # Run the simulation for 60 seconds with 5 concurrent connections
    run_traffic_simulation(duration=60, max_workers=5) 