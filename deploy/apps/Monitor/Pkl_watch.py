import os
import time
import requests
import pickle

# Directory and API settings
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
WATCH_DIRECTORY = os.path.join(base_path, "apps", "Monitor", "output")
TARGET_FILE = 'packets.pkl'
API_URL = 'http://127.0.0.1:5002/analyze_packets'

def send_to_api(file_path):
    """
    Sends the content of a .pkl file to the REST API for analysis.
    """
    try:
        with open(file_path, 'rb') as file:
            packets = pickle.load(file)

        # Wrap the packets under the 'features' key
        payload = {'features': packets}

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            print(f"API Response: {response.json()}")
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error sending file {file_path} to API: {e}")


def monitor_file(file_path):
    """
    Monitors a file for modifications and sends it to the API when updated.
    """
    last_modified_time = None
    print(f"Monitoring file: {file_path}")

    while True:
        try:
            current_modified_time = os.path.getmtime(file_path)
            if last_modified_time is None or current_modified_time > last_modified_time:
                print(f"Detected file update: {file_path}")
                send_to_api(file_path)
                last_modified_time = current_modified_time
        except FileNotFoundError:
            print(f"Waiting for file: {file_path}")
        except Exception as e:
            print(f"Error monitoring file: {e}")

        time.sleep(5)  # Check for changes every 5 seconds

if __name__ == '__main__':
    target_file_path = os.path.join(WATCH_DIRECTORY, TARGET_FILE)
    monitor_file(target_file_path)
