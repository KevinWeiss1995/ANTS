import requests

# Define the URL for the prediction endpoint
url = "http://localhost:5001/predict/phishing"

# Sample input data for the phishing XGBoost model
# This should be a list of feature values in the correct order
sample_data = [
    [
        1,  # having_IP_Address
        0,  # URL_Length
        -1, # Shortining_Service
        1,  # having_At_Symbol
        1,  # double_slash_redirecting
        -1, # Prefix_Suffix
        1,  # having_Sub_Domain
        1,  # SSLfinal_State
        1   # Domain_registeration_length
    ]
]

# Send a POST request to the prediction endpoint
response = requests.post(url, json=sample_data)

# Check the response
if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)  # Changed to response.text for better error visibility
