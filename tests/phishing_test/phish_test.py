import requests

url = "http://localhost:5001/predict/phishing"
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

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)  # Changed to response.text for better error visibility
