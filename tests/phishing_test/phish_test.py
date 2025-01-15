import requests

url = "http://localhost:5000/predict/phishing"
sample_data = [
    [
    0,  # having_IP_Address (0 indicates absence of an IP address in the URL)
    1,   # URL_Length (long URL length, which is often used in legitimate sites)
    -1,  # Shortining_Service (0 indicates no use of a URL shortening service)
    -1,  # having_At_Symbol (absence of "@" in the URL)
    -1,  # double_slash_redirecting (indicates no double slash in the URL)
    1,   # Prefix_Suffix (presence of prefix or suffix, which is common in legitimate URLs)
    -1,  # having_Sub_Domain (absence of subdomain, often used in legitimate sites)
    -1,  # SSLfinal_State (0 indicates SSL is not present, which is common for non-phishing)
    1    # Domain_registeration_length (long registration length, often used in legitimate domains)
    ]
]

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)  # Changed to response.text for better error visibility
