import requests

API_URL = "http://localhost:8000/predict"
payload = {"features": [0]*80}
response = requests.post(API_URL, json=payload)
print(response.status_code)
print(response.text)
