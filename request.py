import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url, json={'experience':10, 'test_score':7, 'interview_score':9})

print(r.json())
