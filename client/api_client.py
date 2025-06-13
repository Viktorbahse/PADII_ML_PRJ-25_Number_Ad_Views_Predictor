import requests

BASE_URL = "http://localhost:8000"

def set_active_model(model_name: str):
    response = requests.post(f"{BASE_URL}/set/{model_name}")
    return response.json()

def predict_one_item(data):
    response = requests.post(f"{BASE_URL}/predict_one_item", json=data)
    return response.json()

def predict_csv(file):
    files = {'file': file}
    response = requests.post(f"{BASE_URL}/predict_csv", files=files)
    return response

def get_models():
    response = requests.get(f"{BASE_URL}/models")
    return response.json()