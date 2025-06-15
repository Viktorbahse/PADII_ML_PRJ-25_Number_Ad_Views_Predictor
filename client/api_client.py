import requests
import logging
from logging.handlers import RotatingFileHandler
import os

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Настройка логирования
log_file = os.path.join(log_dir, "app.log")
handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

BASE_URL = "http://backend:8000"

def set_active_model(model_name: str):
    logger.info(f"Установить модель: {model_name}")
    response = requests.post(f"{BASE_URL}/set/{model_name}")
    return response.json()

def predict_one_item(data):
    logger.info(f"Предсказание для данных: {data}")
    response = requests.post(f"{BASE_URL}/predict_one_item", json=data)
    return response.json()

def predict_csv(file):
    logger.info("Предсказание для набора данных в формате CSV")
    files = {'file': file}
    response = requests.post(f"{BASE_URL}/predict_csv", files=files)
    return response

def get_models():
    logger.info("Выведен список доступных моделей")
    response = requests.get(f"{BASE_URL}/models")
    return response.json()