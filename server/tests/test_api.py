import pytest
from fastapi.testclient import TestClient
from app import app
from classes import DummyModelFirst, DummyModelSecond
import io
import pandas as pd

app.state.models = {
    "DummyModelFirst": DummyModelFirst(),
    "DummyModelSecond": DummyModelSecond()
}
app.state.active_model = None

client = TestClient(app)

def test_get_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert "models" in response.json()
    assert "DummyModelFirst" in response.json()["models"]

def test_set_valid_model():
    response = client.post("/set/DummyModelFirst")
    assert response.status_code == 200
    assert response.json()["active_model"] == "DummyModelFirst"

def test_set_invalid_model():
    response = client.post("/set/NonExistentModel")
    assert response.status_code == 400
    assert "не найдена" in response.json()["detail"]

def test_predict_valid_input():
    client.post("/set/DummyModelFirst")
    payload = {
        "cpm": 100.0,
        "hour_start": 100,
        "hour_end": 300,
        "publishers": "1,2,3",
        "audience_size": 10,
        "user_ids": "101,102,103"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    for key in ["at_least_one", "at_least_two", "at_least_three"]:
        assert key in result

def test_predict_missing_model():
    app.state.active_model = None
    payload = {
        "cpm": 100.0,
        "hour_start": 100,
        "hour_end": 300,
        "publishers": "1,2,3",
        "audience_size": 10,
        "user_ids": "101,102,103"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "не установлена" in response.json()["detail"]

def test_predict_csv():
    client.post("/set/DummyModelFirst")
    df = pd.DataFrame([{
        "cpm": 100,
        "hour_start": 0,
        "hour_end": 100,
        "publishers": "1,2,3",
        "audience_size": 5,
        "user_ids": "10,11"
    }])
    buffer = io.StringIO()
    df.to_csv(buffer, sep='\t', index=False)
    buffer.seek(0)
    files = {"file": ("test.csv", buffer.getvalue(), "text/csv")}
    response = client.post("/predict_csv", files=files)
    assert response.status_code == 200
