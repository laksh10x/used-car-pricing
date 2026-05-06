from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_model_missing_or_ready_response_is_valid():
    payload = {
        "make": "Honda",
        "model": "Accord",
        "year": 2018,
        "mileage": 82000,
        "engine": "2.0L I-4 252HP",
        "fuel_type": "Gasoline",
        "transmission": "Automatic",
        "drivetrain": "Front-wheel Drive",
        "color": "Gray",
        "damage_description": "Rear bumper cracked and engine knocks when cold.",
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code in (200, 503)
