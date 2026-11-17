"""
Tests for FastAPI application endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "F1 WHAT-IF" in response.text
    assert "Singapore Grand Prix" in response.text


def test_race_info_api():
    response = client.get("/api/race-info")
    assert response.status_code == 200
    data = response.json()
    assert "race_info" in data
    assert "baseline_stints" in data
    assert "official_classification" in data
    assert len(data["official_classification"]) == 10


def test_simulate_api():
    payload = {
        "modified_driver": 55,
        "custom_stints": [
            {"stint_number": 1, "lap_start": 1, "lap_end": 20, "compound": "MEDIUM", "start_tyre_age": 0},
            {"stint_number": 2, "lap_start": 21, "lap_end": 62, "compound": "HARD", "start_tyre_age": 0}
        ],
        "remove_safety_car": False,
        "insert_safety_car_lap": None,
        "insert_safety_car_duration": 3,
        "replace_slow_stop": False
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert "chart_data" in data
    assert "explanation" in data
    assert len(data["comparison"]) == 10


def test_validation_endpoints():
    r_rep = client.get("/api/validation/replay")
    assert r_rep.status_code == 200
    assert r_rep.json()["success"] is True

    r_bck = client.get("/api/validation/backtest")
    assert r_bck.status_code == 200
    assert "model_mae_sec" in r_bck.json()

    r_san = client.get("/api/validation/sanity")
    assert r_san.status_code == 200
    assert r_san.json()["all_passed"] is True


def test_ml_lab_page():
    response = client.get("/ml-lab")
    assert response.status_code == 200
    assert "Machine Learning Lab" in response.text
    assert "Traffic & Overtake Model" in response.text


def test_validation_page():
    response = client.get("/validation")
    assert response.status_code == 200
    assert "Verification & Proof Center" in response.text


def test_ml_api_endpoints():
    # Metrics
    res_m = client.get("/api/ml/metrics")
    assert res_m.status_code == 200
    assert "models" in res_m.json()

    # Tyre Curves
    res_c = client.get("/api/ml/tyre-curves")
    assert res_c.status_code == 200
    assert "curves" in res_c.json()
    assert "ages" in res_c.json()["curves"]

    # NLP Prediction
    res_nlp = client.post("/api/ml/predict-nlp", json={"prompt": "make Sainz pit one lap earlier"})
    assert res_nlp.status_code == 200
    assert res_nlp.json()["intent"] == "MODIFY_PIT_LAP"


def test_simulate_with_ml_options():
    payload = {
        "modified_driver": 55,
        "enable_traffic_model": True,
        "use_nonlinear_tyre": True
    }
    response = client.post("/api/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "traffic_events" in data
    assert data["active_assumptions"]["what_if_controls_applied"]["enable_traffic_model"] is True
    assert data["active_assumptions"]["what_if_controls_applied"]["use_nonlinear_tyre"] is True
