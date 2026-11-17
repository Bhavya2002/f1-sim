"""
Tests for Natural Language Strategy Assistant.
"""

import pytest
from fastapi.testclient import TestClient
from src.ai.assistant import StrategyAIAssistant
from src.app import app
from src.engine.simulation_engine import SimulationEngine

client = TestClient(app)


@pytest.fixture
def assistant():
    engine = SimulationEngine(cache_dir="data/cache/singapore_2023")
    return StrategyAIAssistant(engine)


def test_command_pit_one_lap_earlier(assistant):
    res = assistant.process_query("make Sainz pit one lap earlier")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["action"] == "MODIFY_PIT_LAP"
    assert res["state"]["modified_driver"] == 55
    # Stint 1 should end on Lap 19 instead of 20
    assert res["state"]["custom_stints"][0]["lap_end"] == 19
    assert "earlier" in res["response"]
    assert res["simulation_result"] is not None


def test_command_remove_safety_car(assistant):
    res = assistant.process_query("what if there was no safety car?")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["action"] == "REMOVE_SAFETY_CAR"
    assert res["state"]["remove_safety_car"] is True
    assert res["simulation_result"] is not None


def test_command_remove_vsc(assistant):
    res = assistant.process_query("what if there was no virtual safety car?")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["action"] == "REMOVE_VSC"
    assert res["state"]["remove_vsc"] is True
    assert res["simulation_result"] is not None


def test_command_switch_to_2_stop(assistant):
    res = assistant.process_query("switch Norris to a 2-stop strategy")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["action"] == "SET_STOP_COUNT"
    assert res["state"]["modified_driver"] == 4
    assert len(res["state"]["custom_stints"]) == 3  # 2 stops = 3 stints


def test_command_optimize_pit_stop(assistant):
    res = assistant.process_query("optimize pit stop for Leclerc")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["action"] == "OPTIMIZE_PIT_STOP"
    assert res["state"]["modified_driver"] == 16
    assert res["state"]["replace_slow_stop"] is True


def test_question_verstappen_win(assistant):
    res = assistant.process_query("Why did Verstappen win when the safety car was removed?")
    assert res["type"] == "QUESTION_ANSWERED"
    assert res["action"] == "Q_AND_A"
    assert "Verstappen" in res["response"]
    assert "Hard" in res["response"]


def test_question_tyre_degradation(assistant):
    res = assistant.process_query("What is the degradation rate on Mediums vs Hards?")
    assert res["type"] == "QUESTION_ANSWERED"
    assert res["action"] == "Q_AND_A"
    assert "Soft" in res["response"]
    assert "Medium" in res["response"]
    assert "Hard" in res["response"]


def test_api_chat_endpoint():
    payload = {
        "prompt": "make Sainz pit 1 lap earlier",
        "current_state": {"modified_driver": 55}
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "COMMAND_EXECUTED"
    assert data["state"]["modified_driver"] == 55
    assert "ui_directives" in data


def test_command_toggle_traffic(assistant):
    res = assistant.process_query("enable traffic model")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["state"]["enable_traffic_model"] is True
    assert res["ui_directives"]["enable_traffic_model"] is True


def test_command_toggle_nonlinear_tyre(assistant):
    res = assistant.process_query("use nonlinear tyre model")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["state"]["use_nonlinear_tyre"] is True
    assert res["ui_directives"]["use_nonlinear_tyre"] is True


def test_command_switch_driver(assistant):
    res = assistant.process_query("switch to Norris")
    assert res["type"] == "COMMAND_EXECUTED"
    assert res["state"]["modified_driver"] == 4
    assert res["ui_directives"]["modified_driver"] == 4
