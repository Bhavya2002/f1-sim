"""
Tests for ReplayCheck, ModelBacktest, and SanityChecks.
"""

import pytest
from src.engine.simulation_engine import SimulationEngine
from src.validation.replay_check import ReplayCheck
from src.validation.model_backtest import ModelBacktest
from src.validation.sanity_checks import SanityChecks


def test_replay_check():
    checker = ReplayCheck(cache_dir="data/cache/singapore_2023")
    res = checker.run()
    assert res["success"] is True
    assert res["overall_status"] == "EXACT MATCH (10/10)"
    assert len(res["table"]) == 10

    # Ensure positions match 1 to 10
    for r in res["table"]:
        assert r["replayed_position"] == r["official_position"]
        assert r["matches"] is True

    # Check Perez penalty is documented
    assert "Sergio Perez" in res["penalty_reconciliation_note"]
    assert "5.0-second" in res["penalty_reconciliation_note"]


def test_model_backtest():
    backtest = ModelBacktest(cache_dir="data/cache/singapore_2023")
    res = backtest.run()
    assert res["training_laps"] > 400
    assert res["test_laps"] > 300
    assert res["model_mae_sec"] > 0.0
    assert res["baseline_mae_sec"] > 0.0
    # Check that error is reasonable for F1 lap pace (~0.5 - 1.5s/lap)
    assert 0.4 < res["model_mae_sec"] < 2.0
    assert 0.4 < res["baseline_mae_sec"] < 2.0
    assert "compound_breakdown" in res
    assert len(res["compound_breakdown"]) >= 2


def test_sanity_checks():
    engine = SimulationEngine(cache_dir="data/cache/singapore_2023")
    checker = SanityChecks(engine)
    res = checker.run()
    assert res["all_passed"] is True
    for c in res["checks"]:
        assert c["status"] == "PASS"
    assert len(res["fallbacks_active"]) > 0
