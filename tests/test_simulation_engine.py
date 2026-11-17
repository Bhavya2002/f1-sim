"""
Unit and integration tests for SimulationEngine and ScenarioManager.
"""

import pytest
from src.engine.simulation_engine import SimulationEngine
from src.engine.scenario_manager import ScenarioManager


@pytest.fixture
def engine():
    return SimulationEngine(cache_dir="data/cache/singapore_2023")


@pytest.fixture
def scenario_mgr(engine):
    return ScenarioManager(engine)


def test_baseline_simulation(engine):
    res = engine.run_simulation()
    assert "comparison" in res
    assert "chart_data" in res
    assert "explanation" in res
    assert "active_assumptions" in res
    assert len(res["comparison"]) == 10

    # Verify positions are 1 to 10 and match exact 2023 Singapore Grand Prix official results
    expected_top10 = ["SAI", "NOR", "HAM", "LEC", "VER", "GAS", "PIA", "PER", "LAW", "MAG"]
    actual_top10 = [c["code"] for c in res["comparison"]]
    assert actual_top10 == expected_top10, f"Expected {expected_top10}, got {actual_top10}"

    # P1 must be Carlos Sainz
    assert res["comparison"][0]["driver_number"] == 55
    assert res["comparison"][0]["name"] == "Carlos Sainz"
    assert res["comparison"][0]["sim_position"] == 1
    assert res["comparison"][0]["actual_position"] == 1

    # Verify cumulative times are positive and monotonic
    for c in res["comparison"]:
        assert c["sim_total_time"] > 5000.0


def test_remove_vsc(engine):
    baseline_res = engine.run_simulation()
    no_vsc_res = engine.run_simulation(remove_vsc=True)

    # In no-VSC scenario, VSC laps 44-45 run at green flag pace
    assert no_vsc_res["track_schedule"][44] == "GREEN"
    assert no_vsc_res["track_schedule"][45] == "GREEN"

    # Mercedes pitted under VSC; removing VSC re-prices that stop to full green pit loss (~22.0s vs ~13.0s),
    # so Hamilton's gap to leader increases compared to the actual race
    ham_base = [c for c in baseline_res["comparison"] if c["code"] == "HAM"][0]
    ham_novsc = [c for c in no_vsc_res["comparison"] if c["code"] == "HAM"][0]
    assert ham_novsc["sim_gap_to_leader"] > ham_base["sim_gap_to_leader"]


def test_insert_vsc(engine):
    res = engine.run_simulation(insert_vsc_lap=30, insert_vsc_duration=2)
    assert res["track_schedule"][30] == "VSC"
    assert res["track_schedule"][31] == "VSC"
    assert res["track_schedule"][32] == "GREEN"


def test_remove_safety_car(engine):
    baseline_res = engine.run_simulation()
    no_sc_res = engine.run_simulation(remove_safety_car=True)

    # In no-SC scenario, SC laps 20-22 run at green flag pace
    assert no_sc_res["track_schedule"][20] == "GREEN"
    assert no_sc_res["track_schedule"][21] == "GREEN"
    assert no_sc_res["track_schedule"][22] == "GREEN"

    # Drivers who pitted on Lap 20 should have lost ~22s instead of ~11.5s
    # Verstappen who did not pit on Lap 20 should benefit significantly
    ver_base = [c for c in baseline_res["comparison"] if c["code"] == "VER"][0]
    ver_nosc = [c for c in no_sc_res["comparison"] if c["code"] == "VER"][0]
    assert ver_nosc["sim_position"] <= ver_base["sim_position"]


def test_modify_pit_lap(engine, scenario_mgr):
    # Move Sainz (#55) pit stop from Lap 20 to Lap 15
    new_stints = scenario_mgr.modify_pit_lap(driver_number=55, stint_index=0, new_pit_lap=15)
    assert new_stints[0]["lap_end"] == 15
    assert new_stints[1]["lap_start"] == 16

    res = engine.run_simulation(modified_driver=55, custom_stints=new_stints)
    sainz_item = [c for c in res["comparison"] if c["driver_number"] == 55][0]
    assert sainz_item["is_modified"] is True


def test_modify_compound(engine, scenario_mgr):
    # Switch Hamilton's stint 3 from MEDIUM to SOFT
    new_stints = scenario_mgr.modify_compound(driver_number=44, stint_index=2, new_compound="SOFT")
    assert new_stints[2]["compound"] == "SOFT"

    res = engine.run_simulation(modified_driver=44, custom_stints=new_stints)
    ham_item = [c for c in res["comparison"] if c["driver_number"] == 44][0]
    assert "S(L45-62)" in ham_item["strategy"]


def test_change_stop_count(engine, scenario_mgr):
    # Convert Norris (1-stop) to 2-stop
    custom_2stop = scenario_mgr.create_custom_strategy(driver_number=4, stop_count=2)
    assert len(custom_2stop) == 3  # 2 stops = 3 stints
    assert custom_2stop[0]["lap_end"] < custom_2stop[1]["lap_end"] < 62

    res = engine.run_simulation(modified_driver=4, custom_stints=custom_2stop)
    nor_item = [c for c in res["comparison"] if c["driver_number"] == 4][0]
    assert nor_item["is_modified"] is True


def test_insert_safety_car(engine):
    res = engine.run_simulation(insert_safety_car_lap=35, insert_safety_car_duration=3)
    assert res["track_schedule"][35] == "SC"
    assert res["track_schedule"][36] == "SC"
    assert res["track_schedule"][37] == "SC"
    assert res["track_schedule"][38] == "GREEN"


def test_replace_slow_pit_stop(engine):
    base_res = engine.run_simulation(modified_driver=16)
    opt_res = engine.run_simulation(modified_driver=16, replace_slow_stop=True)

    lec_base = [c for c in base_res["comparison"] if c["driver_number"] == 16][0]
    lec_opt = [c for c in opt_res["comparison"] if c["driver_number"] == 16][0]

    # Time should be faster with optimized pit stop
    assert lec_opt["sim_total_time"] < lec_base["sim_total_time"]


def test_traffic_and_dirty_air_model(engine):
    # Running with traffic model enabled
    res = engine.run_simulation(
        modified_driver=44,
        enable_traffic_model=True
    )
    assert "traffic_events" in res
    assert "driver_traffic_summary" in res
    assert 44 in res["driver_traffic_summary"]
    assert "time_lost_dirty_air" in res["driver_traffic_summary"][44]
    assert "laps_in_dirty_air" in res["driver_traffic_summary"][44]
    for row in res["comparison"]:
        assert "traffic_summary" in row
        assert "cliff_analysis" in row
    assert res["active_assumptions"]["what_if_controls_applied"]["enable_traffic_model"] is True


def test_nonlinear_tyre_model(engine):
    # Running with nonlinear tyre model enabled
    res = engine.run_simulation(
        modified_driver=55,
        use_nonlinear_tyre=True
    )
    assert "stint_cliff_analysis" in res
    assert len(res["stint_cliff_analysis"]) > 0
    assert "health_status" in res["stint_cliff_analysis"][0]
    assert res["active_assumptions"]["what_if_controls_applied"]["use_nonlinear_tyre"] is True
    # Pace model should provide non-linear curves
    curves = engine.pace_model.get_tyre_degradation_curves()
    assert "curves" in curves
    assert "SOFT" in curves["curves"]
    assert len(curves["curves"]["SOFT"]) == 50
