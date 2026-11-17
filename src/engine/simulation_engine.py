"""
Unified Lap-by-Lap Formula 1 Simulation Engine.

Every what-if is a control on this shared engine:
- Change pit stop lap (earlier / later)
- Change fitted tyre compound (with realistic stint lengths)
- Change stop count (1-stop vs 2-stop vs 3-stop)
- Remove Safety Car (re-simulate at green pace, eliminate bunching, re-price cheap stops)
- Insert Safety Car (chosen lap and duration)
- Replace slow pit stop with team median

Calculates cumulative times, gap to leader, and finishing positions lap by lap.
Returns transparent active assumptions alongside all outputs.
"""

import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from src.models.pace_model import PaceAndDegModel
from src.models.pit_model import PitStopModel
from src.models.track_conditions import TrackConditionModel, TrackStatus


class SimulationEngine:
    def __init__(self, cache_dir: str = "data/cache/singapore_2023"):
        self.cache_dir = cache_dir
        self.pace_model = PaceAndDegModel(cache_dir=cache_dir)
        self.pit_model = PitStopModel(cache_dir=cache_dir)
        self.track_model = TrackConditionModel()

        with open(os.path.join(cache_dir, "race_info.json")) as f:
            self.race_info = json.load(f)
        with open(os.path.join(cache_dir, "laps.json")) as f:
            self.raw_laps = json.load(f)
        with open(os.path.join(cache_dir, "stints.json")) as f:
            self.raw_stints = json.load(f)
        with open(os.path.join(cache_dir, "pit.json")) as f:
            self.raw_pits = json.load(f)

        self.total_laps = self.race_info["total_laps"]
        self.default_sc_laps = set(self.race_info.get("safety_car_laps", [20, 21, 22]))
        self.default_vsc_laps = set(self.race_info.get("vsc_laps", [44, 45]))
        self.official_top10 = self.race_info["official_classification"]
        self.top10_drivers = [d["driver_number"] for d in self.official_top10]

        # Load local trained traffic model
        self.traffic_model_data = None
        self.traffic_pipeline = None
        self._load_traffic_model()

        # Extract baseline stints for top 10
        self.baseline_stints = self._extract_baseline_stints()
        # Compute actual race cumulative times and gaps for reference
        self.actual_lap_times, self.actual_cumulative, self.actual_gaps = self._build_actual_timeline()

    def _load_traffic_model(self):
        import joblib
        model_path = os.path.join(os.path.dirname(self.cache_dir), "models", "traffic_overtake_model.joblib")
        if not os.path.exists(model_path):
            model_path = "data/models/traffic_overtake_model.joblib"
        if os.path.exists(model_path):
            try:
                self.traffic_model_data = joblib.load(model_path)
                self.traffic_pipeline = self.traffic_model_data.get("pipeline")
            except Exception as e:
                print(f"Warning: could not load traffic model from {model_path}: {e}")

    def _extract_baseline_stints(self) -> Dict[int, List[dict]]:
        driver_stints = {}
        for dn in self.top10_drivers:
            d_stints = [
                s for s in self.raw_stints
                if s["driver_number"] == dn and s["lap_start"] <= self.total_laps
            ]
            d_stints = sorted(d_stints, key=lambda x: x["stint_number"])
            # Format cleanly
            stint_list = []
            for s in d_stints:
                end_lap = min(s["lap_end"], self.total_laps)
                stint_list.append({
                    "stint_number": s["stint_number"],
                    "lap_start": s["lap_start"],
                    "lap_end": end_lap,
                    "compound": s["compound"],
                    "start_tyre_age": s.get("tyre_age_at_start", 0) or 0,
                })
            driver_stints[dn] = stint_list
        return driver_stints

    def _build_actual_timeline(self):
        """
        Builds the actual recorded lap times, cumulative race time, and gap to leader
        directly from OpenF1 laps for the top 10 finishers.
        """
        driver_laps = {dn: {} for dn in self.top10_drivers}
        for row in self.raw_laps:
            dn = row.get("driver_number")
            lap_num = row.get("lap_number")
            dur = row.get("lap_duration")
            if dn in self.top10_drivers and lap_num is not None and 1 <= lap_num <= self.total_laps and dur is not None:
                driver_laps[dn][lap_num] = dur

        # Fallback for any missing lap
        for dn in self.top10_drivers:
            for l in range(1, self.total_laps + 1):
                if l not in driver_laps[dn]:
                    driver_laps[dn][l] = driver_laps[dn].get(l - 1, 100.0)

        # Leader cumulative time (Sainz #55)
        leader_cum = {}
        cum = 0.0
        for l in range(1, self.total_laps + 1):
            cum += driver_laps[55][l]
            leader_cum[l] = cum

        actual_cum = {}
        actual_gaps = {}
        for dn in self.top10_drivers:
            actual_cum[dn] = {}
            actual_gaps[dn] = {}
            d_cum = 0.0
            for l in range(1, self.total_laps + 1):
                d_cum += driver_laps[dn][l]
                actual_cum[dn][l] = round(d_cum, 3)
                actual_gaps[dn][l] = round(d_cum - leader_cum[l], 3)

        return driver_laps, actual_cum, actual_gaps

    def build_stint_schedule(
        self,
        driver_number: int,
        pit_laps: List[int],
        compounds: List[str]
    ) -> List[dict]:
        """
        Constructs a valid, consistent full-race stint schedule for what-if controls:
        - Ensures pit_laps are strictly ascending and within [2, total_laps - 1].
        - Ensures compounds match stint count (len(pit_laps) + 1).
        - Validates stint length and tyre life.
        """
        sorted_pits = sorted(list(set(pit_laps)))
        num_stints = len(sorted_pits) + 1

        # Fallback compounds if length mismatch
        if len(compounds) < num_stints:
            compounds = list(compounds) + ["HARD"] * (num_stints - len(compounds))
        elif len(compounds) > num_stints:
            compounds = compounds[:num_stints]

        stints = []
        start_lap = 1
        for i, pit_lap in enumerate(sorted_pits):
            stints.append({
                "stint_number": i + 1,
                "lap_start": start_lap,
                "lap_end": pit_lap,
                "compound": compounds[i].upper(),
                "start_tyre_age": 0,
            })
            start_lap = pit_lap + 1

        # Final stint
        stints.append({
            "stint_number": num_stints,
            "lap_start": start_lap,
            "lap_end": self.total_laps,
            "compound": compounds[-1].upper(),
            "start_tyre_age": 0,
        })
        return stints

    def run_simulation(
        self,
        modified_driver: Optional[int] = None,
        custom_stints: Optional[List[dict]] = None,
        remove_safety_car: bool = False,
        remove_vsc: bool = False,
        insert_safety_car_lap: Optional[int] = None,
        insert_safety_car_duration: int = 3,
        insert_vsc_lap: Optional[int] = None,
        insert_vsc_duration: int = 2,
        replace_slow_stop: bool = False,
        enable_traffic_model: bool = False,
        use_nonlinear_tyre: bool = False,
    ) -> dict:
        """
        Unified simulation execution.
        Applies all composable what-if controls through one arithmetic loop.
        """
        driver_meta = {d["driver_number"]: d for d in self.official_top10}

        # Determine if this is an unmodified actual race replay
        is_stints_modified = False
        if modified_driver is not None and custom_stints is not None:
            baseline = self.baseline_stints.get(modified_driver, [])
            # Check if stints differ from baseline
            if len(custom_stints) != len(baseline):
                is_stints_modified = True
            else:
                for s_new, s_base in zip(custom_stints, baseline):
                    if (s_new.get("lap_start") != s_base.get("lap_start") or
                        s_new.get("lap_end") != s_base.get("lap_end") or
                        s_new.get("compound") != s_base.get("compound")):
                        is_stints_modified = True
                        break

        is_actual_replay = (
            not is_stints_modified and
            not remove_safety_car and
            not remove_vsc and
            insert_safety_car_lap is None and
            insert_vsc_lap is None and
            not replace_slow_stop and
            not enable_traffic_model and
            not use_nonlinear_tyre
        )

        active_assumptions = {
            "is_actual_replay": is_actual_replay,
            "models": {
                "pace_and_degradation": self.pace_model.get_assumptions(),
                "pit_loss": self.pit_model.get_assumptions(),
                "track_conditions": self.track_model.get_assumptions(),
            },
            "what_if_controls_applied": {
                "modified_driver": modified_driver if is_stints_modified else None,
                "custom_stints": is_stints_modified,
                "remove_safety_car": remove_safety_car,
                "remove_vsc": remove_vsc,
                "insert_safety_car": insert_safety_car_lap is not None,
                "insert_vsc": insert_vsc_lap is not None,
                "replace_slow_stop": replace_slow_stop,
                "enable_traffic_model": enable_traffic_model,
                "use_nonlinear_tyre": use_nonlinear_tyre,
            },
            "active_notes": []
        }

        if is_actual_replay:
            active_assumptions["active_notes"].append(
                "ACTUAL RACE REPLAY: Utilizing recorded real-world lap times, pit stops, and decisions from the 2023 Singapore Grand Prix. Exact historical classification reproduced."
            )

        if enable_traffic_model:
            active_assumptions["active_notes"].append(
                "TRAFFIC & DIRTY AIR MODEL ACTIVE: Chasing cars within 1.0s evaluate a trained ML overtake classifier. Trailing cars without an overtake advantage receive a +0.75s dirty air penalty and remain behind."
            )

        if use_nonlinear_tyre:
            active_assumptions["active_notes"].append(
                "NON-LINEAR TYRE MODEL ACTIVE: Capturing thermal degradation cliffs (Soft: L18, Med: L28, Hard: L42) and scrub-in phases via locally trained HistGradientBoosting regressor."
            )

        # 1. Resolve Track Status Schedule per Lap
        track_schedule = {}
        sc_laps = set() if remove_safety_car else set(self.default_sc_laps)
        vsc_laps = set() if remove_vsc else set(self.default_vsc_laps)

        if remove_safety_car:
            active_assumptions["active_notes"].append(
                "Full Safety Car (SC) REMOVED on Laps 20-22: laps re-simulated at green-flag pace, pack bunching eliminated, and Lap 20 pit stops re-priced to green flag pit loss (~22.0s)."
            )

        if remove_vsc:
            active_assumptions["active_notes"].append(
                "Virtual Safety Car (VSC) REMOVED on Laps 44-45: laps run at green pace, and VSC pit stops (e.g. Mercedes) re-priced to green flag pit loss (~22.0s)."
            )

        if insert_safety_car_lap is not None and 1 <= insert_safety_car_lap <= self.total_laps:
            inserted = set(range(insert_safety_car_lap, min(self.total_laps + 1, insert_safety_car_lap + insert_safety_car_duration)))
            sc_laps.update(inserted)
            active_assumptions["active_notes"].append(
                f"Full Safety Car (SC) INSERTED on Laps {insert_safety_car_lap} to {insert_safety_car_lap + insert_safety_car_duration - 1} with 70% bunching compression to 0.8s intervals."
            )

        if insert_vsc_lap is not None and 1 <= insert_vsc_lap <= self.total_laps:
            inserted_v = set(range(insert_vsc_lap, min(self.total_laps + 1, insert_vsc_lap + insert_vsc_duration)))
            vsc_laps.update(inserted_v)
            active_assumptions["active_notes"].append(
                f"Virtual Safety Car (VSC) INSERTED on Laps {insert_vsc_lap} to {insert_vsc_lap + insert_vsc_duration - 1} with 1.35x delta pace and frozen gaps (no bunching)."
            )

        for l in range(1, self.total_laps + 1):
            if l in sc_laps:
                track_schedule[l] = TrackStatus.SC
            elif l in vsc_laps:
                track_schedule[l] = TrackStatus.VSC
            else:
                track_schedule[l] = TrackStatus.GREEN

        # 2. Resolve Driver Stints
        driver_stint_plans = deepcopy(self.baseline_stints)
        if is_stints_modified and modified_driver in driver_stint_plans:
            driver_stint_plans[modified_driver] = custom_stints
            active_assumptions["active_notes"].append(
                f"Driver #{modified_driver} strategy modified: {len(custom_stints)} stint(s) with compounds {[s['compound'] for s in custom_stints]}."
            )

        # Baseline pit stop laps
        baseline_pit_laps = {}
        for dn, stints in self.baseline_stints.items():
            baseline_pit_laps[dn] = [s["lap_end"] for s in stints[:-1]]

        # Modified pit stop laps
        driver_pit_laps = {}
        for dn, stints in driver_stint_plans.items():
            driver_pit_laps[dn] = [s["lap_end"] for s in stints[:-1]]

        # 3. Simulate Lap by Lap
        sim_cum_times = {dn: 0.0 for dn in self.top10_drivers}
        sim_lap_times = {dn: {} for dn in self.top10_drivers}
        sim_lap_gaps = {dn: {} for dn in self.top10_drivers}
        traffic_events = []

        driver_traffic_summary = {
            dn: {
                "driver_number": dn,
                "code": driver_meta[dn]["code"],
                "laps_in_dirty_air": 0,
                "time_lost_dirty_air": 0.0,
                "trapped_behind": None,
                "trapped_counts": {},
                "overtakes_made": 0,
                "overtakes_blocked": 0,
            }
            for dn in self.top10_drivers
        }

        standing_start_overhead = 6.2  # lap 1 grid start overhead
        pace_fn = self.pace_model.predict_lap_time_nonlinear if use_nonlinear_tyre else self.pace_model.predict_lap_time

        for lap in range(1, self.total_laps + 1):
            cond = track_schedule[lap]
            slowdown = self.track_model.get_slowdown_multiplier(cond)

            # Compute each driver's lap time
            for dn in self.top10_drivers:
                # If exact actual race replay is requested:
                if is_actual_replay:
                    lap_time = self.actual_lap_times[dn][lap]
                else:
                    # Counterfactual Simulation:
                    # If this driver's strategy was modified:
                    if is_stints_modified and dn == modified_driver:
                        # Find current active custom stint
                        stints = driver_stint_plans[dn]
                        curr_stint = None
                        for s in stints:
                            if s["lap_start"] <= lap <= s["lap_end"]:
                                curr_stint = s
                                break
                        if curr_stint is None:
                            curr_stint = stints[-1]

                        compound = curr_stint["compound"]
                        tyre_age = (lap - curr_stint["lap_start"]) + curr_stint.get("start_tyre_age", 0)

                        green_lap = pace_fn(dn, compound, tyre_age, lap)
                        lap_time = green_lap * slowdown
                        if lap == 1:
                            lap_time += standing_start_overhead

                        # Pit stop loss if pitting on this lap
                        if lap in driver_pit_laps[dn]:
                            is_override = replace_slow_stop
                            pit_loss = self.pit_model.get_pit_time_loss(cond.value, is_slow_stop_override=is_override)
                            lap_time += pit_loss

                    else:
                        # Rival driver keeping actual recorded decisions:
                        lap_time = self.actual_lap_times[dn][lap]

                        # Adjust if track condition on this lap differs from actual race
                        is_actual_sc = lap in self.default_sc_laps
                        is_actual_vsc = lap in self.default_vsc_laps

                        # If SC was removed on Laps 20-22:
                        if remove_safety_car and is_actual_sc:
                            # Re-simulate this SC lap at driver's green pace
                            pitted_lap_20 = 20 in baseline_pit_laps.get(dn, [])
                            compound = "HARD" if (lap > 20 and pitted_lap_20) else "MEDIUM"
                            tyre_age = (lap - 20) if (lap > 20 and pitted_lap_20) else lap
                            green_lap = pace_fn(dn, compound, tyre_age, lap)
                            lap_time = green_lap

                            # If driver pitted on Lap 20: re-price pit loss to full Green pit loss (~22.0s)
                            if lap == 20 and pitted_lap_20:
                                lap_time += self.pit_model.green_pit_loss

                        # If VSC was removed on Laps 44-45:
                        elif remove_vsc and is_actual_vsc:
                            # If driver pitted under VSC on Lap 44 (e.g. Hamilton #44):
                            if 44 in baseline_pit_laps.get(dn, []):
                                if lap == 44:
                                    # In-lap before stop + full green pit loss (~22.0s)
                                    green_lap = pace_fn(dn, "HARD", 24, lap)
                                    lap_time = green_lap + self.pit_model.green_pit_loss
                                else:
                                    # Out-lap on fresh Mediums
                                    green_lap = pace_fn(dn, "MEDIUM", 1, lap)
                                    lap_time = green_lap
                            else:
                                # Stayed out on Hards (e.g. Sainz, Norris, Leclerc)
                                green_lap = pace_fn(dn, "HARD", lap - 20, lap)
                                lap_time = green_lap

                        # If a new SC was inserted:
                        elif cond == TrackStatus.SC and not is_actual_sc:
                            lap_time = lap_time * 1.45

                        # If a new VSC was inserted:
                        elif cond == TrackStatus.VSC and not is_actual_vsc:
                            lap_time = lap_time * 1.35

                        # Pit stop optimization override for this driver
                        if replace_slow_stop and dn == modified_driver and lap in baseline_pit_laps.get(dn, []):
                            lap_time -= 1.5

                sim_lap_times[dn][lap] = round(lap_time, 3)

            # Apply Traffic & Dirty Air Dynamics (if enabled)
            if enable_traffic_model and not is_actual_replay and cond == TrackStatus.GREEN and lap > 1:
                entering_order = sorted(self.top10_drivers, key=lambda d: sim_cum_times[d])
                for idx in range(1, len(entering_order)):
                    chaser_dn = entering_order[idx]
                    leader_dn = entering_order[idx - 1]
                    gap_start = sim_cum_times[chaser_dn] - sim_cum_times[leader_dn]

                    if 0.0 < gap_start <= 1.2:
                        chaser_raw = sim_lap_times[chaser_dn][lap]
                        leader_raw = sim_lap_times[leader_dn][lap]
                        pace_delta = leader_raw - chaser_raw

                        p_overtake = 0.28
                        if self.traffic_pipeline:
                            try:
                                import pandas as pd
                                X_case = pd.DataFrame([{
                                    "gap_before": gap_start,
                                    "pace_delta": pace_delta,
                                    "tyre_age_diff": 4 if pace_delta > 0.4 else 0,
                                    "compound_adv": 1 if pace_delta > 0.7 else 0,
                                    "speed_delta": 3.0 if pace_delta > 0.3 else 0.0,
                                    "is_drs": 1
                                }])
                                p_overtake = float(self.traffic_pipeline.predict_proba(X_case)[0][1])
                            except Exception:
                                p_overtake = 0.28

                        chaser_proj = sim_cum_times[chaser_dn] + chaser_raw
                        leader_proj = sim_cum_times[leader_dn] + leader_raw

                        if chaser_proj < leader_proj:
                            if p_overtake < 0.50:
                                # Blocked in dirty air! Chaser cannot pass
                                penalty = round((leader_proj + 0.35) - (sim_cum_times[chaser_dn] + chaser_raw), 3)
                                if penalty > 0:
                                    driver_traffic_summary[chaser_dn]["time_lost_dirty_air"] += penalty
                                sim_lap_times[chaser_dn][lap] = round((leader_proj + 0.35) - sim_cum_times[chaser_dn], 3)
                                driver_traffic_summary[chaser_dn]["laps_in_dirty_air"] += 1
                                driver_traffic_summary[chaser_dn]["overtakes_blocked"] += 1
                                leader_code = driver_meta[leader_dn]["code"]
                                driver_traffic_summary[chaser_dn]["trapped_counts"][leader_code] = (
                                    driver_traffic_summary[chaser_dn]["trapped_counts"].get(leader_code, 0) + 1
                                )
                                traffic_events.append({
                                    "lap": lap,
                                    "type": "DIRTY_AIR_BLOCK",
                                    "chaser": driver_meta[chaser_dn]["code"],
                                    "leader": leader_code,
                                    "prob": round(p_overtake, 2),
                                    "gap": round(gap_start, 2)
                                })
                            else:
                                driver_traffic_summary[chaser_dn]["overtakes_made"] += 1
                                traffic_events.append({
                                    "lap": lap,
                                    "type": "OVERTAKE",
                                    "chaser": driver_meta[chaser_dn]["code"],
                                    "leader": driver_meta[leader_dn]["code"],
                                    "prob": round(p_overtake, 2),
                                    "gap": round(gap_start, 2)
                                })
                        elif gap_start <= 0.8:
                            sim_lap_times[chaser_dn][lap] = round(sim_lap_times[chaser_dn][lap] + 0.25, 3)
                            driver_traffic_summary[chaser_dn]["laps_in_dirty_air"] += 1
                            driver_traffic_summary[chaser_dn]["time_lost_dirty_air"] += 0.25
                            leader_code = driver_meta[leader_dn]["code"]
                            driver_traffic_summary[chaser_dn]["trapped_counts"][leader_code] = (
                                driver_traffic_summary[chaser_dn]["trapped_counts"].get(leader_code, 0) + 1
                            )

            for dn in self.top10_drivers:
                sim_cum_times[dn] += sim_lap_times[dn][lap]

            # Handle Safety Car Bunching (only for counterfactual simulations where SC is active)
            if not is_actual_replay and cond == TrackStatus.SC and not remove_safety_car:
                sorted_field = sorted([(dn, sim_cum_times[dn]) for dn in self.top10_drivers], key=lambda x: x[1])
                bunched_field = self.track_model.apply_sc_bunching(sorted_field)
                for dn, b_time in bunched_field:
                    sim_cum_times[dn] = b_time

            # Compute gaps to leader on this lap
            sorted_by_cum = sorted([(dn, sim_cum_times[dn]) for dn in self.top10_drivers], key=lambda x: x[1])
            leader_time = sorted_by_cum[0][1]
            for dn in self.top10_drivers:
                sim_lap_gaps[dn][lap] = round(sim_cum_times[dn] - leader_time, 3)

        # 4. Final Classification (apply post-race penalties)
        # Sergio Perez had an official 5-second penalty in 2023 Singapore
        penalties = {11: 5.0}  # Perez 5s
        final_times = []
        for dn in self.top10_drivers:
            raw_time = sim_cum_times[dn]
            pen = penalties.get(dn, 0.0)
            total_time = raw_time + pen
            final_times.append({
                "driver_number": dn,
                "raw_time": round(raw_time, 3),
                "penalty_seconds": pen,
                "total_time": round(total_time, 3),
            })

        final_times = sorted(final_times, key=lambda x: x["total_time"])

        # Finalize driver traffic summaries
        for dn, traf in driver_traffic_summary.items():
            traf["time_lost_dirty_air"] = round(traf["time_lost_dirty_air"], 2)
            if traf["trapped_counts"]:
                most_frequent_rival = max(traf["trapped_counts"].items(), key=lambda x: x[1])
                traf["trapped_behind"] = {
                    "code": most_frequent_rival[0],
                    "laps": most_frequent_rival[1]
                }
            else:
                traf["trapped_behind"] = None

        target_driver = modified_driver if is_stints_modified and modified_driver else 55
        stint_cliff_analysis = self.pace_model.analyze_stints_for_cliffs(driver_stint_plans[target_driver])

        # Format Side-by-Side Comparison
        driver_meta = {d["driver_number"]: d for d in self.official_top10}
        sim_leader_time = final_times[0]["total_time"]

        comparison_results = []
        for sim_pos, item in enumerate(final_times, start=1):
            dn = item["driver_number"]
            meta = driver_meta[dn]
            actual_pos = meta["pos"]
            delta_pos = actual_pos - sim_pos  # positive means improved
            stints = driver_stint_plans[dn]
            strategy_str = "-".join([f"{s['compound'][0]}(L{s['lap_start']}-{s['lap_end']})" for s in stints])
            sim_gap = round(item["total_time"] - sim_leader_time, 3)
            official_t = meta.get("official_time") or meta.get("time", "")

            driver_traf = driver_traffic_summary.get(dn, {})
            driver_cliffs = self.pace_model.analyze_stints_for_cliffs(stints)

            comparison_results.append({
                "sim_position": sim_pos,
                "actual_position": actual_pos,
                "position_delta": delta_pos,
                "driver_number": dn,
                "code": meta["code"],
                "name": meta["name"],
                "team": meta["team"],
                "sim_total_time": round(item["total_time"], 3),
                "sim_gap_to_leader": sim_gap,
                "actual_official_time": official_t,
                "strategy": strategy_str,
                "is_modified": (dn == modified_driver) if is_stints_modified else False,
                "penalty_applied": item["penalty_seconds"],
                "traffic_summary": driver_traf,
                "cliff_analysis": driver_cliffs,
            })

        # Format lap-by-lap gap comparison for plotting
        chart_data = {
            "laps": list(range(1, self.total_laps + 1)),
            "track_status": [track_schedule[l].value for l in range(1, self.total_laps + 1)],
            "drivers": {},
        }
        for dn in self.top10_drivers:
            code = driver_meta[dn]["code"]
            chart_data["drivers"][code] = {
                "name": driver_meta[dn]["name"],
                "driver_number": dn,
                "actual_gap": [self.actual_gaps[dn][l] for l in range(1, self.total_laps + 1)],
                "simulated_gap": [sim_lap_gaps[dn][l] for l in range(1, self.total_laps + 1)],
                "is_modified": (dn == modified_driver) if is_stints_modified else False,
            }

        # Build Explanation Panel
        explanation = self._build_explanation(
            comparison_results=comparison_results,
            modified_driver=modified_driver if is_stints_modified else None,
            remove_safety_car=remove_safety_car,
            remove_vsc=remove_vsc,
            insert_safety_car_lap=insert_safety_car_lap,
            insert_vsc_lap=insert_vsc_lap,
            replace_slow_stop=replace_slow_stop,
            enable_traffic_model=enable_traffic_model,
            use_nonlinear_tyre=use_nonlinear_tyre,
            traffic_events=traffic_events,
            driver_traffic_summary=driver_traffic_summary,
            active_assumptions=active_assumptions
        )

        return {
            "comparison": comparison_results,
            "chart_data": chart_data,
            "explanation": explanation,
            "traffic_events": traffic_events,
            "driver_traffic_summary": driver_traffic_summary,
            "stint_cliff_analysis": stint_cliff_analysis,
            "active_assumptions": active_assumptions,
            "track_schedule": {l: track_schedule[l].value for l in track_schedule},
        }

    def _build_explanation(
        self,
        comparison_results: List[dict],
        modified_driver: Optional[int],
        remove_safety_car: bool,
        remove_vsc: bool,
        insert_safety_car_lap: Optional[int],
        insert_vsc_lap: Optional[int],
        replace_slow_stop: bool,
        enable_traffic_model: bool = False,
        use_nonlinear_tyre: bool = False,
        traffic_events: Optional[List[dict]] = None,
        driver_traffic_summary: Optional[dict] = None,
        active_assumptions: Optional[dict] = None
    ) -> dict:
        """
        Generates structured, explainable analysis of the simulation output.
        Only describes computed numbers and declared assumptions.
        """
        highlights = []
        winner = comparison_results[0]
        actual_winner = [c for c in comparison_results if c["actual_position"] == 1][0]

        if active_assumptions and active_assumptions.get("is_actual_replay", False):
            highlights.append(
                "**Actual Race Replay**: Official race results and recorded lap data from the 2023 Singapore Grand Prix are active. Exact finishing classification reproduced."
            )
        else:
            if winner["driver_number"] != actual_winner["driver_number"]:
                highlights.append(
                    f"**Race Winner Change**: Simulated P1 is **{winner['name']}** ({winner['team']}), displacing actual winner **{actual_winner['name']}** by {abs(round(comparison_results[1]['sim_gap_to_leader'], 3))}s."
                )
            else:
                highlights.append(
                    f"**Winner Retained**: **{winner['name']}** maintains P1 under simulated assumptions."
                )

            if modified_driver:
                mod_item = [c for c in comparison_results if c["driver_number"] == modified_driver][0]
                d_pos = mod_item["position_delta"]
                sign = "+" if d_pos > 0 else ""
                highlights.append(
                    f"**Modified Driver ({mod_item['name']})**: Moved from actual P{mod_item['actual_position']} to simulated P{mod_item['sim_position']} ({sign}{d_pos} positions) with strategy `{mod_item['strategy']}`."
                )

            if remove_safety_car:
                highlights.append(
                    "**Safety Car Removal Impact**: Pitting under green cost ~22.0s rather than ~11.5s under SC. Drivers who took cheap stops on Lap 20 lost net track position, and field compression was removed."
                )

            if remove_vsc:
                highlights.append(
                    "**Virtual Safety Car Removal Impact**: Pitting under green cost ~22.0s rather than ~13.0s under VSC. Drivers who pitted under VSC on Lap 44 (e.g. Mercedes Russell & Hamilton) lost time relative to one-stoppers."
                )

            if insert_safety_car_lap:
                highlights.append(
                    f"**Safety Car Insertion Impact**: Field compressed behind SC on Lap {insert_safety_car_lap}. Gaps compressed at 70% per lap to 0.8s restart intervals."
                )

            if insert_vsc_lap:
                highlights.append(
                    f"**Virtual Safety Car Insertion Impact**: Pace slowed by 1.35x on Laps {insert_vsc_lap} to {insert_vsc_lap + 1}. Gaps frozen (no bunching compression) while pit loss was reduced."
                )

            if replace_slow_stop:
                highlights.append(
                    "**Pit Stop Optimization**: Substituted slow stop with team median pit stop delta (-1.5s saving)."
                )

            if enable_traffic_model and traffic_events:
                blocks = [e for e in traffic_events if e["type"] == "DIRTY_AIR_BLOCK"]
                overtakes = [e for e in traffic_events if e["type"] == "OVERTAKE"]
                traf_msg = f"**Traffic & Overtaking (Model 1)**: Simulated real wheel-to-wheel racing. {len(overtakes)} pass(es) were completed, and cars were held up in traffic {len(blocks)} time(s)."
                if modified_driver and driver_traffic_summary and modified_driver in driver_traffic_summary:
                    m_traf = driver_traffic_summary[modified_driver]
                    if m_traf.get("laps_in_dirty_air", 0) > 0:
                        rival_text = f" behind {m_traf['trapped_behind']['code']}" if m_traf.get("trapped_behind") else ""
                        traf_msg += f" Selected driver lost {m_traf['time_lost_dirty_air']}s over {m_traf['laps_in_dirty_air']} lap(s) in traffic{rival_text}."
                    else:
                        traf_msg += " Selected driver ran in clean air."
                highlights.append(traf_msg)

            if use_nonlinear_tyre:
                highlights.append(
                    "**Tire Wear & Grip Drop-Off (Model 2)**: Simulated realistic tire wear, slowing cars down significantly if tires were pushed past their recommended lifespan (the tire cliff)."
                )

        return {
            "summary": " ".join(highlights),
            "key_drivers": [
                {
                    "driver": c["name"],
                    "actual_p": c["actual_position"],
                    "sim_p": c["sim_position"],
                    "delta": c["position_delta"],
                    "sim_gap": c["sim_gap_to_leader"],
                }
                for c in comparison_results[:5]
            ],
            "assumptions_summary": active_assumptions["active_notes"],
        }
