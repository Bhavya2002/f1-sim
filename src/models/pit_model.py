"""
Pit Stop Model.

Models pit stop time loss for Green Flag, Virtual Safety Car (VSC), and Safety Car (SC):
- Analyzes actual pit stops from OpenF1 / FastF1.
- Identifies missing fields (e.g. stop_duration in 2023) and logs explicit fallbacks.
- Calculates pit loss deltas:
  * Green flag pit loss: ~22.0s
  * VSC pit loss: ~13.0s
  * SC pit loss: ~11.5s
- Supports replacing anomalous/slow stops with team median stop time (~2.8s stationary fallback).
- Returns explicit dictionary of pit assumptions and active fallbacks.
"""

import json
import os
import pandas as pd


class PitStopModel:
    def __init__(self, cache_dir: str = "data/cache/singapore_2023"):
        self.cache_dir = cache_dir
        self.df_pit = None
        self.team_median_stationary = 2.8   # fallback stationary time in seconds
        self.pit_lane_transit_time = 26.5   # pit lane travel time excluding stationary stop
        self.green_pit_loss = 22.0          # net time loss vs flying green lap
        self.vsc_pit_loss = 13.0            # net time loss under VSC
        self.sc_pit_loss = 11.5             # net time loss under SC
        self.fallback_notes = []
        self.assumptions = {}

        self._load_and_calibrate()

    def _load_and_calibrate(self):
        pit_file = os.path.join(self.cache_dir, "pit.json")
        if os.path.exists(pit_file):
            with open(pit_file) as f:
                self.df_pit = pd.DataFrame(json.load(f))
        else:
            self.df_pit = pd.DataFrame()

        if not self.df_pit.empty and "lane_duration" in self.df_pit.columns:
            valid_lane = self.df_pit["lane_duration"].dropna()
            if len(valid_lane) > 0:
                median_lane = float(valid_lane.median())
                # Flying straight time ~7.5s, net green loss = lane - flying_straight
                self.green_pit_loss = round(median_lane - 7.5, 1)
                self.pit_lane_transit_time = round(median_lane - self.team_median_stationary, 1)

        # Record fallback information explicitly
        has_stop = False
        if not self.df_pit.empty and "stop_duration" in self.df_pit.columns:
            has_stop = self.df_pit["stop_duration"].notnull().any()

        if not has_stop:
            self.fallback_notes.append({
                "field": "stop_duration (stationary time)",
                "status": "Missing in 2023 OpenF1 dataset (available only >= 2024 US GP)",
                "fallback_applied": f"Team median stationary stop fallback of {self.team_median_stationary}s used; pit transit time {self.pit_lane_transit_time}s derived from lane_duration.",
            })

        self.assumptions = {
            "green_flag_pit_loss_sec": self.green_pit_loss,
            "vsc_pit_loss_sec": self.vsc_pit_loss,
            "sc_pit_loss_sec": self.sc_pit_loss,
            "pit_lane_transit_sec": self.pit_lane_transit_time,
            "team_median_stationary_stop_sec": self.team_median_stationary,
            "fallbacks_used": self.fallback_notes,
        }

    def get_pit_time_loss(self, condition: str = "GREEN", is_slow_stop_override: bool = False) -> float:
        """
        Returns net pit time loss in seconds for the given track condition.
        """
        cond = condition.upper()
        if cond == "SC":
            loss = self.sc_pit_loss
        elif cond == "VSC":
            loss = self.vsc_pit_loss
        else:
            loss = self.green_pit_loss

        if is_slow_stop_override:
            # Replaced slow stop with optimal team median stop (-1.5s delta)
            loss = max(5.0, loss - 1.5)

        return round(float(loss), 2)

    def get_assumptions(self) -> dict:
        return self.assumptions
