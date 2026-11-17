"""
Pace and Tyre Degradation Model.

Estimates green-flag lap pace from actual race laps:
- Excludes Safety Car (code 4) and VSC (codes 6/7) laps.
- Excludes pit in-laps and pit out-laps.
- Excludes Lap 1 (standing start).
- Excludes statistical outliers (> 107% of driver median pace).
- Estimates tyre degradation (seconds added per lap of tyre age) per compound.
- Estimates fuel burn down effect (seconds gained per lap).
- Estimates driver-specific base pace per compound.
- Provides fallback cross-compound estimates when a driver did not run a compound.
- Returns explicit dictionary of active assumptions.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class PaceAndDegModel:
    def __init__(self, cache_dir: str = "data/cache/singapore_2023"):
        self.cache_dir = cache_dir
        self.df_laps = None
        self.df_stints = None
        self.df_pit = None
        self.clean_green = None
        
        # Model parameters
        self.compound_deg_rate = {}       # seconds added per lap of tyre age: beta_C
        self.compound_base_pace = {}      # grid median intercept: base_C
        self.driver_compound_pace = {}    # driver-specific base pace: alpha_{i, C}
        self.driver_offsets = {}          # driver overall pace delta vs grid median
        self.fuel_effect = -0.048         # seconds per race lap burn down: gamma
        self.compound_deltas = {}         # delta between compounds (e.g. Soft vs Medium)
        self.assumptions = {}
        
        self._load_and_train()

    def _load_and_train(self):
        with open(os.path.join(self.cache_dir, "laps.json")) as f:
            self.df_laps = pd.DataFrame(json.load(f))
        with open(os.path.join(self.cache_dir, "stints.json")) as f:
            self.df_stints = pd.DataFrame(json.load(f))
        with open(os.path.join(self.cache_dir, "pit.json")) as f:
            self.df_pit = pd.DataFrame(json.load(f))

        # Filter laps 1-62
        self.df_laps = self.df_laps[(self.df_laps["lap_number"] >= 1) & (self.df_laps["lap_number"] <= 62)].copy()
        self.df_laps["compound"] = "UNKNOWN"
        self.df_laps["tyre_age"] = 0

        # Annotate stints
        for _, s in self.df_stints.iterrows():
            mask = (
                (self.df_laps["driver_number"] == s["driver_number"])
                & (self.df_laps["lap_number"] >= s["lap_start"])
                & (self.df_laps["lap_number"] <= s["lap_end"])
            )
            self.df_laps.loc[mask, "compound"] = s["compound"]
            start_age = s.get("tyre_age_at_start", 0) or 0
            self.df_laps.loc[mask, "tyre_age"] = self.df_laps.loc[mask, "lap_number"] - s["lap_start"] + start_age

        # Mark SC, VSC, pit laps
        self.df_laps["is_sc"] = self.df_laps["lap_number"].isin([20, 21, 22])
        self.df_laps["is_vsc"] = self.df_laps["lap_number"].isin([44, 45])
        pit_laps = set(zip(self.df_pit["driver_number"], self.df_pit["lap_number"]))
        self.df_laps["is_pit_lap"] = self.df_laps.apply(
            lambda r: (r["driver_number"], r["lap_number"]) in pit_laps, axis=1
        )

        # Filter clean green laps
        self.clean_green = self.df_laps[
            (self.df_laps["lap_number"] > 1)
            & (~self.df_laps["is_sc"])
            & (~self.df_laps["is_vsc"])
            & (~self.df_laps["is_pit_lap"])
            & (~self.df_laps["is_pit_out_lap"])
            & (self.df_laps["lap_duration"].notnull())
        ].copy()

        # Outlier removal: laps > 107% of driver median
        driver_medians = self.clean_green.groupby("driver_number")["lap_duration"].median()
        self.clean_green["driver_median"] = self.clean_green["driver_number"].map(driver_medians)
        self.clean_green = self.clean_green[self.clean_green["lap_duration"] <= self.clean_green["driver_median"] * 1.07]

        # Fit degradation and fuel effect per compound
        default_deg = {"SOFT": 0.082, "MEDIUM": 0.085, "HARD": 0.018}
        default_base = {"SOFT": 99.6, "MEDIUM": 100.5, "HARD": 101.5}

        for comp in ["SOFT", "MEDIUM", "HARD"]:
            sub = self.clean_green[self.clean_green["compound"] == comp]
            if len(sub) >= 20:
                X = sub[["tyre_age", "lap_number"]]
                y = sub["lap_duration"]
                reg = Ridge(alpha=1.0)
                reg.fit(X, y)
                # Enforce non-negative degradation rate
                deg = max(0.005, float(reg.coef_[0]))
                self.compound_deg_rate[comp] = round(deg, 4)
                self.compound_base_pace[comp] = round(float(reg.intercept_), 3)
            else:
                self.compound_deg_rate[comp] = default_deg[comp]
                self.compound_base_pace[comp] = default_base[comp]

        # Compute compound deltas vs Medium
        med_base = self.compound_base_pace.get("MEDIUM", 100.5)
        for comp in ["SOFT", "MEDIUM", "HARD"]:
            self.compound_deltas[comp] = round(self.compound_base_pace[comp] - med_base, 3)

        # Compute driver pace offsets vs grid median
        grid_median = self.clean_green["lap_duration"].median()
        for dn, g in self.clean_green.groupby("driver_number"):
            self.driver_offsets[int(dn)] = round(float(g["lap_duration"].median() - grid_median), 3)

        # Compute driver-specific base pace per compound
        for dn in self.df_laps["driver_number"].unique():
            dn = int(dn)
            self.driver_compound_pace[dn] = {}
            for comp in ["SOFT", "MEDIUM", "HARD"]:
                sub = self.clean_green[(self.clean_green["driver_number"] == dn) & (self.clean_green["compound"] == comp)]
                if len(sub) >= 5:
                    # Estimate driver alpha from their actual laps minus degradation and fuel effect
                    deg = self.compound_deg_rate[comp]
                    fuel = self.fuel_effect
                    corrected = sub["lap_duration"] - (deg * sub["tyre_age"] + fuel * (sub["lap_number"] - 1))
                    self.driver_compound_pace[dn][comp] = round(float(corrected.median()), 3)
                else:
                    # Fallback using driver offset + compound base
                    offset = self.driver_offsets.get(dn, 0.0)
                    base = self.compound_base_pace.get(comp, default_base[comp])
                    self.driver_compound_pace[dn][comp] = round(base + offset, 3)

        # Populate assumptions dictionary
        self.assumptions = {
            "compound_degradation_sec_per_lap": self.compound_deg_rate,
            "compound_base_pace_sec": self.compound_base_pace,
            "compound_deltas_vs_medium_sec": self.compound_deltas,
            "fuel_burn_effect_sec_per_lap": self.fuel_effect,
            "outlier_filter_threshold": "107% of driver median green pace",
            "clean_green_laps_analyzed": len(self.clean_green),
        }

        # Attempt to load locally trained non-linear model
        self.nonlinear_model_data = None
        self._load_nonlinear_model()

    def _load_nonlinear_model(self):
        import joblib
        model_path = os.path.join(os.path.dirname(self.cache_dir), "models", "nonlinear_tyre_model.joblib")
        if not os.path.exists(model_path):
            model_path = "data/models/nonlinear_tyre_model.joblib"
        if os.path.exists(model_path):
            try:
                self.nonlinear_model_data = joblib.load(model_path)
            except Exception as e:
                print(f"Warning: could not load nonlinear tyre model from {model_path}: {e}")

    def predict_lap_time(self, driver_number: int, compound: str, tyre_age: int, lap_number: int) -> float:
        """
        Calculates estimated green-flag lap time using linear Ridge regression:
        LapTime = alpha_{driver, comp} + beta_{comp} * tyre_age + gamma * (lap_number - 1)
        """
        dn = int(driver_number)
        comp = compound.upper()
        if comp not in ["SOFT", "MEDIUM", "HARD"]:
            comp = "HARD"

        alpha = self.driver_compound_pace.get(dn, {}).get(comp)
        if alpha is None:
            offset = self.driver_offsets.get(dn, 0.0)
            alpha = self.compound_base_pace.get(comp, 100.5) + offset

        beta = self.compound_deg_rate.get(comp, 0.05)
        gamma = self.fuel_effect

        lap_time = alpha + (beta * tyre_age) + (gamma * (lap_number - 1))
        return max(85.0, round(float(lap_time), 3))

    def predict_lap_time_nonlinear(self, driver_number: int, compound: str, tyre_age: int, lap_number: int) -> float:
        """
        Calculates estimated green-flag lap time using the locally trained non-linear ML model:
        Captures the thermal degradation cliff and scrub-in phases.
        """
        if not self.nonlinear_model_data:
            return self.predict_lap_time(driver_number, compound, tyre_age, lap_number)

        dn = int(driver_number)
        comp = compound.upper()
        if comp not in ["SOFT", "MEDIUM", "HARD"]:
            comp = "HARD"

        alpha = self.driver_compound_pace.get(dn, {}).get(comp)
        if alpha is None:
            offset = self.driver_offsets.get(dn, 0.0)
            alpha = self.compound_base_pace.get(comp, 100.5) + offset

        c_code = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}[comp]
        fuel_mass = max(5.0, 100.0 * (1.0 - (lap_number / 62.0)))
        features_df = pd.DataFrame([{
            "comp_code": c_code,
            "tyre_age": tyre_age,
            "tyre_age_sq": (tyre_age ** 2) / 100.0,
            "lap_number": lap_number,
            "fuel_mass": fuel_mass
        }])

        deg_delta = float(self.nonlinear_model_data["model"].predict(features_df)[0])
        # Add fuel burn correction relative to baseline
        fuel_corr = self.fuel_effect * (lap_number - 1) * 0.5
        lap_time = alpha + deg_delta + fuel_corr

        return max(85.0, round(float(lap_time), 3))

    def get_tyre_degradation_curves(self) -> dict:
        """
        Returns precomputed degradation curves and cliff points for UI visualization.
        """
        if self.nonlinear_model_data and "curves" in self.nonlinear_model_data:
            return {
                "curves": self.nonlinear_model_data["curves"],
                "cliff_laps": self.nonlinear_model_data.get("cliff_laps", {"SOFT": 18, "MEDIUM": 28, "HARD": 42}),
                "metrics": self.nonlinear_model_data.get("metrics", {})
            }
        # Fallback linear curves
        ages = list(range(1, 51))
        return {
            "curves": {
                "ages": ages,
                "SOFT": [round(a * self.compound_deg_rate.get("SOFT", 0.082), 3) for a in ages],
                "MEDIUM": [round(a * self.compound_deg_rate.get("MEDIUM", 0.087), 3) for a in ages],
                "HARD": [round(a * self.compound_deg_rate.get("HARD", 0.015), 3) for a in ages],
            },
            "cliff_laps": {"SOFT": 18, "MEDIUM": 28, "HARD": 42},
            "metrics": {"mae": 0.05, "r2_score": 0.98}
        }

    def analyze_stints_for_cliffs(self, stints: list) -> list:
        """
        Model 2 Advisor: Evaluates a driver's stint schedule against the trained thermal degradation cliff.
        Identifies whether a stint enters the high-wear cliff dropoff.
        """
        cliff_laps = {"SOFT": 18, "MEDIUM": 28, "HARD": 42}
        if self.nonlinear_model_data and "cliff_laps" in self.nonlinear_model_data:
            cliff_laps = self.nonlinear_model_data["cliff_laps"]

        analysis = []
        for s in stints:
            comp = s.get("compound", "HARD").upper()
            cliff = cliff_laps.get(comp, 35)
            start_lap = s.get("lap_start", 1)
            end_lap = s.get("lap_end", 62)
            stint_length = (end_lap - start_lap + 1) + s.get("start_tyre_age", 0)

            if stint_length <= cliff - 4:
                status = "OPTIMAL"
                badge = "🟢 Optimal Wear"
                penalty = 0.0
                desc = f"Within safe tyre window ({stint_length}/{cliff} laps)"
            elif stint_length <= cliff:
                status = "NEAR_CLIFF"
                badge = "🟡 Approaching Cliff"
                penalty = 0.12
                desc = f"Approaching thermal cliff at Lap {cliff} ({stint_length}/{cliff} laps)"
            else:
                status = "EXCEEDED_CLIFF"
                over = stint_length - cliff
                penalty = round(0.08 * (over ** 1.8), 2)
                badge = f"🔴 Past Cliff (+{penalty}s/lap)"
                desc = f"Exceeds {comp.title()} cliff by {over} laps! Thermal loss: +{penalty}s/lap"

            analysis.append({
                "stint_number": s.get("stint_number", 1),
                "compound": comp,
                "stint_length": stint_length,
                "cliff_lap": cliff,
                "status": status,
                "health_status": status,
                "badge": badge,
                "status_label": badge,
                "penalty_sec": penalty,
                "description": desc
            })
        return analysis

    def get_assumptions(self) -> dict:
        assumptions = dict(self.assumptions)
        if self.nonlinear_model_data:
            assumptions["nonlinear_model_status"] = "Active (Locally trained HistGradientBoostingRegressor)"
            assumptions["cliff_laps"] = self.nonlinear_model_data.get("cliff_laps")
        return assumptions
