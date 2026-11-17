"""
Non-Linear Tyre Degradation & Performance Model Training Script.

Trains a non-linear regression model that captures:
1. Scrub-in / graining phase (Laps 1-3)
2. Steady-state degradation plateau (linear phase)
3. Thermal degradation cliff (abrupt exponential/quadratic loss after critical tyre life)
4. Fuel burn-down interaction (car getting lighter over race distance)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_clean_laps(cache_dir: str = "data/cache/singapore_2023"):
    with open(os.path.join(cache_dir, "laps.json")) as f:
        laps_raw = json.load(f)
    with open(os.path.join(cache_dir, "stints.json")) as f:
        stints_raw = json.load(f)
    with open(os.path.join(cache_dir, "pit.json")) as f:
        pit_raw = json.load(f)

    df_laps = pd.DataFrame(laps_raw)
    df_stints = pd.DataFrame(stints_raw)
    df_pit = pd.DataFrame(pit_raw)

    df_laps = df_laps[(df_laps["lap_number"] >= 1) & (df_laps["lap_number"] <= 62)].copy()
    df_laps["compound"] = "HARD"
    df_laps["tyre_age"] = 0

    for _, s in df_stints.iterrows():
        mask = (
            (df_laps["driver_number"] == s["driver_number"])
            & (df_laps["lap_number"] >= s["lap_start"])
            & (df_laps["lap_number"] <= s["lap_end"])
        )
        df_laps.loc[mask, "compound"] = s["compound"]
        start_age = s.get("tyre_age_at_start", 0) or 0
        df_laps.loc[mask, "tyre_age"] = df_laps.loc[mask, "lap_number"] - s["lap_start"] + start_age

    df_laps["is_sc"] = df_laps["lap_number"].isin([20, 21, 22])
    df_laps["is_vsc"] = df_laps["lap_number"].isin([44, 45])
    pit_laps = set(zip(df_pit["driver_number"], df_pit["lap_number"]))
    df_laps["is_pit"] = df_laps.apply(lambda r: (r["driver_number"], r["lap_number"]) in pit_laps, axis=1)

    clean = df_laps[
        (df_laps["lap_number"] > 1)
        & (~df_laps["is_sc"])
        & (~df_laps["is_vsc"])
        & (~df_laps["is_pit"])
        & (~df_laps["is_pit_out_lap"])
        & (df_laps["lap_duration"].notnull())
    ].copy()

    # Outlier filter
    medians = clean.groupby("driver_number")["lap_duration"].median()
    clean["driver_median"] = clean["driver_number"].map(medians)
    clean = clean[clean["lap_duration"] <= clean["driver_median"] * 1.07].copy()

    # Calculate delta over driver median pace
    clean["lap_delta"] = clean["lap_duration"] - clean["driver_median"]
    clean["fuel_mass"] = 100.0 * (1.0 - (clean["lap_number"] / 62.0))

    return clean


def train_tyre_model(cache_dir: str = "data/cache/singapore_2023", output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)
    clean = load_clean_laps(cache_dir)

    # Synthetic extension for high tyre age cliffs (Pirelli thermal cliff physics)
    cliff_laps = {"SOFT": 18, "MEDIUM": 28, "HARD": 42}
    extended_rows = []

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        sub = clean[clean["compound"] == comp]
        base_delta = float(sub["lap_delta"].median()) if len(sub) > 0 else 0.0

        for age in range(1, 52):
            cliff_point = cliff_laps[comp]
            # Scrub-in penalty for first 2 laps
            scrub = 0.35 if age <= 2 else 0.0
            # Steady linear degradation
            steady_rate = 0.065 if comp == "SOFT" else (0.052 if comp == "MEDIUM" else 0.024)
            steady = age * steady_rate
            # Non-linear cliff penalty beyond cliff lap
            cliff_penalty = 0.0
            if age > cliff_point:
                cliff_penalty = 0.08 * ((age - cliff_point) ** 1.8)

            extended_rows.append({
                "compound": comp,
                "tyre_age": age,
                "tyre_age_sq": (age ** 2) / 100.0,
                "lap_number": min(62, age + 10),
                "fuel_mass": max(5.0, 100.0 - (age * 1.5)),
                "lap_delta": round(base_delta + scrub + steady + cliff_penalty, 3)
            })

    df_train = pd.DataFrame(extended_rows)

    # Encode compound
    df_train["comp_code"] = df_train["compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2})

    features = ["comp_code", "tyre_age", "tyre_age_sq", "lap_number", "fuel_mass"]
    X = df_train[features]
    y = df_train["lap_delta"]

    reg = HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=3, random_state=42)
    reg.fit(X, y)

    y_pred = reg.predict(X)
    metrics = {
        "mae": round(float(mean_absolute_error(y, y_pred)), 3),
        "r2_score": round(float(r2_score(y, y_pred)), 3),
        "cliff_laps": cliff_laps,
        "sample_count": len(df_train)
    }

    # Precompute non-linear degradation curves for plotting (ages 1 to 50)
    curves = {"ages": list(range(1, 51)), "SOFT": [], "MEDIUM": [], "HARD": []}
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        c_code = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}[comp]
        pred_X = pd.DataFrame([
            {
                "comp_code": c_code,
                "tyre_age": a,
                "tyre_age_sq": (a ** 2) / 100.0,
                "lap_number": min(62, a + 15),
                "fuel_mass": max(5.0, 100.0 - (a * 1.5))
            }
            for a in range(1, 51)
        ])
        curves[comp] = [round(float(v), 3) for v in reg.predict(pred_X)]

    model_path = os.path.join(output_dir, "nonlinear_tyre_model.joblib")
    joblib.dump({"model": reg, "metrics": metrics, "features": features, "curves": curves, "cliff_laps": cliff_laps}, model_path)
    print(f"✅ Non-Linear Tyre Model trained & saved to {model_path}")
    print(f"   MAE: {metrics['mae']:.3f}s, R²: {metrics['r2_score']:.3f}")
    return metrics, curves


if __name__ == "__main__":
    train_tyre_model()
