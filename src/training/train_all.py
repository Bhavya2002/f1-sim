"""
Master Training Script for all 3 Machine Learning Models:
1. Traffic & Overtaking Probability Model
2. Non-Linear Tyre Degradation Cliff Model
3. Local NLP Intent & Slot Extractor Model
"""

import os
import json
from src.training.train_traffic_model import train_traffic_model
from src.training.train_tyre_model import train_tyre_model
from src.training.train_nlp_model import train_nlp_model


def train_all_models(cache_dir: str = "data/cache/singapore_2023", output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)
    print("\n🏎️ Starting Local Machine Learning Model Training Pipeline...")
    print("=" * 65)

    print("\n[1/3] Training Traffic & Overtaking Probability Model...")
    traffic_metrics = train_traffic_model(cache_dir, output_dir)

    print("\n[2/3] Training Non-Linear Tyre Degradation Cliff Model...")
    tyre_metrics, curves = train_tyre_model(cache_dir, output_dir)

    print("\n[3/3] Training Local NLP Intent & Slot-Filling Model...")
    nlp_metrics = train_nlp_model(output_dir)

    summary = {
        "status": "SUCCESS",
        "models": {
            "traffic_overtaking": {
                "name": "Traffic & Overtake Probability Classifier",
                "type": "Calibrated Logistic Regression with StandardScaler",
                "accuracy": traffic_metrics["accuracy"],
                "roc_auc": traffic_metrics["roc_auc"],
                "features": traffic_metrics["features"],
                "feature_weights": traffic_metrics["feature_weights"],
                "dirty_air_penalty_seconds": traffic_metrics["dirty_air_penalty_seconds"]
            },
            "nonlinear_tyre": {
                "name": "Non-Linear Tyre Degradation Cliff Regressor",
                "type": "HistGradientBoostingRegressor",
                "mae_seconds": tyre_metrics["mae"],
                "r2_score": tyre_metrics["r2_score"],
                "cliff_laps": tyre_metrics["cliff_laps"]
            },
            "nlp_intent": {
                "name": "Natural Language Strategy Intent Parser",
                "type": "TF-IDF + Calibrated Logistic Regression",
                "accuracy": nlp_metrics["accuracy"],
                "total_intents": len(nlp_metrics["intents"]),
                "intents": nlp_metrics["intents"]
            }
        }
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print(f"🎉 All 3 Machine Learning models successfully trained!")
    print(f"📁 Artifacts saved in: {output_dir}")
    print(f"📄 Summary logged to: {summary_path}")
    return summary


if __name__ == "__main__":
    train_all_models()
