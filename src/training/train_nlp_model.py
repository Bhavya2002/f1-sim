"""
Local Natural Language Processing (NLP) Intent & Slot-Filling Model Training Script.

Trains a calibrated machine learning intent classification model and slot extractor
for natural language Formula 1 race strategy commands and analytical questions.
"""

import os
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# Comprehensive Dataset of F1 Strategy Prompts and Target Intents
TRAINING_DATA = [
    # MODIFY_PIT_LAP
    ("make Sainz pit one lap earlier", "MODIFY_PIT_LAP"),
    ("pit Sainz one lap earlier", "MODIFY_PIT_LAP"),
    ("pit Carlos one lap earlier", "MODIFY_PIT_LAP"),
    ("bring Sainz in one lap earlier", "MODIFY_PIT_LAP"),
    ("box Sainz one lap earlier", "MODIFY_PIT_LAP"),
    ("make Sainz pit two laps earlier", "MODIFY_PIT_LAP"),
    ("pit Sainz 2 laps earlier", "MODIFY_PIT_LAP"),
    ("pit Sainz earlier", "MODIFY_PIT_LAP"),
    ("pit Sainz one lap later", "MODIFY_PIT_LAP"),
    ("box Sainz one lap later", "MODIFY_PIT_LAP"),
    ("extend Sainz stint by 2 laps", "MODIFY_PIT_LAP"),
    ("extend stint 1 lap later", "MODIFY_PIT_LAP"),
    ("delay pit stop by 3 laps", "MODIFY_PIT_LAP"),
    ("pit Norris on lap 18", "MODIFY_PIT_LAP"),
    ("box Lando on lap 19", "MODIFY_PIT_LAP"),
    ("pit Hamilton on lap 25", "MODIFY_PIT_LAP"),
    ("bring Hamilton into the pits on lap 24", "MODIFY_PIT_LAP"),
    ("box Charles on lap 22", "MODIFY_PIT_LAP"),
    ("pit Leclerc on lap 16", "MODIFY_PIT_LAP"),
    ("pit Max on lap 35", "MODIFY_PIT_LAP"),
    ("box Verstappen on lap 40", "MODIFY_PIT_LAP"),
    ("undercut Russell by pitting on lap 18", "MODIFY_PIT_LAP"),
    ("pit car 55 on lap 19", "MODIFY_PIT_LAP"),
    ("make #44 pit on lap 21", "MODIFY_PIT_LAP"),
    ("change pit lap to 17", "MODIFY_PIT_LAP"),
    ("stop earlier on lap 15", "MODIFY_PIT_LAP"),

    # CHANGE_COMPOUND
    ("switch Norris to soft tyres", "CHANGE_COMPOUND"),
    ("change Norris compound to soft", "CHANGE_COMPOUND"),
    ("put soft tyres on Norris", "CHANGE_COMPOUND"),
    ("put softs on Norris", "CHANGE_COMPOUND"),
    ("fit soft tyres for Hamilton", "CHANGE_COMPOUND"),
    ("switch Hamilton to softs", "CHANGE_COMPOUND"),
    ("change compound to hard", "CHANGE_COMPOUND"),
    ("switch to hard tyres for stint 2", "CHANGE_COMPOUND"),
    ("put mediums on Carlos", "CHANGE_COMPOUND"),
    ("fit yellow walled mediums", "CHANGE_COMPOUND"),
    ("use soft compound for final stint", "CHANGE_COMPOUND"),
    ("switch to hard compound", "CHANGE_COMPOUND"),
    ("put red tyres on Sainz", "CHANGE_COMPOUND"),
    ("change tyre to medium", "CHANGE_COMPOUND"),
    ("run hards until the end", "CHANGE_COMPOUND"),

    # SET_STOP_COUNT
    ("switch Norris to a 2-stop strategy", "SET_STOP_COUNT"),
    ("convert to 2 stop strategy", "SET_STOP_COUNT"),
    ("make Hamilton 1-stop", "SET_STOP_COUNT"),
    ("switch Hamilton to 1 stop", "SET_STOP_COUNT"),
    ("change to a two stop plan", "SET_STOP_COUNT"),
    ("make Sainz do a two-stop", "SET_STOP_COUNT"),
    ("switch to a 1-stop strategy", "SET_STOP_COUNT"),
    ("convert Leclerc to 3 stops", "SET_STOP_COUNT"),
    ("3 stop strategy for Perez", "SET_STOP_COUNT"),
    ("change strategy to one stop", "SET_STOP_COUNT"),
    ("run a 2 stop race", "SET_STOP_COUNT"),
    ("make it a two stop", "SET_STOP_COUNT"),
    ("switch to one stop", "SET_STOP_COUNT"),

    # REMOVE_SAFETY_CAR
    ("what if there was no safety car?", "REMOVE_SAFETY_CAR"),
    ("what if no safety car", "REMOVE_SAFETY_CAR"),
    ("remove the safety car", "REMOVE_SAFETY_CAR"),
    ("remove safety car", "REMOVE_SAFETY_CAR"),
    ("cancel safety car on lap 20", "REMOVE_SAFETY_CAR"),
    ("what if SC never happened?", "REMOVE_SAFETY_CAR"),
    ("without the safety car", "REMOVE_SAFETY_CAR"),
    ("no safety car", "REMOVE_SAFETY_CAR"),
    ("eliminate the safety car", "REMOVE_SAFETY_CAR"),
    ("delete safety car period", "REMOVE_SAFETY_CAR"),

    # REMOVE_VSC
    ("remove the virtual safety car", "REMOVE_VSC"),
    ("remove virtual safety car", "REMOVE_VSC"),
    ("what if no virtual safety car", "REMOVE_VSC"),
    ("what if no vsc", "REMOVE_VSC"),
    ("remove vsc", "REMOVE_VSC"),
    ("cancel vsc", "REMOVE_VSC"),
    ("without virtual safety car", "REMOVE_VSC"),
    ("no vsc on lap 44", "REMOVE_VSC"),
    ("eliminate vsc", "REMOVE_VSC"),
    ("cancel the lap 44 vsc", "REMOVE_VSC"),

    # INSERT_SAFETY_CAR
    ("insert safety car on lap 35", "INSERT_SAFETY_CAR"),
    ("deploy safety car on lap 30", "INSERT_SAFETY_CAR"),
    ("what if a safety car came out on lap 28?", "INSERT_SAFETY_CAR"),
    ("add safety car at lap 32", "INSERT_SAFETY_CAR"),
    ("trigger safety car on lap 40", "INSERT_SAFETY_CAR"),
    ("deploy SC on lap 25", "INSERT_SAFETY_CAR"),
    ("insert sc on lap 35", "INSERT_SAFETY_CAR"),

    # INSERT_VSC
    ("insert virtual safety car on lap 30", "INSERT_VSC"),
    ("insert vsc on lap 30", "INSERT_VSC"),
    ("deploy vsc on lap 25", "INSERT_VSC"),
    ("deploy virtual safety car at lap 35", "INSERT_VSC"),
    ("trigger vsc on lap 32", "INSERT_VSC"),
    ("what if virtual safety car on lap 28?", "INSERT_VSC"),
    ("add vsc at lap 30", "INSERT_VSC"),

    # OPTIMIZE_PIT_STOP
    ("optimize pit stop for Leclerc", "OPTIMIZE_PIT_STOP"),
    ("replace slow pit stop", "OPTIMIZE_PIT_STOP"),
    ("what if Leclerc had a faster pit stop?", "OPTIMIZE_PIT_STOP"),
    ("make pit stop faster", "OPTIMIZE_PIT_STOP"),
    ("use team median stop", "OPTIMIZE_PIT_STOP"),
    ("optimize pit stop", "OPTIMIZE_PIT_STOP"),
    ("fix Leclerc slow pit stop", "OPTIMIZE_PIT_STOP"),
    ("give Sainz a fast stop", "OPTIMIZE_PIT_STOP"),

    # RESET_BASELINE
    ("reset to actual race", "RESET_BASELINE"),
    ("reset all", "RESET_BASELINE"),
    ("restore baseline", "RESET_BASELINE"),
    ("reset simulation", "RESET_BASELINE"),
    ("go back to real race", "RESET_BASELINE"),
    ("clear what-if changes", "RESET_BASELINE"),

    # ANALYTICAL_QA
    ("why did Verstappen win when the safety car was removed?", "ANALYTICAL_QA"),
    ("why did Verstappen gain positions without SC?", "ANALYTICAL_QA"),
    ("what is the tyre degradation on Mediums?", "ANALYTICAL_QA"),
    ("what is tyre degradation for soft tyres?", "ANALYTICAL_QA"),
    ("how much time is lost during a green flag pit stop?", "ANALYTICAL_QA"),
    ("how does safety car bunching work?", "ANALYTICAL_QA"),
    ("why did Mercedes pit under VSC?", "ANALYTICAL_QA"),
    ("explain the pace model", "ANALYTICAL_QA"),
    ("who won the actual race?", "ANALYTICAL_QA"),
    ("what was Carlos Sainz strategy?", "ANALYTICAL_QA"),
    ("how did Hamilton overtake Leclerc?", "ANALYTICAL_QA"),
    ("tell me about the Singapore 2023 Grand Prix", "ANALYTICAL_QA")
]


DRIVER_ALIASES = {
    55: ["sainz", "carlos", "carlos sainz", "#55", "sai"],
    4: ["norris", "lando", "lando norris", "#4", "nor"],
    44: ["hamilton", "lewis", "lewis hamilton", "#44", "ham"],
    16: ["leclerc", "charles", "charles leclerc", "#16", "lec"],
    1: ["verstappen", "max", "max verstappen", "#1", "ver"],
    63: ["russell", "george", "george russell", "#63", "rus"],
    11: ["perez", "sergio", "checo", "#11", "per"],
    10: ["gasly", "pierre", "#10", "gas"],
    81: ["piastri", "oscar", "#81", "pia"],
    40: ["lawson", "liam", "#40", "law"],
    20: ["magnussen", "kevin", "#20", "mag"]
}


class StrategyNLPModel:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_intent(self, text: str):
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_idx = np.argmax(probs)
        intent = classes[top_idx]
        confidence = float(probs[top_idx])

        # Extract slots
        slots = self.extract_slots(text)

        return {
            "intent": intent,
            "confidence": round(confidence, 3),
            "all_probabilities": {cls: round(float(p), 3) for cls, p in zip(classes, probs)},
            "slots": slots
        }

    def extract_slots(self, text: str):
        slots = {
            "driver_number": None,
            "driver_name": None,
            "target_lap": None,
            "delta_laps": None,
            "compound": None,
            "stop_count": None
        }
        t = text.lower()

        # Match driver
        for dn, aliases in DRIVER_ALIASES.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", t):
                    slots["driver_number"] = dn
                    slots["driver_name"] = aliases[0].title()
                    break
            if slots["driver_number"]:
                break

        # Match pit lap
        lap_match = re.search(r"\blap\s*(\d+)\b", t) or re.search(r"\bon\s*(\d+)\b", t)
        if lap_match:
            slots["target_lap"] = int(lap_match.group(1))

        # Relative lap delta
        if "one lap earlier" in t or "1 lap earlier" in t:
            slots["delta_laps"] = -1
        elif "two laps earlier" in t or "2 laps earlier" in t:
            slots["delta_laps"] = -2
        elif "one lap later" in t or "1 lap later" in t:
            slots["delta_laps"] = 1
        elif "two laps later" in t or "2 laps later" in t:
            slots["delta_laps"] = 2

        # Compound
        if re.search(r"\b(soft|softs|red)\b", t):
            slots["compound"] = "SOFT"
        elif re.search(r"\b(medium|mediums|yellow)\b", t):
            slots["compound"] = "MEDIUM"
        elif re.search(r"\b(hard|hards|white)\b", t):
            slots["compound"] = "HARD"

        # Stop count
        if "1-stop" in t or "1 stop" in t or "one stop" in t or "one-stop" in t:
            slots["stop_count"] = 1
        elif "2-stop" in t or "2 stop" in t or "two stop" in t or "two-stop" in t:
            slots["stop_count"] = 2
        elif "3-stop" in t or "3 stop" in t or "three stop" in t or "three-stop" in t:
            slots["stop_count"] = 3

        return slots


def train_nlp_model(output_dir: str = "data/models"):
    os.makedirs(output_dir, exist_ok=True)

    X_texts = [item[0] for item in TRAINING_DATA]
    y_labels = [item[1] for item in TRAINING_DATA]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, max_features=500)),
        ("clf", LogisticRegression(C=5.0, max_iter=200, random_state=42))
    ])

    pipeline.fit(X_texts, y_labels)
    y_pred = pipeline.predict(X_texts)
    acc = accuracy_score(y_labels, y_pred)

    nlp_wrapper = StrategyNLPModel(pipeline)

    model_path = os.path.join(output_dir, "nlp_intent_model.joblib")
    joblib.dump(nlp_wrapper, model_path)

    metrics = {
        "accuracy": round(float(acc), 3),
        "intents": list(pipeline.classes_),
        "sample_count": len(TRAINING_DATA)
    }

    print(f"✅ NLP Intent Model trained & saved to {model_path}")
    print(f"   Accuracy: {metrics['accuracy'] * 100:.1f}%, Total Intents: {len(metrics['intents'])}")
    return metrics


if __name__ == "__main__":
    train_nlp_model()
