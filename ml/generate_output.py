import pandas as pd
import os
import json
import pickle

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "conflict_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "results.json")


def generate_output():

    print("🚀 Generating output for frontend...")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    X = compute_features(df)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X)[:, 1]

    # -----------------------------
    # LOAD METRICS
    # -----------------------------
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "confusion_matrix": [[0, 0], [0, 0]]
        }

    # -----------------------------
    # SAMPLE RISK OUTPUT
    # -----------------------------
    sample_results = []

    for i in range(min(10, len(df))):
        risk_score = float(probs[i])

        status = "Critical" if risk_score > 0.8 else "Safe"

        sample_results.append(
            f"Pair {i+1} → Risk: {round(risk_score, 3)} ({status})"
        )

    # -----------------------------
    # FINAL JSON STRUCTURE
    # -----------------------------
    output = {
        "model_accuracy": float(metrics["accuracy"]),

        "metrics": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"])
        },

        "confusion_matrix": metrics["confusion_matrix"],

        "summary": {
            "total": int(len(df)),
            "conflicts": int(df["is_conflict"].sum()),
            "safe": int(len(df) - df["is_conflict"].sum())
        },

        "sample_risks": sample_results
    }

    # -----------------------------
    # SAVE JSON
    # -----------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("✅ results.json updated successfully")


if __name__ == "__main__":
    generate_output()