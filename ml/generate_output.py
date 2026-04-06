import pandas as pd
import os
import json
import pickle
import numpy as np

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "conflict_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "results.json")


def classify_risk(score):
    if score > 0.8:
        return "Critical"
    elif score > 0.5:
        return "High"
    elif score > 0.2:
        return "Medium"
    else:
        return "Low"


def generate_output():

    print("🚀 Generating intelligent output...")

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
    # RISK DISTRIBUTION
    # -----------------------------
    distribution = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0
    }

    for score in probs:
        label = classify_risk(score)
        distribution[label] += 1

    # -----------------------------
    # TOP RISKY CASES
    # -----------------------------
    top_indices = np.argsort(probs)[-10:][::-1]

    top_cases = []

    for i in top_indices:
        score = float(probs[i])
        label = classify_risk(score)

        top_cases.append({
            "pair_id": int(i),
            "risk_score": round(score, 3),
            "risk_level": label
        })

    # -----------------------------
    # INSIGHT GENERATION
    # -----------------------------
    insight = ""

    if distribution["Critical"] > 0:
        insight = "Critical near-miss risks detected. Immediate attention required."
    elif distribution["High"] > 5:
        insight = "Multiple high-risk aircraft interactions observed."
    elif distribution["Medium"] > distribution["Low"]:
        insight = "Moderate convergence trends detected in air traffic."
    else:
        insight = "Airspace appears largely safe with low-risk interactions."

    # -----------------------------
    # FINAL OUTPUT
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

        "risk_distribution": distribution,

        "top_risks": top_cases,

        "insight": insight
    }

    # -----------------------------
    # SAVE
    # -----------------------------
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("✅ Intelligent results.json generated")


if __name__ == "__main__":
    generate_output()