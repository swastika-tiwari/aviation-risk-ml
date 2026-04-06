import pandas as pd
import os
import json
import pickle
from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "conflict_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "results.json")

def generate_output():

    df = pd.read_csv(DATA_PATH)

    X = compute_features(df)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(X)[:, 1]

    results = []

    for i in range(min(50, len(df))):

        risk_score = float(probs[i])

        results.append({
            "aircraft_1": "A1",
            "aircraft_2": "A2",
            "risk_score": risk_score,
            "status": "Critical" if risk_score > 0.8 else "Safe"
        })

    output = {
        "total_samples": len(df),
        "high_risk_cases": sum([1 for r in results if r["status"] == "Critical"]),
        "sample_results": results
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("results.json updated")

if __name__ == "__main__":
    generate_output()