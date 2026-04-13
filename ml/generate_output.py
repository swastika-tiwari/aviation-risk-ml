import pandas as pd
import os
import json
import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, roc_curve, auc

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "conflict_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "results.json")


# ==============================
# RISK CLASSIFICATION
# ==============================
def classify_risk(score):
    if score > 0.8:
        return "Critical"
    elif score > 0.5:
        return "High"
    elif score > 0.2:
        return "Medium"
    else:
        return "Low"


# ==============================
# MAIN FUNCTION
# ==============================
def generate_output():

    print("🚀 Generating intelligent output...")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_PATH)
    X = compute_features(df)

    # -----------------------------
    # LOAD MODEL + SCALER
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        model, scaler = pickle.load(f)

    X = scaler.transform(X)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    probs = model.predict_proba(X)[:, 1]

    # 🔥 LOWER THRESHOLD FOR SAFETY
    y_true = df["is_conflict"]
    y_pred = (probs > 0.3).astype(int)

    # -----------------------------
    # LOAD METRICS (fallback safe)
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
    # CLASSIFICATION REPORT
    # -----------------------------
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    report_path = os.path.join(BASE_DIR, "web", "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred))

    # -----------------------------
    # RISK DISTRIBUTION
    # -----------------------------
    distribution = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}

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
    if distribution["Critical"] > 0:
        insight = "Critical near-miss risks detected. Immediate attention required."
    elif distribution["High"] > 5:
        insight = "Multiple high-risk aircraft interactions observed."
    elif distribution["Medium"] > distribution["Low"]:
        insight = "Moderate convergence trends detected in air traffic."
    else:
        insight = "Airspace appears largely safe with low-risk interactions."

    # -----------------------------
    # VISUALIZATIONS
    # -----------------------------
    print("📊 Generating visualizations...")

    # 1. Confusion Matrix
    cm = np.array(metrics["confusion_matrix"])

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(BASE_DIR, "web", "confusion_matrix.png"))
    plt.close()

    # 2. Risk Distribution
    plt.figure()
    plt.bar(distribution.keys(), distribution.values())
    plt.title("Risk Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")
    plt.savefig(os.path.join(BASE_DIR, "web", "risk_distribution.png"))
    plt.close()

    # 3. Top Risk Scores
    top_scores = [case["risk_score"] for case in top_cases]

    plt.figure()
    plt.plot(range(len(top_scores)), top_scores, marker='o')
    plt.title("Top 10 Risk Scores")
    plt.xlabel("Case Rank")
    plt.ylabel("Risk Score")
    plt.savefig(os.path.join(BASE_DIR, "web", "top_risks.png"))
    plt.close()

    # 4. Classification Report Heatmap
    report_df = pd.DataFrame(report_dict).transpose().iloc[:-1, :-1]

    plt.figure()
    sns.heatmap(report_df, annot=True)
    plt.title("Classification Report Heatmap")
    plt.savefig(os.path.join(BASE_DIR, "web", "classification_report.png"))
    plt.close()

    # 5. ROC Curve + AUC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "web", "roc_curve.png"))
    plt.close()

    print("📈 All visualizations saved in /web folder")

    # -----------------------------
    # FINAL OUTPUT JSON
    # -----------------------------
    output = {
        "model_accuracy": float(metrics["accuracy"]),
        "auc_score": float(roc_auc),

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

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print("✅ Intelligent results.json generated")


if __name__ == "__main__":
    generate_output()