import pandas as pd
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "conflict_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")


def train_model():

    print("🚀 Training model...")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded:", df.shape)

    # -----------------------------
    # FEATURES + LABEL
    # -----------------------------
    X = compute_features(df)
    y = df["is_conflict"]

    print("Class distribution:")
    print(y.value_counts())

    # -----------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # -----------------------------
    # MODEL (ANTI-OVERFITTING)
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_split=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # EVALUATION
    # -----------------------------
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report_dict = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))

    print("\n🎯 Accuracy:", acc)

    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved")

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    metrics = {
        "accuracy": float(acc),
        "precision": float(report_dict["1"]["precision"]),
        "recall": float(report_dict["1"]["recall"]),
        "f1_score": float(report_dict["1"]["f1-score"]),
        "confusion_matrix": cm.tolist()
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Metrics saved")


if __name__ == "__main__":
    train_model()