import pandas as pd
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "conflict_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "metrics.json")


def train_model():
    print("🚀 Training model...")

    df = pd.read_csv(DATA_PATH)

    X = compute_features(df)
    y = df["is_conflict"]

    print("Class distribution:\n", y.value_counts())

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # SMOTE
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)

    print(classification_report(y_test, preds))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, scaler), f)

    metrics = {
        "accuracy": acc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Model + metrics saved")


if __name__ == "__main__":
    train_model()