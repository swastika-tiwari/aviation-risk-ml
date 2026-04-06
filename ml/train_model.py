import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import shap

from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "conflict_model.pkl")

def train_model():

    df = pd.read_csv(DATA_PATH)

    X = compute_features(df)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = XGBClassifier(scale_pos_weight=3)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))

    # Precision-Recall
    probs = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()

    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:100])

    shap.plots.bar(shap_values)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved")

if __name__ == "__main__":
    train_model()