import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "conflict_model.pkl")

def train_model():

    df = pd.read_csv(DATA_PATH)

    X = compute_features(df)
    y = df["is_conflict"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=10)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    print("Accuracy:", accuracy_score(y_test, preds))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("Model saved")

if __name__ == "__main__":
    train_model()