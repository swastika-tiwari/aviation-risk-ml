import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/flights.csv")

# Convert timestamps
df["firstseen"] = pd.to_datetime(df["firstseen"], unit="s")
df["lastseen"] = pd.to_datetime(df["lastseen"], unit="s")

# Create flight duration feature
df["duration_minutes"] = (df["lastseen"] - df["firstseen"]).dt.total_seconds() / 60

# Drop rows with missing airports
df = df.dropna(subset=["estdepartureairport", "estarrivalairport"])

# Create simple risk label (long flights + unknown aircraft model)
df["risk"] = ((df["duration_minutes"] > 180) | (df["model"].isna())).astype(int)

# Features
X = df[["duration_minutes"]]
y = df["risk"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save results
results = {
    "model_accuracy": float(acc),
    "samples_used": int(len(df)),
    "avg_duration": float(df["duration_minutes"].mean()),
}

with open("output/results.json", "w") as f:
    json.dump(results, f)

print("Model accuracy:", acc)
print("Results saved!")