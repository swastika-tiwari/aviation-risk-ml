import pandas as pd
import numpy as np
import math
import json
import os
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "flight.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "web", "results.json")

HORIZONTAL_THRESHOLD_KM = 9.26
VERTICAL_THRESHOLD_FT = 1000

# ==============================
# HAVERSINE
# ==============================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


# ==============================
# PREPROCESSING
# ==============================

def preprocess():
    df = pd.read_csv(DATA_PATH)

    df = df[df['onground'] == False]

    df = df.dropna(subset=['lat', 'lon', 'geoaltitude', 'velocity', 'time'])

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna()

    df['hour'] = (df['time'] // 3600).astype(int)

    return df


# ==============================
# POTENTIAL PAIRS
# ==============================

def get_pairs(df):
    pairs = []

    grouped = df.groupby('hour')

    for _, group in grouped:

        if len(group) > 50:
            group = group.sample(50, random_state=42)

        records = group.to_dict('records')

        for a1, a2 in combinations(records, 2):

            if abs(a1['lat'] - a2['lat']) > 1:
                continue

            pairs.append((a1, a2))

    return pairs


# ==============================
# CPA CALCULATION (IMPROVED)
# ==============================

def compute_cpa(a1, a2):
    x1, y1 = a1['lat'], a1['lon']
    x2, y2 = a2['lat'], a2['lon']

    v1 = a1.get('velocity', 0)
    v2 = a2.get('velocity', 0)

    theta1 = np.random.uniform(0, 2*np.pi)
    theta2 = np.random.uniform(0, 2*np.pi)

    vx1, vy1 = v1 * np.cos(theta1), v1 * np.sin(theta1)
    vx2, vy2 = v2 * np.cos(theta2), v2 * np.sin(theta2)

    dx, dy = x2 - x1, y2 - y1
    dvx, dvy = vx2 - vx1, vy2 - vy1

    denom = dvx**2 + dvy**2
    if denom == 0:
        return haversine(x1, y1, x2, y2)

    t = -(dx*dvx + dy*dvy) / denom

    cx1 = x1 + vx1 * t
    cy1 = y1 + vy1 * t
    cx2 = x2 + vx2 * t
    cy2 = y2 + vy2 * t

    return haversine(cx1, cy1, cx2, cy2)


# ==============================
# SYNTHETIC DATA GENERATION
# ==============================

def generate_data(pairs):
    data = []

    for a1, a2 in pairs:

        # original pair
        dist = haversine(a1['lat'], a1['lon'], a2['lat'], a2['lon'])
        alt_diff = abs(a1['geoaltitude'] - a2['geoaltitude'])

        label_real = 1 if (dist < HORIZONTAL_THRESHOLD_KM and alt_diff < VERTICAL_THRESHOLD_FT) else 0
        data.append((a1, a2, dist, alt_diff, label_real))

        # synthetic (realistic)
        a2_syn = a2.copy()

        a2_syn['lat'] = a1['lat'] + np.random.uniform(-0.05, 0.05)
        a2_syn['lon'] = a1['lon'] + np.random.uniform(-0.05, 0.05)
        a2_syn['geoaltitude'] = a1['geoaltitude'] + np.random.uniform(-1500, 1500)

        dist_syn = haversine(a1['lat'], a1['lon'], a2_syn['lat'], a2_syn['lon'])
        alt_syn = abs(a1['geoaltitude'] - a2_syn['geoaltitude'])

        label_syn = 1 if (dist_syn < HORIZONTAL_THRESHOLD_KM and alt_syn < VERTICAL_THRESHOLD_FT) else 0

        data.append((a1, a2_syn, dist_syn, alt_syn, label_syn))

    return data


# ==============================
# FEATURE ENGINEERING
# ==============================

def compute_features(data):
    X, y, pair_names = [], [], []

    for a1, a2, dist, alt_diff, label in data:

        vel1 = a1.get('velocity', 0)
        vel2 = a2.get('velocity', 0)

        closure_rate = abs(vel1 - vel2)

        cpa = compute_cpa(a1, a2)

        time_diff = abs(a1['time'] - a2['time'])

        X.append([dist, alt_diff, closure_rate, cpa, time_diff])
        y.append(label)

        pair_names.append((a1.get('callsign', 'UNK'), a2.get('callsign', 'UNK')))

    return np.array(X), np.array(y), pair_names


# ==============================
# MODEL TRAINING
# ==============================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, preds))

    return model, acc


# ==============================
# DETECT RISKS
# ==============================

def detect_risks(model, X, pair_names):
    preds = model.predict(X)

    detected = []
    for i, p in enumerate(preds):
        if p == 1:
            detected.append(pair_names[i])

    return detected


# ==============================
# SAVE RESULTS
# ==============================

def save_results(acc, detected_pairs, y):
    results = {
    "model_accuracy": float(acc),

    "metrics": {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"]
    },

    "confusion_matrix": cm.tolist(),

    "summary": {
        "total": int(len(y)),
        "conflicts": int(sum(y)),
        "safe": int(len(y) - sum(y))
    },

    "sample_risks": detected_pairs[:10]
}

    print("Writing results to:", OUTPUT_PATH)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ results.json updated")


# ==============================
# MAIN
# ==============================

def main():
    print("🚀 Starting pipeline...")

    df = preprocess()
    print(f"Loaded {len(df)} rows")

    pairs = get_pairs(df)
    print(f"Generated {len(pairs)} potential pairs")

    data = generate_data(pairs)

    X, y, pair_names = compute_features(data)

    model, acc = train_model(X, y)
    print(f"\n🎯 Model Accuracy: {acc:.4f}")

    detected = detect_risks(model, X, pair_names)

    save_results(acc, detected, y)

    print("🎯 Done.")


if __name__ == "__main__":
    main()