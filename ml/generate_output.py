import pandas as pd
import numpy as np
import math
import json
import os
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# CONFIG
# ==============================

DATA_PATH = "data/flights.csv"
OUTPUT_PATH = "web/results.json"

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

    # Filter airborne only
    df = df[df['onground'] == False]

    # Drop missing
    df = df.dropna(subset=['lat', 'lon', 'geoaltitude', 'velocity'])

    # Convert time
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna()

    # Add hour bin
    df['hour'] = (df['time'] // 3600).astype(int)

    return df


# ==============================
# POTENTIAL PAIRS
# ==============================

def get_pairs(df):
    pairs = []

    grouped = df.groupby('hour')

    for _, group in grouped:

        # sample to control explosion
        if len(group) > 50:
            group = group.sample(50, random_state=42)

        records = group.to_dict('records')

        for a1, a2 in combinations(records, 2):

            # spatial filter
            if abs(a1['lat'] - a2['lat']) > 1:
                continue

            pairs.append((a1, a2))

    return pairs


# ==============================
# SYNTHETIC CONFLICT GENERATION
# ==============================

def generate_synthetic_pairs(pairs):
    data = []

    for a1, a2 in pairs:

        # original safe pair
        dist = haversine(a1['lat'], a1['lon'], a2['lat'], a2['lon'])
        alt_diff = abs(a1['geoaltitude'] - a2['geoaltitude'])

        # label safe
        data.append((a1, a2, dist, alt_diff, 0))

        # create synthetic conflict (nudge)
        a2_syn = a2.copy()

        # force closer position
        a2_syn['lat'] = a1['lat'] + np.random.uniform(-0.01, 0.01)
        a2_syn['lon'] = a1['lon'] + np.random.uniform(-0.01, 0.01)
        a2_syn['geoaltitude'] = a1['geoaltitude'] + np.random.uniform(-500, 500)

        dist_syn = haversine(a1['lat'], a1['lon'], a2_syn['lat'], a2_syn['lon'])
        alt_syn = abs(a1['geoaltitude'] - a2_syn['geoaltitude'])

        data.append((a1, a2_syn, dist_syn, alt_syn, 1))

    return data


# ==============================
# FEATURE ENGINEERING
# ==============================

def compute_features(data):
    X = []
    y = []
    pair_names = []

    for a1, a2, dist, alt_diff, label in data:

        # closure rate (simplified)
        vel1 = a1.get('velocity', 0)
        vel2 = a2.get('velocity', 0)
        closure_rate = abs(vel1 - vel2)

        # CPA (simplified proxy)
        cpa = dist - closure_rate * 0.01

        X.append([
            dist,
            alt_diff,
            closure_rate,
            cpa
        ])

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

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

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

    return detected, preds


# ==============================
# SAVE RESULTS
# ==============================

def save_results(acc, detected_pairs, y):
    results = {
        "model_accuracy": float(acc),
        "detected_risks": list(set([f"{a}-{b}" for a, b in detected_pairs])),
        "separation_breach_count": {
            "total": int(len(y)),
            "synthetic": int(sum(y)),
            "safe": int(len(y) - sum(y))
        }
    }

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

    data = generate_synthetic_pairs(pairs)

    X, y, pair_names = compute_features(data)

    model, acc = train_model(X, y)
    print(f"Model Accuracy: {acc:.4f}")

    detected, preds = detect_risks(model, X, pair_names)

    save_results(acc, detected, y)

    print("🎯 Done.")


if __name__ == "__main__":
    main()