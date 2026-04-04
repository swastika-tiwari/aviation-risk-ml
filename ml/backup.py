import pandas as pd
import json
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("Loading ADS-B trajectory dataset...")

df = pd.read_csv("data/flights.csv")

# Keep required columns
df = df[["time","icao24","lat","lon","geoaltitude","velocity"]]

df = df.dropna()

print("Rows after cleaning:", len(df))

# Convert time
df["time"] = pd.to_datetime(df["time"], unit="s")

# Haversine distance function
def distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians,[lat1,lon1,lat2,lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

print("Detecting near-miss events...")

near_miss_events = []

sample_df = df.sample(min(20000, len(df)))  # keep it fast

for i in range(len(sample_df)-1):
    a = sample_df.iloc[i]
    b = sample_df.iloc[i+1]

    if a["icao24"] != b["icao24"]:
        dist = distance(a["lat"], a["lon"], b["lat"], b["lon"])
        alt_diff = abs(a["geoaltitude"] - b["geoaltitude"])

        if dist < 5 and alt_diff < 300:
            near_miss_events.append({
                "distance_km": dist,
                "altitude_diff": alt_diff,
                "velocity": a["velocity"]
            })

print("Near miss events detected:", len(near_miss_events))

events_df = pd.DataFrame(near_miss_events)

if len(events_df) < 10:
    print("Dataset needs more dense traffic for ML training")

# Create ML dataset
events_df["risk"] = (events_df["distance_km"] < 3).astype(int)

X = events_df[["distance_km","altitude_diff","velocity"]]
y = events_df["risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

results = {
    "near_miss_events_detected": int(len(events_df)),
    "model_accuracy": round(float(acc),3),
    "dataset_size": int(len(df))
}

print("Results:", results)

os.makedirs("web", exist_ok=True)

with open("web/results.json","w") as f:
    json.dump(results,f,indent=4)

print("Dashboard updated.")