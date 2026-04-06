import pandas as pd
import numpy as np
import os
import math

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed_flights.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")

EARTH_RADIUS = 6371

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS * math.atan2(math.sqrt(a), math.sqrt(1-a))

def generate_shadow(a):
    # Start ~20 miles away (~32 km)
    offset_lat = np.random.uniform(-0.3, 0.3)
    offset_lon = np.random.uniform(-0.3, 0.3)

    shadow = a.copy()

    shadow["lat"] = a["lat"] + offset_lat
    shadow["lon"] = a["lon"] + offset_lon

    return shadow

def add_noise(val, scale):
    return val + np.random.normal(0, scale)

def generate_dataset():
    df = pd.read_csv(INPUT_PATH)

    sample = df.sample(5000, random_state=42)

    data = []

    for _, a in sample.iterrows():

        shadow = generate_shadow(a)

        # simulate convergence
        dist = haversine(a["lat"], a["lon"], shadow["lat"], shadow["lon"])

        # add noise
        shadow["lat"] = add_noise(shadow["lat"], 0.01)
        shadow["lon"] = add_noise(shadow["lon"], 0.01)
        shadow["baroaltitude"] = add_noise(a["baroaltitude"], 200)

        # decide conflict vs near-miss
        if np.random.rand() < 0.5:
            # conflict
            label = 1
        else:
            # near miss (divert)
            shadow["lat"] += np.random.uniform(0.02, 0.05)
            label = 0

        data.append({
            "lat1": a["lat"],
            "lon1": a["lon"],
            "alt1": a["baroaltitude"],
            "vel1": a["velocity"],
            "lat2": shadow["lat"],
            "lon2": shadow["lon"],
            "alt2": shadow["baroaltitude"],
            "vel2": shadow["velocity"],
            "time": a["time"],
            "label": label
        })

    df_out = pd.DataFrame(data)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Training dataset created → {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_dataset()