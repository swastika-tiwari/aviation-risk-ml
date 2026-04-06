import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed_flights.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")

def generate():
    df = pd.read_csv(INPUT_PATH)

    df["hour"] = (df["time"] // 3600).astype(int)

    pairs = []

    grouped = df.groupby("hour")

    for _, group in grouped:
        group = group.sample(min(len(group), 50))

        for i in range(len(group)):
            for j in range(i+1, len(group)):
                a1 = group.iloc[i]
                a2 = group.iloc[j]

                pairs.append({
                    "lat1": a1["lat"],
                    "lon1": a1["lon"],
                    "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"],
                    "vx1": a1["vx"],
                    "vy1": a1["vy"],

                    "lat2": a2["lat"],
                    "lon2": a2["lon"],
                    "alt2": a2["geoaltitude"],
                    "velocity2": a2["velocity"],
                    "vx2": a2["vx"],
                    "vy2": a2["vy"],

                    "is_conflict": 0
                })

                # synthetic conflict
                pairs.append({
                    "lat1": a1["lat"],
                    "lon1": a1["lon"],
                    "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"],
                    "vx1": a1["vx"],
                    "vy1": a1["vy"],

                    "lat2": a1["lat"] + np.random.uniform(-0.02, 0.02),
                    "lon2": a1["lon"] + np.random.uniform(-0.02, 0.02),
                    "alt2": a1["geoaltitude"] + np.random.uniform(-800, 800),
                    "velocity2": a2["velocity"],
                    "vx2": a2["vx"],
                    "vy2": a2["vy"],

                    "is_conflict": 1
                })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(OUTPUT_PATH, index=False)

    print("Training dataset created")

if __name__ == "__main__":
    generate()