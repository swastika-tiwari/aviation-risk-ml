import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed_flights.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "training_set.csv")

H_THRESHOLD = 9.26   # km
V_THRESHOLD = 1000   # ft

def generate():
    df = pd.read_csv(INPUT_PATH)

    df["hour"] = (df["time"] // 3600).astype(int)

    pairs = []
    grouped = df.groupby("hour")

    for _, group in grouped:
        group = group.sample(min(len(group), 50), random_state=42)

        for i in range(len(group)):
            for j in range(i+1, len(group)):

                a1 = group.iloc[i]
                a2 = group.iloc[j]

                # -----------------------------
                # REAL PAIR (NO LABEL FORCED)
                # -----------------------------
                dist = np.sqrt((a1["lat"] - a2["lat"])**2 + (a1["lon"] - a2["lon"])**2)
                alt_diff = abs(a1["geoaltitude"] - a2["geoaltitude"])

                label = 1 if (dist < 0.08 and alt_diff < 1200) else 0  # slightly relaxed

                pairs.append({
                    "lat1": a1["lat"], "lon1": a1["lon"], "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"], "vx1": a1["vx"], "vy1": a1["vy"],

                    "lat2": a2["lat"], "lon2": a2["lon"], "alt2": a2["geoaltitude"],
                    "velocity2": a2["velocity"], "vx2": a2["vx"], "vy2": a2["vy"],

                    "is_conflict": label
                })

                # -----------------------------
                # HARD SYNTHETIC (AMBIGUOUS)
                # -----------------------------
                a2_syn = a2.copy()

                # move closer BUT not always conflict
                a2_syn["lat"] = a1["lat"] + np.random.uniform(-0.08, 0.08)
                a2_syn["lon"] = a1["lon"] + np.random.uniform(-0.08, 0.08)
                a2_syn["geoaltitude"] = a1["geoaltitude"] + np.random.uniform(-1500, 1500)

                dist_syn = np.sqrt((a1["lat"] - a2_syn["lat"])**2 + (a1["lon"] - a2_syn["lon"])**2)
                alt_syn = abs(a1["geoaltitude"] - a2_syn["geoaltitude"])

                # probabilistic labeling (IMPORTANT)
                prob_conflict = np.exp(-dist_syn * 10) * np.exp(-alt_syn / 1000)

                label_syn = 1 if np.random.rand() < prob_conflict else 0

                pairs.append({
                    "lat1": a1["lat"], "lon1": a1["lon"], "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"], "vx1": a1["vx"], "vy1": a1["vy"],

                    "lat2": a2_syn["lat"], "lon2": a2_syn["lon"], "alt2": a2_syn["geoaltitude"],
                    "velocity2": a2["velocity"], "vx2": a2["vx"], "vy2": a2["vy"],

                    "is_conflict": label_syn
                })

    df_pairs = pd.DataFrame(pairs)
    df_pairs.to_csv(OUTPUT_PATH, index=False)

    print("Realistic training dataset created")

if __name__ == "__main__":
    generate()