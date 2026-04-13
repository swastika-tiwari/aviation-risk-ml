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
        group = group.sample(min(len(group), 50), random_state=42)

        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                a1 = group.iloc[i]
                a2 = group.iloc[j]

                # ---------------- REAL PAIR ----------------
                dist = np.sqrt((a1["lat"] - a2["lat"])**2 + (a1["lon"] - a2["lon"])**2)
                alt_diff = abs(a1["geoaltitude"] - a2["geoaltitude"])

                label = 1 if (dist < 0.08 and alt_diff < 1200) else 0

                pairs.append({
                    "lat1": a1["lat"], "lon1": a1["lon"], "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"], "vx1": a1["vx"], "vy1": a1["vy"],

                    "lat2": a2["lat"], "lon2": a2["lon"], "alt2": a2["geoaltitude"],
                    "velocity2": a2["velocity"], "vx2": a2["vx"], "vy2": a2["vy"],

                    "is_conflict": label
                })

                # ---------------- SYNTHETIC ----------------
                a2_syn = a2.copy()

                direction = np.random.choice([-1, 1])

                a2_syn["lat"] = a1["lat"] + direction * np.random.uniform(0.01, 0.08)
                a2_syn["lon"] = a1["lon"] + direction * np.random.uniform(0.01, 0.08)
                a2_syn["geoaltitude"] = a1["geoaltitude"] + np.random.uniform(-1500, 1500)

                # Head-on velocity simulation
                a2_syn["vx"] = -a1["vx"] + np.random.normal(0, 0.001)
                a2_syn["vy"] = -a1["vy"] + np.random.normal(0, 0.001)

                dist_syn = np.sqrt((a1["lat"] - a2_syn["lat"])**2 + (a1["lon"] - a2_syn["lon"])**2)
                alt_syn = abs(a1["geoaltitude"] - a2_syn["geoaltitude"])

                prob_conflict = np.exp(-dist_syn * 8) * np.exp(-alt_syn / 800)
                label_syn = 1 if np.random.rand() < prob_conflict else 0

                pairs.append({
                    "lat1": a1["lat"], "lon1": a1["lon"], "alt1": a1["geoaltitude"],
                    "velocity1": a1["velocity"], "vx1": a1["vx"], "vy1": a1["vy"],

                    "lat2": a2_syn["lat"], "lon2": a2_syn["lon"], "alt2": a2_syn["geoaltitude"],
                    "velocity2": a2["velocity"], "vx2": a2_syn["vx"], "vy2": a2_syn["vy"],

                    "is_conflict": label_syn
                })

    pd.DataFrame(pairs).to_csv(OUTPUT_PATH, index=False)
    print("✅ Training dataset created")

if __name__ == "__main__":
    generate()