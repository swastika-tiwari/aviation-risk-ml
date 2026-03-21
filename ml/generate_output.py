import pandas as pd
import numpy as np
import json
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("../data/flights.csv")

# Clean
df = df.dropna(subset=['latitude', 'longitude', 'baro_altitude'])
df = df.sample(n=2000, random_state=42)

# Distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Pairwise analysis
pairs = []

for i in range(len(df)):
    for j in range(i+1, len(df)):
        d = haversine(
            df.iloc[i]['latitude'], df.iloc[i]['longitude'],
            df.iloc[j]['latitude'], df.iloc[j]['longitude']
        )
        alt_diff = abs(df.iloc[i]['baro_altitude'] - df.iloc[j]['baro_altitude'])

        if d < 5 and alt_diff < 1000:
            pairs.append({
                "distance": float(d),
                "alt_diff": float(alt_diff)
            })

pairs_df = pd.DataFrame(pairs)

# ML Model
model = IsolationForest(contamination=0.05)
pairs_df['anomaly'] = model.fit_predict(pairs_df)

# Convert to JSON
results = pairs_df.to_dict(orient="records")

with open("../output/results.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results generated!")