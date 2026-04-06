import numpy as np
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


def compute_cpa(row):
    p1 = np.array([row["lat1"], row["lon1"]])
    p2 = np.array([row["lat2"], row["lon2"]])

    v1 = np.array([row["vx1"], row["vy1"]])
    v2 = np.array([row["vx2"], row["vy2"]])

    dp = p2 - p1
    dv = v2 - v1

    dv_norm = np.dot(dv, dv)

    if dv_norm == 0:
        return 0, np.linalg.norm(dp)

    t_cpa = -np.dot(dp, dv) / dv_norm
    t_cpa = max(t_cpa, 0)

    d_cpa = np.linalg.norm(dp + dv * t_cpa)

    return t_cpa, d_cpa


def compute_features(df):
    X = []

    for _, row in df.iterrows():

        Dh = haversine(row["lat1"], row["lon1"], row["lat2"], row["lon2"])
        Dv = abs(row["alt1"] - row["alt2"])
        rel_vel = abs(row["velocity1"] - row["velocity2"])

        t_cpa, d_cpa = compute_cpa(row)

        X.append([Dh, Dv, rel_vel, t_cpa, d_cpa])

    return np.array(X)