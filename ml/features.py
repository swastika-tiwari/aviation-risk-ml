import numpy as np

# ==============================
# HAVERSINE (VECTOR SAFE)
# ==============================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ==============================
# CPA CALCULATION (TRAJECTORY BASED)
# ==============================

def compute_cpa(row):
    # Positions (lat, lon)
    p1 = np.array([row["lat1"], row["lon1"]])
    p2 = np.array([row["lat2"], row["lon2"]])

    # Velocity vectors
    v1 = np.array([row["vx1"], row["vy1"]])
    v2 = np.array([row["vx2"], row["vy2"]])

    dp = p2 - p1      # relative position
    dv = v2 - v1      # relative velocity

    dv_norm = np.dot(dv, dv)

    # Handle zero relative velocity
    if dv_norm == 0 or np.isnan(dv_norm):
        return 0.0, np.linalg.norm(dp)

    # Time to CPA
    t_cpa = -np.dot(dp, dv) / dv_norm

    # Only future prediction (no past)
    t_cpa = max(t_cpa, 0)

    # Distance at CPA
    d_cpa = np.linalg.norm(dp + dv * t_cpa)

    return t_cpa, d_cpa


# ==============================
# FEATURE ENGINEERING
# ==============================

def compute_features(df):
    X = []

    for _, row in df.iterrows():

        # --------------------------
        # BASE FEATURES
        # --------------------------
        Dh = haversine(row["lat1"], row["lon1"], row["lat2"], row["lon2"])
        Dv = abs(row["alt1"] - row["alt2"])
        rel_vel = abs(row["velocity1"] - row["velocity2"])

        # --------------------------
        # CPA FEATURES
        # --------------------------
        t_cpa, d_cpa = compute_cpa(row)

        # --------------------------
        # REALISM: NOISE INJECTION
        # --------------------------
        Dh += np.random.normal(0, 0.02)        # ~20m GPS noise
        Dv += np.random.normal(0, 50)          # altitude noise
        rel_vel += np.random.normal(0, 5)      # speed noise

        # --------------------------
        # SAFETY CLIPPING
        # --------------------------
        Dh = max(Dh, 0)
        Dv = max(Dv, 0)
        rel_vel = max(rel_vel, 0)
        t_cpa = max(t_cpa, 0)
        d_cpa = max(d_cpa, 0)

        # --------------------------
        # FINAL FEATURE VECTOR
        # --------------------------
        X.append([
            Dh,        # horizontal distance
            Dv,        # vertical separation
            rel_vel,   # relative velocity
            t_cpa,     # time to closest approach
            d_cpa      # distance at closest approach
        ])

    return np.array(X)