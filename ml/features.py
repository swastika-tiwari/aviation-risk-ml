import numpy as np

# ==============================
# HAVERSINE (ACCURATE DISTANCE)
# ==============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ==============================
# CPA CALCULATION (ROBUST)
# ==============================
def compute_cpa(row):
    p1 = np.array([row["lat1"], row["lon1"]])
    p2 = np.array([row["lat2"], row["lon2"]])

    v1 = np.array([row["vx1"], row["vy1"]])
    v2 = np.array([row["vx2"], row["vy2"]])

    dp = p2 - p1
    dv = v2 - v1

    dv_norm = np.dot(dv, dv)

    # Avoid division issues
    if dv_norm == 0 or np.isnan(dv_norm):
        return 0.0, np.linalg.norm(dp)

    # Time to CPA
    t_cpa = -np.dot(dp, dv) / dv_norm
    t_cpa = max(t_cpa, 0)

    # Distance at CPA
    d_cpa = np.linalg.norm(dp + dv * t_cpa)

    return t_cpa, d_cpa


# ==============================
# FEATURE ENGINEERING (ENHANCED)
# ==============================
def compute_features(df):
    X = []

    for _, row in df.iterrows():

        # --------------------------
        # BASE FEATURES
        # --------------------------
        Dh = haversine(row["lat1"], row["lon1"], row["lat2"], row["lon2"])
        Dv = abs(row["alt1"] - row["alt2"])
        rel_vel_mag = abs(row["velocity1"] - row["velocity2"])

        # --------------------------
        # RELATIVE VELOCITY VECTOR
        # --------------------------
        rel_vx = row["vx1"] - row["vx2"]
        rel_vy = row["vy1"] - row["vy2"]
        rel_speed = np.sqrt(rel_vx**2 + rel_vy**2)

        # --------------------------
        # HEADING DIFFERENCE
        # --------------------------
        heading1 = np.arctan2(row["vy1"], row["vx1"])
        heading2 = np.arctan2(row["vy2"], row["vx2"])
        heading_diff = abs(heading1 - heading2)

        # Normalize angle (0 to pi)
        if heading_diff > np.pi:
            heading_diff = 2 * np.pi - heading_diff

        # --------------------------
        # CPA FEATURES
        # --------------------------
        t_cpa, d_cpa = compute_cpa(row)

        # --------------------------
        # CLOSING RATE (VERY IMPORTANT)
        # --------------------------
        closing_rate = Dh / (t_cpa + 1e-5)

        # --------------------------
        # RISK SCORE FEATURE
        # --------------------------
        risk_score = np.exp(-Dh) * np.exp(-Dv / 1000)

        # --------------------------
        # NOISE INJECTION (REALISM)
        # --------------------------
        Dh += np.random.normal(0, 0.02)        # ~20m GPS noise
        Dv += np.random.normal(0, 50)          # altitude noise
        rel_vel_mag += np.random.normal(0, 5)
        rel_speed += np.random.normal(0, 0.001)

        # --------------------------
        # SAFETY CLIPPING
        # --------------------------
        Dh = max(Dh, 0)
        Dv = max(Dv, 0)
        rel_vel_mag = max(rel_vel_mag, 0)
        rel_speed = max(rel_speed, 0)
        t_cpa = max(t_cpa, 0)
        d_cpa = max(d_cpa, 0)
        closing_rate = max(closing_rate, 0)
        risk_score = max(risk_score, 0)

        # --------------------------
        # FINAL FEATURE VECTOR
        # --------------------------
        X.append([
            Dh,             # horizontal distance
            Dv,             # vertical separation
            rel_vel_mag,    # speed difference
            rel_speed,      # relative velocity vector magnitude
            heading_diff,   # direction difference
            closing_rate,   # how fast aircraft are approaching
            t_cpa,          # time to closest approach
            d_cpa,          # distance at CPA
            risk_score      # engineered risk feature
        ])

    return np.array(X)