import numpy as np

EARTH_RADIUS = 6371  # km

def haversine(lat1, lon1, lat2, lon2):
    # Convert to radians (VECTORISED)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return EARTH_RADIUS * c


def compute_features(df):

    # Horizontal distance
    Dh = haversine(df["lat1"], df["lon1"], df["lat2"], df["lon2"])

    # Vertical distance
    Dv = np.abs(df["alt1"] - df["alt2"])

    # Relative velocity
    rel_vel = np.abs(df["vel1"] - df["vel2"])

    # Time to CPA (avoid divide by zero)
    tcpa = Dh / (rel_vel + 1e-5)

    # Distance at CPA
    dcpa = Dh - rel_vel * tcpa

    return np.column_stack([Dh, Dv, rel_vel, tcpa, dcpa])