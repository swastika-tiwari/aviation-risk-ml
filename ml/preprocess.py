import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "flight.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed_flights.csv")

def preprocess_data():
    print("🔄 Preprocessing raw ADS-B data...")

    df = pd.read_csv(INPUT_PATH)

    # 1. Filter invalid rows
    df = df[
        (df["onground"] == False) &
        df["lat"].notna() &
        df["lon"].notna() &
        df["baroaltitude"].notna()
    ]

    # 2. Convert time to numeric timestamp
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    # 3. Sort by aircraft and time
    df = df.sort_values(by=["icao24", "time"])

    # 4. Interpolation (per aircraft)
    df["baroaltitude"] = df.groupby("icao24")["baroaltitude"].transform(lambda x: x.interpolate())
    df["velocity"] = df.groupby("icao24")["velocity"].transform(lambda x: x.interpolate())

    # 5. Save processed file
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Saved processed data → {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_data()