import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, "data", "flight.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed_flights.csv")

def preprocess():
    df = pd.read_csv(INPUT_PATH)

    # Filter airborne
    df = df[df["onground"] == False]

    # Drop missing
    df = df.dropna(subset=["lat", "lon", "geoaltitude", "velocity", "time"])

    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna()

    # Sort
    df = df.sort_values(by=["icao24", "time"])

    # Compute velocity vectors
    df["lat_prev"] = df.groupby("icao24")["lat"].shift(1)
    df["lon_prev"] = df.groupby("icao24")["lon"].shift(1)
    df["time_prev"] = df.groupby("icao24")["time"].shift(1)

    dt = df["time"] - df["time_prev"]

    df["vx"] = (df["lat"] - df["lat_prev"]) / dt
    df["vy"] = (df["lon"] - df["lon_prev"]) / dt

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df.to_csv(OUTPUT_PATH, index=False)
    print("Processed data saved")

if __name__ == "__main__":
    preprocess()