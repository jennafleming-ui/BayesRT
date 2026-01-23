import fastf1 as ff1
import pandas as pd
import os

print("Downloading F1 data...")

# Create cache directory if it doesn't exist
cache_dir = 'data/cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
    print(f"Created cache directory: {cache_dir}")

# Enable cache to speed up future downloads
ff1.Cache.enable_cache(cache_dir)

# Download 2024 Monaco Grand Prix data
print("Loading 2024 Monaco GP...")
session = ff1.get_session(2024, 'Monaco', 'R')  # 'R' = Race
session.load()

print("Data loaded successfully!")
print(f"Found {len(session.laps)} laps of data")
print(f"Drivers in race: {list(session.drivers)}")

# Save basic data to CSV for easy viewing
laps = session.laps[['Driver', 'LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']]
laps.to_csv('data/monaco_2024_laps.csv', index=False)
print("Saved data to data/monaco_2024_laps.csv")

# Quick analysis
fastest_lap = laps.loc[laps['LapTime'].idxmin()]
print(f"\nFastest lap: {fastest_lap['Driver']} - Lap {fastest_lap['LapNumber']}")
print(f"Time: {fastest_lap['LapTime']}")

