"""
BayesRT - Latest F1 Data Downloader
Downloads the most recent F1 races and combines with existing data
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Enable cache for faster loading
cache_dir = 'data/cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def download_race_data(year, race_name, session_type='R'):
    """
    Download data for a specific F1 race
    
    Args:
        year: Season year (e.g., 2024)
        race_name: Race name (e.g., 'Monaco', 'Singapore')
        session_type: 'R' for Race, 'Q' for Qualifying, 'FP1', etc.
    """
    print(f"\n Downloading {year} {race_name} ({session_type})...")
    
    try:
        # Load the session
        session = fastf1.get_session(year, race_name, session_type)
        session.load()
        
        # Get all laps
        laps = session.laps
        
        # Convert timedelta to seconds
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
        
        # Calculate mean and std for outlier filtering
        mean_time = laps['LapTimeSeconds'].mean()
        std_time = laps['LapTimeSeconds'].std()
        
        print(f"   Average lap time: {mean_time:.2f}s (±{std_time:.2f}s)")
        
        # Remove outliers (pit laps, crashes, safety cars)
        # Keep laps within 3 standard deviations
        laps_clean = laps[
            (laps['LapTimeSeconds'] > mean_time - 3*std_time) & 
            (laps['LapTimeSeconds'] < mean_time + 3*std_time) &
            (laps['LapTimeSeconds'].notna())
        ].copy()
        
        # Add race identifier
        laps_clean['Race'] = race_name
        laps_clean['Year'] = year
        
        print(f"    Found {len(laps_clean)} valid laps (removed {len(laps) - len(laps_clean)} outliers)")
        
        return laps_clean
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def get_latest_races(year=2024, num_races=5):
    """
    Get the most recent completed F1 races
    
    Args:
        year: Season year
        num_races: Number of recent races to download
    """
    print(f"\n Finding latest {num_races} races from {year} season...")
    
    try:
        # Get the full season schedule
        schedule = fastf1.get_event_schedule(year)
        
        # Filter to races that have already happened
        today = datetime.now()
        past_races = schedule[schedule['EventDate'] < today]
        
        # Get the most recent races
        recent_races = past_races.tail(num_races)
        
        print(f"\n Most Recent Races:")
        for idx, race in recent_races.iterrows():
            print(f"   {race['EventDate'].strftime('%Y-%m-%d')} - {race['EventName']} ({race['Country']})")
        
        return recent_races['EventName'].tolist()
        
    except Exception as e:
        print(f" Error getting schedule: {e}")
        return None


def download_multiple_races(races_list):
    """
    Download multiple races and combine them
    
    Args:
        races_list: List of (year, race_name) tuples
    """
    all_laps = []
    successful_downloads = 0
    
    print(f"\n Starting download of {len(races_list)} races...\n")
    print("=" * 60)
    
    for year, race_name in races_list:
        laps = download_race_data(year, race_name, 'R')
        
        if laps is not None and len(laps) > 0:
            all_laps.append(laps)
            successful_downloads += 1
    
    print("\n" + "=" * 60)
    print(f" Successfully downloaded {successful_downloads}/{len(races_list)} races")
    
    if len(all_laps) == 0:
        print(" No data downloaded. Check your internet connection or race names.")
        return None
    
    # Combine all races
    combined_data = pd.concat(all_laps, ignore_index=True)
    
    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total laps: {len(combined_data):,}")
    print(f"   Unique drivers: {combined_data['Driver'].nunique()}")
    print(f"   Races: {combined_data['Race'].nunique()}")
    print(f"   Average lap time: {combined_data['LapTimeSeconds'].mean():.2f}s")
    
    return combined_data


def main():
    """Main function to download latest F1 data"""
    
    print("\n" + "=" * 60)
    print("  BayesRT - Latest F1 Data Downloader")
    print("=" * 60)
    
    # Option 1: Download specific races (customize this list)
    races_to_download = [
        (2024, 'Monaco'),        # Your original data
        (2024, 'Bahrain'),       # Season opener
        (2024, 'Saudi Arabia'),  # Fast street circuit
        (2024, 'Australia'),     # Melbourne
        (2024, 'Japan'),         # Suzuka
        (2024, 'China'),         # Shanghai
        (2024, 'Miami'),         # Street circuit
        (2024, 'Emilia Romagna'),# Imola
        (2024, 'Spain'),         # Barcelona
        (2024, 'Canada'),        # Montreal
    ]
    
    # Option 2: Automatically get latest races (uncomment to use)
    # latest_race_names = get_latest_races(year=2024, num_races=10)
    # if latest_race_names:
    #     races_to_download = [(2024, race) for race in latest_race_names]
    
    # Download all races
    combined_data = download_multiple_races(races_to_download)
    
    if combined_data is not None:
        # Save to CSV
        output_file = 'data/f1_multi_track_2024.csv'
        os.makedirs('data', exist_ok=True)
        combined_data.to_csv(output_file, index=False)
        
        print(f"\n Data saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        # Show breakdown by race
        print(f"\n Laps per Race:")
        race_counts = combined_data.groupby('Race')['LapTimeSeconds'].count().sort_values(ascending=False)
        for race, count in race_counts.items():
            print(f"   {race}: {count} laps")
        
        print("\n" + "=" * 60)
        print(" Download complete! You can now use this data with BayesRT.")
        print("=" * 60)
        
        print("\n Next Steps:")
        print("   1. Update risk_calc_pro.py to use 'data/f1_multi_track_2024.csv'")
        print("   2. Retrain models: python3 risk_calc_pro.py")
        print("   3. Run dashboard: python3 dashboard_server.py")
        
    else:
        print("\n No data was downloaded. Please check the race names and try again.")


if __name__ == '__main__':
    main()