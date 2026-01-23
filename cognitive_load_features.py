import pandas as pd
import numpy as np

class CognitiveLoadExtractor:
    def __init__(self):
        pass
    
    def calculate_stress_indicators(self, driver_laps):
        """
        Extract cognitive load proxy variables from lap data
        """
        features = {}
        
        # Consistency metrics (higher variance = higher cognitive load)
        features['lap_time_std'] = driver_laps['LapTimeSeconds'].std()
        features['lap_time_cv'] = features['lap_time_std'] / driver_laps['LapTimeSeconds'].mean()
        
        # Performance degradation over time
        features['performance_trend'] = self._calculate_performance_slope(driver_laps)
        
        # Reaction to pressure (lap time spikes)
        features['spike_frequency'] = self._count_performance_spikes(driver_laps)
        
        return features
    
    def _calculate_performance_slope(self, laps):
        """Calculate if lap times are getting worse over time"""
        if len(laps) < 10:
            return 0
            
        x = laps['LapNumber'].values
        y = laps['LapTimeSeconds'].values
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope  # Positive = getting slower (fatigue/stress)
    
    def _count_performance_spikes(self, laps):
        """Count sudden lap time increases"""
        lap_times = laps['LapTimeSeconds'].values
        mean_time = np.mean(lap_times)
        std_time = np.std(lap_times)
        
        spikes = np.sum(lap_times > (mean_time + 2 * std_time))
        return spikes / len(lap_times)  # Normalize by total laps

# Test the class
if __name__ == "__main__":
    print("Testing CognitiveLoadExtractor...")
    
    # Load your existing Monaco data
    try:
        df = pd.read_csv('data/monaco_2024_laps.csv')
        df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
        
        # Clean the data first
        clean_df = df[(df['LapTimeSeconds'] > 70) & (df['LapTimeSeconds'] < 95)]
        
        # Test on one driver
        test_driver = 'HAM'  # Hamilton
        driver_laps = clean_df[clean_df['Driver'] == test_driver]
        
        if len(driver_laps) > 0:
            print(f"Testing with {test_driver}: {len(driver_laps)} laps")
            
            extractor = CognitiveLoadExtractor()
            features = extractor.calculate_stress_indicators(driver_laps)
            
            print("Extracted features:")
            for key, value in features.items():
                print(f"  {key}: {value:.4f}")
        else:
            print(f"No data found for driver {test_driver}")
            print("Available drivers:", clean_df['Driver'].unique())
            
    except FileNotFoundError:
        print("Monaco data file not found. Run your data download script first.")
    except Exception as e:
        print(f"Error: {e}")

       