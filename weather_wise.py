import numpy as np
from datetime import datetime
from typing import Dict

class WeatherWise:
    def __init__(self):
        pass
    
    def fetch_current_weather(self, lat: float = 43.7347, lon: float = 7.4206) -> Dict:
        """Simulate weather data"""
        return {
            'temp': 22.0 + np.random.randn() * 2,
            'humidity': 60 + np.random.randn() * 10,
            'pressure': 1013 + np.random.randn() * 5,
            'wind_speed': 15 + np.random.randn() * 5,
            'clouds': 30 + np.random.randn() * 20,
            'timestamp': datetime.now()
        }
    
    def predict_track_temperature(self, current_weather: Dict, minutes_ahead: int = 30) -> Dict:
        """Predict track surface temperature"""
        air_temp = current_weather['temp']
        cloud_cover = current_weather['clouds'] / 100.0
        
        sun_factor = 1.0 - (cloud_cover * 0.5)
        base_track_temp = air_temp + (12 * sun_factor)
        
        hour = datetime.now().hour
        trend = 0.5 if 10 <= hour <= 16 else -0.3
        
        predicted_temp = base_track_temp + (trend * minutes_ahead / 60)
        uncertainty = 1.2 + (0.05 * minutes_ahead)
        
        return {
            'predicted_temp': predicted_temp,
            'uncertainty_range': uncertainty,
            'confidence_lower': predicted_temp - uncertainty,
            'confidence_upper': predicted_temp + uncertainty,
            'minutes_ahead': minutes_ahead
        }
    
    def predict_precipitation(self, current_weather: Dict, minutes_ahead: int = 30) -> Dict:
        """Predict probability of precipitation"""
        humidity = current_weather['humidity']
        clouds = current_weather['clouds']
        
        base_probability = (humidity - 40) / 60.0
        base_probability = max(0, min(1, base_probability))
        
        cloud_factor = clouds / 100.0
        rain_probability = (base_probability * 0.6 + cloud_factor * 0.4)
        rain_probability = max(0, min(1, rain_probability))
        
        uncertainty = 0.1 + (0.01 * minutes_ahead)
        
        return {
            'rain_probability': rain_probability,
            'uncertainty': uncertainty,
            'confidence_lower': max(0, rain_probability - uncertainty),
            'confidence_upper': min(1, rain_probability + uncertainty),
            'minutes_ahead': minutes_ahead
        }
    
    def get_circuit_forecast(self, circuit_name: str = "Monaco") -> Dict:
        """Generate comprehensive forecast"""
        current = self.fetch_current_weather()
        forecasts = []
        
        for minutes in [15, 30, 45, 60]:
            temp_forecast = self.predict_track_temperature(current, minutes)
            precip_forecast = self.predict_precipitation(current, minutes)
            
            forecasts.append({
                'time_horizon': minutes,
                'track_temp': temp_forecast,
                'precipitation': precip_forecast
            })
        
        return {
            'circuit': circuit_name,
            'current_conditions': current,
            'forecasts': forecasts,
            'generated_at': datetime.now()
        }

if __name__ == "__main__":
    print("Testing WeatherWise...")
    
    weather = WeatherWise()
    forecast = weather.get_circuit_forecast("Monaco")
    
    print(f"\n🌤️ Weather Forecast for {forecast['circuit']}:")
    print(f"Current: {forecast['current_conditions']['temp']:.1f}°C, "
          f"Humidity: {forecast['current_conditions']['humidity']:.0f}%")
    
    print("\nTrack Temp Predictions:")
    for f in forecast['forecasts']:
        temp = f['track_temp']
        print(f"  +{f['time_horizon']}min: {temp['predicted_temp']:.1f}°C")
    
    print("WeatherWise complete!")