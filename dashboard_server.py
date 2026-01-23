from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
import threading
from datetime import datetime
import tensorflow as tf
import numpy as np

# Import AI engines
from neuro_strategy import NeuroStrategy, RaceState
from weather_wise import WeatherWise

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bayesrt-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize engines
print("Initializing AI Engines...")
neuro_strategy = NeuroStrategy()
weather_wise = WeatherWise()

# Try to load trained models
try:
    risk_calc_model = tf.keras.models.load_model('models/risk_calc_model.keras')
    print("RiskCalc Pro model loaded")
except:
    risk_calc_model = None
    print("RiskCalc Pro model not found - using defaults")

try:
    cognitive_model = tf.keras.models.load_model('models/cognitive_load_model.keras')
    print("CognitiveLoad model loaded")
except:
    cognitive_model = None
    print("CognitiveLoad model not found - using defaults")

dashboard_active = False
update_rate = 1.0

@app.route('/')
def index():
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_dashboard')
def start_dashboard():
    global dashboard_active
    dashboard_active = True
    print('Dashboard started')
    thread = threading.Thread(target=update_dashboard)
    thread.daemon = True
    thread.start()

@socketio.on('stop_dashboard')
def stop_dashboard():
    global dashboard_active
    dashboard_active = False
    print('Dashboard stopped')

def update_dashboard():
    """Main update loop - runs all AI engines"""
    global dashboard_active
    lap_counter = 10
    
    while dashboard_active:
        try:
            lap_counter += 1
            if lap_counter > 50:
                lap_counter = 10
            
            # Simulate race state
            race_state = RaceState(
                current_lap=lap_counter,
                total_laps=50,
                current_position=5,
                fuel_remaining=50.0 - (lap_counter * 0.8),
                tire_age=min(lap_counter % 25, 20),
                tire_compound='medium',
                gap_to_leader=12.5,
                gap_to_next=2.3
            )
            
            # RiskCalc Pro - simulated predictions
            risk_data = {
                'predicted_lap_time': 75.3 + (np.random.randn() * 0.1),
                'confidence_95_lower': 74.8,
                'confidence_95_upper': 75.8,
                'uncertainty': 0.4 + (np.random.rand() * 0.2)
            }
            
            # NeuroStrategy
            strategies = neuro_strategy.optimize_strategy(
                race_state,
                objectives={'lap_time': 0.5, 'tire_life': 0.3, 'position': 0.2}
            )
            strategy_data = {
                'top_strategy': {
                    'pit_lap': strategies[0].pit_on_lap,
                    'compound': strategies[0].new_tire_compound,
                    'expected_time': strategies[0].expected_lap_time,
                    'confidence': strategies[0].confidence
                },
                'alternatives': len(strategies)
            }
            
            # WeatherWise
            weather_forecast = weather_wise.get_circuit_forecast("Monaco")
            weather_data = {
                'current_temp': weather_forecast['current_conditions']['temp'],
                'track_temp': weather_forecast['forecasts'][1]['track_temp']['predicted_temp'],
                'rain_probability': weather_forecast['forecasts'][1]['precipitation']['rain_probability']
            }
            
            # CognitiveLoad - simulated
            cognitive_data = {
                'load_level': 0.55 + (np.random.rand() * 0.2),
                'status': 'normal' if lap_counter < 35 else 'elevated',
                'trend': 'stable' if lap_counter < 30 else 'increasing'
            }
            
            # Combine all data
            dashboard_update = {
                'timestamp': datetime.now().isoformat(),
                'race_state': {
                    'current_lap': race_state.current_lap,
                    'position': race_state.current_position,
                    'tire_age': race_state.tire_age
                },
                'risk_calc': risk_data,
                'strategy': strategy_data,
                'weather': weather_data,
                'cognitive': cognitive_data
            }
            
            # Emit to all connected clients
            socketio.emit('dashboard_update', dashboard_update)
            
            # Sleep until next update
            time.sleep(update_rate)
            
        except Exception as e:
            print(f"Error in update loop: {e}")
            import traceback
            traceback.print_exc()
            dashboard_active = False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("BayesRT Dashboard Server Starting...")
    print("="*60)
    print("Open browser to: http://localhost:5001")
    print("="*60 + "\n")
socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
