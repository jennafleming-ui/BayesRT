# BayesRT - Formula 1 Race Strategy Optimization System

An integrated machine learning framework for real-time Formula 1 race strategy optimization, combining neural networks, weather prediction, risk assessment, and cognitive load monitoring.

## Overview

BayesRT is an AI-powered decision support system designed to optimize race strategy in Formula 1. The system analyzes real-time race data, weather conditions, and risk factors to recommend optimal pit stop timing, tire compound selection, and strategic decisions.

## Key Features

- **NeuroStrategy Engine**: Neural network-based strategy optimization using multi-objective optimization
- **WeatherWise**: Real-time weather prediction and circuit-specific forecasting
- **RiskCalc Pro**: Predictive risk assessment with confidence intervals
- **Cognitive Load Monitoring**: Driver performance and fatigue analysis
- **Real-time Dashboard**: Live visualization with WebSocket updates
- **Multi-objective Optimization**: Balances lap time, tire life, and track position

## Technologies

- **Python 3.9+**
- **TensorFlow/Keras**: Deep learning models
- **Flask**: Web application framework
- **Flask-SocketIO**: Real-time bidirectional communication
- **NumPy**: Numerical computing
- **HTML/CSS/JavaScript**: Dashboard interface

## System Architecture

```
BayesRT/
├── dashboard_server.py      # Main Flask server with SocketIO
├── neuro_strategy.py        # Neural strategy optimization engine
├── weather_wise.py          # Weather prediction system
├── risk_calc_pro.py         # Risk assessment module
├── cognitive_load_model.py  # Driver cognitive load analysis
├── templates/               # HTML dashboard templates
├── models/                  # Trained ML models
│   ├── risk_calc_model.keras
│   └── cognitive_load_model.keras
└── data/                    # Race and training data
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jennafleming-ui/BayesRT.git
cd BayesRT
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install flask flask-socketio flask-cors tensorflow numpy
```

## Usage

1. Start the server:
```bash
python3 dashboard_server.py
```

2. Open your browser and navigate to:
```
http://localhost:5001
```

3. The dashboard will display:
   - Current race state (lap, position, tire age)
   - Strategy recommendations
   - Weather forecasts
   - Risk assessments
   - Cognitive load metrics

## AI Components

### NeuroStrategy
- Multi-layer neural network for strategy optimization
- Considers tire degradation, fuel consumption, and track position
- Provides confidence scores for each recommendation
- Generates multiple strategy alternatives

### WeatherWise
- Circuit-specific weather prediction
- Track temperature forecasting
- Rain probability analysis
- Integration with strategy recommendations

### RiskCalc Pro
- Predictive lap time modeling
- Uncertainty quantification
- 95% confidence intervals
- Real-time risk assessment

### CognitiveLoad Monitor
- Driver fatigue analysis
- Performance trend detection
- Alert system for elevated cognitive load

## Dashboard Features

- **Real-time Updates**: Live data streaming via WebSocket
- **Strategy Visualization**: Top recommendations with confidence scores
- **Weather Display**: Current conditions and forecasts
- **Risk Metrics**: Predicted lap times and uncertainty
- **Race Progress**: Lap counter, position, tire age

## Research Applications

This system was developed as part of research into:
- Machine learning for motorsport strategy
- Real-time decision support systems
- Multi-objective optimization in racing
- Integration of weather, risk, and cognitive factors

## Paper

For detailed methodology and results, see the accompanying research paper: *"Machine Learning Approaches to Formula 1 Race Strategy Optimization: Integrating Weather Analysis, Risk Assessment, and Neural Strategy Models"*

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue or contact via LinkedIn.

## License

This project is available for academic and research purposes.

## Author

**Jenna Fleming**
- GitHub: [@jennafleming-ui](https://github.com/jennafleming-ui)
- LinkedIn:[www.linkedin.com/in/jf-jennafleming]
