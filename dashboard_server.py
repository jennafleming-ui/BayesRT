"""
BayesRT dashboard server.

Streams strategy, risk, weather, and cognitive-load intelligence to the
browser over WebSockets. Where trained artifacts and their dependencies are
available, the numbers are REAL: RiskCalc Pro predicts lap times with
MC-Dropout uncertainty over an actual replayed race, and the cognitive-load
index scores the driver's real trailing laps. When an artifact can't be
loaded, that panel falls back to a clearly-labelled simulation instead of
silently faking a model.

Every payload carries a `data_source` string so the UI never misrepresents
whether a value came from a model or a fallback.
"""

import os
import logging

import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from neuro_strategy import NeuroStrategy, RaceState
from weather_wise import WeatherWise
from data_utils import clean_racing_laps

log = logging.getLogger("bayesrt")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("BAYESRT_SECRET_KEY", "bayesrt-dev-key")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# --------------------------------------------------------------------------- #
# Engine + model initialisation (each guarded so one failure never kills app)
# --------------------------------------------------------------------------- #
log.info("Initializing AI engines...")
neuro_strategy = NeuroStrategy()
weather_wise = WeatherWise()

DATA_PATH = "data/multi_track_2024.csv"


def _try_load_risk():
    # Escape hatch for environments where the TensorFlow install is broken:
    # importing it can abort the process at the C-library level (uncatchable),
    # so allow skipping straight to the labelled simulation fallback.
    if os.environ.get("BAYESRT_SKIP_RISK", "0") == "1":
        log.info("BAYESRT_SKIP_RISK=1; using simulated risk.")
        return None
    try:
        from risk_calc_pro import RiskCalcPro
        model = RiskCalcPro.load()
        log.info("RiskCalc Pro model loaded (real predictions).")
        return model
    except Exception as e:  # noqa: BLE001 - report and degrade, don't crash
        log.info(f"RiskCalc Pro unavailable ({type(e).__name__}); using simulated risk.")
        return None


def _try_load_cognitive():
    try:
        from cognitive_load_model import CognitiveLoadIndex
        model = CognitiveLoadIndex.load()
        log.info("Cognitive-load index loaded (real scoring).")
        return model
    except Exception as e:  # noqa: BLE001
        log.info(f"Cognitive-load index unavailable ({type(e).__name__}); using simulated load.")
        return None


risk_calc = _try_load_risk()
cognitive_model = _try_load_cognitive()


class RaceReplay:
    """Feeds real laps from a stored race one at a time.

    Picks the driver with the most laps in the chosen race so the panels have
    a continuous stint to reason over. Returns None if the dataset is missing,
    in which case the update loop falls back to a synthetic race.
    """

    def __init__(self, data_path=DATA_PATH, race=None, driver=None):
        self.ok = False
        try:
            df = pd.read_csv(data_path)
            df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
            df = clean_racing_laps(df)
            self.race = race or df["Race"].iloc[0]
            race_df = df[df["Race"] == self.race]
            self.driver = driver or race_df["Driver"].value_counts().index[0]
            self.laps = (race_df[race_df["Driver"] == self.driver]
                         .sort_values("LapNumber").reset_index(drop=True))
            self.total_laps = int(race_df["LapNumber"].max())
            self.ok = len(self.laps) > 5
        except Exception as e:  # noqa: BLE001
            log.info(f"Race replay data unavailable ({type(e).__name__}).")

    def window_up_to(self, i, size=10):
        """Trailing window of real laps ending at lap index i (inclusive)."""
        start = max(0, i - size + 1)
        return self.laps.iloc[start:i + 1]


replay = RaceReplay()

dashboard_active = False
_loop_running = False        # guards against stacking multiple feed threads
UPDATE_RATE = 1.0


@app.route("/")
def index():
    return render_template("dashboard.html")


@socketio.on("connect")
def handle_connect():
    log.info("Client connected")
    emit("connection_response", {"status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    log.info("Client disconnected")


@socketio.on("start_dashboard")
def start_dashboard():
    global dashboard_active, _loop_running
    dashboard_active = True
    if _loop_running:
        return  # a feed is already running; don't stack another
    _loop_running = True
    log.info("Dashboard started")
    socketio.start_background_task(update_dashboard)


@socketio.on("stop_dashboard")
def stop_dashboard():
    global dashboard_active
    dashboard_active = False
    log.info("Dashboard stopped")


# --------------------------------------------------------------------------- #
# Per-panel builders
# --------------------------------------------------------------------------- #
def build_risk(replay_lap_index):
    """Real MC-Dropout lap-time prediction, or a labelled simulation."""
    if risk_calc is not None and replay.ok:
        try:
            window = replay.window_up_to(replay_lap_index)
            out = risk_calc.predict_with_uncertainty(window)
            mean = float(np.ravel(out["mean"])[-1])
            lower = float(np.ravel(out["lower"])[-1])
            upper = float(np.ravel(out["upper"])[-1])
            std = float(np.ravel(out["std"])[-1])
            return {
                "predicted_lap_time": mean,
                "confidence_95_lower": lower,
                "confidence_95_upper": upper,
                "uncertainty": std,
            }, True
        except Exception as e:  # noqa: BLE001
            log.info(f"Risk prediction failed ({type(e).__name__}); simulating.")
    # Fallback
    return {
        "predicted_lap_time": 75.3 + np.random.randn() * 0.1,
        "confidence_95_lower": 74.8,
        "confidence_95_upper": 75.8,
        "uncertainty": 0.4 + np.random.rand() * 0.2,
    }, False


def build_cognitive(replay_lap_index, lap, total_laps):
    """Real cognitive-load index over trailing laps, or a labelled simulation."""
    if cognitive_model is not None and replay.ok:
        try:
            window = replay.window_up_to(replay_lap_index, size=8)
            score = cognitive_model.score(window)
            return {
                "load_level": score["load_index"] / 100.0,
                "status": score["status"],
                "trend": score["trend"],
            }, True
        except Exception as e:  # noqa: BLE001
            log.info(f"Cognitive scoring failed ({type(e).__name__}); simulating.")
    return {
        "load_level": 0.55 + np.random.rand() * 0.2,
        "status": "normal" if lap < total_laps * 0.7 else "elevated",
        "trend": "stable" if lap < total_laps * 0.6 else "increasing",
    }, False


def build_race_state(replay_lap_index, lap, total_laps):
    """RaceState from real telemetry when available, else synthetic."""
    if replay.ok:
        row = replay.laps.iloc[replay_lap_index]
        position = int(row["Position"]) if pd.notna(row.get("Position")) else 5
        tire_age = int(row["TyreLife"]) if pd.notna(row.get("TyreLife")) else lap % 25
        compound = str(row.get("Compound", "MEDIUM")).lower()
        return RaceState(
            current_lap=lap, total_laps=total_laps, current_position=position,
            fuel_remaining=max(0.0, 100.0 - lap * 1.5), tire_age=tire_age,
            tire_compound=compound if compound in ("soft", "medium", "hard") else "medium",
            gap_to_leader=0.0, gap_to_next=0.0,
        )
    return RaceState(
        current_lap=lap, total_laps=total_laps, current_position=5,
        fuel_remaining=50.0 - lap * 0.8, tire_age=min(lap % 25, 20),
        tire_compound="medium", gap_to_leader=12.5, gap_to_next=2.3,
    )


def build_strategy(race_state):
    strategies = neuro_strategy.optimize_strategy(
        race_state, objectives={"lap_time": 0.5, "tire_life": 0.3, "position": 0.2}
    )
    if not strategies:  # guard: near race end the pit-lap range can be empty
        return {"top_strategy": None, "alternatives": 0}
    top = strategies[0]
    return {
        "top_strategy": {
            "pit_lap": top.pit_on_lap,
            "compound": top.new_tire_compound or "current",
            "expected_time": top.expected_lap_time,
            "confidence": top.confidence,
        },
        "alternatives": len(strategies),
    }


def build_weather():
    forecast = weather_wise.get_circuit_forecast(replay.race if replay.ok else "Monaco")
    return {
        "current_temp": forecast["current_conditions"]["temp"],
        "track_temp": forecast["forecasts"][1]["track_temp"]["predicted_temp"],
        "rain_probability": forecast["forecasts"][1]["precipitation"]["rain_probability"],
    }


# --------------------------------------------------------------------------- #
# Main streaming loop
# --------------------------------------------------------------------------- #
def update_dashboard():
    """Background task: advance the race and emit a combined update per tick."""
    global _loop_running
    total_laps = replay.total_laps if replay.ok else 50
    i = 0  # index into replay.laps (or a synthetic counter)

    try:
        while dashboard_active:
            if replay.ok:
                i = (i + 1) % len(replay.laps)
                lap = int(replay.laps.iloc[i]["LapNumber"])
            else:
                i = 10 if i >= 50 else i + 1
                lap = i

            # Any single panel failing must not tear down the whole feed.
            try:
                risk_data, risk_real = build_risk(i)
                cognitive_data, cog_real = build_cognitive(i, lap, total_laps)
                race_state = build_race_state(i, lap, total_laps)
                strategy_data = build_strategy(race_state)
                weather_data = build_weather()
            except Exception as e:  # noqa: BLE001
                log.info(f"Tick error ({type(e).__name__}); skipping this update.")
                socketio.sleep(UPDATE_RATE)
                continue

            if replay.ok:
                src = f"Replay: {replay.driver} @ {replay.race}"
                src += "  ·  risk={}  ·  cognitive={}".format(
                    "model" if risk_real else "sim", "model" if cog_real else "sim")
                src += "  ·  weather=sim"
            else:
                src = "Simulated race (no dataset loaded)"

            socketio.emit("dashboard_update", {
                "data_source": src,
                "race_state": {
                    "current_lap": race_state.current_lap,
                    "total_laps": total_laps,
                    "position": race_state.current_position,
                    "tire_age": race_state.tire_age,
                },
                "risk_calc": risk_data,
                "strategy": strategy_data,
                "weather": weather_data,
                "cognitive": cognitive_data,
            })
            socketio.sleep(UPDATE_RATE)
    finally:
        _loop_running = False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BayesRT Dashboard Server Starting...")
    print("=" * 60)
    print("Open browser to: http://localhost:5001")
    print("=" * 60 + "\n")
    debug = os.environ.get("BAYESRT_DEBUG", "0") == "1"
    socketio.run(app, debug=debug, host="127.0.0.1", port=5001,
                 allow_unsafe_werkzeug=True)
