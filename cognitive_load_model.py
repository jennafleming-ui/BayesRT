"""
Cognitive Load Index — an *unsupervised* driver-strain estimate.

Design note (important for honesty): Formula 1 timing data contains no direct
label for a driver's cognitive load. Rather than invent a fake target, this
model treats cognitive load as a latent construct and estimates it from
observable *instability proxies* drawn from lap data:

  * lap_time_cv       — coefficient of variation (inconsistency)
  * performance_trend — slope of lap time vs lap number (positive = fading)
  * spike_frequency   — rate of large lap-time excursions

The model `fit`s a baseline distribution over these proxies across the whole
field (StandardScaler), then scores any window of a driver's laps as a
0–100 index: each proxy is z-scored against the field, combined with
interpretable weights, and squashed with a logistic. 50 ≈ field-average
strain; higher = more unstable/fatigued relative to peers.

It is a *relative, proxy* measure — documented as such — not a claim to
measure literal neural cognitive load.
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from cognitive_load_features import CognitiveLoadExtractor
from data_utils import clean_racing_laps

MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "cognitive_load_scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "cognitive_load_meta.json")

# Proxies that increase with strain, and how much weight each carries.
LOAD_FEATURES = ["lap_time_cv", "performance_trend", "spike_frequency"]
LOAD_WEIGHTS = {"lap_time_cv": 0.4, "performance_trend": 0.3, "spike_frequency": 0.3}
MIN_LAPS = 5  # need a few laps for a stable estimate


class CognitiveLoadIndex:
    def __init__(self):
        self.scaler = StandardScaler()
        self.extractor = CognitiveLoadExtractor()
        self.fitted = False

    # ------------------------------------------------------------------ #
    # Fitting the field baseline
    # ------------------------------------------------------------------ #
    def _feature_vector(self, driver_laps):
        f = self.extractor.calculate_stress_indicators(driver_laps)
        return np.array([f[k] for k in LOAD_FEATURES], dtype=float)

    def fit(self, laps_df):
        """Learn the field's normal range of instability proxies."""
        group_keys = [c for c in ("Race", "Driver") if c in laps_df.columns] or ["Driver"]
        rows = []
        for _, grp in laps_df.groupby(group_keys):
            if len(grp) >= MIN_LAPS:
                vec = self._feature_vector(grp)
                if np.all(np.isfinite(vec)):
                    rows.append(vec)
        if len(rows) < 2:
            raise ValueError("Not enough driver-race groups to fit the index.")
        X = np.vstack(rows)
        self.scaler.fit(X)
        self.fitted = True
        print(f"CognitiveLoadIndex fitted on {len(rows)} driver-race windows "
              f"from {laps_df.get('Driver', pd.Series()).nunique()} drivers.")
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    def score(self, driver_laps):
        """Return a cognitive-load estimate for one driver's laps."""
        if not self.fitted:
            raise RuntimeError("Call fit() or load() before scoring.")
        if len(driver_laps) < MIN_LAPS:
            return {"load_index": 50.0, "status": "insufficient_data",
                    "trend": "unknown", "components": {}}

        vec = self._feature_vector(driver_laps).reshape(1, -1)
        z = self.scaler.transform(vec).flatten()  # field-relative z-scores
        weights = np.array([LOAD_WEIGHTS[k] for k in LOAD_FEATURES])
        combined = float(np.dot(weights, z))
        load_index = round(float(self._logistic(combined) * 100), 1)

        status = ("high" if load_index >= 70
                  else "elevated" if load_index >= 55
                  else "normal")
        trend = ("increasing" if z[LOAD_FEATURES.index("performance_trend")] > 0.5
                 else "stable")

        return {
            "load_index": load_index,
            "status": status,
            "trend": trend,
            "components": {k: round(float(zi), 3) for k, zi in zip(LOAD_FEATURES, z)},
        }

    def rolling_load(self, driver_laps, window=8):
        """Time series of the load index over a race (trailing window)."""
        laps = driver_laps.sort_values("LapNumber")
        out = []
        for end in range(len(laps)):
            start = max(0, end - window + 1)
            w = laps.iloc[start:end + 1]
            idx = self.score(w)["load_index"] if len(w) >= MIN_LAPS else None
            out.append({"lap": int(laps.iloc[end]["LapNumber"]), "load_index": idx})
        return out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
        with open(META_PATH, "w") as f:
            json.dump({"features": LOAD_FEATURES, "weights": LOAD_WEIGHTS,
                       "min_laps": MIN_LAPS}, f)
        print(f"   Saved cognitive-load index to '{MODEL_DIR}/'")

    @classmethod
    def load(cls):
        obj = cls()
        obj.scaler = joblib.load(SCALER_PATH)
        obj.fitted = True
        return obj


if __name__ == "__main__":
    print("Training CognitiveLoadIndex...")
    df = pd.read_csv("data/multi_track_2024.csv")
    df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    clean = clean_racing_laps(df)

    model = CognitiveLoadIndex().fit(clean)
    model.save()

    # Demo: rank drivers in the first race by estimated load.
    race = clean["Race"].iloc[0]
    race_laps = clean[clean["Race"] == race]
    print(f"\nCognitive load ranking — {race}:")
    scores = []
    for drv, grp in race_laps.groupby("Driver"):
        s = model.score(grp)
        if s["status"] != "insufficient_data":
            scores.append((drv, s["load_index"], s["status"]))
    for drv, idx, status in sorted(scores, key=lambda t: -t[1]):
        print(f"   {drv}: {idx:5.1f}  ({status})")
