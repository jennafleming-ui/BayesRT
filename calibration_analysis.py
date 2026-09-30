"""
Deeper validation of RiskCalc Pro's uncertainty quantification, beyond the
single 95%-coverage number reported elsewhere.

Three analyses, none of which require changing the shipped model:

1. Calibration curve — empirical coverage at six nominal confidence levels
   (50/68/80/90/95/99%), not just 95%. A model can hit one target coverage
   by luck; a full curve either confirms genuine calibration or exposes
   where it breaks down.

2. 5-fold leave-one-circuit-out cross-validation — rotates the held-out
   circuit across all 5 races (not just the one random GroupShuffleSplit
   fold used elsewhere) and reports mean +/- SD, so the generalization
   number isn't an artifact of which single circuit happened to be held out.

3. MC-Dropout vs. split-conformal prediction — a much simpler, distribution-
   free baseline with a theoretical marginal-coverage guarantee. If a
   fixed-width conformal interval matches MC-Dropout's coverage at a
   fraction of the complexity, that's worth knowing and reporting honestly.

Run: python3 calibration_analysis.py
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from data_utils import clean_racing_laps
from risk_calc_pro import RiskCalcPro, RANDOM_STATE

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

Z_BY_LEVEL = {0.50: 0.674, 0.68: 0.994, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


def build_features_and_target(df):
    rc = RiskCalcPro()
    X = rc.prepare_features(df, fit=True)
    y = df["LapTimeSeconds"].values
    return rc, X, y


def train_core(X_train, y_train, epochs=200, batch_size=32):
    """Mirrors RiskCalcPro.train()'s core steps for a given train split."""
    rc = RiskCalcPro()
    rc.feature_columns = list(X_train.columns)
    X_train_scaled = rc.scaler.fit_transform(X_train)
    y_train_scaled = rc.y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    rc.model = rc.build_model(X_train_scaled.shape[1])
    rc.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    rc.model.fit(
        X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size,
        validation_split=0.2, verbose=0, callbacks=[early_stop],
    )
    train_mean = rc._mc_samples(X_train_scaled, n_samples=20).mean(axis=0)
    rc.aleatoric_std = float(np.std(y_train - train_mean))
    return rc


def mc_predict(rc, X, n_samples=20):
    X_scaled = rc.scaler.transform(X.reindex(columns=rc.feature_columns, fill_value=0.0))
    samples = rc._mc_samples(X_scaled, n_samples=n_samples)
    mean = samples.mean(axis=0)
    epistemic = samples.std(axis=0)
    total_std = np.sqrt(epistemic ** 2 + rc.aleatoric_std ** 2)
    return mean, total_std


# --------------------------------------------------------------------------- #
# Load and clean data once
# --------------------------------------------------------------------------- #
df = pd.read_csv("data/multi_track_2024.csv")
df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
df = clean_racing_laps(df)
df = df.sort_values(["Race", "Driver", "LapNumber"]).reset_index(drop=True)
races = sorted(df["Race"].unique())
print(f"Loaded {len(df)} laps across {len(races)} circuits: {races}\n")

results = {}

# --------------------------------------------------------------------------- #
# 1 & 3: temporal-holdout model -> calibration curve + conformal comparison
# --------------------------------------------------------------------------- #
print("=" * 70)
print("STEP 1: temporal-holdout model (deployment scenario)")
print("=" * 70)

group_keys = ["Race", "Driver"]
test_mask = np.zeros(len(df), dtype=bool)
positions = np.arange(len(df))
for _, idx in df.groupby(group_keys).groups.items():
    pos = positions[df.index.get_indexer(idx)]
    n_test = max(1, int(round(len(pos) * 0.2)))
    test_mask[pos[-n_test:]] = True
train_idx, test_idx = positions[~test_mask], positions[test_mask]

rc_full, X_all, y_all = build_features_and_target(df)
rc_temporal = train_core(X_all.iloc[train_idx], y_all[train_idx])
mean_test, std_test = mc_predict(rc_temporal, X_all.iloc[test_idx])
y_test = y_all[test_idx]

mae_temporal = float(np.mean(np.abs(mean_test - y_test)))
print(f"Temporal holdout: n_test={len(y_test)}, MAE={mae_temporal:.3f}s\n")

# --- calibration curve ---
print("Calibration curve (nominal vs. empirical coverage):")
calib_curve = []
for level, z in Z_BY_LEVEL.items():
    lower, upper = mean_test - z * std_test, mean_test + z * std_test
    emp = float(np.mean((y_test >= lower) & (y_test <= upper)))
    width = float(np.mean(upper - lower))
    calib_curve.append({"nominal": level, "empirical": emp, "mean_width": width})
    print(f"   nominal {level*100:5.1f}%  ->  empirical {emp*100:5.1f}%   (mean width {width:.2f}s)")
results["calibration_curve"] = calib_curve
results["temporal_mae"] = mae_temporal

# --- split-conformal comparison on the same test set, split calib/eval ---
print("\nMC-Dropout vs. split-conformal (same point predictions, same eval half):")
calib_mask = np.zeros(len(test_idx), dtype=bool)
tpos = np.arange(len(test_idx))
test_df_idx = df.iloc[test_idx]
for _, idx in test_df_idx.groupby(group_keys).groups.items():
    local_pos = tpos[test_df_idx.index.get_indexer(idx)]
    n_calib = max(1, len(local_pos) // 2)
    calib_mask[local_pos[:n_calib]] = True  # earlier half of each stint's test window
eval_mask = ~calib_mask

alpha = 0.05
n_calib = calib_mask.sum()
nonconformity = np.abs(mean_test[calib_mask] - y_test[calib_mask])
q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)
q_hat = float(np.quantile(nonconformity, q_level))

mean_eval, std_eval, y_eval = mean_test[eval_mask], std_test[eval_mask], y_test[eval_mask]
mae_eval = float(np.mean(np.abs(mean_eval - y_eval)))

mcd_lower, mcd_upper = mean_eval - 1.96 * std_eval, mean_eval + 1.96 * std_eval
mcd_cov = float(np.mean((y_eval >= mcd_lower) & (y_eval <= mcd_upper)))
mcd_width = float(np.mean(mcd_upper - mcd_lower))

conf_lower, conf_upper = mean_eval - q_hat, mean_eval + q_hat
conf_cov = float(np.mean((y_eval >= conf_lower) & (y_eval <= conf_upper)))
conf_width = 2 * q_hat

print(f"   shared point-predictor MAE (eval half): {mae_eval:.3f}s")
print(f"   MC-Dropout   : 95% target -> {mcd_cov*100:5.1f}% coverage, mean width {mcd_width:.2f}s")
print(f"   Split-conformal: 95% target -> {conf_cov*100:5.1f}% coverage, fixed width {conf_width:.2f}s")
results["conformal_comparison"] = {
    "mae_eval": mae_eval,
    "mc_dropout": {"coverage": mcd_cov, "mean_width": mcd_width},
    "split_conformal": {"coverage": conf_cov, "width": conf_width},
    "note": "eval half is the later, harder-to-predict portion of each test "
            "stint; split-conformal's i.i.d./exchangeability assumption is "
            "only approximate here since laps are temporally ordered.",
}

# --------------------------------------------------------------------------- #
# 2: 5-fold leave-one-circuit-out cross-validation
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("STEP 2: leave-one-circuit-out, rotated across all 5 circuits")
print("=" * 70)

cv_results = []
for held_out in races:
    train_mask_cv = (df["Race"] != held_out).values
    test_mask_cv = ~train_mask_cv

    rc_i, X_i, y_i = build_features_and_target(df)
    rc_i = train_core(X_i.iloc[train_mask_cv], y_i[train_mask_cv])
    mean_i, std_i = mc_predict(rc_i, X_i.iloc[test_mask_cv])
    y_i_test = y_i[test_mask_cv]

    mae_i = float(np.mean(np.abs(mean_i - y_i_test)))
    rmse_i = float(np.sqrt(np.mean((mean_i - y_i_test) ** 2)))
    lower_i, upper_i = mean_i - 1.96 * std_i, mean_i + 1.96 * std_i
    cov_i = float(np.mean((y_i_test >= lower_i) & (y_i_test <= upper_i)))

    print(f"   held out {held_out:10s}: n={test_mask_cv.sum():4d}  MAE={mae_i:.3f}s  "
          f"RMSE={rmse_i:.3f}s  95% CI coverage={cov_i*100:.1f}%")
    cv_results.append({"held_out": held_out, "n": int(test_mask_cv.sum()),
                        "mae": mae_i, "rmse": rmse_i, "coverage": cov_i})

maes = [r["mae"] for r in cv_results]
covs = [r["coverage"] for r in cv_results]
print(f"\n   Across all 5 circuits: MAE = {np.mean(maes):.3f}s +/- {np.std(maes):.3f}s   "
      f"coverage = {np.mean(covs)*100:.1f}% +/- {np.std(covs)*100:.1f}%")
results["cv_leave_one_circuit_out"] = {
    "per_circuit": cv_results,
    "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
    "coverage_mean": float(np.mean(covs)), "coverage_std": float(np.std(covs)),
}

with open("calibration_analysis_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved full results to calibration_analysis_results.json")
