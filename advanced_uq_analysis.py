"""
A more complete uncertainty-quantification study: three complementary paradigms,
scored with proper scoring rules rather than eyeballed coverage/width pairs.

Paradigms compared:
  1. MC-Dropout            — Bayesian approximate inference (Gal & Ghahramani 2016)
  2. Deep Ensemble          — a second, *independent* epistemic-uncertainty estimate
                              (Lakshminarayanan et al. 2017): 5 separately-initialized
                              networks; disagreement across members stands in for
                              epistemic variance instead of dropout-sample spread.
  3. Split-conformal        — distribution-free, finite-sample marginal coverage
                              guarantee, no distributional assumption at all.
  4. Normalized conformal   — a hybrid: conformalize residuals *scaled* by
                              MC-Dropout's own predicted std, so the resulting
                              interval is both adaptive (varies per-lap, like
                              MC-Dropout) and carries conformal's coverage
                              guarantee (unlike MC-Dropout, whose Gaussian
                              interval has no such guarantee under misspecification).

Scoring:
  - CRPS (closed-form, Gaussian): the standard strictly-proper scoring rule for
    probabilistic forecasts (Gneiting & Raftery, 2007) — rewards both calibration
    and sharpness in one number, unlike coverage alone.
  - NLL: Gaussian negative log-likelihood.
  - PIT + Kolmogorov-Smirnov test: for a well-calibrated Gaussian forecast, the
    probability-integral-transformed residuals should be Uniform(0,1). A KS test
    against Uniform(0,1) gives a p-value for "is this forecast calibrated," not
    just a single coverage percentage at one confidence level.

CRPS/NLL/PIT require an actual predictive density, so they're reported only for
MC-Dropout and the Deep Ensemble. Conformal methods (3 & 4) are distribution-free
by design and are scored on coverage + width instead — noted explicitly, not
glossed over.

Run: python3 advanced_uq_analysis.py  (reuses the same temporal split as
calibration_analysis.py; trains 1 MC-Dropout model + 5 ensemble members)
"""

import json

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from data_utils import clean_racing_laps
from risk_calc_pro import RiskCalcPro

SQRT_PI = np.sqrt(np.pi)


def train_core(X_train, y_train, seed, epochs=200, batch_size=32):
    tf.keras.utils.set_random_seed(seed)
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


def mc_dropout_predict(rc, X, n_samples=20):
    X_scaled = rc.scaler.transform(X.reindex(columns=rc.feature_columns, fill_value=0.0))
    samples = rc._mc_samples(X_scaled, n_samples=n_samples)
    mean = samples.mean(axis=0)
    epistemic = samples.std(axis=0)
    total_std = np.sqrt(epistemic ** 2 + rc.aleatoric_std ** 2)
    return mean, total_std


def point_predict(rc, X):
    """Deterministic forward pass (dropout OFF) -- for ensemble members."""
    X_scaled = rc.scaler.transform(X.reindex(columns=rc.feature_columns, fill_value=0.0))
    y_scaled = rc.model.predict(X_scaled, verbose=0).flatten()
    return rc.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * stats.norm.cdf(z) - 1)
                                   + 2 * stats.norm.pdf(z) - 1 / SQRT_PI)))


def nll_gaussian(y, mu, sigma):
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def pit_ks_test(y, mu, sigma):
    pit = stats.norm.cdf((y - mu) / sigma)
    ks_stat, p_value = stats.kstest(pit, "uniform")
    return float(ks_stat), float(p_value)


def coverage_width(y, mu, sigma, z=1.96):
    lower, upper = mu - z * sigma, mu + z * sigma
    return float(np.mean((y >= lower) & (y <= upper))), float(np.mean(upper - lower))


# --------------------------------------------------------------------------- #
df = pd.read_csv("data/multi_track_2024.csv")
df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
df = clean_racing_laps(df)
df = df.sort_values(["Race", "Driver", "LapNumber"]).reset_index(drop=True)

group_keys = ["Race", "Driver"]
test_mask = np.zeros(len(df), dtype=bool)
positions = np.arange(len(df))
for _, idx in df.groupby(group_keys).groups.items():
    pos = positions[df.index.get_indexer(idx)]
    n_test = max(1, int(round(len(pos) * 0.2)))
    test_mask[pos[-n_test:]] = True
train_idx, test_idx = positions[~test_mask], positions[test_mask]

rc0 = RiskCalcPro()
X_all = rc0.prepare_features(df, fit=True)
y_all = df["LapTimeSeconds"].values
X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]

print("=" * 72)
print("1. MC-Dropout (single model, 20 stochastic forward passes)")
print("=" * 72)
rc_mcd = train_core(X_train, y_train, seed=42)
mu_mcd, sigma_mcd = mc_dropout_predict(rc_mcd, X_test)
mae_mcd = float(np.mean(np.abs(mu_mcd - y_test)))
crps_mcd = crps_gaussian(y_test, mu_mcd, sigma_mcd)
nll_mcd = nll_gaussian(y_test, mu_mcd, sigma_mcd)
ks_mcd, p_mcd = pit_ks_test(y_test, mu_mcd, sigma_mcd)
cov_mcd, width_mcd = coverage_width(y_test, mu_mcd, sigma_mcd)
print(f"   MAE {mae_mcd:.3f}s | CRPS {crps_mcd:.3f} | NLL {nll_mcd:.3f} | "
      f"PIT-KS p={p_mcd:.4f} | 95% coverage {cov_mcd*100:.1f}% | width {width_mcd:.2f}s")

print("\n" + "=" * 72)
print("2. Deep Ensemble (5 independently-initialized members)")
print("=" * 72)
ensemble = [rc_mcd] + [train_core(X_train, y_train, seed=100 + i) for i in range(4)]
member_preds = np.array([point_predict(m, X_test) for m in ensemble])  # (5, n_test)
mu_ens = member_preds.mean(axis=0)
epistemic_ens = member_preds.std(axis=0)
aleatoric_ens = float(np.mean([m.aleatoric_std for m in ensemble]))
sigma_ens = np.sqrt(epistemic_ens ** 2 + aleatoric_ens ** 2)
mae_ens = float(np.mean(np.abs(mu_ens - y_test)))
crps_ens = crps_gaussian(y_test, mu_ens, sigma_ens)
nll_ens = nll_gaussian(y_test, mu_ens, sigma_ens)
ks_ens, p_ens = pit_ks_test(y_test, mu_ens, sigma_ens)
cov_ens, width_ens = coverage_width(y_test, mu_ens, sigma_ens)
print(f"   MAE {mae_ens:.3f}s | CRPS {crps_ens:.3f} | NLL {nll_ens:.3f} | "
      f"PIT-KS p={p_ens:.4f} | 95% coverage {cov_ens*100:.1f}% | width {width_ens:.2f}s")

print("\n" + "=" * 72)
print("3 & 4. Split-conformal vs. normalized (MC-Dropout-scaled) conformal")
print("=" * 72)
calib_mask = np.zeros(len(test_idx), dtype=bool)
tpos = np.arange(len(test_idx))
test_df_idx = df.iloc[test_idx]
for _, idx in test_df_idx.groupby(group_keys).groups.items():
    local_pos = tpos[test_df_idx.index.get_indexer(idx)]
    n_calib = max(1, len(local_pos) // 2)
    calib_mask[local_pos[:n_calib]] = True
eval_mask = ~calib_mask
n_calib = calib_mask.sum()
alpha = 0.05
q_level = min(1.0, np.ceil((n_calib + 1) * (1 - alpha)) / n_calib)

# plain split-conformal: fixed-width interval
resid_calib = np.abs(mu_mcd[calib_mask] - y_test[calib_mask])
q_fixed = float(np.quantile(resid_calib, q_level))
cov_fixed = float(np.mean(np.abs(mu_mcd[eval_mask] - y_test[eval_mask]) <= q_fixed))
width_fixed = 2 * q_fixed

# normalized conformal: scale residuals by MC-Dropout's own predicted std
scaled_resid_calib = resid_calib / sigma_mcd[calib_mask]
q_scaled = float(np.quantile(scaled_resid_calib, q_level))
eval_width = 2 * q_scaled * sigma_mcd[eval_mask]
cov_norm = float(np.mean(np.abs(mu_mcd[eval_mask] - y_test[eval_mask]) <= q_scaled * sigma_mcd[eval_mask]))
width_norm = float(np.mean(eval_width))

print(f"   Split-conformal (fixed width)      : 95% target -> {cov_fixed*100:.1f}% coverage, width {width_fixed:.2f}s (constant)")
print(f"   Normalized conformal (MCD-scaled)  : 95% target -> {cov_norm*100:.1f}% coverage, mean width {width_norm:.2f}s (adaptive)")
print("   (conformal methods are distribution-free: no CRPS/NLL/PIT -- those need a density)")

results = {
    "mc_dropout": {"mae": mae_mcd, "crps": crps_mcd, "nll": nll_mcd, "pit_ks_stat": ks_mcd,
                   "pit_ks_pvalue": p_mcd, "coverage_95": cov_mcd, "mean_width": width_mcd},
    "deep_ensemble": {"mae": mae_ens, "crps": crps_ens, "nll": nll_ens, "pit_ks_stat": ks_ens,
                       "pit_ks_pvalue": p_ens, "coverage_95": cov_ens, "mean_width": width_ens},
    "split_conformal": {"coverage_95": cov_fixed, "width": width_fixed},
    "normalized_conformal": {"coverage_95": cov_norm, "mean_width": width_norm},
}
with open("advanced_uq_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to advanced_uq_results.json")
