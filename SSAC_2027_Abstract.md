# BayesRT: Quantifying Cross-Circuit Generalization for Real-Time F1 Strategy AI

**Introduction**

Race engineers must decide pit timing and tire strategy in seconds, under uncertainty about tire degradation, weather, and rivals. Prior motorsports ML work — neural lap-time prediction, ensemble tire-degradation models, RL pit-stop policies — reports point-estimate accuracy on a single circuit and rarely tests whether a model trained on one track transfers to another, despite teams needing support at unfamiliar circuits. We ask: how much does uncertainty-aware lap-time prediction degrade on a genuinely new circuit, and does the model's stated uncertainty stay honest as it degrades?

**Methods**

BayesRT is an MC-Dropout Bayesian neural network (two hidden layers, 64/32 units) predicting per-lap time from causal features only — prior lap time, 3-lap rolling statistics (shifted to stay causal), tire age, compound, and driver identity. Training data: 5,112 laps from five 2024 Formula 1 races (Monaco, Bahrain, Australia, Japan, Spain) via the FastF1 API, cleaned with a per-circuit IQR filter — a fixed absolute cutoff discards up to 95% of laps at circuits slower than the one it was tuned on. We evaluate three ways: a temporal holdout (the deployment scenario); leave-one-circuit-out cross-validation rotating the held-out circuit across all 5 races, not a single random split; and a calibration curve across six nominal confidence levels (50-99%), benchmarked against split-conformal prediction. Predictive intervals combine MC-Dropout epistemic variance with residual-based aleatoric variance, added in quadrature.

**Results**

Within-race prediction achieves MAE 0.77-0.85s. Across all six confidence levels tested, empirical coverage exceeds the nominal target by 10-20 points — the model is consistently conservative, never overconfident. A split-conformal baseline nearly matches this calibration (92% vs. 95% target) with a 30% narrower interval — MC-Dropout's added complexity buys safety margin more than sharper calibration. Cross-validated leave-one-circuit-out gives MAE 2.49s ± 0.72s and 74.2% ± 15.2% coverage; per-circuit MAE ranges from 1.52s (Australia) to 3.39s (Monaco) — the one circuit prior work here evaluated on is, by this measure, the hardest to generalize to.

| Evaluation | MAE | 95% CI Coverage |
|---|---|---|
| Temporal (same race) | 0.77-0.85s | 98.1% |
| Leave-one-circuit-out (5-fold CV, mean ± SD) | 2.49s ± 0.72s | 74.2% ± 15.2% |

Twelve stakeholder interviews with race engineers, strategists, and data analysts independently identified calibrated uncertainty, not raw accuracy, as the adoption-critical property: as one team strategist put it, "a prediction without uncertainty is actually less useful."

**Conclusion**

Real-time race-strategy AI should report generalization gaps — and their variance across circuits, not just the mean — rather than single-circuit accuracy. Which new tracks a model generalizes to and which it doesn't is itself operationally useful: it tells a team where to distrust the system, not just by how much on average. The natural next step is propagating that uncertainty into the strategy decision itself — ranking pit options by expected outcome under the model's predictive distribution rather than a point estimate — so calibration changes what the system recommends, not just how it explains the recommendation. Code, trained models, and the multi-circuit dataset are open-sourced at github.com/jennafleming-ui/BayesRT.
