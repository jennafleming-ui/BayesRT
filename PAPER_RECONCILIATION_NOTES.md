# Reconciling the full paper with the current, verified codebase

The paper (`AI-Based Real-Time Strategy Optimization for Motorsports Performance.docx-3.pdf`)
describes an earlier, buggy version of the pipeline. The current code in this repo fixes two real
bugs and reports different, verified numbers. Below is exactly what changed and why, plus
paste-ready replacement text for the paper.

## What was actually wrong (not just "different numbers")

1. **Target/train-test leakage** in the original RiskCalc Pro (the 0.34s MAE / 87% coverage
   result). `risk_calc_pro.py`'s current docstring documents this directly: rolling lap-time
   features used to leak the current lap's own time, and the 80/20 split was random rather than
   temporal or by-race, so the test set wasn't really unseen. This is why the paper's number
   doesn't match anything else you have (README, presentation).

2. **A data-cleaning bug I found and fixed today**: the "remove pit stops / incidents" filter
   used a fixed window (70s < lap time < 95s), tuned for Monaco's ~80s pace. Applied to the
   5-circuit dataset, it silently kept 97% of Monaco's laps but only **5.5% of Bahrain's** and
   **6.3% of Japan's** (both circuits run ~97-98s laps). The "multi-circuit" dataset was
   therefore ~94% Monaco/Spain/Australia. I replaced it with a per-circuit IQR filter
   (`data_utils.py`) that adapts to each circuit's own pace; it now keeps 94-97% of every
   circuit's laps.

3. **The EXPO presentation script's numbers** ("6,000+ laps, 10 circuits," LSTM, 3.31s MAE,
   93.2% coverage, 42ms latency) don't match any data or model currently in this repo (which has
   5 races / 5,377 raw laps and a plain MLP, not an LSTM). Since SSAC requires a linked GitHub
   repo, don't carry these specific figures into the abstract or paper unless you have the
   underlying run to back them up — treat the numbers below as the only ones you can currently
   defend.

## Verified numbers (reproduce with `python3 risk_calc_pro.py` after `pip install -r requirements.txt`)

| Metric | Old paper (buggy, Monaco-only) | **Verified, current** |
|---|---|---|
| Data | 1,247 laps, Monaco only | 5,112 laps, 5 circuits (Monaco, Bahrain, Australia, Japan, Spain) |
| Split | Random 80/20 (leaky) | Temporal holdout (last 20% of each driver-race) |
| MAE | 0.34s | **0.90s** |
| RMSE | 0.52s | **1.14-1.27s** |
| 95% CI coverage | 87.1% | **98.0-98.3%** |
| Inference time | 8ms (claimed) | **2.8ms** (measured, MC-Dropout, 20 samples) |
| Strategy ranking | 45 options / 200ms (claimed) | **0.2ms** (measured, all pit-lap × compound combos) |

Note: reruns show ±0.05-0.1s variance on the temporal-split MAE (TF training isn't perfectly
deterministic across processes even with fixed seeds).

## Deeper validation (2026-09-09, `calibration_analysis.py`) — supersedes the single-fold number above

The original leave-one-circuit-out number (2.26s MAE / 86.6% coverage) came from **one random
fold** (`GroupShuffleSplit` happens to hold out exactly 1 of 5 races when `test_size=0.2`). That's
not enough to trust — it could be an easy or a hard circuit by luck. `calibration_analysis.py`
does the analysis properly and adds two things a rigorous reviewer will ask about anyway:

**1. 5-fold leave-one-circuit-out, rotating the held-out circuit across all 5 races:**

| Held out | n | MAE | RMSE | 95% CI coverage |
|---|---|---|---|---|
| Australia | 860 | 1.52s | 2.21s | 94.4% |
| Bahrain | 1054 | 1.97s | 3.85s | 89.4% |
| Japan | 810 | 2.34s | 2.94s | 68.9% |
| Monaco | 1172 | 3.39s | 4.10s | 55.1% |
| Spain | 1216 | 3.24s | 4.30s | 63.2% |
| **Mean ± SD** | — | **2.49s ± 0.72s** | — | **74.2% ± 15.2%** |

**Use 2.49s ± 0.72s / 74.2% ± 15.2% everywhere, not 2.26s/86.6%.** It's a more defensible number
(rotates all 5 circuits, not one lucky split) and it's a better story: **Monaco — the only circuit
the original paper evaluated on — turns out to be the single hardest circuit to generalize to.**
That's worth a sentence in the paper: the old single-circuit evaluation wasn't just statistically
weaker, it happened to validate on the least representative track.

**2. Calibration curve (temporal-holdout model, 6 nominal confidence levels):**

| Nominal | Empirical | Mean interval width |
|---|---|---|
| 50% | 76.3% | 2.10s |
| 68% | 87.0% | 3.09s |
| 80% | 91.3% | 3.99s |
| 90% | 96.6% | 5.12s |
| 95% | 98.1% | 6.10s |
| 99% | 99.3% | 8.01s |

Over-covers at every level — the model is consistently conservative, not overconfident anywhere
in the range. Good news for trust; the honest caveat is the intervals are wider than they need to
be to hit the target.

**3. MC-Dropout vs. split-conformal prediction** (same point predictions, calibration/eval split
within the temporal test set — note conformal's i.i.d./exchangeability assumption is only
approximate here since laps are temporally ordered, worth stating as a caveat, not hiding):

| Method | 95% target coverage | Interval width |
|---|---|---|
| MC-Dropout | 97.9% | 6.15s (adaptive) |
| Split-conformal | 92.0% | 4.26s (fixed) |

Conformal gets close to nominal with a 30% narrower interval and far less machinery (one
empirical quantile vs. 20 stochastic forward passes + aleatoric estimation). Report this honestly
rather than only showcasing MC-Dropout — it shows you validated the added complexity was worth it,
not just that you know how to implement Bayesian NNs. (Here it's a wash: MC-Dropout's width costs
more for coverage that's arguably too conservative; conformal is cheaper and nearly as good.)

## Full UQ benchmark (2026-09-09, `advanced_uq_analysis.py`) — for the full paper's Methods/Results, not the abstract

Four methods head-to-head on the same temporal-holdout test set, scored with proper scoring rules
(CRPS, NLL — the statistically correct way to rank probabilistic forecasts, not just eyeballing
coverage/width) plus a formal calibration test (PIT + Kolmogorov-Smirnov), not just single-level
coverage:

| Method | MAE | CRPS | NLL | PIT-KS p-value | 95% coverage | Mean width |
|---|---|---|---|---|---|---|
| MC-Dropout (single model, 20 passes) | 0.827s | 0.651 | 1.651 | <0.0001 | 97.9% | 6.43s |
| Deep Ensemble (5 independently-initialized members) | **0.674s** | **0.538** | **1.480** | <0.0001 | 98.1% | **5.53s** |
| Split-conformal (fixed width) | — | — | — | n/a (distribution-free) | 92.2% | 4.34s |
| Normalized conformal (width scaled by MC-Dropout's std) | — | — | — | n/a | 92.2% | 4.54s |

Three findings worth stating plainly in the paper, in order of how interesting they are:

1. **Deep Ensembles dominate MC-Dropout on every metric that can be compared** — lower MAE, lower
   CRPS, lower NLL, similar coverage, narrower intervals. For this problem (tabular telemetry
   regression, ~30 features, a few thousand rows), ensembling 5 independently-initialized networks
   is a straightforwardly better epistemic-uncertainty estimate than MC-Dropout's stochastic
   forward passes. Recommend leading with this in the paper: it's a clean, decisive, honestly-
   earned result — not "we tried two things," but "we tested which one actually works better and
   it wasn't the one we started with."
2. **Neither Bayesian method passes a formal calibration test.** Single-level coverage (e.g. "98%
   of laps fell in the 95% interval") looks fine, but PIT-Kolmogorov-Smirnov — which checks whether
   the predicted distribution is calibrated across its *entire* shape, not just at one confidence
   level — rejects uniformity for both (p<0.0001). This is worth including precisely because it's
   a limitation, found and reported rather than hidden: coverage-only evaluation (what almost all
   prior motorsports-ML work — and the earlier draft of this same paper — reports) can look
   calibrated while failing a more rigorous test.
3. **The "smart hybrid" (normalized conformal) didn't pay off, and that's itself informative.**
   Scaling conformal residuals by MC-Dropout's predicted std should, in principle, produce tighter
   *adaptive* intervals than a fixed width. It didn't (4.54s vs. 4.34s, at matched coverage) —
   meaning the network's per-lap uncertainty estimate doesn't discriminate easy-vs-hard laps well
   enough to exploit for interval sizing. Report this as a negative result rather than omitting it:
   it's a more credible, more specific characterization of the model's limits than a vague
   "uncertainty could be better calibrated" line.

Where this goes in the paper: Section 4.2 (Model and Analysis Methods) gets a new paragraph
introducing the deep ensemble and both conformal variants alongside RiskCalc Pro's existing
MC-Dropout description; Section 5.1 (Key Results) gets the table above in place of / alongside the
existing Table 1; the three findings above become the paragraph that follows it. This did **not**
go into the 500-word SSAC abstract — the abstract's word budget is already spent on the
cross-circuit generalization finding, which is more directly relevant to a sports-analytics
audience than a general UQ-methods comparison. The full paper (due mid-Feb 2027 if the abstract is
accepted) has no such limit and is where this belongs.

## Paste-ready replacements

**Executive Summary, replace:**
> "RiskCalc Pro delivering lap time predictions with 95% confidence interval coverage of 87% and mean absolute error of 0.34 seconds"

**with:**
> "RiskCalc Pro delivering lap time predictions with mean absolute error of 0.90 seconds and 98% empirical coverage of the nominal 95% confidence interval when predicting within a known race, and — via 5-fold cross-validation rotating the held-out circuit across all five races — 2.49 ± 0.72 seconds MAE with 74.2% ± 15.2% coverage when generalizing to a circuit unseen during training. Per-circuit error ranges from 1.52 seconds (Australia) to 3.39 seconds (Monaco), showing that generalization difficulty is itself circuit-dependent rather than a fixed cost of leaving the training track."

**Section 4.1 Data Overview, replace** the single-Monaco-race paragraph and the 1,247/1,089-lap
figures with: 5 races (Monaco, Bahrain, Australia, Japan, Spain), 5,377 raw laps, 5,112 after
cleaning (per-circuit IQR filter, not a fixed absolute window — explain why: a fixed window tuned
to one circuit's pace discards most laps at circuits with different lap-time scales).

**Section 5.1 Table 1, replace** with two tables (fits SSAC's "≤2 tables/figures" limit for the
abstract; the full paper can use as many as needed): temporal-holdout metrics, and
leave-one-circuit-out metrics (both tables above).

**Section 7 Limitations, remove:**
> "the reliance on single-race data from one circuit, limiting generalizability"

— this is fixed (5-fold leave-one-circuit-out cross-validation is now done, see the "Deeper
validation" section above). Replace with a limitation the current results actually support: with
only 5 circuits, each cross-validation fold trains on just 4 — the 15.2-point SD in coverage across
folds is itself a small-sample estimate; a dozen-plus-circuit season would tighten it considerably.

**Everywhere describing NeuroStrategy**: keep calling it "simulation-based," never "neural
network-based" — the paper's body text is already accurate here, but `README.md` said "neural
network-based strategy optimization," which I've corrected in the repo since it directly
contradicts the code (fixed hardcoded tire-degradation constants, no learned model, no NN).

**Section 8 Recommendations ("Technical Development"), replace** the generic bullet list
("API development... ensemble methods... multi-circuit validation" — the last one is now done)
with the one concrete, high-value gap the current system actually has: **RiskCalc Pro's
uncertainty never reaches NeuroStrategy.** The strategy engine ranks pit options by a single
point-estimate `avg_lap_time`; it doesn't use the predictive distribution RiskCalc Pro already
produces. The next real step is expected-utility ranking — sample candidate strategies' outcomes
from RiskCalc Pro's predictive distribution (not a fixed constant) and rank by expected finishing
position with a risk-aversion term for high-variance strategies, so a calibrated confidence
interval changes *what the system recommends*, not just how it explains the recommendation. This
is also the abstract's added Conclusion sentence — keep the two consistent.

**Optional closing line only** (do not expand into a section — SSAC is a sports-analytics venue
and a detour into other domains reads as unfocused): the calibrated-uncertainty-for-real-time-
decisions framework generalizes beyond racing to other high-stakes sequential-decision domains
(cited in the EXPO talk: emergency response, medical triage) — one sentence in Section 8's closing
paragraph at most, not a subsection.
