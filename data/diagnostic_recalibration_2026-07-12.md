# Conviction-Weight Recalibration — 2026-07-12 22:51

Refit of `FLOW_TRACKER_WEIGHTS_ACCUM` against the bar-by-bar replay `realized_r` target produced by `scripts/build_replay_backtest.py`. Method: chronological 60/40 train-validate split, NNLS fit on the train slice, OOS Spearman rank correlation on the validate slice, weights normalized to sum 1.0.

**Acceptance criteria (sample-size-aware):**

- *Loose regime* (n_train < 60): OOS Spearman > 0 AND OOS Spearman ≥ legacy OOS Spearman.
- *Tight regime* (n_train ≥ 60): OOS Spearman ≥ 0.10 AND OOS Spearman ≥ legacy + 0.05.

If either fails, legacy weights are kept and `accept=False` is recorded.

---

## 1. Global fit

| n_train | n_val | Confidence | Regime | OOS Spearman (new) | OOS Spearman (legacy) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 272 | 182 | high | tight | +0.011 | +0.045 | ❌ reject (tight_threshold_unmet (need spearman>=0.10 and lift>=0.05, got new=0.011, legacy=0.045)) |

**New weights (global):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

**Legacy weights (for comparison):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

---

## 2. Per-bucket fits

Each DTE bucket gets an independent fit. Buckets with low n fall back to the global fit (or legacy if global was rejected).

| Bucket | n_train | n_val | Conf | Regime | OOS new | OOS legacy | Accept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 0 | low | loose | — | — | ❌ |
| swing | 21 | 0 | low | loose | — | — | ❌ |
| position | 100 | 67 | high | tight | +0.076 | +0.059 | ❌ |
| leap | 51 | 35 | medium | loose | -0.236 | +0.160 | ❌ |
| unknown | 106 | 71 | high | tight | +0.015 | +0.027 | ❌ |


---

## 3. Honest caveats

- **Sample size**: at the time of writing, the panel has ~150 replayed rows. Per-bucket fits below n=30 are LOW confidence and explicitly fall back to legacy. Re-run this recalibration after Stage A's sequencing fix produces 4-6 weeks of clean per-bucket data.

- **Proxy features ≠ in-flight components**: the production `conviction_score` is built from in-flight `_norm` intermediates that are not persisted in `grade_history.csv`. We refit against the closest available *persisted* approximations (persistence_ratio, log-normalized prem_mcap_bps, |accumulation_score|, accel_ratio_today, log-normalized cumulative_premium, |latest_oi_change|). The fitted weights are interpreted as a re-weighting recommendation; production may still keep the legacy formula if the recommendation isn't strong enough.

- **OOS Spearman alone isn't enough**: a positive OOS rank correlation says the score *orders* trades correctly more often than not. It does not guarantee that the *level* of the score (and therefore grade boundaries) maps to the right hit rates. Stage D.4 separately recalibrates the grade-tier thresholds against realized R quantiles.


---

## Grade-history input audit

Inspecting `data/grade_history.csv` (855 rows). For each of the six raw inputs that feed conviction_recalibration's proxies, we count blank/null vs exact-zero vs non-zero rows. Columns flagged as `⚠ suspicious` have > 50% blank-or-zero values, which usually means an upstream feature is silently producing zeros and the fit is treating them as low-signal.

| Column | n | blank | zero | non-zero | blank+zero % | flag |
| --- | --- | --- | --- | --- | --- | --- |
| `persistence_ratio` | 855 | 0 | 0 | 855 | 0.0% | ok |
| `prem_mcap_bps` | 855 | 0 | 0 | 855 | 0.0% | ok |
| `accumulation_score` | 855 | 0 | 0 | 855 | 0.0% | ok |
| `accel_ratio_today` | 855 | 0 | 803 | 52 | 93.9% | ⚠ suspicious |
| `cumulative_premium` | 855 | 0 | 0 | 855 | 0.0% | ok |
| `latest_oi_change` | 855 | 0 | 1 | 854 | 0.1% | ok |

**Action:** investigate the producers of `accel_ratio_today`. Most likely the feature is being computed before its inputs are populated, or is hitting a default branch in `compute_multi_day_flow`.
