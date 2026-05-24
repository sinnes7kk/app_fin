# Conviction-Weight Recalibration — 2026-05-24 22:59

Refit of `FLOW_TRACKER_WEIGHTS_ACCUM` against the bar-by-bar replay `realized_r` target produced by `scripts/build_replay_backtest.py`. Method: chronological 60/40 train-validate split, NNLS fit on the train slice, OOS Spearman rank correlation on the validate slice, weights normalized to sum 1.0.

**Acceptance criteria (sample-size-aware):**

- *Loose regime* (n_train < 60): OOS Spearman > 0 AND OOS Spearman ≥ legacy OOS Spearman.
- *Tight regime* (n_train ≥ 60): OOS Spearman ≥ 0.10 AND OOS Spearman ≥ legacy + 0.05.

If either fails, legacy weights are kept and `accept=False` is recorded.

---

## 1. Global fit

| n_train | n_val | Confidence | Regime | OOS Spearman (new) | OOS Spearman (legacy) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 154 | 104 | high | tight | +0.030 | +0.062 | ❌ reject (tight_threshold_unmet (need spearman>=0.10 and lift>=0.05, got new=0.030, legacy=0.062)) |

**New weights (global):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

**Legacy weights (for comparison):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

---

## 2. Per-bucket fits

Each DTE bucket gets an independent fit. Buckets with low n fall back to the global fit (or legacy if global was rejected).

| Bucket | n_train | n_val | Conf | Regime | OOS new | OOS legacy | Accept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 0 | 0 | low | loose | — | — | ❌ |
| swing | 13 | 0 | low | loose | — | — | ❌ |
| position | 54 | 36 | medium | loose | +0.346 | +0.256 | ✅ |
| leap | 30 | 7 | medium | loose | +0.541 | +0.727 | ❌ |
| unknown | 70 | 48 | high | tight | +0.038 | +0.183 | ❌ |

**`position` weights:** `persistence=0.43, intensity=0.07, consistency=0.16, accel=0.25, mass=0.09, oi_change=0.00`  

---

## 3. Honest caveats

- **Sample size**: at the time of writing, the panel has ~150 replayed rows. Per-bucket fits below n=30 are LOW confidence and explicitly fall back to legacy. Re-run this recalibration after Stage A's sequencing fix produces 4-6 weeks of clean per-bucket data.

- **Proxy features ≠ in-flight components**: the production `conviction_score` is built from in-flight `_norm` intermediates that are not persisted in `grade_history.csv`. We refit against the closest available *persisted* approximations (persistence_ratio, log-normalized prem_mcap_bps, |accumulation_score|, accel_ratio_today, log-normalized cumulative_premium, |latest_oi_change|). The fitted weights are interpreted as a re-weighting recommendation; production may still keep the legacy formula if the recommendation isn't strong enough.

- **OOS Spearman alone isn't enough**: a positive OOS rank correlation says the score *orders* trades correctly more often than not. It does not guarantee that the *level* of the score (and therefore grade boundaries) maps to the right hit rates. Stage D.4 separately recalibrates the grade-tier thresholds against realized R quantiles.


---

## Grade-history input audit

Inspecting `data/grade_history.csv` (375 rows). For each of the six raw inputs that feed conviction_recalibration's proxies, we count blank/null vs exact-zero vs non-zero rows. Columns flagged as `⚠ suspicious` have > 50% blank-or-zero values, which usually means an upstream feature is silently producing zeros and the fit is treating them as low-signal.

| Column | n | blank | zero | non-zero | blank+zero % | flag |
| --- | --- | --- | --- | --- | --- | --- |
| `persistence_ratio` | 375 | 0 | 0 | 375 | 0.0% | ok |
| `prem_mcap_bps` | 375 | 0 | 0 | 375 | 0.0% | ok |
| `accumulation_score` | 375 | 0 | 0 | 375 | 0.0% | ok |
| `accel_ratio_today` | 375 | 0 | 360 | 15 | 96.0% | ⚠ suspicious |
| `cumulative_premium` | 375 | 0 | 0 | 375 | 0.0% | ok |
| `latest_oi_change` | 375 | 0 | 0 | 375 | 0.0% | ok |

**Action:** investigate the producers of `accel_ratio_today`. Most likely the feature is being computed before its inputs are populated, or is hitting a default branch in `compute_multi_day_flow`.
