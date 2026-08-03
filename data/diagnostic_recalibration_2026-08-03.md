# Conviction-Weight Recalibration — 2026-08-03 07:50

Refit of `FLOW_TRACKER_WEIGHTS_ACCUM` against the bar-by-bar replay `realized_r` target produced by `scripts/build_replay_backtest.py`. Method: chronological 60/40 train-validate split, NNLS fit on the train slice, OOS Spearman rank correlation on the validate slice, weights normalized to sum 1.0.

**Acceptance criteria (sample-size-aware):**

- *Loose regime* (n_train < 60): OOS Spearman > 0 AND OOS Spearman ≥ legacy OOS Spearman.
- *Tight regime* (n_train ≥ 60): OOS Spearman ≥ 0.10 AND OOS Spearman ≥ legacy + 0.05.

If either fails, legacy weights are kept and `accept=False` is recorded.

---

## 1. Global fit

| n_train | n_val | Confidence | Regime | OOS Spearman (new) | OOS Spearman (legacy) | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 634 | 423 | high | tight | -0.065 | -0.080 | ❌ reject (tight_threshold_unmet (need spearman>=0.10 and lift>=0.05, got new=-0.065, legacy=-0.080)) |

**New weights (global):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

**Legacy weights (for comparison):**

`persistence=0.25, intensity=0.20, consistency=0.25, accel=0.20, mass=0.05, oi_change=0.05`

---

## 2. Per-bucket fits

Each DTE bucket gets an independent fit. Buckets with low n fall back to the global fit (or legacy if global was rejected).

| Bucket | n_train | n_val | Conf | Regime | OOS new | OOS legacy | Accept |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 4 | 0 | low | loose | — | — | ❌ |
| swing | 30 | 10 | medium | loose | +0.116 | +0.794 | ❌ |
| position | 327 | 219 | high | tight | -0.088 | -0.072 | ❌ |
| leap | 139 | 93 | high | tight | +0.036 | +0.157 | ❌ |
| unknown | 141 | 94 | high | tight | -0.270 | -0.223 | ❌ |


---

## 3. Honest caveats

- **Sample size**: at the time of writing, the panel has ~150 replayed rows. Per-bucket fits below n=30 are LOW confidence and explicitly fall back to legacy. Re-run this recalibration after Stage A's sequencing fix produces 4-6 weeks of clean per-bucket data.

- **Proxy features ≠ in-flight components**: the production `conviction_score` is built from in-flight `_norm` intermediates that are not persisted in `grade_history.csv`. We refit against the closest available *persisted* approximations (persistence_ratio, log-normalized prem_mcap_bps, |accumulation_score|, accel_ratio_today, log-normalized cumulative_premium, |latest_oi_change|). The fitted weights are interpreted as a re-weighting recommendation; production may still keep the legacy formula if the recommendation isn't strong enough.

- **OOS Spearman alone isn't enough**: a positive OOS rank correlation says the score *orders* trades correctly more often than not. It does not guarantee that the *level* of the score (and therefore grade boundaries) maps to the right hit rates. Stage D.4 separately recalibrates the grade-tier thresholds against realized R quantiles.


---

## Grade-history input audit

Inspecting `data/grade_history.csv` (1080 rows). For each of the six raw inputs that feed conviction_recalibration's proxies, we count blank/null vs exact-zero vs non-zero rows. Columns flagged as `⚠ suspicious` have > 50% blank-or-zero values, which usually means an upstream feature is silently producing zeros and the fit is treating them as low-signal.

| Column | n | blank | zero | non-zero | blank+zero % | flag |
| --- | --- | --- | --- | --- | --- | --- |
| `persistence_ratio` | 1080 | 0 | 0 | 1080 | 0.0% | ok |
| `prem_mcap_bps` | 1080 | 0 | 0 | 1080 | 0.0% | ok |
| `accumulation_score` | 1080 | 0 | 0 | 1080 | 0.0% | ok |
| `accel_ratio_today` | 1080 | 0 | 1007 | 73 | 93.2% | ⚠ suspicious |
| `cumulative_premium` | 1080 | 0 | 0 | 1080 | 0.0% | ok |
| `latest_oi_change` | 1080 | 0 | 2 | 1078 | 0.2% | ok |

**Action:** investigate the producers of `accel_ratio_today`. Most likely the feature is being computed before its inputs are populated, or is hitting a default branch in `compute_multi_day_flow`.
