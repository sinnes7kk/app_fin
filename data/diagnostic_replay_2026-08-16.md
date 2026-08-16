# Faithful Replay Backtest — 2026-08-16 22:21

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1079 / 1213**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 159 | 13.1% |
| T1_then_stop | 268 | 22.1% |
| stop | 561 | 46.2% |
| ema20_trail | 0 | 0.0% |
| time_stop | 3 | 0.2% |
| no_exit_yet | 222 | 18.3% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1079 | 45.2% | +0.12 | -0.28 | +1.41 | +3.00 | -4.39 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | +5.17 | -4.38 |
| A | 64 | 50.0% | +0.36 | +2.01 | -1.65 |
| A- | 127 | 44.9% | +0.14 | +0.69 | -0.55 |
| B+ | 456 | 45.6% | +0.12 | +0.67 | -0.54 |
| B | 411 | 44.8% | +0.08 | +0.86 | -0.78 |
| B- | 17 | 29.4% | -0.38 | -0.20 | -0.18 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 195 | 46.7% | +0.23 | +1.23 |
| B | 884 | 44.9% | +0.10 | +0.74 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 5 | 80.0% | +0.60 | +1.39 | 1.8 | 50.0% | 0.0% | 62.5% |
| swing | 43 | 41.9% | -0.13 | +1.12 | 2.7 | 31.2% | 12.5% | 68.8% |
| position | 551 | 44.1% | +0.09 | +1.37 | 3.2 | 30.9% | 11.0% | 68.6% |
| leap | 233 | 43.8% | +0.05 | +1.30 | 3.4 | 31.9% | 11.3% | 72.0% |
| unknown | 247 | 49.0% | +0.29 | +1.60 | 3.5 | 34.6% | 20.5% | 64.3% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 94 | 48.9% | +0.22 |
| A | leap | 46 | 41.3% | +0.16 |
| A | unknown | 52 | 48.1% | +0.33 |
| B | lottery | 5 | 80.0% | +0.60 |
| B | swing | 40 | 42.5% | -0.11 |
| B | position | 457 | 43.1% | +0.06 |
| B | leap | 187 | 44.4% | +0.03 |
| B | unknown | 195 | 49.2% | +0.27 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 8 | 2.5 | 1.5 | 3.0 | 6 |
| swing | 48 | 2.8 | 1.0 | 3.5 | 15 |
| position | 637 | 3.4 | 2.0 | 4.0 | 17 |
| leap | 257 | 3.2 | 2.0 | 3.0 | 20 |
| unknown | 263 | 3.3 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 8 | 50.0% | 50.0% | 0.0% | 0.0% |
| swing | 48 | 47.9% | 39.6% | 14.6% | 10.4% |
| position | 637 | 53.4% | 43.0% | 18.1% | 8.9% |
| leap | 257 | 54.5% | 42.4% | 22.6% | 8.9% |
| unknown | 263 | 59.3% | 49.8% | 24.3% | 18.6% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 4 | 0 | 0.0% |
| swing | 19 | 7 | 36.8% |
| position | 274 | 115 | 42.0% |
| leap | 109 | 58 | 53.2% |
| unknown | 131 | 64 | 48.9% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 8 | LOW | 4 | 1.0 | 1.8 | 1.5 | +0.60 |
| swing | 48 | HIGH | 5 | 1.0 | 2.6 | 1.0 | -0.13 |
| position | 637 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.09 |
| leap | 257 | HIGH | 15 | 1.0 | 2.3 | 2.0 | +0.05 |
| unknown | 263 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.29 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 13 | 53.8% | +0.06 | +0.50 |
| All-only | 124 | 43.5% | +0.02 | -0.14 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 85 | 45.9% | +0.10 |
| 3 days | 39 | 43.6% | -0.04 |
| 4 days | 32 | 46.9% | -0.02 |
| 5+ days | 1 | 0.0% | -1.00 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 42 | 45.2% | +0.10 |
| 0.75–0.99 | 2 | 50.0% | +0.34 |
| 0.50–0.74 | 42 | 54.8% | +0.23 |
| < 0.50 | 71 | 39.4% | -0.13 |
