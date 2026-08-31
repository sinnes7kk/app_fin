# Faithful Replay Backtest — 2026-08-31 00:14

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1283 / 1363**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 187 | 13.7% |
| T1_then_stop | 324 | 23.8% |
| stop | 679 | 49.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 5 | 0.4% |
| no_exit_yet | 168 | 12.3% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1283 | 45.1% | +0.12 | -0.34 | +1.42 | +3.00 | -4.77 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | +5.17 | -4.38 |
| A | 79 | 50.6% | +0.39 | +1.90 | -1.51 |
| A- | 149 | 45.6% | +0.17 | +0.92 | -0.75 |
| B+ | 545 | 45.9% | +0.11 | +0.80 | -0.69 |
| B | 489 | 43.6% | +0.07 | +0.84 | -0.77 |
| B- | 17 | 29.4% | -0.38 | -0.20 | -0.18 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 232 | 47.4% | +0.25 | +1.33 |
| B | 1051 | 44.5% | +0.09 | +0.80 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 7 | 71.4% | +0.67 | +1.02 | 1.9 | 44.4% | 0.0% | 66.7% |
| swing | 59 | 40.7% | -0.15 | +1.16 | 3.0 | 32.4% | 11.8% | 67.6% |
| position | 667 | 45.3% | +0.13 | +1.40 | 3.8 | 34.4% | 12.9% | 74.6% |
| leap | 278 | 43.5% | +0.06 | +1.29 | 3.6 | 33.4% | 11.4% | 74.9% |
| unknown | 272 | 46.3% | +0.20 | +1.54 | 3.7 | 34.6% | 19.1% | 71.4% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 123 | 50.4% | +0.30 |
| A | leap | 51 | 43.1% | +0.19 |
| A | unknown | 55 | 45.5% | +0.24 |
| B | lottery | 7 | 71.4% | +0.67 |
| B | swing | 56 | 41.1% | -0.14 |
| B | position | 544 | 44.1% | +0.09 |
| B | leap | 227 | 43.6% | +0.03 |
| B | unknown | 217 | 46.5% | +0.19 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 2.2 | 1.0 | 2.0 | 6 |
| swing | 68 | 3.3 | 2.0 | 5.0 | 15 |
| position | 704 | 3.5 | 2.0 | 5.0 | 17 |
| leap | 299 | 3.3 | 2.0 | 3.0 | 20 |
| unknown | 283 | 3.4 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 44.4% | 44.4% | 0.0% | 0.0% |
| swing | 68 | 47.1% | 36.8% | 13.2% | 8.8% |
| position | 704 | 59.9% | 48.4% | 20.0% | 10.8% |
| leap | 299 | 56.9% | 43.1% | 22.1% | 8.7% |
| unknown | 283 | 60.1% | 49.5% | 23.7% | 17.3% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 4 | 0 | 0.0% |
| swing | 25 | 9 | 36.0% |
| position | 341 | 141 | 41.3% |
| leap | 129 | 66 | 51.2% |
| unknown | 140 | 67 | 47.9% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 9 | LOW | 3 | 1.0 | 1.9 | 1.0 | +0.67 |
| swing | 68 | HIGH | 6 | 1.0 | 2.6 | 2.0 | -0.15 |
| position | 704 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.13 |
| leap | 299 | HIGH | 15 | 1.0 | 2.3 | 2.0 | +0.06 |
| unknown | 283 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.20 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 35 | 57.1% | +0.26 | +0.71 |
| All-only | 298 | 43.6% | +0.08 | -0.30 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 177 | 42.4% | +0.01 |
| 3 days | 115 | 46.1% | +0.18 |
| 4 days | 65 | 50.8% | +0.19 |
| 5+ days | 4 | 25.0% | -0.53 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 94 | 48.9% | +0.14 |
| 0.75–0.99 | 3 | 66.7% | +0.58 |
| 0.50–0.74 | 104 | 38.5% | -0.04 |
| < 0.50 | 160 | 46.2% | +0.14 |
