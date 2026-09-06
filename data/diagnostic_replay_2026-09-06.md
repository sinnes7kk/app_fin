# Faithful Replay Backtest — 2026-09-06 23:33

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1296 / 1442**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 188 | 13.0% |
| T1_then_stop | 324 | 22.5% |
| stop | 682 | 47.3% |
| ema20_trail | 0 | 0.0% |
| time_stop | 5 | 0.3% |
| no_exit_yet | 243 | 16.9% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1296 | 44.8% | +0.11 | -0.34 | +1.41 | +3.00 | -4.77 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | +5.17 | -4.38 |
| A | 79 | 50.6% | +0.39 | +1.92 | -1.53 |
| A- | 149 | 45.6% | +0.17 | +0.78 | -0.61 |
| B+ | 553 | 45.6% | +0.11 | +0.77 | -0.66 |
| B | 494 | 43.3% | +0.07 | +0.78 | -0.71 |
| B- | 17 | 29.4% | -0.38 | -0.20 | -0.18 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 232 | 47.4% | +0.25 | +1.23 |
| B | 1064 | 44.3% | +0.08 | +0.76 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 7 | 71.4% | +0.67 | +1.02 | 1.9 | 44.4% | 0.0% | 66.7% |
| swing | 63 | 39.7% | -0.12 | +1.16 | 2.7 | 30.3% | 11.8% | 60.5% |
| position | 671 | 45.0% | +0.12 | +1.39 | 3.5 | 32.4% | 12.2% | 70.5% |
| leap | 278 | 43.5% | +0.06 | +1.29 | 3.3 | 31.3% | 10.7% | 70.2% |
| unknown | 277 | 46.2% | +0.19 | +1.52 | 3.7 | 33.7% | 18.6% | 69.8% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 123 | 50.4% | +0.30 |
| A | leap | 51 | 43.1% | +0.19 |
| A | unknown | 55 | 45.5% | +0.24 |
| B | lottery | 7 | 71.4% | +0.67 |
| B | swing | 60 | 40.0% | -0.11 |
| B | position | 548 | 43.8% | +0.08 |
| B | leap | 227 | 43.6% | +0.03 |
| B | unknown | 222 | 46.4% | +0.18 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 2.2 | 1.0 | 2.0 | 6 |
| swing | 76 | 3.2 | 2.0 | 5.0 | 15 |
| position | 747 | 3.5 | 2.0 | 5.0 | 17 |
| leap | 319 | 3.3 | 2.0 | 3.0 | 20 |
| unknown | 291 | 3.4 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 44.4% | 44.4% | 0.0% | 0.0% |
| swing | 76 | 43.4% | 34.2% | 13.2% | 9.2% |
| position | 747 | 56.5% | 45.6% | 18.9% | 10.2% |
| leap | 319 | 53.3% | 40.4% | 20.7% | 8.2% |
| unknown | 291 | 59.1% | 48.5% | 23.0% | 16.8% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 4 | 0 | 0.0% |
| swing | 26 | 10 | 38.5% |
| position | 341 | 141 | 41.3% |
| leap | 129 | 66 | 51.2% |
| unknown | 141 | 67 | 47.5% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 9 | LOW | 3 | 1.0 | 1.9 | 1.0 | +0.67 |
| swing | 76 | HIGH | 6 | 1.0 | 2.6 | 2.0 | -0.12 |
| position | 747 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.12 |
| leap | 319 | HIGH | 15 | 1.0 | 2.3 | 2.0 | +0.06 |
| unknown | 291 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.19 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 37 | 56.8% | +0.30 | +0.71 |
| All-only | 305 | 43.0% | +0.07 | -0.30 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 188 | 41.0% | -0.02 |
| 3 days | 115 | 46.1% | +0.18 |
| 4 days | 67 | 50.7% | +0.21 |
| 5+ days | 4 | 25.0% | -0.53 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 104 | 46.2% | +0.08 |
| 0.75–0.99 | 3 | 66.7% | +0.58 |
| 0.50–0.74 | 105 | 38.1% | -0.04 |
| < 0.50 | 162 | 46.3% | +0.15 |
