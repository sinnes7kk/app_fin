# Faithful Replay Backtest — 2026-09-28 00:20

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1478 / 1484**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 210 | 14.2% |
| T1_then_stop | 402 | 27.1% |
| stop | 831 | 56.0% |
| ema20_trail | 0 | 0.0% |
| time_stop | 6 | 0.4% |
| no_exit_yet | 35 | 2.4% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1478 | 43.1% | +0.08 | -0.47 | +1.40 | +3.00 | -4.77 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | +5.17 | -4.38 |
| A | 81 | 48.1% | +0.33 | +1.92 | -1.59 |
| A- | 171 | 43.3% | +0.13 | +0.78 | -0.65 |
| B+ | 607 | 44.0% | +0.08 | +0.77 | -0.69 |
| B | 596 | 41.6% | +0.04 | +0.78 | -0.74 |
| B- | 19 | 36.8% | -0.13 | -0.20 | +0.07 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 256 | 44.9% | +0.20 | +1.23 |
| B | 1222 | 42.7% | +0.06 | +0.76 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 7 | 71.4% | +0.43 | +1.44 | 2.3 | 55.6% | 0.0% | 77.8% |
| swing | 81 | 38.3% | -0.09 | +1.22 | 4.2 | 35.8% | 16.0% | 84.0% |
| position | 762 | 42.9% | +0.08 | +1.39 | 4.4 | 37.0% | 12.9% | 84.8% |
| leap | 328 | 41.5% | +0.03 | +1.31 | 4.2 | 36.2% | 12.8% | 84.5% |
| unknown | 300 | 46.0% | +0.19 | +1.52 | 4.2 | 37.0% | 18.7% | 77.0% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 134 | 47.0% | +0.23 |
| A | leap | 61 | 41.0% | +0.16 |
| A | unknown | 58 | 44.8% | +0.21 |
| B | lottery | 7 | 71.4% | +0.43 |
| B | swing | 78 | 38.5% | -0.08 |
| B | position | 628 | 42.0% | +0.04 |
| B | leap | 267 | 41.6% | -0.00 |
| B | unknown | 242 | 46.3% | +0.19 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 2.0 | 1.0 | 1.8 | 6 |
| swing | 81 | 3.8 | 2.0 | 6.0 | 17 |
| position | 765 | 3.8 | 2.0 | 5.0 | 20 |
| leap | 329 | 3.5 | 2.0 | 4.0 | 20 |
| unknown | 300 | 3.5 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 9 | 55.6% | 55.6% | 11.1% | 0.0% |
| swing | 81 | 54.3% | 43.2% | 13.6% | 9.9% |
| position | 765 | 63.1% | 50.3% | 20.1% | 10.6% |
| leap | 329 | 62.3% | 46.5% | 24.0% | 10.0% |
| unknown | 300 | 62.7% | 51.7% | 23.7% | 16.7% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 5 | 1 | 20.0% |
| swing | 35 | 11 | 31.4% |
| position | 385 | 154 | 40.0% |
| leap | 153 | 79 | 51.6% |
| unknown | 155 | 71 | 45.8% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 9 | LOW | 3 | 1.0 | 1.9 | 1.0 | +0.43 |
| swing | 81 | HIGH | 8 | 1.0 | 2.3 | 2.0 | -0.09 |
| position | 765 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.08 |
| leap | 329 | HIGH | 15 | 1.0 | 2.4 | 2.0 | +0.03 |
| unknown | 300 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.19 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 55 | 45.5% | +0.14 | -0.13 |
| All-only | 447 | 38.7% | -0.02 | -0.60 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 265 | 40.8% | -0.01 |
| 3 days | 182 | 38.5% | -0.00 |
| 4 days | 98 | 42.9% | +0.12 |
| 5+ days | 10 | 10.0% | -0.59 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 156 | 39.1% | +0.01 |
| 0.75–0.99 | 7 | 71.4% | +0.80 |
| 0.50–0.74 | 170 | 35.9% | -0.11 |
| < 0.50 | 222 | 42.3% | +0.06 |
