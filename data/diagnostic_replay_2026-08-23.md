# Faithful Replay Backtest — 2026-08-23 22:22

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1211 / 1291**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 176 | 13.6% |
| T1_then_stop | 303 | 23.5% |
| stop | 630 | 48.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 5 | 0.4% |
| no_exit_yet | 177 | 13.7% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1211 | 44.8% | +0.12 | -0.30 | +1.41 | +3.00 | -4.77 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | +5.17 | -4.38 |
| A | 75 | 49.3% | +0.41 | +1.93 | -1.53 |
| A- | 143 | 46.9% | +0.18 | +0.90 | -0.72 |
| B+ | 513 | 46.4% | +0.13 | +0.74 | -0.61 |
| B | 459 | 42.0% | +0.05 | +0.82 | -0.77 |
| B- | 17 | 29.4% | -0.38 | -0.20 | -0.18 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 222 | 47.7% | +0.27 | +1.32 |
| B | 989 | 44.1% | +0.08 | +0.76 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | 66.7% | +0.33 | +1.16 | 2.0 | 50.0% | 0.0% | 75.0% |
| swing | 54 | 46.3% | +0.01 | +1.18 | 2.9 | 35.0% | 10.0% | 66.7% |
| position | 630 | 44.8% | +0.12 | +1.39 | 3.6 | 32.8% | 12.7% | 73.0% |
| leap | 258 | 41.5% | +0.03 | +1.28 | 3.5 | 32.4% | 11.0% | 74.0% |
| unknown | 263 | 47.1% | +0.22 | +1.57 | 3.7 | 35.1% | 19.9% | 69.7% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 115 | 53.0% | +0.35 |
| A | leap | 49 | 38.8% | +0.12 |
| A | unknown | 55 | 45.5% | +0.26 |
| B | lottery | 6 | 66.7% | +0.33 |
| B | swing | 51 | 47.1% | +0.03 |
| B | position | 515 | 42.9% | +0.06 |
| B | leap | 209 | 42.1% | +0.01 |
| B | unknown | 208 | 47.6% | +0.21 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 8 | 2.5 | 1.5 | 3.0 | 6 |
| swing | 60 | 3.2 | 2.0 | 5.0 | 15 |
| position | 671 | 3.4 | 2.0 | 4.0 | 17 |
| leap | 281 | 3.2 | 2.0 | 3.0 | 20 |
| unknown | 271 | 3.3 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 8 | 50.0% | 50.0% | 0.0% | 0.0% |
| swing | 60 | 50.0% | 41.7% | 15.0% | 8.3% |
| position | 671 | 58.9% | 47.5% | 19.8% | 10.7% |
| leap | 281 | 55.2% | 42.0% | 21.4% | 8.9% |
| unknown | 271 | 60.9% | 50.9% | 24.4% | 18.1% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 4 | 0 | 0.0% |
| swing | 25 | 9 | 36.0% |
| position | 319 | 133 | 41.7% |
| leap | 118 | 60 | 50.8% |
| unknown | 138 | 66 | 47.8% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 8 | LOW | 4 | 1.0 | 2.0 | 1.5 | +0.33 |
| swing | 60 | HIGH | 6 | 1.0 | 2.5 | 2.0 | +0.01 |
| position | 671 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.12 |
| leap | 281 | HIGH | 15 | 1.0 | 2.3 | 2.0 | +0.03 |
| unknown | 271 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.22 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 28 | 53.6% | +0.25 | +0.54 |
| All-only | 237 | 42.6% | +0.07 | -0.24 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 141 | 42.6% | +0.05 |
| 3 days | 91 | 42.9% | +0.12 |
| 4 days | 55 | 49.1% | +0.16 |
| 5+ days | 2 | 0.0% | -1.00 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 73 | 42.5% | +0.10 |
| 0.75–0.99 | 2 | 50.0% | +0.15 |
| 0.50–0.74 | 84 | 34.5% | -0.07 |
| < 0.50 | 130 | 50.0% | +0.17 |
