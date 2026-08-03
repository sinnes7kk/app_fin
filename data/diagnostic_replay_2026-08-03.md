# Faithful Replay Backtest — 2026-08-03 07:50

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 1057 / 1079**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 156 | 14.5% |
| T1_then_stop | 266 | 24.7% |
| stop | 555 | 51.4% |
| ema20_trail | 0 | 0.0% |
| time_stop | 3 | 0.3% |
| no_exit_yet | 99 | 9.2% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 1057 | 45.1% | +0.12 | -0.30 | +1.41 | +3.00 | -4.39 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A+ | 4 | 50.0% | +0.78 | — | — |
| A | 64 | 50.0% | +0.36 | +36.41 | -36.06 |
| A- | 124 | 45.2% | +0.15 | +0.58 | -0.43 |
| B+ | 446 | 45.5% | +0.12 | -2.01 | +2.13 |
| B | 402 | 44.5% | +0.08 | +0.96 | -0.88 |
| B- | 17 | 29.4% | -0.38 | +3.73 | -4.11 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 192 | 46.9% | +0.23 | +2.82 |
| B | 865 | 44.7% | +0.09 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 4 | 75.0% | +0.50 | +1.30 | 2.2 | 50.0% | 0.0% | 66.7% |
| swing | 40 | 40.0% | -0.21 | +1.10 | 3.0 | 35.7% | 11.9% | 76.2% |
| position | 546 | 44.0% | +0.09 | +1.37 | 3.7 | 35.4% | 12.4% | 78.3% |
| leap | 232 | 44.0% | +0.05 | +1.30 | 3.7 | 34.9% | 12.3% | 78.7% |
| unknown | 235 | 49.4% | +0.30 | +1.65 | 3.7 | 37.2% | 22.2% | 68.6% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 3 | 33.3% | -0.39 |
| A | position | 92 | 48.9% | +0.23 |
| A | leap | 46 | 41.3% | +0.16 |
| A | unknown | 51 | 49.0% | +0.34 |
| B | lottery | 4 | 75.0% | +0.50 |
| B | swing | 37 | 40.5% | -0.19 |
| B | position | 454 | 43.0% | +0.06 |
| B | leap | 186 | 44.6% | +0.03 |
| B | unknown | 184 | 49.5% | +0.28 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 3.0 | 2.0 | 4.0 | 6 |
| swing | 42 | 2.9 | 1.0 | 3.8 | 15 |
| position | 557 | 3.4 | 2.0 | 4.0 | 17 |
| leap | 235 | 3.2 | 2.0 | 3.0 | 20 |
| unknown | 239 | 3.4 | 2.0 | 4.0 | 18 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 50.0% | 50.0% | 0.0% | 0.0% |
| swing | 42 | 52.4% | 42.9% | 14.3% | 9.5% |
| position | 557 | 60.9% | 49.0% | 20.5% | 10.1% |
| leap | 235 | 59.6% | 46.4% | 24.7% | 9.8% |
| unknown | 239 | 63.2% | 53.6% | 25.9% | 20.1% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 3 | 0 | 0.0% |
| swing | 18 | 6 | 33.3% |
| position | 273 | 114 | 41.8% |
| leap | 109 | 58 | 53.2% |
| unknown | 128 | 62 | 48.4% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 5 | 1.0 | 1.9 | 2.0 | +0.50 |
| swing | 42 | HIGH | 5 | 1.0 | 2.6 | 1.0 | -0.21 |
| position | 557 | HIGH | 10 | 1.0 | 2.3 | 2.0 | +0.09 |
| leap | 235 | HIGH | 15 | 1.0 | 2.3 | 2.0 | +0.05 |
| unknown | 239 | HIGH | 5 | 1.0 | 2.3 | 2.0 | +0.30 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 13 | 53.8% | +0.06 | +0.50 |
| All-only | 106 | 42.5% | -0.05 | -0.22 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 72 | 45.8% | +0.09 |
| 3 days | 33 | 39.4% | -0.11 |
| 4 days | 29 | 48.3% | -0.07 |
| 5+ days | 1 | 0.0% | -1.00 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 29 | 48.3% | +0.17 |
| 0.75–0.99 | 1 | 0.0% | -1.00 |
| 0.50–0.74 | 39 | 51.3% | +0.07 |
| < 0.50 | 66 | 39.4% | -0.11 |
