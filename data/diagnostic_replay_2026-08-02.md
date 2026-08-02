# Faithful Replay Backtest — 2026-08-02 22:57

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 497 / 1067**.

---


## 1. Replay summary by exit_reason

_Fills are **conservative** (2026-07-21): on a bar spanning both the stop and a target the stop is credited first (`intrabar_priority="stop_first"`), and stops that gap through fill at the worse open (`gap_fill`). Realized R is therefore a lower bound, not the optimistic target-first estimate used previously._

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 41 | 3.8% |
| T1_then_stop | 60 | 5.6% |
| stop | 147 | 13.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 819 | 76.8% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 497 | 53.1% | +0.20 | +0.23 | +1.30 | +3.00 | -4.42 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 26 | 61.5% | +0.36 | +36.41 | -36.05 |
| A- | 64 | 54.7% | +0.18 | +0.58 | -0.41 |
| B+ | 205 | 53.2% | +0.14 | -2.01 | +2.14 |
| B | 192 | 51.0% | +0.24 | +0.96 | -0.72 |
| B- | 10 | 60.0% | +0.32 | +3.73 | -3.41 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 90 | 56.7% | +0.23 | +2.82 |
| B | 407 | 52.3% | +0.19 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 25 | 48.0% | +0.08 | +0.80 | 0.8 | 15.0% | 5.0% | 25.0% |
| position | 185 | 52.4% | +0.16 | +1.15 | 0.6 | 8.3% | 2.5% | 15.2% |
| leap | 94 | 48.9% | -0.02 | +0.79 | 0.5 | 7.2% | 1.7% | 17.4% |
| unknown | 190 | 56.8% | +0.35 | +1.26 | 1.3 | 19.8% | 9.1% | 31.0% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 1 | 0.0% | -0.33 |
| A | position | 27 | 48.1% | +0.01 |
| A | leap | 19 | 73.7% | +0.47 |
| A | unknown | 43 | 55.8% | +0.28 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 24 | 50.0% | +0.10 |
| B | position | 158 | 53.2% | +0.19 |
| B | leap | 75 | 42.7% | -0.14 |
| B | unknown | 147 | 57.1% | +0.37 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 40 | 1.4 | 1.0 | 2.0 | 3 |
| position | 554 | 1.7 | 1.0 | 2.0 | 7 |
| leap | 235 | 1.3 | 1.0 | 2.0 | 3 |
| unknown | 232 | 1.5 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 40 | 30.0% | 22.5% | 7.5% | 5.0% |
| position | 554 | 18.2% | 14.3% | 7.0% | 2.9% |
| leap | 235 | 17.4% | 11.5% | 7.2% | 1.7% |
| unknown | 232 | 46.1% | 33.2% | 16.4% | 9.1% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 9 | 3 | 33.3% |
| position | 79 | 39 | 49.4% |
| leap | 27 | 17 | 63.0% |
| unknown | 77 | 38 | 49.4% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 40 | HIGH | 5 | 0.5 | 2.0 | 1.0 | +0.08 |
| position | 554 | HIGH | 10 | 0.5 | 2.2 | 1.0 | +0.16 |
| leap | 235 | HIGH | 15 | 0.5 | 2.1 | 1.0 | -0.02 |
| unknown | 232 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.35 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.

---

## 8. Flow-tracker mode / streak realized R (forward-only)

The Strong ⊂ Activity ⊂ All gates and the multi-day streak fields (`active_days`, `day_persistence`) are stamped onto `grade_history` since 2026-07-21. Rows written before then have blank flags and are excluded here. **The core question:** does tightening the mode gate (All → Activity → Strong) actually raise realized R?

| Mode tier | n | Hit | Mean R | Median R |
| --- | --- | --- | --- | --- |
| Strong | 0 | — | — | — |
| Activity-only | 0 | — | — | — |
| All-only | 6 | 50.0% | -0.27 | -0.09 |

**Read this as:** if the mode gates add value, `Mean R` should climb monotonically from All-only → Activity-only → Strong. If Strong's R is no better (or worse) than the looser tiers at comparable `n`, the Strong gate is costing signal without improving quality.

**Active-day streak vs realized R** (does a longer directional streak predict a better trade?):

| Streak | n | Hit | Mean R |
| --- | --- | --- | --- |
| 2 days | 11 | 81.8% | +0.20 |
| 3 days | 4 | 50.0% | -0.10 |
| 4 days | 1 | 0.0% | -0.50 |

**Day-persistence vs realized R** (higher = more of the window's days leaned the trade's way):

| Persistence | n | Hit | Mean R |
| --- | --- | --- | --- |
| 1.0 (pure) | 12 | 75.0% | +0.23 |
| 0.50–0.74 | 1 | 0.0% | -1.21 |
| < 0.50 | 3 | 66.7% | -0.08 |
