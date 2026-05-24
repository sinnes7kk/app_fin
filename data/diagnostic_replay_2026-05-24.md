# Faithful Replay Backtest — 2026-05-24 22:59

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 258 / 334**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 26 | 7.8% |
| T1_then_stop | 30 | 9.0% |
| stop | 90 | 26.9% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 188 | 56.3% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 258 | 49.2% | +0.28 | +0.00 | +1.25 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 6 | 66.7% | +0.81 | +36.41 | -35.61 |
| A- | 21 | 61.9% | +0.34 | +0.58 | -0.24 |
| B+ | 118 | 49.2% | +0.19 | -2.01 | +2.20 |
| B | 108 | 45.4% | +0.31 | +0.96 | -0.65 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 27 | 63.0% | +0.44 | +2.82 |
| B | 231 | 47.6% | +0.26 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 13 | 15.4% | -0.53 | +0.33 | 1.0 | 0.0% | 0.0% | 46.7% |
| position | 90 | 45.6% | +0.20 | +1.17 | 1.5 | 14.5% | 7.3% | 36.3% |
| leap | 37 | 45.9% | +0.15 | +0.88 | 0.6 | 9.8% | 4.9% | 29.5% |
| unknown | 118 | 56.8% | +0.47 | +1.38 | 1.5 | 20.1% | 10.4% | 37.3% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 6 | 66.7% | +0.98 |
| A | leap | 8 | 75.0% | +0.41 |
| A | unknown | 13 | 53.8% | +0.21 |
| B | swing | 13 | 15.4% | -0.53 |
| B | position | 84 | 44.0% | +0.14 |
| B | leap | 29 | 37.9% | +0.08 |
| B | unknown | 105 | 57.1% | +0.50 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 15 | 1.2 | 1.0 | 1.0 | 2 |
| position | 124 | 2.0 | 1.0 | 3.0 | 7 |
| leap | 61 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 134 | 1.7 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| swing | 15 | 26.7% | 6.7% | 0.0% | 0.0% |
| position | 124 | 41.1% | 30.6% | 12.1% | 7.3% |
| leap | 61 | 26.2% | 19.7% | 13.1% | 4.9% |
| unknown | 134 | 50.7% | 36.6% | 20.1% | 10.4% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 1 | 0 | 0.0% |
| position | 38 | 15 | 39.5% |
| leap | 12 | 8 | 66.7% |
| unknown | 49 | 27 | 55.1% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 15 | MEDIUM | 5 | 0.5 | 2.2 | 1.0 | -0.53 |
| position | 124 | HIGH | 10 | 1.0 | 2.3 | 1.0 | +0.20 |
| leap | 61 | HIGH | 15 | 0.5 | 2.2 | 1.0 | +0.15 |
| unknown | 134 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.47 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
