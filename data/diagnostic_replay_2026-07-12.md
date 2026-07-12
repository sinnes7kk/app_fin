# Faithful Replay Backtest — 2026-07-12 22:51

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 454 / 837**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 40 | 4.8% |
| T1_then_stop | 57 | 6.8% |
| stop | 142 | 17.0% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 598 | 71.4% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 454 | 51.5% | +0.30 | +0.15 | +1.18 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 26 | 57.7% | +0.35 | +36.41 | -36.06 |
| A- | 59 | 55.9% | +0.31 | +0.58 | -0.28 |
| B+ | 195 | 52.3% | +0.27 | -2.01 | +2.28 |
| B | 165 | 47.9% | +0.31 | +0.96 | -0.65 |
| B- | 9 | 55.6% | +0.73 | +3.73 | -3.00 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 85 | 56.5% | +0.32 | +2.82 |
| B | 369 | 50.4% | +0.30 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 21 | 33.3% | -0.11 | +0.67 | 0.9 | 13.3% | 3.3% | 33.3% |
| position | 167 | 52.7% | +0.31 | +1.19 | 0.7 | 10.1% | 3.8% | 19.1% |
| leap | 86 | 47.7% | +0.17 | +0.80 | 0.5 | 8.6% | 2.0% | 20.3% |
| unknown | 177 | 54.8% | +0.41 | +1.28 | 1.3 | 20.3% | 9.7% | 35.3% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 1 | 0.0% | -0.33 |
| A | position | 24 | 54.2% | +0.30 |
| A | leap | 19 | 73.7% | +0.56 |
| A | unknown | 41 | 51.2% | +0.24 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 20 | 35.0% | -0.10 |
| B | position | 143 | 52.4% | +0.32 |
| B | leap | 67 | 40.3% | +0.06 |
| B | unknown | 136 | 55.9% | +0.46 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 30 | 1.2 | 1.0 | 1.0 | 2 |
| position | 397 | 1.6 | 1.0 | 2.0 | 7 |
| leap | 197 | 1.2 | 1.0 | 1.0 | 3 |
| unknown | 207 | 1.5 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 30 | 30.0% | 16.7% | 6.7% | 3.3% |
| position | 397 | 23.9% | 18.4% | 9.1% | 3.8% |
| leap | 197 | 17.8% | 12.7% | 8.6% | 2.0% |
| unknown | 207 | 48.3% | 34.3% | 17.4% | 9.7% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 5 | 2 | 40.0% |
| position | 73 | 36 | 49.3% |
| leap | 25 | 17 | 68.0% |
| unknown | 71 | 36 | 50.7% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 30 | HIGH | 5 | 0.5 | 2.1 | 1.0 | -0.11 |
| position | 397 | HIGH | 10 | 0.5 | 2.1 | 1.0 | +0.31 |
| leap | 197 | HIGH | 15 | 0.5 | 2.0 | 1.0 | +0.17 |
| unknown | 207 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.41 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
