# Faithful Replay Backtest — 2026-06-28 23:03

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 412 / 691**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 38 | 5.5% |
| T1_then_stop | 55 | 8.0% |
| stop | 134 | 19.4% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 464 | 67.1% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 412 | 50.0% | +0.29 | +0.01 | +1.20 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 9 | 66.7% | +0.49 | +36.41 | -35.92 |
| A- | 48 | 52.1% | +0.22 | +0.58 | -0.37 |
| B+ | 190 | 52.1% | +0.28 | -2.01 | +2.28 |
| B | 156 | 45.5% | +0.30 | +0.96 | -0.66 |
| B- | 9 | 55.6% | +0.73 | +3.73 | -3.00 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 57 | 54.4% | +0.26 | +2.82 |
| B | 355 | 49.3% | +0.30 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 17 | 29.4% | -0.22 | +0.63 | 0.9 | 13.0% | 4.3% | 43.5% |
| position | 161 | 52.2% | +0.31 | +1.19 | 0.8 | 12.2% | 4.7% | 23.1% |
| leap | 76 | 43.4% | +0.13 | +0.79 | 0.5 | 9.9% | 2.5% | 24.2% |
| unknown | 155 | 53.5% | +0.41 | +1.29 | 1.4 | 20.4% | 9.9% | 36.5% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 0 | — | — |
| A | position | 20 | 50.0% | +0.25 |
| A | leap | 13 | 76.9% | +0.54 |
| A | unknown | 24 | 45.8% | +0.12 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 17 | 29.4% | -0.22 |
| B | position | 141 | 52.5% | +0.32 |
| B | leap | 63 | 36.5% | +0.04 |
| B | unknown | 131 | 55.0% | +0.47 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 23 | 1.2 | 1.0 | 1.0 | 2 |
| position | 320 | 1.7 | 1.0 | 2.0 | 7 |
| leap | 161 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 181 | 1.6 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 23 | 30.4% | 17.4% | 4.3% | 4.3% |
| position | 320 | 28.1% | 21.9% | 10.9% | 4.7% |
| leap | 161 | 18.6% | 13.7% | 9.9% | 2.5% |
| unknown | 181 | 48.1% | 34.8% | 18.2% | 9.9% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 4 | 1 | 25.0% |
| position | 70 | 35 | 50.0% |
| leap | 22 | 16 | 72.7% |
| unknown | 63 | 33 | 52.4% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 23 | MEDIUM | 5 | 0.5 | 2.2 | 1.0 | -0.22 |
| position | 320 | HIGH | 10 | 0.5 | 2.1 | 1.0 | +0.31 |
| leap | 161 | HIGH | 15 | 0.5 | 2.1 | 1.0 | +0.13 |
| unknown | 181 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.41 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
