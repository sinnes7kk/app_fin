# Faithful Replay Backtest — 2026-06-21 23:15

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 388 / 628**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 38 | 6.1% |
| T1_then_stop | 54 | 8.6% |
| stop | 126 | 20.1% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 410 | 65.3% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 388 | 51.0% | +0.33 | +0.07 | +1.21 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 6 | 66.7% | +0.81 | +36.41 | -35.61 |
| A- | 36 | 63.9% | +0.42 | +0.58 | -0.16 |
| B+ | 185 | 52.4% | +0.29 | -2.01 | +2.30 |
| B | 152 | 45.4% | +0.30 | +0.96 | -0.66 |
| B- | 9 | 55.6% | +0.73 | +3.73 | -3.00 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 42 | 64.3% | +0.48 | +2.82 |
| B | 346 | 49.4% | +0.31 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 17 | 29.4% | -0.22 | +0.63 | 1.0 | 14.3% | 4.8% | 47.6% |
| position | 152 | 52.0% | +0.32 | +1.23 | 0.9 | 13.1% | 5.2% | 24.8% |
| leap | 69 | 43.5% | +0.18 | +0.85 | 0.5 | 11.5% | 2.9% | 25.2% |
| unknown | 147 | 56.5% | +0.46 | +1.35 | 1.4 | 21.5% | 10.5% | 36.6% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 13 | 53.8% | +0.42 |
| A | leap | 11 | 81.8% | +0.72 |
| A | unknown | 18 | 61.1% | +0.37 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 17 | 29.4% | -0.22 |
| B | position | 139 | 51.8% | +0.31 |
| B | leap | 58 | 36.2% | +0.07 |
| B | unknown | 129 | 55.8% | +0.48 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 21 | 1.2 | 1.0 | 1.0 | 2 |
| position | 290 | 1.7 | 1.0 | 2.0 | 7 |
| leap | 139 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 172 | 1.6 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 21 | 33.3% | 19.0% | 4.8% | 4.8% |
| position | 290 | 30.0% | 23.8% | 11.7% | 5.2% |
| leap | 139 | 20.1% | 15.8% | 11.5% | 2.9% |
| unknown | 172 | 49.4% | 36.6% | 19.2% | 10.5% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 4 | 1 | 25.0% |
| position | 69 | 34 | 49.3% |
| leap | 22 | 16 | 72.7% |
| unknown | 63 | 33 | 52.4% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 21 | MEDIUM | 5 | 0.5 | 2.2 | 1.0 | -0.22 |
| position | 290 | HIGH | 10 | 0.5 | 2.2 | 1.0 | +0.32 |
| leap | 139 | HIGH | 15 | 0.5 | 2.0 | 1.0 | +0.18 |
| unknown | 172 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.46 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
