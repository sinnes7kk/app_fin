# Faithful Replay Backtest — 2026-07-18 09:00

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 471 / 918**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 40 | 4.4% |
| T1_then_stop | 57 | 6.2% |
| stop | 147 | 16.0% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 674 | 73.4% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 471 | 51.6% | +0.30 | +0.15 | +1.17 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 26 | 57.7% | +0.35 | +36.41 | -36.06 |
| A- | 59 | 55.9% | +0.31 | +0.58 | -0.28 |
| B+ | 197 | 51.8% | +0.26 | -2.01 | +2.27 |
| B | 180 | 48.9% | +0.32 | +0.96 | -0.64 |
| B- | 9 | 55.6% | +0.73 | +3.73 | -3.00 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 85 | 56.5% | +0.32 | +2.82 |
| B | 386 | 50.5% | +0.30 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 24 | 41.7% | -0.04 | +0.68 | 0.8 | 11.1% | 2.8% | 27.8% |
| position | 175 | 52.6% | +0.31 | +1.18 | 0.7 | 9.6% | 3.4% | 17.9% |
| leap | 87 | 47.1% | +0.17 | +0.80 | 0.4 | 7.9% | 1.9% | 18.7% |
| unknown | 182 | 54.4% | +0.41 | +1.26 | 1.3 | 19.4% | 9.3% | 34.3% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 1 | 0.0% | -0.33 |
| A | position | 24 | 54.2% | +0.30 |
| A | leap | 19 | 73.7% | +0.56 |
| A | unknown | 41 | 51.2% | +0.24 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 23 | 43.5% | -0.03 |
| B | position | 151 | 52.3% | +0.31 |
| B | leap | 68 | 39.7% | +0.06 |
| B | unknown | 141 | 55.3% | +0.46 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 36 | 1.2 | 1.0 | 1.0 | 2 |
| position | 446 | 1.6 | 1.0 | 2.0 | 7 |
| leap | 214 | 1.2 | 1.0 | 1.0 | 3 |
| unknown | 216 | 1.5 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 36 | 30.6% | 19.4% | 5.6% | 2.8% |
| position | 446 | 22.2% | 17.3% | 8.3% | 3.4% |
| leap | 214 | 16.8% | 12.1% | 7.9% | 1.9% |
| unknown | 216 | 47.2% | 33.8% | 16.7% | 9.3% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 7 | 2 | 28.6% |
| position | 77 | 37 | 48.1% |
| leap | 26 | 17 | 65.4% |
| unknown | 73 | 36 | 49.3% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 36 | HIGH | 5 | 0.5 | 2.1 | 1.0 | -0.04 |
| position | 446 | HIGH | 10 | 0.5 | 2.1 | 1.0 | +0.31 |
| leap | 214 | HIGH | 15 | 0.5 | 2.0 | 1.0 | +0.17 |
| unknown | 216 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.41 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
