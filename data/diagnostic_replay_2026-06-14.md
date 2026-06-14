# Faithful Replay Backtest — 2026-06-14 23:10

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 350 / 549**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 36 | 6.6% |
| T1_then_stop | 48 | 8.7% |
| stop | 116 | 21.1% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 349 | 63.6% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 350 | 51.1% | +0.33 | +0.07 | +1.23 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 6 | 66.7% | +0.81 | +36.41 | -35.61 |
| A- | 33 | 60.6% | +0.36 | +0.58 | -0.23 |
| B+ | 172 | 53.5% | +0.31 | -2.01 | +2.32 |
| B | 134 | 44.8% | +0.31 | +0.96 | -0.65 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 39 | 61.5% | +0.42 | +2.82 |
| B | 311 | 49.8% | +0.32 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 0 | 0.0% | — | — | 0.0 | 0.0% | 0.0% | 0.0% |
| swing | 17 | 29.4% | -0.22 | +0.63 | 1.0 | 14.3% | 4.8% | 47.6% |
| position | 142 | 52.1% | +0.35 | +1.26 | 1.0 | 15.4% | 6.1% | 27.9% |
| leap | 58 | 39.7% | +0.10 | +0.71 | 0.5 | 7.3% | 2.4% | 22.6% |
| unknown | 133 | 57.9% | +0.49 | +1.41 | 1.4 | 21.3% | 11.0% | 36.8% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 13 | 53.8% | +0.42 |
| A | leap | 10 | 80.0% | +0.61 |
| A | unknown | 16 | 56.2% | +0.31 |
| B | lottery | 0 | — | — |
| B | swing | 17 | 29.4% | -0.22 |
| B | position | 129 | 51.9% | +0.34 |
| B | leap | 48 | 31.2% | -0.01 |
| B | unknown | 117 | 58.1% | +0.51 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 21 | 1.2 | 1.0 | 1.0 | 2 |
| position | 247 | 1.7 | 1.0 | 2.0 | 7 |
| leap | 124 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 155 | 1.6 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 2 | 0.0% | 0.0% | 0.0% | 0.0% |
| swing | 21 | 33.3% | 19.0% | 4.8% | 4.8% |
| position | 247 | 32.4% | 26.7% | 13.8% | 6.1% |
| leap | 124 | 16.9% | 12.1% | 8.9% | 2.4% |
| unknown | 155 | 49.7% | 37.4% | 20.6% | 11.0% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 4 | 1 | 25.0% |
| position | 66 | 34 | 51.5% |
| leap | 15 | 11 | 73.3% |
| unknown | 58 | 32 | 55.2% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 2 | LOW | 3 | 0.5 | 1.5 | — | — |
| swing | 21 | MEDIUM | 5 | 0.5 | 2.2 | 1.0 | -0.22 |
| position | 247 | HIGH | 10 | 0.5 | 2.2 | 1.0 | +0.35 |
| leap | 124 | HIGH | 15 | 0.5 | 2.2 | 1.0 | +0.10 |
| unknown | 155 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.49 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
