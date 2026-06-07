# Faithful Replay Backtest — 2026-06-07 23:06

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 330 / 490**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 35 | 7.1% |
| T1_then_stop | 43 | 8.8% |
| stop | 107 | 21.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 305 | 62.2% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 330 | 52.4% | +0.36 | +0.19 | +1.24 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 6 | 66.7% | +0.81 | +36.41 | -35.61 |
| A- | 31 | 64.5% | +0.44 | +0.58 | -0.14 |
| B+ | 161 | 54.7% | +0.33 | -2.01 | +2.34 |
| B | 127 | 45.7% | +0.34 | +0.96 | -0.62 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 37 | 64.9% | +0.50 | +2.82 |
| B | 293 | 50.9% | +0.34 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 0 | 0.0% | — | — | 0.0 | 0.0% | 0.0% | 0.0% |
| swing | 15 | 26.7% | -0.33 | +0.50 | 0.9 | 10.5% | 0.0% | 47.4% |
| position | 126 | 54.8% | +0.42 | +1.34 | 1.1 | 16.3% | 7.4% | 28.1% |
| leap | 57 | 40.4% | +0.11 | +0.72 | 0.5 | 8.0% | 2.7% | 23.9% |
| unknown | 132 | 58.3% | +0.49 | +1.41 | 1.4 | 21.4% | 11.0% | 37.0% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 11 | 63.6% | +0.68 |
| A | leap | 10 | 80.0% | +0.61 |
| A | unknown | 16 | 56.2% | +0.31 |
| B | lottery | 0 | — | — |
| B | swing | 15 | 26.7% | -0.33 |
| B | position | 115 | 53.9% | +0.39 |
| B | leap | 47 | 31.9% | +0.00 |
| B | unknown | 116 | 58.6% | +0.52 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 19 | 1.1 | 1.0 | 1.0 | 2 |
| position | 203 | 1.8 | 1.0 | 2.0 | 7 |
| leap | 113 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 154 | 1.6 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| swing | 19 | 31.6% | 15.8% | 0.0% | 0.0% |
| position | 203 | 36.9% | 30.0% | 15.3% | 7.4% |
| leap | 113 | 18.6% | 13.3% | 9.7% | 2.7% |
| unknown | 154 | 49.4% | 37.0% | 20.8% | 11.0% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 3 | 0 | 0.0% |
| position | 61 | 31 | 50.8% |
| leap | 15 | 11 | 73.3% |
| unknown | 57 | 32 | 56.1% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 1 | LOW | 3 | 0.5 | 1.5 | — | — |
| swing | 19 | MEDIUM | 5 | 0.5 | 2.1 | 1.0 | -0.33 |
| position | 203 | HIGH | 10 | 0.5 | 2.1 | 1.0 | +0.42 |
| leap | 113 | HIGH | 15 | 0.5 | 2.1 | 1.0 | +0.11 |
| unknown | 154 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.49 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
