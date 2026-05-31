# Faithful Replay Backtest — 2026-05-31 23:01

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 296 / 402**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 31 | 7.7% |
| T1_then_stop | 33 | 8.2% |
| stop | 100 | 24.9% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 238 | 59.2% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 296 | 50.3% | +0.32 | +0.03 | +1.25 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 6 | 66.7% | +0.81 | +36.41 | -35.61 |
| A- | 23 | 60.9% | +0.34 | +0.58 | -0.25 |
| B+ | 137 | 51.8% | +0.25 | -2.01 | +2.26 |
| B | 125 | 45.6% | +0.34 | +0.96 | -0.62 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 29 | 62.1% | +0.43 | +2.82 |
| B | 267 | 49.1% | +0.30 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 0 | 0.0% | — | — | 0.0 | 0.0% | 0.0% | 0.0% |
| swing | 14 | 21.4% | -0.42 | +0.42 | 1.0 | 6.2% | 0.0% | 50.0% |
| position | 110 | 50.9% | +0.36 | +1.30 | 1.3 | 15.3% | 8.9% | 30.6% |
| leap | 49 | 40.8% | +0.10 | +0.78 | 0.6 | 9.5% | 3.6% | 29.8% |
| unknown | 123 | 56.9% | +0.45 | +1.34 | 1.5 | 18.8% | 9.7% | 36.1% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 8 | 62.5% | +0.81 |
| A | leap | 8 | 75.0% | +0.41 |
| A | unknown | 13 | 53.8% | +0.21 |
| B | lottery | 0 | — | — |
| B | swing | 14 | 21.4% | -0.42 |
| B | position | 102 | 50.0% | +0.32 |
| B | leap | 41 | 34.1% | +0.03 |
| B | unknown | 110 | 57.3% | +0.48 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 16 | 1.1 | 1.0 | 1.0 | 2 |
| position | 157 | 1.8 | 1.0 | 2.0 | 7 |
| leap | 84 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 144 | 1.7 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 1 | 0.0% | 0.0% | 0.0% | 0.0% |
| swing | 16 | 31.2% | 12.5% | 0.0% | 0.0% |
| position | 157 | 40.8% | 31.8% | 14.0% | 8.9% |
| leap | 84 | 22.6% | 16.7% | 11.9% | 3.6% |
| unknown | 144 | 47.2% | 34.0% | 18.8% | 9.7% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 2 | 0 | 0.0% |
| position | 50 | 22 | 44.0% |
| leap | 14 | 10 | 71.4% |
| unknown | 49 | 27 | 55.1% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 1 | LOW | 3 | 0.5 | 1.5 | — | — |
| swing | 16 | MEDIUM | 5 | 0.5 | 2.1 | 1.0 | -0.42 |
| position | 157 | HIGH | 10 | 1.0 | 2.1 | 1.0 | +0.36 |
| leap | 84 | HIGH | 15 | 0.5 | 2.2 | 1.0 | +0.10 |
| unknown | 144 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.45 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
