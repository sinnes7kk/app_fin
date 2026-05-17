# Faithful Replay Backtest — 2026-05-17 22:55

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 205 / 251**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 25 | 10.0% |
| T1_then_stop | 23 | 9.2% |
| stop | 72 | 28.7% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 131 | 52.2% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 205 | 51.7% | +0.37 | +0.07 | +1.27 | +3.00 | -1.47 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 2 | 100.0% | +1.62 | +36.41 | -34.80 |
| A- | 19 | 63.2% | +0.38 | +0.58 | -0.20 |
| B+ | 93 | 51.6% | +0.28 | -2.01 | +2.29 |
| B | 86 | 47.7% | +0.40 | +0.96 | -0.56 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 21 | 66.7% | +0.50 | +2.82 |
| B | 184 | 50.0% | +0.35 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 10 | 20.0% | -0.47 | +0.24 | 0.8 | 0.0% | 0.0% | 41.7% |
| position | 70 | 50.0% | +0.34 | +1.30 | 1.9 | 14.6% | 10.1% | 38.2% |
| leap | 24 | 50.0% | +0.15 | +0.92 | 0.7 | 8.3% | 5.6% | 30.6% |
| unknown | 101 | 56.4% | +0.51 | +1.50 | 1.6 | 21.9% | 12.3% | 39.5% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 5 | 80.0% | +1.29 |
| A | leap | 4 | 100.0% | +0.53 |
| A | unknown | 12 | 50.0% | +0.16 |
| B | swing | 10 | 20.0% | -0.47 |
| B | position | 65 | 47.7% | +0.27 |
| B | leap | 20 | 40.0% | +0.08 |
| B | unknown | 89 | 57.3% | +0.56 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 12 | 1.0 | 1.0 | 1.0 | 1 |
| position | 89 | 2.2 | 1.0 | 3.0 | 7 |
| leap | 36 | 1.1 | 1.0 | 1.0 | 2 |
| unknown | 114 | 1.6 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| swing | 12 | 16.7% | 8.3% | 0.0% | 0.0% |
| position | 89 | 46.1% | 34.8% | 16.9% | 10.1% |
| leap | 36 | 30.6% | 22.2% | 13.9% | 5.6% |
| unknown | 114 | 51.8% | 40.4% | 21.9% | 12.3% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 1 | 0 | 0.0% |
| position | 31 | 15 | 48.4% |
| leap | 8 | 5 | 62.5% |
| unknown | 46 | 25 | 54.3% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 12 | LOW | 5 | 0.5 | 2.1 | 1.0 | -0.47 |
| position | 89 | HIGH | 10 | 1.0 | 2.3 | 1.0 | +0.34 |
| leap | 36 | HIGH | 15 | 0.5 | 2.1 | 1.0 | +0.15 |
| unknown | 114 | HIGH | 5 | 1.0 | 2.2 | 1.0 | +0.51 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
