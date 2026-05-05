# Faithful Replay Backtest — 2026-05-05 10:02

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 149 / 164**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 19 | 11.6% |
| T1_then_stop | 15 | 9.1% |
| stop | 57 | 34.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 73 | 44.5% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 149 | 49.0% | +0.34 | +0.00 | +1.30 | +3.00 | -1.09 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 1 | 100.0% | +3.00 | +36.41 | -33.41 |
| A- | 15 | 60.0% | +0.31 | +0.30 | +0.02 |
| B+ | 48 | 45.8% | +0.15 | -2.11 | +2.25 |
| B | 80 | 47.5% | +0.39 | +1.88 | -1.49 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 16 | 62.5% | +0.48 | +2.70 |
| B | 133 | 47.4% | +0.32 | +0.33 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 7 | 28.6% | -0.40 | +0.31 | 0.9 | 0.0% | 0.0% | 50.0% |
| position | 63 | 52.4% | +0.40 | +1.38 | 2.5 | 18.5% | 13.8% | 44.6% |
| leap | 8 | 25.0% | -0.21 | +0.91 | 1.2 | 0.0% | 11.1% | 44.4% |
| unknown | 71 | 50.7% | +0.41 | +1.53 | 1.9 | 23.2% | 11.0% | 42.7% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 5 | 80.0% | +1.29 |
| A | leap | 1 | 100.0% | +0.46 |
| A | unknown | 10 | 50.0% | +0.08 |
| B | swing | 7 | 28.6% | -0.40 |
| B | position | 58 | 50.0% | +0.33 |
| B | leap | 7 | 14.3% | -0.31 |
| B | unknown | 61 | 50.8% | +0.47 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 8 | 1.0 | 1.0 | 1.0 | 1 |
| position | 65 | 2.2 | 1.0 | 3.0 | 7 |
| leap | 9 | 1.0 | 1.0 | 1.0 | 1 |
| unknown | 82 | 1.8 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| swing | 8 | 25.0% | 12.5% | 0.0% | 0.0% |
| position | 65 | 58.5% | 43.1% | 23.1% | 13.8% |
| leap | 9 | 33.3% | 22.2% | 11.1% | 11.1% |
| unknown | 82 | 51.2% | 41.5% | 23.2% | 11.0% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| swing | 1 | 0 | 0.0% |
| position | 28 | 15 | 53.6% |
| leap | 2 | 1 | 50.0% |
| unknown | 34 | 19 | 55.9% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 8 | LOW | 7 | 0.5 | 2.1 | 1.0 | -0.40 |
| position | 65 | HIGH | 10 | 1.0 | 2.3 | 1.0 | +0.40 |
| leap | 9 | LOW | 15 | 0.5 | 2.1 | 1.0 | -0.21 |
| unknown | 82 | HIGH | 5 | 1.0 | 2.4 | 1.0 | +0.41 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
