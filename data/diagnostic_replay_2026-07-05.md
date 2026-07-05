# Faithful Replay Backtest — 2026-07-05 23:02

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 443 / 770**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 40 | 5.2% |
| T1_then_stop | 57 | 7.4% |
| stop | 142 | 18.4% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 531 | 69.0% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 443 | 50.6% | +0.30 | +0.06 | +1.19 | +3.00 | -1.84 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 25 | 56.0% | +0.32 | +36.41 | -36.09 |
| A- | 58 | 55.2% | +0.30 | +0.58 | -0.29 |
| B+ | 192 | 52.1% | +0.27 | -2.01 | +2.28 |
| B | 159 | 45.9% | +0.31 | +0.96 | -0.65 |
| B- | 9 | 55.6% | +0.73 | +3.73 | -3.00 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 83 | 55.4% | +0.30 | +2.82 |
| B | 360 | 49.4% | +0.30 | -0.01 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 3 | 33.3% | +0.16 | +1.00 | 0.5 | 16.7% | 0.0% | 0.0% |
| swing | 21 | 33.3% | -0.11 | +0.67 | 0.9 | 13.3% | 3.3% | 33.3% |
| position | 164 | 52.4% | +0.32 | +1.20 | 0.8 | 11.2% | 4.2% | 21.3% |
| leap | 82 | 45.1% | +0.16 | +0.82 | 0.5 | 9.6% | 2.2% | 22.5% |
| unknown | 173 | 53.8% | +0.40 | +1.30 | 1.4 | 21.0% | 10.0% | 36.5% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | swing | 1 | 0.0% | -0.33 |
| A | position | 23 | 52.2% | +0.28 |
| A | leap | 19 | 73.7% | +0.56 |
| A | unknown | 40 | 50.0% | +0.21 |
| B | lottery | 3 | 33.3% | +0.16 |
| B | swing | 20 | 35.0% | -0.10 |
| B | position | 141 | 52.5% | +0.32 |
| B | leap | 63 | 36.5% | +0.04 |
| B | unknown | 133 | 54.9% | +0.46 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 1.5 | 1.5 | 1.8 | 2 |
| swing | 30 | 1.2 | 1.0 | 1.0 | 2 |
| position | 356 | 1.7 | 1.0 | 2.0 | 7 |
| leap | 178 | 1.2 | 1.0 | 1.0 | 2 |
| unknown | 200 | 1.5 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| lottery | 6 | 33.3% | 16.7% | 16.7% | 0.0% |
| swing | 30 | 30.0% | 16.7% | 6.7% | 3.3% |
| position | 356 | 26.1% | 20.2% | 10.1% | 4.2% |
| leap | 178 | 19.1% | 14.0% | 9.6% | 2.2% |
| unknown | 200 | 48.5% | 35.0% | 18.0% | 10.0% |

Conditional probability: of trades that hit +1R, what fraction then go on to +2R? This separates 'small wins' from 'runners.'

| Bucket | Hit +1R | Hit +2R | P(+2R | +1R) |
| --- | --- | --- | --- |
| lottery | 1 | 1 | 100.0% |
| swing | 5 | 2 | 40.0% |
| position | 72 | 36 | 50.0% |
| leap | 25 | 17 | 68.0% |
| unknown | 70 | 36 | 51.4% |

---

## 7. Concrete per-bucket config recommendations

Recommended values are derived from observed time-to-MFE distributions and exit-reason mix. **Where sample size is small (n < 15), the recommendation is marked LOW-CONFIDENCE — these come from a thin panel and should be re-derived after Stage A's sequencing fix produces clean per-bucket data over 4-6 weeks.**

| Bucket | n | Confidence | MAX_HOLD_DAYS | TIME_STOP_MIN_R | ATR_TRAIL_MULT | Median d-to-MFE | Observed Mean R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lottery | 6 | LOW | 3 | 0.5 | 1.5 | 1.5 | +0.16 |
| swing | 30 | HIGH | 5 | 0.5 | 2.1 | 1.0 | -0.11 |
| position | 356 | HIGH | 10 | 0.5 | 2.1 | 1.0 | +0.32 |
| leap | 178 | HIGH | 15 | 0.5 | 2.1 | 1.0 | +0.16 |
| unknown | 200 | HIGH | 5 | 1.0 | 2.1 | 1.0 | +0.40 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
