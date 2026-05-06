# Faithful Replay Backtest — 2026-05-06 05:37

Source: `data/grade_history.csv` replayed bar-by-bar via `app/analytics/trade_replay.py`. Production exit logic (T2 hit, ATR trail, EMA20 trail, hybrid trail, T1 partial + post-T1 tighten, time stop) is faithfully reproduced; health-based / gamma / wall exits are skipped (no historical data).

**Rows replayed: 154 / 179**.

---


## 1. Replay summary by exit_reason

| Exit reason | n | % of replayed |
| --- | --- | --- |
| T2 | 19 | 10.6% |
| T1_then_stop | 15 | 8.4% |
| stop | 57 | 31.8% |
| ema20_trail | 0 | 0.0% |
| time_stop | 0 | 0.0% |
| no_exit_yet | 88 | 49.2% |

**Aggregate realized-R (all rows):**

| n | Hit | Mean R | Median R | Std | Best | Worst |
| --- | --- | --- | --- | --- | --- | --- |
| 154 | 49.4% | +0.33 | +0.00 | +1.29 | +3.00 | -1.47 |

---

## 2. Per-grade tier with realized R (vs old 5d close-to-close)

Side-by-side comparison: the legacy metric (`forward_excess_return / 0.02`) vs the new bar-by-bar replay (`realized_r`). The two diverge when the trade plan would have exited intraday before the 5d close was reached.


| Grade | n | Hit (replay) | Mean R (replay) | Mean R (legacy 5d) | Δ (new - legacy) |
| --- | --- | --- | --- | --- | --- |
| A | 2 | 100.0% | +1.62 | +36.41 | -34.80 |
| A- | 15 | 60.0% | +0.31 | +0.58 | -0.27 |
| B+ | 51 | 47.1% | +0.16 | -2.13 | +2.29 |
| B | 81 | 46.9% | +0.37 | +1.40 | -1.03 |
| B- | 5 | 60.0% | +0.85 | +3.73 | -2.88 |

**Coarse-grade view (matches dashboard headline):**

| Coarse | n | Hit (replay) | Mean R (replay) | Mean R (legacy) |
| --- | --- | --- | --- | --- |
| A | 17 | 64.7% | +0.47 | +2.82 |
| B | 137 | 47.4% | +0.31 | +0.14 |

---

## 3. Per-DTE-bucket performance

| Bucket | n | Hit | Mean R | Mean MFE | Avg days | % T1 hit | % T2 hit | % stopped |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swing | 7 | 28.6% | -0.40 | +0.31 | 0.9 | 0.0% | 0.0% | 50.0% |
| position | 65 | 52.3% | +0.40 | +1.33 | 2.2 | 16.2% | 12.2% | 39.2% |
| leap | 9 | 33.3% | -0.16 | +0.81 | 1.0 | 0.0% | 9.1% | 36.4% |
| unknown | 73 | 50.7% | +0.39 | +1.49 | 1.8 | 22.1% | 10.5% | 40.7% |

---

## 4. DTE-bucket × grade interaction

| Grade | Bucket | n | Hit | Mean R |
| --- | --- | --- | --- | --- |
| A | position | 5 | 80.0% | +1.29 |
| A | leap | 2 | 100.0% | +0.35 |
| A | unknown | 10 | 50.0% | +0.08 |
| B | swing | 7 | 28.6% | -0.40 |
| B | position | 60 | 50.0% | +0.33 |
| B | leap | 7 | 14.3% | -0.31 |
| B | unknown | 63 | 50.8% | +0.44 |

**Read this as:** a row with high `n` and positive `Mean R` is a profitable cohort. Sparse rows (low n) are inconclusive — *do not* read trends from them.

---

## 5. Time-to-MFE distribution per bucket

| Bucket | n | Mean d-to-MFE | Median | p75 | Max |
| --- | --- | --- | --- | --- | --- |
| swing | 8 | 1.0 | 1.0 | 1.0 | 1 |
| position | 74 | 2.2 | 1.0 | 3.0 | 7 |
| leap | 11 | 1.0 | 1.0 | 1.0 | 1 |
| unknown | 86 | 1.8 | 1.0 | 2.0 | 8 |

**Interpretation:** if `Median d-to-MFE` is lower than the per-bucket `MAX_HOLD_DAYS` config, your time stop is reasonable. If `Median d-to-MFE` is higher than `MAX_HOLD_DAYS`, you are exiting before the typical move plays out.

---

## 6. Path metrics (% reaching +0.5R / +1R / +2R / +3R MFE)

| Bucket | n | +0.5R/3d | +1R/5d | +2R/5d | +3R/10d |
| --- | --- | --- | --- | --- | --- |
| swing | 8 | 25.0% | 12.5% | 0.0% | 0.0% |
| position | 74 | 51.4% | 37.8% | 20.3% | 12.2% |
| leap | 11 | 27.3% | 18.2% | 9.1% | 9.1% |
| unknown | 86 | 48.8% | 39.5% | 22.1% | 10.5% |

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
| position | 74 | HIGH | 10 | 1.0 | 2.3 | 1.0 | +0.40 |
| leap | 11 | LOW | 15 | 0.5 | 2.0 | 1.0 | -0.16 |
| unknown | 86 | HIGH | 5 | 1.0 | 2.3 | 1.0 | +0.39 |

Machine-readable config written to: `data/replay_recommended_config.json` (consumed by Stage C config refactor).

**Honest caveat:** with the current panel size (~104 rows; ~15 Grade A; ~50% unknown DTE pre-Stage-A-fix), per-bucket lottery and leap recommendations are LOW-CONFIDENCE. Values for `swing` and `unknown` are most reliable; `lottery`/`leap` should be re-derived after the sequencing fix produces 4-6 weeks of clean data.
