# Unified feature sweep — 2026-09-13 23:55

Panel: **1478 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1414** matured rows carry `replay_realized_r`, **1478** carry `replay_forward_return_5d`.

Every feature is scored against **two labels**:

- **realized R** (`replay_realized_r`, matured rows only) — the outcome after the replay's stop/target/trail policy. This is what the account actually earns, but it confounds feature quality with exit-policy quality: a feature can predict the move correctly and still score flat because the stop took the trade out first.
- **forward return** (`replay_forward_return_5d`, all rows) — plain 5-day close-to-close, independent of any entry or exit rule. This isolates *does the feature predict price*, and needs no matured trade, so it reaches a usable sample far sooner on newly-added features.

Read them together. Agreement in sign is the strong signal. A feature with a good forward IC but a poor realized-R IC is a hint that the **exit policy**, not the feature, is the thing to fix.

`spearman` = pooled rank IC (in-sample). `oos` = chronological 60/40 walk-forward rank IC. `r_spread` = mean realized R of top tercile − bottom tercile (the $ edge, in R). Sorted by sign-agreeing OOS IC on realized R so in-sample-only flukes sink.

**Caveat:** one bull-market regime, small OOS slices. Treat this as a hypothesis watchlist, not a hit list. A feature needs |IC| that holds OOS across fresh weeks before it earns a place in a live score.

---

## Full ranking (all scorers, both targets)

| Feature | Family | Live? | n | Spearman | p | OOS | n_val | R-spread | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vrp_proxy` | feature-lab | shadow | 1168 | +0.068 | 0.021 | +0.171 | 468 | +0.26 | 1222 | +0.034 | +0.132 |
| `iv_skew_25d` | UW options | shadow | 200 | +0.070 | 0.322 | +0.099 | 80 | +0.19 | 211 | +0.141 | +0.092 |
| `window_return_pct` | flow-tracker (unused) | — | 1414 | +0.012 | 0.661 | +0.087 | 566 | +0.12 | 1478 | -0.060 | +0.082 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 708 | +0.068 | 0.069 | +0.076 | 284 | +0.26 | 766 | -0.025 | -0.018 |
| `aggressor_net_prem_bps` | aggressor | shadow | 857 | +0.002 | 0.948 | +0.072 | 343 | +0.04 | 898 | +0.025 | +0.023 |
| `max_pain_dist_pct` | UW options | shadow | 1218 | +0.012 | 0.678 | +0.065 | 488 | +0.11 | 1279 | -0.059 | +0.040 |
| `directional_sweep_share` | aggressor | shadow | 857 | +0.029 | 0.400 | +0.064 | 343 | -0.01 | 898 | +0.011 | +0.051 |
| `gex_total` | UW options | shadow | 1204 | +0.021 | 0.471 | +0.064 | 482 | +0.06 | 1265 | -0.021 | +0.038 |
| `sector_relative_pct` | feature-lab | shadow | 761 | +0.092 | 0.011 | +0.059 | 305 | +0.35 | 805 | +0.035 | -0.001 |
| `vanna_total` | UW options | shadow | 1204 | +0.026 | 0.362 | +0.054 | 482 | +0.09 | 1265 | -0.041 | +0.016 |
| `dealer_net_delta_at_spot` | UW options | shadow | 708 | +0.018 | 0.636 | +0.052 | 284 | +0.02 | 766 | +0.003 | +0.010 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1414 | +0.016 | 0.550 | +0.051 | 566 | +0.13 | 1478 | -0.042 | -0.029 |
| `persistence_ratio` | conviction_score component | LIVE | 1414 | +0.056 | 0.036 | +0.046 | 566 | +0.16 | 1478 | +0.009 | -0.006 |
| `aggressor_bull_share` | aggressor | shadow | 851 | +0.018 | 0.603 | +0.042 | 341 | +0.02 | 892 | +0.006 | +0.024 |
| `conviction_score` | composite | LIVE grade | 1414 | +0.045 | 0.090 | +0.040 | 566 | +0.17 | 1478 | -0.008 | +0.025 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1414 | +0.014 | 0.602 | +0.037 | 566 | +0.00 | 1478 | -0.042 | +0.026 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 858 | +0.016 | 0.640 | +0.036 | 344 | +0.07 | 899 | +0.016 | +0.135 |
| `prem_momentum_z3d` | feature-lab | shadow | 833 | +0.025 | 0.477 | +0.033 | 334 | -0.02 | 866 | -0.074 | -0.105 |
| `momentum_score` | composite | shadow | 1186 | +0.013 | 0.661 | +0.005 | 475 | -0.01 | 1246 | -0.029 | -0.002 |
| `momentum_composite` | composite | shadow | 1186 | +0.013 | 0.661 | +0.005 | 475 | -0.01 | 1246 | -0.029 | -0.002 |
| `realized_vol_regime` | feature-lab | shadow | 1170 | -0.011 | 0.717 | +0.048 | 468 | -0.06 | 1224 | +0.010 | +0.020 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1414 | -0.006 | 0.815 | +0.039 | 566 | +0.03 | 1478 | -0.050 | -0.045 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1414 | -0.006 | 0.815 | +0.039 | 566 | +0.03 | 1478 | -0.050 | -0.045 |
| `accumulation_score` | flow-tracker (unused) | — | 1414 | -0.003 | 0.907 | +0.002 | 566 | -0.04 | 1478 | +0.012 | +0.055 |
| `far_otm_call_share` | feature-lab | shadow | 858 | -0.018 | 0.607 | +0.062 | 344 | -0.08 | 899 | -0.025 | +0.011 |
| `bollinger_z` | price/vol (T1) | shadow | 487 | -0.001 | 0.988 | +0.057 | 195 | -0.02 | 540 | +0.055 | -0.044 |
| `rsi_14` | price/vol (T1) | shadow | 487 | -0.007 | 0.874 | +0.059 | 195 | -0.02 | 540 | +0.034 | -0.109 |
| `ret_21d` | price/vol (T1) | shadow | 487 | -0.010 | 0.831 | +0.020 | 195 | +0.09 | 540 | +0.038 | -0.118 |
| `beta_63d` | cross-sectional (T2) | shadow | 487 | -0.040 | 0.377 | +0.139 | 195 | -0.21 | 540 | -0.111 | -0.022 |
| `px_vs_sma50` | price/vol (T1) | shadow | 484 | -0.028 | 0.537 | +0.034 | 194 | -0.05 | 537 | +0.044 | -0.106 |
| `ret_5d` | price/vol (T1) | shadow | 487 | -0.011 | 0.803 | +0.015 | 195 | -0.04 | 540 | +0.047 | -0.057 |
| `atr_pct` | price/vol (T1) | shadow | 463 | -0.015 | 0.746 | +0.139 | 186 | -0.13 | 513 | -0.061 | -0.042 |
| `atm_iv_30d` | UW options | shadow | 1204 | -0.036 | 0.207 | +0.010 | 482 | -0.08 | 1265 | -0.056 | -0.023 |
| `expiry_concentration_top1` | UW options | shadow | 1202 | +0.009 | 0.748 | -0.001 | 481 | +0.05 | 1263 | +0.050 | +0.030 |
| `cumulative_premium` | conviction_score component | LIVE | 1414 | -0.023 | 0.387 | -0.004 | 566 | -0.12 | 1478 | -0.090 | -0.065 |
| `latest_oi_change` | conviction_score component | LIVE | 1414 | +0.002 | 0.945 | -0.009 | 566 | -0.02 | 1478 | +0.022 | -0.013 |
| `charm_total` | UW options | shadow | 1204 | +0.030 | 0.299 | -0.012 | 482 | +0.09 | 1265 | +0.008 | -0.033 |
| `bullish_premium_share` | feature-lab | shadow | 1204 | +0.005 | 0.863 | -0.014 | 482 | -0.00 | 1265 | -0.058 | -0.025 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 487 | -0.011 | 0.803 | -0.017 | 195 | +0.05 | 540 | +0.025 | -0.162 |
| `atm_iv_60d` | UW options | shadow | 1204 | -0.041 | 0.155 | -0.017 | 482 | -0.14 | 1265 | -0.055 | -0.035 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1414 | +0.019 | 0.467 | -0.024 | 566 | +0.06 | 1478 | -0.023 | -0.017 |
| `atm_iv_90d` | UW options | shadow | 1204 | -0.044 | 0.130 | -0.029 | 482 | -0.12 | 1265 | -0.054 | -0.039 |
| `ask_side_ratio` | aggressor | shadow | 854 | +0.016 | 0.645 | -0.035 | 342 | +0.05 | 895 | -0.050 | -0.121 |
| `unusual_premium_share` | feature-lab | shadow | 851 | +0.044 | 0.198 | -0.037 | 341 | +0.15 | 895 | +0.041 | +0.090 |
| `sweep_share` | flow-tracker (unused) | — | 1414 | -0.014 | 0.609 | -0.038 | 566 | -0.02 | 1478 | -0.018 | -0.033 |
| `far_otm_put_share` | feature-lab | shadow | 858 | +0.021 | 0.532 | -0.039 | 344 | +0.00 | 899 | -0.046 | -0.038 |
| `gap_pct` | price/vol (T1) | shadow | 487 | -0.042 | 0.356 | -0.040 | 195 | -0.04 | 540 | -0.009 | -0.056 |
| `ret_126d` | price/vol (T1) | shadow | 473 | -0.196 | 0.000 | -0.072 | 190 | -0.69 | 525 | -0.138 | -0.217 |
| `multileg_share` | flow-tracker (unused) | — | 1414 | -0.021 | 0.432 | -0.075 | 566 | -0.01 | 1478 | -0.013 | -0.020 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1414 | -0.041 | 0.126 | -0.082 | 566 | -0.24 | 1478 | +0.019 | -0.007 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 477 | -0.122 | 0.008 | -0.084 | 191 | -0.32 | 530 | -0.080 | -0.179 |
| `ret_63d` | price/vol (T1) | shadow | 478 | -0.128 | 0.005 | -0.088 | 192 | -0.28 | 531 | -0.075 | -0.189 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 478 | -0.134 | 0.003 | -0.088 | 192 | -0.35 | 531 | -0.078 | -0.191 |
| `rel_volume` | price/vol (T1) | shadow | 487 | -0.117 | 0.010 | -0.099 | 195 | -0.11 | 540 | -0.043 | +0.049 |
| `term_slope_30_90` | UW options | shadow | 1204 | -0.028 | 0.335 | -0.107 | 482 | -0.11 | 1265 | +0.017 | -0.022 |
| `px_vs_sma200` | price/vol (T1) | shadow | 469 | -0.227 | 0.000 | -0.161 | 188 | -0.78 | 520 | -0.149 | -0.289 |
| `dist_52w_high` | price/vol (T1) | shadow | 487 | -0.154 | 0.001 | -0.186 | 195 | -0.46 | 540 | -0.115 | -0.287 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `ret_126d` (price/vol (T1), shadow): Spearman -0.196, OOS -0.072, R-spread -0.69, n=473
- `rel_strength_sector_63d` (cross-sectional (T2), shadow): Spearman -0.122, OOS -0.084, R-spread -0.32, n=477
- `ret_63d` (price/vol (T1), shadow): Spearman -0.128, OOS -0.088, R-spread -0.28, n=478
- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.134, OOS -0.088, R-spread -0.35, n=478
- `rel_volume` (price/vol (T1), shadow): Spearman -0.117, OOS -0.099, R-spread -0.11, n=487
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.227, OOS -0.161, R-spread -0.78, n=469
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.154, OOS -0.186, R-spread -0.46, n=487

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

_No feature shows that split on current data._
