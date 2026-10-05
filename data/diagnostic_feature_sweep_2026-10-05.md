# Unified feature sweep — 2026-10-05 00:29

Panel: **1478 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1466** matured rows carry `replay_realized_r`, **1478** carry `replay_forward_return_5d`.

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
| `iv_skew_25d` | UW options | shadow | 210 | +0.085 | 0.221 | +0.150 | 84 | +0.28 | 211 | +0.158 | +0.140 |
| `vrp_proxy` | feature-lab | shadow | 1214 | +0.049 | 0.089 | +0.121 | 486 | +0.23 | 1222 | +0.032 | +0.124 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 892 | +0.022 | 0.517 | +0.079 | 357 | +0.09 | 899 | +0.012 | +0.122 |
| `window_return_pct` | flow-tracker (unused) | — | 1466 | +0.005 | 0.863 | +0.062 | 587 | +0.10 | 1478 | -0.064 | +0.067 |
| `max_pain_dist_pct` | UW options | shadow | 1270 | +0.002 | 0.952 | +0.051 | 508 | +0.08 | 1279 | -0.062 | +0.028 |
| `directional_sweep_share` | aggressor | shadow | 891 | +0.028 | 0.399 | +0.047 | 357 | +0.01 | 898 | +0.004 | +0.030 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1466 | +0.016 | 0.538 | +0.045 | 587 | +0.01 | 1478 | -0.045 | +0.017 |
| `aggressor_bull_share` | aggressor | shadow | 885 | +0.024 | 0.477 | +0.032 | 354 | +0.01 | 892 | -0.001 | +0.005 |
| `gex_total` | UW options | shadow | 1256 | +0.010 | 0.716 | +0.031 | 503 | +0.04 | 1265 | -0.026 | +0.025 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1466 | +0.005 | 0.839 | +0.028 | 587 | +0.09 | 1478 | -0.041 | -0.032 |
| `prem_momentum_z3d` | feature-lab | shadow | 859 | +0.021 | 0.530 | +0.021 | 344 | -0.04 | 866 | -0.076 | -0.112 |
| `vanna_total` | UW options | shadow | 1256 | +0.019 | 0.491 | +0.019 | 503 | +0.07 | 1265 | -0.047 | -0.000 |
| `momentum_composite` | composite | shadow | 1237 | +0.020 | 0.473 | +0.018 | 495 | +0.01 | 1246 | -0.027 | +0.004 |
| `momentum_score` | composite | shadow | 1237 | +0.020 | 0.473 | +0.018 | 495 | +0.01 | 1246 | -0.027 | +0.004 |
| `persistence_ratio` | conviction_score component | LIVE | 1466 | +0.045 | 0.087 | +0.017 | 587 | +0.13 | 1478 | +0.004 | -0.018 |
| `accumulation_score` | flow-tracker (unused) | — | 1466 | +0.002 | 0.948 | +0.014 | 587 | -0.04 | 1478 | +0.015 | +0.064 |
| `conviction_score` | composite | LIVE grade | 1466 | +0.032 | 0.220 | +0.013 | 587 | +0.14 | 1478 | -0.009 | +0.021 |
| `latest_oi_change` | conviction_score component | LIVE | 1466 | +0.007 | 0.785 | +0.011 | 587 | -0.02 | 1478 | +0.028 | +0.004 |
| `charm_total` | UW options | shadow | 1256 | +0.032 | 0.254 | +0.005 | 503 | +0.10 | 1265 | +0.014 | -0.018 |
| `unusual_premium_share` | feature-lab | shadow | 890 | +0.054 | 0.107 | +0.001 | 356 | +0.14 | 895 | +0.042 | +0.089 |
| `aggressor_net_prem_bps` | aggressor | shadow | 891 | -0.001 | 0.980 | +0.039 | 357 | +0.00 | 898 | +0.021 | +0.011 |
| `realized_vol_regime` | feature-lab | shadow | 1216 | -0.017 | 0.560 | +0.043 | 487 | -0.08 | 1224 | +0.014 | +0.033 |
| `atr_pct` | price/vol (T1) | shadow | 507 | -0.013 | 0.778 | +0.125 | 203 | -0.08 | 513 | -0.068 | -0.058 |
| `px_vs_sma50` | price/vol (T1) | shadow | 530 | -0.032 | 0.465 | +0.004 | 212 | -0.11 | 537 | +0.031 | -0.138 |
| `far_otm_call_share` | feature-lab | shadow | 892 | -0.024 | 0.478 | +0.029 | 357 | -0.10 | 899 | -0.022 | +0.017 |
| `atm_iv_30d` | UW options | shadow | 1256 | -0.048 | 0.092 | +0.001 | 503 | -0.11 | 1265 | -0.057 | -0.031 |
| `beta_63d` | cross-sectional (T2) | shadow | 533 | -0.055 | 0.202 | +0.024 | 214 | -0.22 | 540 | -0.126 | -0.064 |
| `rsi_14` | price/vol (T1) | shadow | 533 | -0.023 | 0.589 | -0.000 | 214 | -0.05 | 540 | +0.018 | -0.146 |
| `dealer_net_delta_at_spot` | UW options | shadow | 758 | +0.000 | 0.999 | -0.002 | 304 | -0.02 | 766 | -0.004 | -0.012 |
| `expiry_concentration_top1` | UW options | shadow | 1254 | +0.009 | 0.749 | -0.003 | 502 | +0.05 | 1263 | +0.053 | +0.040 |
| `ask_side_ratio` | aggressor | shadow | 888 | +0.013 | 0.695 | -0.008 | 356 | +0.03 | 895 | -0.047 | -0.115 |
| `sector_relative_pct` | feature-lab | shadow | 798 | +0.071 | 0.046 | -0.009 | 320 | +0.28 | 805 | +0.033 | -0.005 |
| `bollinger_z` | price/vol (T1) | shadow | 533 | -0.023 | 0.590 | -0.013 | 214 | -0.06 | 540 | +0.038 | -0.086 |
| `sweep_share` | flow-tracker (unused) | — | 1466 | -0.009 | 0.728 | -0.014 | 587 | -0.01 | 1478 | -0.019 | -0.035 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1466 | +0.023 | 0.370 | -0.017 | 587 | +0.06 | 1478 | -0.016 | -0.001 |
| `far_otm_put_share` | feature-lab | shadow | 892 | +0.026 | 0.441 | -0.017 | 357 | +0.01 | 899 | -0.049 | -0.043 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1466 | -0.024 | 0.355 | -0.017 | 587 | -0.02 | 1478 | -0.056 | -0.059 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1466 | -0.024 | 0.355 | -0.017 | 587 | -0.02 | 1478 | -0.056 | -0.059 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 523 | -0.094 | 0.032 | -0.019 | 210 | -0.23 | 530 | -0.077 | -0.166 |
| `atm_iv_60d` | UW options | shadow | 1256 | -0.051 | 0.069 | -0.021 | 503 | -0.15 | 1265 | -0.056 | -0.045 |
| `atm_iv_90d` | UW options | shadow | 1256 | -0.052 | 0.063 | -0.028 | 503 | -0.18 | 1265 | -0.056 | -0.047 |
| `cumulative_premium` | conviction_score component | LIVE | 1466 | -0.030 | 0.252 | -0.034 | 587 | -0.14 | 1478 | -0.096 | -0.081 |
| `ret_5d` | price/vol (T1) | shadow | 533 | -0.039 | 0.374 | -0.040 | 214 | -0.10 | 540 | +0.035 | -0.084 |
| `rel_volume` | price/vol (T1) | shadow | 533 | -0.100 | 0.021 | -0.044 | 214 | -0.11 | 540 | -0.023 | +0.095 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 524 | -0.110 | 0.012 | -0.045 | 210 | -0.25 | 531 | -0.075 | -0.181 |
| `ret_21d` | price/vol (T1) | shadow | 533 | -0.020 | 0.650 | -0.046 | 214 | +0.03 | 540 | +0.023 | -0.153 |
| `ret_63d` | price/vol (T1) | shadow | 524 | -0.107 | 0.014 | -0.050 | 210 | -0.21 | 531 | -0.073 | -0.180 |
| `bullish_premium_share` | feature-lab | shadow | 1256 | -0.006 | 0.830 | -0.050 | 503 | -0.02 | 1265 | -0.063 | -0.040 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 758 | +0.048 | 0.190 | -0.052 | 304 | +0.19 | 766 | -0.032 | -0.039 |
| `term_slope_30_90` | UW options | shadow | 1256 | -0.007 | 0.810 | -0.052 | 503 | -0.02 | 1265 | +0.017 | -0.014 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1466 | -0.032 | 0.214 | -0.053 | 587 | -0.22 | 1478 | +0.020 | -0.000 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 533 | -0.024 | 0.586 | -0.062 | 214 | -0.01 | 540 | +0.012 | -0.194 |
| `gap_pct` | price/vol (T1) | shadow | 533 | -0.060 | 0.164 | -0.069 | 214 | -0.11 | 540 | -0.015 | -0.069 |
| `ret_126d` | price/vol (T1) | shadow | 518 | -0.177 | 0.000 | -0.087 | 208 | -0.60 | 525 | -0.147 | -0.242 |
| `multileg_share` | flow-tracker (unused) | — | 1466 | -0.023 | 0.383 | -0.092 | 587 | -0.01 | 1478 | -0.010 | -0.009 |
| `px_vs_sma200` | price/vol (T1) | shadow | 513 | -0.213 | 0.000 | -0.169 | 206 | -0.73 | 520 | -0.164 | -0.331 |
| `dist_52w_high` | price/vol (T1) | shadow | 533 | -0.160 | 0.000 | -0.179 | 214 | -0.50 | 540 | -0.115 | -0.284 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.110, OOS -0.045, R-spread -0.25, n=524
- `ret_63d` (price/vol (T1), shadow): Spearman -0.107, OOS -0.050, R-spread -0.21, n=524
- `ret_126d` (price/vol (T1), shadow): Spearman -0.177, OOS -0.087, R-spread -0.60, n=518
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.213, OOS -0.169, R-spread -0.73, n=513
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.160, OOS -0.179, R-spread -0.50, n=533

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

_No feature shows that split on current data._
