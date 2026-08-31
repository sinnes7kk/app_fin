# Unified feature sweep — 2026-08-31 00:14

Panel: **1283 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1204** matured rows carry `replay_realized_r`, **1267** carry `replay_forward_return_5d`.

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
| `dealer_net_gamma_at_spot` | UW options | shadow | 531 | +0.113 | 0.009 | +0.189 | 213 | +0.40 | 562 | -0.007 | +0.126 |
| `vrp_proxy` | feature-lab | shadow | 966 | +0.071 | 0.028 | +0.163 | 387 | +0.30 | 1026 | +0.037 | +0.115 |
| `iv_skew_25d` | UW options | shadow | 168 | +0.064 | 0.406 | +0.138 | 68 | +0.10 | 172 | +0.138 | +0.063 |
| `gex_total` | UW options | shadow | 994 | +0.034 | 0.290 | +0.136 | 398 | +0.08 | 1054 | -0.007 | +0.101 |
| `sector_relative_pct` | feature-lab | shadow | 638 | +0.111 | 0.005 | +0.083 | 256 | +0.41 | 676 | +0.040 | +0.043 |
| `max_pain_dist_pct` | UW options | shadow | 1009 | +0.002 | 0.957 | +0.079 | 404 | +0.03 | 1068 | -0.050 | +0.117 |
| `vanna_total` | UW options | shadow | 994 | +0.038 | 0.233 | +0.078 | 398 | +0.10 | 1054 | -0.055 | -0.009 |
| `prem_momentum_z3d` | feature-lab | shadow | 728 | +0.040 | 0.277 | +0.076 | 292 | -0.00 | 755 | -0.067 | -0.061 |
| `persistence_ratio` | conviction_score component | LIVE | 1204 | +0.067 | 0.021 | +0.073 | 482 | +0.20 | 1267 | +0.010 | +0.023 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1204 | +0.010 | 0.738 | +0.068 | 482 | -0.01 | 1267 | -0.037 | -0.027 |
| `directional_sweep_share` | aggressor | shadow | 714 | +0.038 | 0.314 | +0.060 | 286 | +0.03 | 754 | +0.013 | +0.041 |
| `window_return_pct` | flow-tracker (unused) | — | 1204 | +0.007 | 0.814 | +0.050 | 482 | +0.09 | 1267 | -0.059 | +0.087 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1204 | +0.019 | 0.505 | +0.048 | 482 | +0.01 | 1267 | -0.050 | +0.027 |
| `dealer_net_delta_at_spot` | UW options | shadow | 531 | +0.025 | 0.564 | +0.038 | 213 | +0.06 | 562 | +0.027 | +0.063 |
| `expiry_concentration_top1` | UW options | shadow | 992 | +0.014 | 0.664 | +0.036 | 397 | +0.08 | 1052 | +0.056 | +0.074 |
| `far_otm_put_share` | feature-lab | shadow | 715 | +0.034 | 0.370 | +0.033 | 286 | +0.02 | 755 | -0.042 | -0.034 |
| `bullish_premium_share` | feature-lab | shadow | 994 | +0.030 | 0.346 | +0.030 | 398 | +0.04 | 1054 | -0.030 | +0.024 |
| `conviction_score` | composite | LIVE grade | 1204 | +0.053 | 0.066 | +0.029 | 482 | +0.18 | 1267 | -0.007 | +0.008 |
| `aggressor_bull_share` | aggressor | shadow | 708 | +0.010 | 0.785 | +0.026 | 284 | +0.00 | 748 | -0.009 | -0.057 |
| `momentum_score` | composite | shadow | 976 | +0.022 | 0.492 | +0.021 | 391 | +0.00 | 1036 | -0.039 | +0.009 |
| `momentum_composite` | composite | shadow | 976 | +0.022 | 0.492 | +0.021 | 391 | +0.00 | 1036 | -0.039 | +0.009 |
| `charm_total` | UW options | shadow | 994 | +0.031 | 0.325 | +0.003 | 398 | +0.10 | 1054 | +0.015 | +0.031 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1204 | -0.024 | 0.410 | +0.009 | 482 | -0.05 | 1267 | -0.060 | -0.045 |
| `aggressor_net_prem_bps` | aggressor | shadow | 714 | -0.010 | 0.784 | +0.021 | 286 | +0.03 | 754 | +0.024 | +0.013 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1204 | -0.024 | 0.410 | +0.009 | 482 | -0.05 | 1267 | -0.060 | -0.045 |
| `bollinger_z` | price/vol (T1) | shadow | 285 | -0.034 | 0.567 | +0.006 | 114 | -0.15 | 345 | +0.116 | -0.067 |
| `ret_21d` | price/vol (T1) | shadow | 285 | -0.009 | 0.876 | +0.033 | 114 | -0.04 | 345 | +0.124 | -0.124 |
| `far_otm_call_share` | feature-lab | shadow | 715 | -0.022 | 0.566 | +0.076 | 286 | -0.09 | 755 | -0.006 | +0.081 |
| `cumulative_premium` | conviction_score component | LIVE | 1204 | -0.015 | 0.599 | +0.005 | 482 | -0.11 | 1267 | -0.094 | -0.064 |
| `rsi_14` | price/vol (T1) | shadow | 285 | -0.045 | 0.447 | +0.009 | 114 | -0.21 | 345 | +0.118 | -0.052 |
| `beta_63d` | cross-sectional (T2) | shadow | 285 | -0.064 | 0.278 | +0.023 | 114 | -0.26 | 345 | -0.159 | -0.087 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 285 | -0.007 | 0.905 | -0.001 | 114 | -0.06 | 345 | +0.129 | -0.111 |
| `accumulation_score` | flow-tracker (unused) | — | 1204 | +0.004 | 0.902 | -0.005 | 482 | -0.04 | 1267 | +0.014 | +0.035 |
| `unusual_premium_share` | feature-lab | shadow | 688 | +0.067 | 0.077 | -0.006 | 276 | +0.23 | 731 | +0.018 | +0.031 |
| `sweep_share` | flow-tracker (unused) | — | 1204 | -0.015 | 0.599 | -0.014 | 482 | -0.02 | 1267 | -0.036 | -0.095 |
| `atm_iv_30d` | UW options | shadow | 994 | -0.049 | 0.125 | -0.023 | 398 | -0.16 | 1054 | -0.049 | -0.035 |
| `ask_side_ratio` | aggressor | shadow | 711 | +0.016 | 0.679 | -0.026 | 285 | +0.01 | 751 | -0.031 | -0.092 |
| `multileg_share` | flow-tracker (unused) | — | 1204 | -0.012 | 0.668 | -0.040 | 482 | -0.00 | 1267 | -0.011 | -0.020 |
| `atr_pct` | price/vol (T1) | shadow | 274 | -0.060 | 0.321 | -0.041 | 110 | -0.16 | 333 | -0.074 | -0.046 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1204 | +0.019 | 0.514 | -0.042 | 482 | +0.07 | 1267 | -0.019 | +0.009 |
| `px_vs_sma50` | price/vol (T1) | shadow | 282 | -0.086 | 0.150 | -0.047 | 113 | -0.21 | 342 | +0.127 | -0.020 |
| `ret_5d` | price/vol (T1) | shadow | 285 | -0.019 | 0.752 | -0.047 | 114 | -0.06 | 345 | +0.108 | -0.087 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 715 | +0.001 | 0.989 | -0.054 | 286 | +0.08 | 755 | +0.020 | +0.151 |
| `atm_iv_60d` | UW options | shadow | 994 | -0.051 | 0.108 | -0.055 | 398 | -0.18 | 1054 | -0.046 | -0.047 |
| `realized_vol_regime` | feature-lab | shadow | 968 | -0.033 | 0.301 | -0.057 | 388 | -0.11 | 1028 | +0.022 | +0.018 |
| `atm_iv_90d` | UW options | shadow | 994 | -0.053 | 0.097 | -0.062 | 398 | -0.17 | 1054 | -0.048 | -0.052 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1204 | -0.039 | 0.174 | -0.067 | 482 | -0.24 | 1267 | +0.003 | -0.026 |
| `gap_pct` | price/vol (T1) | shadow | 285 | -0.068 | 0.250 | -0.071 | 114 | +0.00 | 345 | +0.015 | -0.107 |
| `latest_oi_change` | conviction_score component | LIVE | 1204 | -0.018 | 0.538 | -0.078 | 482 | -0.13 | 1267 | +0.022 | -0.015 |
| `term_slope_30_90` | UW options | shadow | 994 | -0.026 | 0.422 | -0.182 | 398 | -0.11 | 1054 | +0.006 | -0.066 |
| `dist_52w_high` | price/vol (T1) | shadow | 285 | -0.172 | 0.004 | -0.192 | 114 | -0.43 | 345 | -0.029 | -0.129 |
| `rel_volume` | price/vol (T1) | shadow | 285 | -0.168 | 0.005 | -0.238 | 114 | -0.30 | 345 | -0.094 | -0.150 |
| `ret_126d` | price/vol (T1) | shadow | 276 | -0.238 | 0.000 | -0.246 | 111 | -0.83 | 336 | -0.094 | -0.067 |
| `ret_63d` | price/vol (T1) | shadow | 276 | -0.195 | 0.001 | -0.273 | 111 | -0.58 | 336 | -0.016 | -0.154 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 275 | -0.205 | 0.001 | -0.276 | 110 | -0.58 | 335 | -0.032 | -0.161 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 276 | -0.196 | 0.001 | -0.279 | 111 | -0.57 | 336 | -0.020 | -0.159 |
| `px_vs_sma200` | price/vol (T1) | shadow | 276 | -0.266 | 0.000 | -0.306 | 111 | -0.88 | 336 | -0.091 | -0.124 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `dealer_net_gamma_at_spot` (UW options, shadow): Spearman +0.113, OOS +0.189, R-spread +0.40, n=531
- `sector_relative_pct` (feature-lab, shadow): Spearman +0.111, OOS +0.083, R-spread +0.41, n=638
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.172, OOS -0.192, R-spread -0.43, n=285
- `rel_volume` (price/vol (T1), shadow): Spearman -0.168, OOS -0.238, R-spread -0.30, n=285
- `ret_126d` (price/vol (T1), shadow): Spearman -0.238, OOS -0.246, R-spread -0.83, n=276
- `ret_63d` (price/vol (T1), shadow): Spearman -0.195, OOS -0.273, R-spread -0.58, n=276
- `rel_strength_sector_63d` (cross-sectional (T2), shadow): Spearman -0.205, OOS -0.276, R-spread -0.58, n=275
- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.196, OOS -0.279, R-spread -0.57, n=276
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.266, OOS -0.306, R-spread -0.88, n=276

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

- `bollinger_z` (price/vol (T1), shadow): forward IC +0.116 (n=345) vs realized-R IC -0.034 (n=285)
- `ret_21d` (price/vol (T1), shadow): forward IC +0.124 (n=345) vs realized-R IC -0.009 (n=285)
- `rsi_14` (price/vol (T1), shadow): forward IC +0.118 (n=345) vs realized-R IC -0.045 (n=285)
- `resid_mom_21d` (cross-sectional (T2), shadow): forward IC +0.129 (n=345) vs realized-R IC -0.007 (n=285)
- `px_vs_sma50` (price/vol (T1), shadow): forward IC +0.127 (n=342) vs realized-R IC -0.086 (n=282)
- `ret_5d` (price/vol (T1), shadow): forward IC +0.108 (n=345) vs realized-R IC -0.019 (n=285)
