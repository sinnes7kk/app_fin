# Unified feature sweep — 2026-09-06 23:33

Panel: **1296 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1208** matured rows carry `replay_realized_r`, **1276** carry `replay_forward_return_5d`.

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
| `dealer_net_gamma_at_spot` | UW options | shadow | 535 | +0.113 | 0.009 | +0.193 | 214 | +0.39 | 571 | -0.005 | +0.137 |
| `vrp_proxy` | feature-lab | shadow | 970 | +0.071 | 0.026 | +0.169 | 388 | +0.30 | 1035 | +0.038 | +0.118 |
| `iv_skew_25d` | UW options | shadow | 168 | +0.064 | 0.406 | +0.138 | 68 | +0.10 | 172 | +0.138 | +0.063 |
| `gex_total` | UW options | shadow | 998 | +0.034 | 0.286 | +0.132 | 400 | +0.08 | 1063 | -0.008 | +0.093 |
| `sector_relative_pct` | feature-lab | shadow | 641 | +0.113 | 0.004 | +0.085 | 257 | +0.41 | 683 | +0.040 | +0.040 |
| `persistence_ratio` | conviction_score component | LIVE | 1208 | +0.070 | 0.015 | +0.081 | 484 | +0.21 | 1276 | +0.012 | +0.031 |
| `max_pain_dist_pct` | UW options | shadow | 1013 | +0.003 | 0.922 | +0.079 | 406 | +0.04 | 1077 | -0.051 | +0.111 |
| `vanna_total` | UW options | shadow | 998 | +0.038 | 0.235 | +0.077 | 400 | +0.10 | 1063 | -0.054 | +0.001 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1208 | +0.011 | 0.715 | +0.069 | 484 | -0.01 | 1276 | -0.036 | -0.028 |
| `prem_momentum_z3d` | feature-lab | shadow | 729 | +0.038 | 0.303 | +0.066 | 292 | -0.01 | 756 | -0.067 | -0.062 |
| `directional_sweep_share` | aggressor | shadow | 716 | +0.039 | 0.293 | +0.060 | 287 | +0.04 | 759 | +0.012 | +0.040 |
| `window_return_pct` | flow-tracker (unused) | — | 1208 | +0.008 | 0.770 | +0.054 | 484 | +0.10 | 1276 | -0.059 | +0.094 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1208 | +0.020 | 0.497 | +0.050 | 484 | +0.01 | 1276 | -0.049 | +0.027 |
| `dealer_net_delta_at_spot` | UW options | shadow | 535 | +0.027 | 0.532 | +0.048 | 214 | +0.06 | 571 | +0.032 | +0.072 |
| `expiry_concentration_top1` | UW options | shadow | 996 | +0.012 | 0.704 | +0.031 | 399 | +0.07 | 1061 | +0.057 | +0.075 |
| `conviction_score` | composite | LIVE grade | 1208 | +0.055 | 0.058 | +0.030 | 484 | +0.18 | 1276 | -0.006 | +0.014 |
| `far_otm_put_share` | feature-lab | shadow | 717 | +0.034 | 0.365 | +0.029 | 287 | +0.03 | 760 | -0.041 | -0.029 |
| `bullish_premium_share` | feature-lab | shadow | 998 | +0.025 | 0.428 | +0.025 | 400 | +0.02 | 1063 | -0.033 | +0.017 |
| `aggressor_bull_share` | aggressor | shadow | 710 | +0.010 | 0.780 | +0.025 | 284 | +0.01 | 753 | -0.010 | -0.049 |
| `momentum_score` | composite | shadow | 980 | +0.019 | 0.546 | +0.013 | 392 | +0.00 | 1045 | -0.042 | +0.006 |
| `momentum_composite` | composite | shadow | 980 | +0.019 | 0.546 | +0.013 | 392 | +0.00 | 1045 | -0.042 | +0.006 |
| `cumulative_premium` | conviction_score component | LIVE | 1208 | -0.013 | 0.648 | +0.011 | 484 | -0.11 | 1276 | -0.092 | -0.064 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1208 | -0.019 | 0.519 | +0.018 | 484 | -0.03 | 1276 | -0.058 | -0.046 |
| `aggressor_net_prem_bps` | aggressor | shadow | 716 | -0.007 | 0.850 | +0.029 | 287 | +0.04 | 759 | +0.025 | +0.022 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1208 | -0.019 | 0.519 | +0.018 | 484 | -0.03 | 1276 | -0.058 | -0.046 |
| `bollinger_z` | price/vol (T1) | shadow | 289 | -0.035 | 0.557 | +0.028 | 116 | -0.18 | 354 | +0.107 | -0.080 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 289 | -0.008 | 0.893 | +0.010 | 116 | -0.07 | 354 | +0.117 | -0.147 |
| `far_otm_call_share` | feature-lab | shadow | 717 | -0.021 | 0.575 | +0.068 | 287 | -0.09 | 760 | -0.006 | +0.077 |
| `ret_21d` | price/vol (T1) | shadow | 289 | -0.008 | 0.889 | +0.055 | 116 | -0.02 | 354 | +0.113 | -0.151 |
| `rsi_14` | price/vol (T1) | shadow | 289 | -0.046 | 0.434 | +0.029 | 116 | -0.20 | 354 | +0.109 | -0.089 |
| `beta_63d` | cross-sectional (T2) | shadow | 289 | -0.050 | 0.399 | +0.065 | 116 | -0.25 | 354 | -0.143 | -0.042 |
| `charm_total` | UW options | shadow | 998 | +0.030 | 0.344 | -0.002 | 400 | +0.09 | 1063 | +0.014 | +0.017 |
| `unusual_premium_share` | feature-lab | shadow | 691 | +0.071 | 0.061 | -0.004 | 277 | +0.25 | 735 | +0.019 | +0.029 |
| `atm_iv_30d` | UW options | shadow | 998 | -0.044 | 0.166 | -0.006 | 400 | -0.14 | 1063 | -0.048 | -0.030 |
| `sweep_share` | flow-tracker (unused) | — | 1208 | -0.014 | 0.618 | -0.007 | 484 | -0.02 | 1276 | -0.035 | -0.100 |
| `px_vs_sma50` | price/vol (T1) | shadow | 286 | -0.083 | 0.163 | -0.009 | 115 | -0.17 | 351 | +0.120 | -0.047 |
| `atr_pct` | price/vol (T1) | shadow | 278 | -0.046 | 0.439 | -0.011 | 112 | -0.15 | 342 | -0.067 | -0.010 |
| `accumulation_score` | flow-tracker (unused) | — | 1208 | +0.000 | 0.991 | -0.016 | 484 | -0.05 | 1276 | +0.013 | +0.043 |
| `ret_5d` | price/vol (T1) | shadow | 289 | -0.018 | 0.754 | -0.018 | 116 | -0.10 | 354 | +0.098 | -0.107 |
| `ask_side_ratio` | aggressor | shadow | 713 | +0.014 | 0.707 | -0.032 | 286 | +0.01 | 756 | -0.034 | -0.106 |
| `atm_iv_60d` | UW options | shadow | 998 | -0.046 | 0.142 | -0.039 | 400 | -0.17 | 1063 | -0.045 | -0.043 |
| `atm_iv_90d` | UW options | shadow | 998 | -0.048 | 0.129 | -0.045 | 400 | -0.16 | 1063 | -0.047 | -0.046 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1208 | +0.015 | 0.591 | -0.051 | 484 | +0.06 | 1276 | -0.020 | +0.011 |
| `realized_vol_regime` | feature-lab | shadow | 972 | -0.031 | 0.337 | -0.052 | 389 | -0.13 | 1037 | +0.022 | +0.022 |
| `multileg_share` | flow-tracker (unused) | — | 1208 | -0.014 | 0.629 | -0.052 | 484 | -0.01 | 1276 | -0.011 | -0.015 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 717 | +0.001 | 0.983 | -0.058 | 287 | +0.08 | 760 | +0.019 | +0.155 |
| `gap_pct` | price/vol (T1) | shadow | 289 | -0.068 | 0.252 | -0.061 | 116 | -0.03 | 354 | +0.020 | -0.056 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1208 | -0.039 | 0.180 | -0.062 | 484 | -0.24 | 1276 | +0.004 | -0.026 |
| `latest_oi_change` | conviction_score component | LIVE | 1208 | -0.015 | 0.601 | -0.072 | 484 | -0.12 | 1276 | +0.023 | -0.009 |
| `dist_52w_high` | price/vol (T1) | shadow | 289 | -0.187 | 0.002 | -0.180 | 116 | -0.47 | 354 | -0.035 | -0.147 |
| `term_slope_30_90` | UW options | shadow | 998 | -0.029 | 0.366 | -0.184 | 400 | -0.12 | 1063 | +0.005 | -0.066 |
| `ret_126d` | price/vol (T1) | shadow | 280 | -0.232 | 0.000 | -0.196 | 112 | -0.84 | 345 | -0.096 | -0.039 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 279 | -0.202 | 0.001 | -0.219 | 112 | -0.61 | 344 | -0.032 | -0.145 |
| `rel_volume` | price/vol (T1) | shadow | 289 | -0.161 | 0.006 | -0.233 | 116 | -0.26 | 354 | -0.086 | -0.087 |
| `ret_63d` | price/vol (T1) | shadow | 280 | -0.196 | 0.001 | -0.244 | 112 | -0.61 | 345 | -0.018 | -0.138 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 280 | -0.198 | 0.001 | -0.250 | 112 | -0.60 | 345 | -0.022 | -0.144 |
| `px_vs_sma200` | price/vol (T1) | shadow | 280 | -0.261 | 0.000 | -0.265 | 112 | -0.86 | 345 | -0.093 | -0.107 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `dealer_net_gamma_at_spot` (UW options, shadow): Spearman +0.113, OOS +0.193, R-spread +0.39, n=535
- `sector_relative_pct` (feature-lab, shadow): Spearman +0.113, OOS +0.085, R-spread +0.41, n=641
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.187, OOS -0.180, R-spread -0.47, n=289
- `ret_126d` (price/vol (T1), shadow): Spearman -0.232, OOS -0.196, R-spread -0.84, n=280
- `rel_strength_sector_63d` (cross-sectional (T2), shadow): Spearman -0.202, OOS -0.219, R-spread -0.61, n=279
- `rel_volume` (price/vol (T1), shadow): Spearman -0.161, OOS -0.233, R-spread -0.26, n=289
- `ret_63d` (price/vol (T1), shadow): Spearman -0.196, OOS -0.244, R-spread -0.61, n=280
- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.198, OOS -0.250, R-spread -0.60, n=280
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.261, OOS -0.265, R-spread -0.86, n=280

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

- `bollinger_z` (price/vol (T1), shadow): forward IC +0.107 (n=354) vs realized-R IC -0.035 (n=289)
- `resid_mom_21d` (cross-sectional (T2), shadow): forward IC +0.117 (n=354) vs realized-R IC -0.008 (n=289)
- `ret_21d` (price/vol (T1), shadow): forward IC +0.113 (n=354) vs realized-R IC -0.008 (n=289)
- `rsi_14` (price/vol (T1), shadow): forward IC +0.109 (n=354) vs realized-R IC -0.046 (n=289)
- `px_vs_sma50` (price/vol (T1), shadow): forward IC +0.120 (n=351) vs realized-R IC -0.083 (n=286)
