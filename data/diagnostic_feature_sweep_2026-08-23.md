# Unified feature sweep — 2026-08-23 22:22

Panel: **1211 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1123** matured rows carry `replay_realized_r`, **1195** carry `replay_forward_return_5d`.

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
| `dealer_net_gamma_at_spot` | UW options | shadow | 480 | +0.129 | 0.005 | +0.218 | 192 | +0.44 | 518 | +0.002 | +0.170 |
| `sector_relative_pct` | feature-lab | shadow | 588 | +0.132 | 0.001 | +0.163 | 236 | +0.45 | 630 | +0.069 | +0.135 |
| `gex_total` | UW options | shadow | 913 | +0.036 | 0.279 | +0.136 | 366 | +0.08 | 982 | +0.011 | +0.153 |
| `vrp_proxy` | feature-lab | shadow | 885 | +0.048 | 0.154 | +0.112 | 354 | +0.18 | 954 | +0.019 | +0.047 |
| `expiry_concentration_top1` | UW options | shadow | 911 | +0.041 | 0.218 | +0.111 | 365 | +0.15 | 980 | +0.065 | +0.132 |
| `window_return_pct` | flow-tracker (unused) | — | 1123 | +0.009 | 0.774 | +0.093 | 450 | +0.09 | 1195 | -0.044 | +0.117 |
| `prem_momentum_z3d` | feature-lab | shadow | 692 | +0.048 | 0.210 | +0.091 | 277 | +0.02 | 722 | -0.048 | -0.011 |
| `dealer_net_delta_at_spot` | UW options | shadow | 480 | +0.021 | 0.652 | +0.074 | 192 | +0.01 | 518 | +0.018 | +0.063 |
| `conviction_score` | composite | LIVE grade | 1123 | +0.059 | 0.048 | +0.073 | 450 | +0.22 | 1195 | +0.006 | +0.035 |
| `iv_skew_25d` | UW options | shadow | 153 | +0.056 | 0.491 | +0.072 | 62 | -0.01 | 158 | +0.132 | +0.030 |
| `charm_total` | UW options | shadow | 913 | +0.067 | 0.043 | +0.071 | 366 | +0.23 | 982 | +0.047 | +0.141 |
| `max_pain_dist_pct` | UW options | shadow | 928 | +0.009 | 0.774 | +0.057 | 372 | +0.10 | 997 | -0.037 | +0.170 |
| `momentum_score` | composite | shadow | 895 | +0.039 | 0.242 | +0.056 | 358 | +0.05 | 964 | -0.052 | -0.014 |
| `momentum_composite` | composite | shadow | 895 | +0.039 | 0.242 | +0.056 | 358 | +0.05 | 964 | -0.052 | -0.014 |
| `aggressor_bull_share` | aggressor | shadow | 658 | +0.005 | 0.897 | +0.053 | 264 | -0.04 | 697 | -0.002 | -0.060 |
| `directional_sweep_share` | aggressor | shadow | 664 | +0.025 | 0.517 | +0.051 | 266 | -0.01 | 703 | +0.014 | -0.000 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1123 | +0.008 | 0.789 | +0.039 | 450 | +0.00 | 1195 | -0.070 | -0.025 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1123 | +0.045 | 0.131 | +0.031 | 450 | +0.16 | 1195 | -0.003 | +0.035 |
| `vanna_total` | UW options | shadow | 913 | +0.017 | 0.615 | +0.030 | 366 | +0.06 | 982 | -0.060 | -0.056 |
| `accumulation_score` | flow-tracker (unused) | — | 1123 | +0.017 | 0.564 | +0.027 | 450 | +0.03 | 1195 | +0.026 | +0.052 |
| `persistence_ratio` | conviction_score component | LIVE | 1123 | +0.058 | 0.052 | +0.011 | 450 | +0.18 | 1195 | -0.012 | -0.035 |
| `bullish_premium_share` | feature-lab | shadow | 913 | +0.033 | 0.325 | +0.006 | 366 | +0.06 | 982 | -0.019 | -0.004 |
| `ask_side_ratio` | aggressor | shadow | 661 | +0.028 | 0.469 | +0.001 | 265 | +0.07 | 700 | -0.012 | -0.024 |
| `aggressor_net_prem_bps` | aggressor | shadow | 664 | -0.018 | 0.651 | +0.021 | 266 | -0.02 | 703 | +0.021 | -0.021 |
| `far_otm_call_share` | feature-lab | shadow | 665 | -0.047 | 0.224 | +0.017 | 266 | -0.25 | 704 | -0.021 | +0.060 |
| `gap_pct` | price/vol (T1) | shadow | 204 | -0.051 | 0.468 | +0.135 | 82 | +0.10 | 273 | +0.056 | +0.091 |
| `beta_63d` | cross-sectional (T2) | shadow | 204 | -0.112 | 0.109 | +0.007 | 82 | -0.35 | 273 | -0.166 | -0.023 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1123 | -0.009 | 0.753 | -0.004 | 450 | -0.07 | 1195 | -0.037 | -0.043 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 204 | +0.054 | 0.446 | -0.005 | 82 | +0.07 | 273 | +0.220 | +0.179 |
| `far_otm_put_share` | feature-lab | shadow | 665 | +0.012 | 0.752 | -0.007 | 266 | -0.05 | 704 | -0.051 | -0.035 |
| `ret_21d` | price/vol (T1) | shadow | 204 | +0.047 | 0.500 | -0.008 | 82 | +0.08 | 273 | +0.214 | +0.200 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1123 | -0.024 | 0.420 | -0.008 | 450 | -0.01 | 1195 | -0.047 | +0.001 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1123 | -0.024 | 0.420 | -0.008 | 450 | -0.01 | 1195 | -0.047 | +0.001 |
| `rsi_14` | price/vol (T1) | shadow | 204 | -0.010 | 0.889 | -0.009 | 82 | -0.10 | 273 | +0.187 | +0.182 |
| `multileg_share` | flow-tracker (unused) | — | 1123 | -0.005 | 0.872 | -0.018 | 450 | -0.00 | 1195 | -0.007 | -0.025 |
| `cumulative_premium` | conviction_score component | LIVE | 1123 | -0.024 | 0.419 | -0.027 | 450 | -0.14 | 1195 | -0.096 | -0.039 |
| `px_vs_sma50` | price/vol (T1) | shadow | 201 | -0.030 | 0.676 | -0.028 | 81 | -0.05 | 270 | +0.166 | +0.178 |
| `atm_iv_30d` | UW options | shadow | 913 | -0.065 | 0.048 | -0.029 | 366 | -0.22 | 982 | -0.048 | -0.036 |
| `latest_oi_change` | conviction_score component | LIVE | 1123 | -0.001 | 0.975 | -0.029 | 450 | -0.04 | 1195 | +0.028 | -0.003 |
| `bollinger_z` | price/vol (T1) | shadow | 204 | -0.007 | 0.925 | -0.040 | 82 | -0.05 | 273 | +0.205 | +0.154 |
| `ret_5d` | price/vol (T1) | shadow | 204 | +0.022 | 0.756 | -0.047 | 82 | -0.10 | 273 | +0.206 | +0.112 |
| `atm_iv_60d` | UW options | shadow | 913 | -0.065 | 0.050 | -0.048 | 366 | -0.24 | 982 | -0.046 | -0.046 |
| `dist_52w_high` | price/vol (T1) | shadow | 204 | -0.100 | 0.156 | -0.050 | 82 | -0.21 | 273 | -0.013 | -0.035 |
| `atm_iv_90d` | UW options | shadow | 913 | -0.067 | 0.045 | -0.053 | 366 | -0.23 | 982 | -0.049 | -0.050 |
| `sweep_share` | flow-tracker (unused) | — | 1123 | -0.026 | 0.390 | -0.056 | 450 | -0.03 | 1195 | -0.033 | -0.074 |
| `unusual_premium_share` | feature-lab | shadow | 635 | +0.055 | 0.167 | -0.057 | 254 | +0.21 | 678 | +0.008 | -0.040 |
| `atr_pct` | price/vol (T1) | shadow | 199 | -0.144 | 0.042 | -0.062 | 80 | -0.58 | 261 | -0.070 | +0.017 |
| `realized_vol_regime` | feature-lab | shadow | 887 | -0.039 | 0.242 | -0.082 | 355 | -0.13 | 956 | +0.019 | -0.015 |
| `ret_63d` | price/vol (T1) | shadow | 197 | -0.183 | 0.010 | -0.087 | 79 | -0.54 | 264 | +0.006 | -0.092 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 197 | -0.184 | 0.010 | -0.090 | 79 | -0.50 | 264 | +0.004 | -0.091 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1123 | -0.052 | 0.079 | -0.108 | 450 | -0.28 | 1195 | -0.009 | -0.057 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 665 | -0.016 | 0.685 | -0.110 | 266 | +0.03 | 704 | -0.009 | +0.096 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 196 | -0.184 | 0.010 | -0.122 | 79 | -0.61 | 263 | +0.003 | -0.066 |
| `rel_volume` | price/vol (T1) | shadow | 204 | -0.139 | 0.047 | -0.156 | 82 | -0.16 | 273 | -0.094 | -0.172 |
| `term_slope_30_90` | UW options | shadow | 913 | -0.011 | 0.742 | -0.162 | 366 | -0.02 | 982 | -0.003 | -0.085 |
| `px_vs_sma200` | price/vol (T1) | shadow | 197 | -0.242 | 0.001 | -0.182 | 79 | -0.77 | 264 | -0.096 | -0.108 |
| `ret_126d` | price/vol (T1) | shadow | 197 | -0.254 | 0.000 | -0.222 | 79 | -0.80 | 264 | -0.129 | -0.159 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `dealer_net_gamma_at_spot` (UW options, shadow): Spearman +0.129, OOS +0.218, R-spread +0.44, n=480
- `sector_relative_pct` (feature-lab, shadow): Spearman +0.132, OOS +0.163, R-spread +0.45, n=588
- `atr_pct` (price/vol (T1), shadow): Spearman -0.144, OOS -0.062, R-spread -0.58, n=199
- `ret_63d` (price/vol (T1), shadow): Spearman -0.183, OOS -0.087, R-spread -0.54, n=197
- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.184, OOS -0.090, R-spread -0.50, n=197
- `rel_strength_sector_63d` (cross-sectional (T2), shadow): Spearman -0.184, OOS -0.122, R-spread -0.61, n=196
- `rel_volume` (price/vol (T1), shadow): Spearman -0.139, OOS -0.156, R-spread -0.16, n=204
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.242, OOS -0.182, R-spread -0.77, n=197
- `ret_126d` (price/vol (T1), shadow): Spearman -0.254, OOS -0.222, R-spread -0.80, n=197

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

- `ret_21d` (price/vol (T1), shadow): forward IC +0.214 (n=273) vs realized-R IC +0.047 (n=204)
- `rsi_14` (price/vol (T1), shadow): forward IC +0.187 (n=273) vs realized-R IC -0.010 (n=204)
- `px_vs_sma50` (price/vol (T1), shadow): forward IC +0.166 (n=270) vs realized-R IC -0.030 (n=201)
- `bollinger_z` (price/vol (T1), shadow): forward IC +0.205 (n=273) vs realized-R IC -0.007 (n=204)
- `ret_5d` (price/vol (T1), shadow): forward IC +0.206 (n=273) vs realized-R IC +0.022 (n=204)
