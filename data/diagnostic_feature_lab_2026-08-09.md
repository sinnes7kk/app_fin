# Feature lab — Spearman ranking — 2026-08-09 22:32

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **956 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | 153 | -0.228 | -0.166 | 62 | 136 | -0.020 | -0.009 |
| `setup_momentum` | 141 | -0.218 | -0.384 | 57 | 124 | -0.226 | -0.033 |
| `rel_strength_spy_63d` | 153 | -0.197 | -0.106 | 62 | 136 | -0.052 | +0.058 |
| `ret_63d` | 153 | -0.196 | -0.105 | 62 | 136 | -0.047 | +0.076 |
| `ret_126d` | 153 | -0.191 | -0.147 | 62 | 136 | -0.025 | -0.004 |
| `setup_price_score` | 141 | -0.185 | -0.250 | 57 | 124 | -0.150 | -0.107 |
| `rel_strength_sector_63d` | 152 | -0.178 | -0.101 | 61 | 135 | -0.063 | +0.048 |
| `atr_pct` | 156 | -0.167 | -0.134 | 63 | 139 | -0.165 | -0.278 |
| `gap_pct` | 156 | -0.161 | -0.135 | 63 | 139 | -0.033 | +0.155 |
| `setup_state_rank` | 141 | -0.144 | -0.187 | 57 | 124 | -0.194 | -0.038 |
| `beta_63d` | 156 | -0.140 | -0.169 | 63 | 139 | -0.149 | -0.277 |
| `setup_pattern_pts` | 141 | -0.123 | -0.177 | 57 | 124 | -0.032 | -0.137 |
| `setup_is_reversal` | 141 | -0.119 | -0.138 | 57 | 124 | -0.033 | +0.183 |
| `sector_relative_pct` | 555 | +0.113 | +0.107 | 222 | 546 | +0.058 | +0.077 |
| `up_down_vol_ratio_10d` | 141 | -0.111 | +0.052 | 57 | 124 | +0.044 | +0.203 |
| `setup_is_breakout` | 141 | -0.105 | -0.111 | 57 | 124 | -0.153 | -0.210 |
| `setup_trend` | 141 | -0.093 | -0.231 | 57 | 124 | -0.067 | -0.223 |
| `unusual_premium_share` | 607 | +0.082 | +0.047 | 243 | 594 | +0.029 | +0.062 |
| `dist_52w_high` | 156 | -0.082 | -0.000 | 63 | 139 | +0.111 | +0.108 |
| `dealer_net_gamma_at_spot` | 467 | +0.079 | +0.120 | 187 | 454 | -0.081 | -0.028 |
| `atm_iv_30d` | 865 | -0.078 | -0.089 | 346 | 848 | -0.063 | -0.063 |
| `rsi_14` | 156 | -0.073 | -0.000 | 63 | 139 | +0.061 | +0.223 |
| `vol_trend_10d` | 141 | -0.073 | -0.026 | 57 | 124 | +0.070 | +0.175 |
| `atm_iv_60d` | 865 | -0.073 | -0.091 | 346 | 848 | -0.055 | -0.058 |
| `atm_iv_90d` | 865 | -0.072 | -0.090 | 346 | 848 | -0.054 | -0.058 |
| `bollinger_z` | 156 | -0.072 | -0.039 | 63 | 139 | +0.074 | +0.218 |
| `setup_ext_cap_atr` | 141 | -0.066 | -0.148 | 57 | 124 | +0.003 | -0.183 |
| `iv_skew_25d` | 153 | +0.063 | +0.073 | 62 | 142 | +0.161 | +0.164 |
| `charm_total` | 865 | +0.059 | +0.027 | 346 | 848 | +0.040 | +0.053 |
| `px_vs_sma50` | 153 | -0.057 | +0.029 | 62 | 136 | +0.055 | +0.239 |
| `realized_vol_regime` | 839 | -0.052 | -0.127 | 336 | 822 | +0.011 | -0.057 |
| `ask_side_ratio` | 629 | +0.049 | +0.079 | 252 | 622 | +0.020 | +0.078 |
| `prem_momentum_z3d` | 667 | +0.047 | +0.121 | 267 | 663 | -0.052 | -0.048 |
| `far_otm_put_share` | 633 | +0.047 | +0.060 | 254 | 626 | -0.050 | -0.014 |
| `setup_is_pullback` | 141 | +0.045 | +0.130 | 57 | 124 | -0.021 | -0.050 |
| `far_otm_call_share` | 633 | -0.041 | +0.039 | 254 | 626 | -0.033 | +0.001 |
| `expiry_concentration_top1` | 863 | +0.040 | +0.094 | 346 | 846 | +0.058 | +0.124 |
| `momentum_composite` | 847 | +0.039 | +0.089 | 339 | 830 | -0.049 | -0.022 |
| `momentum_score` | 847 | +0.039 | +0.089 | 339 | 830 | -0.049 | -0.022 |
| `ret_5d` | 156 | -0.037 | +0.016 | 63 | 139 | +0.177 | +0.278 |
| `resid_mom_21d` | 156 | +0.033 | +0.105 | 63 | 139 | +0.128 | +0.257 |
| `ret_21d` | 156 | +0.032 | +0.091 | 63 | 139 | +0.134 | +0.259 |
| `setup_is_trend_cont` | 141 | -0.031 | -0.134 | 57 | 124 | +0.113 | +0.021 |
| `vrp_proxy` | 837 | +0.030 | +0.069 | 335 | 820 | +0.026 | +0.058 |
| `rel_volume` | 156 | -0.030 | +0.014 | 63 | 139 | -0.002 | +0.095 |
| `setup_room` | 141 | -0.029 | +0.041 | 57 | 124 | -0.071 | -0.067 |
| `aggressor_bull_share` | 626 | +0.029 | +0.130 | 251 | 619 | -0.003 | -0.052 |
| `max_pain_dist_pct` | 880 | -0.027 | -0.020 | 352 | 863 | -0.091 | -0.050 |
| `dollar_delta_weighted_flow` | 633 | -0.023 | -0.104 | 254 | 626 | -0.018 | +0.029 |
| `setup_extended` | 141 | +0.023 | +0.049 | 57 | 124 | -0.013 | +0.016 |
| `rel_vol_3d_20d` | 141 | -0.020 | +0.027 | 57 | 124 | +0.063 | +0.187 |
| `setup_confirm_vol` | 141 | -0.019 | +0.007 | 57 | 124 | -0.027 | +0.134 |
| `sweep_share` | 880 | -0.017 | -0.036 | 352 | 863 | -0.025 | -0.100 |
| `aggressor_net_prem_bps` | 632 | -0.013 | +0.043 | 253 | 625 | +0.026 | +0.000 |
| `dealer_net_delta_at_spot` | 467 | -0.010 | -0.005 | 187 | 454 | -0.005 | +0.085 |
| `bullish_premium_share` | 865 | +0.010 | -0.056 | 346 | 848 | -0.078 | -0.207 |
| `setup_extension` | 141 | +0.008 | -0.008 | 57 | 124 | +0.074 | +0.061 |
| `directional_sweep_share` | 632 | +0.007 | -0.008 | 253 | 625 | +0.002 | -0.048 |
| `term_slope_30_90` | 865 | +0.005 | -0.089 | 346 | 848 | +0.021 | -0.045 |
| `multileg_share` | 880 | +0.005 | -0.004 | 352 | 863 | +0.002 | +0.003 |
| `rel_vol_5d_20d` | 141 | +0.005 | +0.034 | 57 | 124 | +0.069 | +0.262 |
| `gex_total` | 865 | -0.001 | +0.068 | 346 | 848 | -0.037 | +0.039 |
| `vanna_total` | 865 | -0.000 | +0.017 | 346 | 848 | -0.080 | -0.047 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | — | -0.34 (n=7) | -0.27 (n=98) | -0.17 (n=18) | -0.26 (n=29) |
| `setup_momentum` | — | -0.13 (n=7) | -0.25 (n=90) | +0.21 (n=15) | -0.37 (n=28) |
| `rel_strength_spy_63d` | — | -0.14 (n=7) | -0.23 (n=98) | -0.26 (n=18) | -0.26 (n=29) |
| `ret_63d` | — | -0.14 (n=7) | -0.23 (n=98) | -0.26 (n=18) | -0.28 (n=29) |
| `ret_126d` | — | -0.14 (n=7) | -0.24 (n=98) | -0.10 (n=18) | -0.19 (n=29) |
| `setup_price_score` | — | -0.14 (n=7) | -0.18 (n=90) | +0.15 (n=15) | -0.54 (n=28) |
| `rel_strength_sector_63d` | — | -0.07 (n=7) | -0.19 (n=98) | -0.28 (n=18) | -0.27 (n=28) |
| `atr_pct` | — | -0.02 (n=7) | -0.22 (n=101) | +0.21 (n=18) | -0.10 (n=29) |
| `gap_pct` | — | -0.43 (n=7) | -0.18 (n=101) | -0.37 (n=18) | -0.01 (n=29) |
| `setup_state_rank` | — | +0.18 (n=7) | -0.20 (n=90) | +0.24 (n=15) | -0.34 (n=28) |
| `beta_63d` | — | +0.41 (n=7) | -0.22 (n=101) | +0.34 (n=18) | -0.26 (n=29) |
| `setup_pattern_pts` | — | +0.16 (n=7) | -0.12 (n=90) | -0.06 (n=15) | -0.28 (n=28) |
| `setup_is_reversal` | — | -0.21 (n=7) | -0.17 (n=90) | — | +0.05 (n=28) |
| `sector_relative_pct` | — | +0.36 (n=14) | +0.12 (n=317) | +0.07 (n=146) | +0.11 (n=74) |
| `up_down_vol_ratio_10d` | — | -0.16 (n=7) | -0.06 (n=90) | -0.54 (n=15) | -0.22 (n=28) |
| `setup_is_breakout` | — | — | +0.03 (n=90) | — | -0.37 (n=28) |
| `setup_trend` | — | +0.35 (n=7) | -0.01 (n=90) | +0.23 (n=15) | -0.62 (n=28) |
| `unusual_premium_share` | — | +0.43 (n=7) | +0.09 (n=440) | +0.05 (n=157) | — |
| `dist_52w_high` | — | -0.45 (n=7) | -0.08 (n=101) | -0.43 (n=18) | -0.07 (n=29) |
| `dealer_net_gamma_at_spot` | — | -0.31 (n=9) | +0.07 (n=332) | +0.13 (n=113) | -0.20 (n=9) |
| `atm_iv_30d` | +0.34 (n=5) | -0.01 (n=31) | -0.08 (n=471) | -0.02 (n=215) | -0.07 (n=143) |
| `rsi_14` | — | -0.41 (n=7) | -0.01 (n=101) | -0.29 (n=18) | -0.36 (n=29) |
| `vol_trend_10d` | — | +0.09 (n=7) | -0.07 (n=90) | -0.54 (n=15) | -0.20 (n=28) |
| `atm_iv_60d` | +0.52 (n=5) | -0.09 (n=31) | -0.06 (n=471) | -0.01 (n=215) | -0.09 (n=143) |
| `atm_iv_90d` | +0.34 (n=5) | -0.12 (n=31) | -0.06 (n=471) | -0.01 (n=215) | -0.10 (n=143) |
| `bollinger_z` | — | -0.13 (n=7) | -0.00 (n=101) | -0.17 (n=18) | -0.33 (n=29) |
| `setup_ext_cap_atr` | — | +0.22 (n=7) | -0.09 (n=90) | +0.35 (n=15) | -0.46 (n=28) |
| `iv_skew_25d` | — | — | +0.11 (n=104) | +0.18 (n=37) | -0.30 (n=11) |
| `charm_total` | +0.45 (n=5) | -0.19 (n=31) | +0.06 (n=471) | +0.04 (n=215) | +0.02 (n=143) |
| `px_vs_sma50` | — | +0.04 (n=7) | -0.02 (n=98) | -0.41 (n=18) | -0.28 (n=29) |
| `realized_vol_regime` | +0.45 (n=5) | +0.18 (n=31) | -0.06 (n=452) | -0.08 (n=208) | -0.04 (n=143) |
| `ask_side_ratio` | -0.45 (n=5) | +0.17 (n=23) | +0.05 (n=354) | +0.00 (n=159) | +0.07 (n=88) |
| `prem_momentum_z3d` | — | +0.45 (n=23) | +0.09 (n=376) | -0.09 (n=172) | +0.02 (n=92) |
| `far_otm_put_share` | — | -0.05 (n=21) | +0.08 (n=371) | +0.11 (n=157) | -0.03 (n=80) |
| `setup_is_pullback` | — | — | +0.07 (n=90) | -0.28 (n=15) | +0.25 (n=28) |
| `far_otm_call_share` | — | -0.08 (n=21) | -0.05 (n=371) | -0.00 (n=157) | -0.13 (n=80) |
| `expiry_concentration_top1` | +0.67 (n=5) | +0.21 (n=31) | +0.04 (n=471) | +0.02 (n=214) | -0.06 (n=142) |
| `momentum_composite` | -0.89 (n=5) | -0.21 (n=31) | +0.10 (n=460) | -0.09 (n=218) | +0.09 (n=133) |
| `momentum_score` | -0.89 (n=5) | -0.21 (n=31) | +0.10 (n=460) | -0.09 (n=218) | +0.09 (n=133) |
| `ret_5d` | — | +0.04 (n=7) | +0.00 (n=101) | -0.18 (n=18) | -0.22 (n=29) |
| `resid_mom_21d` | — | +0.04 (n=7) | +0.11 (n=101) | -0.37 (n=18) | -0.29 (n=29) |
| `ret_21d` | — | +0.23 (n=7) | +0.10 (n=101) | -0.39 (n=18) | -0.28 (n=29) |
| `setup_is_trend_cont` | — | +0.22 (n=7) | -0.12 (n=90) | +0.28 (n=15) | -0.14 (n=28) |
| `vrp_proxy` | +0.11 (n=5) | +0.17 (n=31) | -0.02 (n=450) | +0.08 (n=208) | +0.12 (n=143) |
| `rel_volume` | — | +0.09 (n=7) | -0.06 (n=101) | -0.18 (n=18) | -0.12 (n=29) |
| `setup_room` | — | — | -0.06 (n=90) | +0.06 (n=15) | -0.11 (n=28) |
| `aggressor_bull_share` | -0.45 (n=5) | -0.30 (n=22) | +0.06 (n=354) | +0.03 (n=157) | +0.04 (n=88) |
| `max_pain_dist_pct` | -0.22 (n=5) | +0.14 (n=32) | -0.01 (n=475) | -0.03 (n=218) | -0.19 (n=150) |
| `dollar_delta_weighted_flow` | — | -0.06 (n=21) | -0.01 (n=371) | -0.05 (n=157) | +0.06 (n=80) |
| `setup_extended` | — | +0.00 (n=7) | +0.08 (n=90) | — | -0.12 (n=28) |
| `rel_vol_3d_20d` | — | +0.29 (n=7) | -0.02 (n=90) | -0.50 (n=15) | -0.16 (n=28) |
| `setup_confirm_vol` | — | +0.05 (n=7) | -0.02 (n=90) | -0.22 (n=15) | -0.18 (n=28) |
| `sweep_share` | -0.63 (n=5) | -0.16 (n=32) | +0.02 (n=475) | -0.01 (n=218) | — |
| `aggressor_net_prem_bps` | -0.11 (n=5) | -0.34 (n=23) | +0.07 (n=355) | -0.06 (n=160) | -0.18 (n=89) |
| `dealer_net_delta_at_spot` | — | +0.22 (n=9) | -0.02 (n=332) | +0.07 (n=113) | -0.11 (n=9) |
| `bullish_premium_share` | +0.11 (n=5) | +0.42 (n=31) | -0.08 (n=471) | +0.03 (n=215) | +0.06 (n=143) |
| `setup_extension` | — | +0.41 (n=7) | -0.06 (n=90) | +0.08 (n=15) | +0.25 (n=28) |
| `directional_sweep_share` | -0.34 (n=5) | -0.34 (n=23) | +0.02 (n=355) | +0.12 (n=160) | -0.12 (n=89) |
| `term_slope_30_90` | -0.34 (n=5) | -0.21 (n=31) | -0.01 (n=471) | +0.10 (n=215) | -0.04 (n=143) |
| `multileg_share` | +0.40 (n=5) | +0.18 (n=32) | -0.05 (n=475) | +0.10 (n=218) | — |
| `rel_vol_5d_20d` | — | +0.41 (n=7) | -0.05 (n=90) | -0.35 (n=15) | -0.07 (n=28) |
| `gex_total` | +0.34 (n=5) | +0.04 (n=31) | +0.00 (n=471) | +0.04 (n=215) | -0.08 (n=143) |
| `vanna_total` | -0.89 (n=5) | +0.42 (n=31) | +0.01 (n=471) | +0.05 (n=215) | -0.07 (n=143) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `sector_relative_pct` | 555 | +0.113 | +0.107 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `up_down_vol_ratio_10d` | 141 | -0.111 | +0.052 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
