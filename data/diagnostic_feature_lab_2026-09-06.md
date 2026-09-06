# Feature lab — Spearman ranking — 2026-09-06 23:33

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **1244 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | 364 | -0.187 | -0.116 | 146 | 345 | -0.093 | -0.107 |
| `setup_ext_cap_atr` | 359 | -0.182 | -0.179 | 144 | 339 | -0.189 | -0.167 |
| `setup_trend` | 359 | -0.170 | -0.042 | 144 | 339 | -0.154 | -0.004 |
| `setup_is_trend_cont` | 359 | -0.168 | -0.204 | 144 | 339 | -0.171 | -0.215 |
| `ret_126d` | 364 | -0.157 | -0.040 | 146 | 345 | -0.096 | -0.039 |
| `setup_price_score` | 359 | -0.156 | -0.032 | 144 | 339 | -0.152 | -0.026 |
| `rel_strength_spy_63d` | 365 | -0.155 | -0.200 | 146 | 345 | -0.022 | -0.144 |
| `ret_63d` | 365 | -0.154 | -0.197 | 146 | 345 | -0.018 | -0.138 |
| `rel_strength_sector_63d` | 364 | -0.146 | -0.197 | 146 | 344 | -0.032 | -0.145 |
| `dist_52w_high` | 374 | -0.123 | -0.157 | 150 | 354 | -0.035 | -0.147 |
| `setup_pattern_pts` | 359 | -0.114 | -0.096 | 144 | 339 | -0.101 | -0.105 |
| `dealer_net_gamma_at_spot` | 585 | +0.105 | +0.176 | 234 | 571 | -0.005 | +0.137 |
| `setup_room` | 359 | +0.105 | +0.122 | 144 | 339 | +0.120 | +0.091 |
| `setup_state_rank` | 359 | -0.098 | +0.011 | 144 | 339 | -0.135 | -0.034 |
| `sector_relative_pct` | 696 | +0.097 | +0.085 | 279 | 683 | +0.040 | +0.040 |
| `setup_momentum` | 359 | -0.097 | -0.048 | 144 | 339 | -0.130 | -0.088 |
| `rel_volume` | 374 | -0.084 | -0.049 | 150 | 354 | -0.086 | -0.087 |
| `atm_iv_90d` | 1083 | -0.071 | -0.111 | 434 | 1063 | -0.047 | -0.046 |
| `atr_pct` | 362 | -0.071 | +0.029 | 145 | 342 | -0.067 | -0.010 |
| `atm_iv_60d` | 1083 | -0.071 | -0.111 | 434 | 1063 | -0.045 | -0.043 |
| `setup_is_reversal` | 359 | +0.071 | +0.181 | 144 | 339 | +0.062 | +0.124 |
| `setup_confirm_vol` | 359 | -0.070 | -0.007 | 144 | 339 | -0.098 | -0.089 |
| `atm_iv_30d` | 1083 | -0.070 | -0.096 | 434 | 1063 | -0.048 | -0.030 |
| `beta_63d` | 374 | -0.068 | +0.048 | 150 | 354 | -0.143 | -0.042 |
| `unusual_premium_share` | 749 | +0.060 | -0.004 | 300 | 735 | +0.019 | +0.029 |
| `gap_pct` | 374 | -0.052 | -0.069 | 150 | 354 | +0.020 | -0.056 |
| `vrp_proxy` | 1055 | +0.047 | +0.112 | 422 | 1035 | +0.038 | +0.118 |
| `ret_21d` | 374 | +0.046 | -0.032 | 150 | 354 | +0.113 | -0.151 |
| `setup_is_pullback` | 359 | +0.041 | +0.047 | 144 | 339 | +0.116 | +0.169 |
| `setup_extension` | 359 | +0.040 | +0.050 | 144 | 339 | +0.051 | +0.146 |
| `iv_skew_25d` | 187 | +0.040 | +0.122 | 75 | 172 | +0.138 | +0.063 |
| `prem_momentum_z3d` | 766 | +0.035 | +0.049 | 307 | 756 | -0.067 | -0.062 |
| `resid_mom_21d` | 374 | +0.034 | -0.074 | 150 | 354 | +0.117 | -0.147 |
| `vanna_total` | 1083 | +0.033 | +0.073 | 434 | 1063 | -0.054 | +0.001 |
| `vol_trend_10d` | 359 | -0.032 | -0.027 | 144 | 339 | -0.092 | -0.177 |
| `charm_total` | 1083 | +0.031 | +0.002 | 434 | 1063 | +0.014 | +0.017 |
| `rel_vol_3d_20d` | 359 | -0.030 | -0.059 | 144 | 339 | -0.061 | -0.171 |
| `directional_sweep_share` | 772 | +0.029 | +0.066 | 309 | 759 | +0.012 | +0.040 |
| `far_otm_put_share` | 773 | +0.028 | -0.001 | 310 | 760 | -0.041 | -0.029 |
| `setup_is_breakout` | 359 | +0.026 | +0.046 | 144 | 339 | -0.028 | -0.071 |
| `gex_total` | 1083 | +0.023 | +0.085 | 434 | 1063 | -0.008 | +0.093 |
| `ret_5d` | 374 | -0.023 | -0.084 | 150 | 354 | +0.098 | -0.107 |
| `setup_extended` | 359 | +0.023 | +0.094 | 144 | 339 | +0.033 | -0.022 |
| `realized_vol_regime` | 1057 | -0.021 | +0.031 | 423 | 1037 | +0.022 | +0.022 |
| `px_vs_sma50` | 371 | -0.019 | -0.092 | 149 | 351 | +0.120 | -0.047 |
| `dealer_net_delta_at_spot` | 585 | +0.018 | +0.041 | 234 | 571 | +0.032 | +0.072 |
| `expiry_concentration_top1` | 1081 | +0.016 | +0.016 | 433 | 1061 | +0.057 | +0.075 |
| `far_otm_call_share` | 773 | -0.015 | +0.084 | 310 | 760 | -0.006 | +0.077 |
| `rel_vol_5d_20d` | 359 | +0.014 | -0.005 | 144 | 339 | -0.018 | -0.065 |
| `sweep_share` | 1098 | -0.014 | -0.070 | 440 | 1078 | -0.028 | -0.093 |
| `momentum_score` | 1065 | +0.013 | -0.000 | 426 | 1045 | -0.042 | +0.006 |
| `momentum_composite` | 1065 | +0.013 | -0.000 | 426 | 1045 | -0.042 | +0.006 |
| `aggressor_bull_share` | 766 | +0.012 | +0.051 | 307 | 753 | -0.010 | -0.049 |
| `multileg_share` | 1098 | -0.011 | -0.064 | 440 | 1078 | -0.012 | -0.016 |
| `bullish_premium_share` | 1083 | +0.009 | +0.004 | 434 | 1063 | -0.033 | +0.017 |
| `up_down_vol_ratio_10d` | 359 | -0.009 | -0.008 | 144 | 339 | +0.061 | -0.036 |
| `bollinger_z` | 374 | +0.008 | -0.014 | 150 | 354 | +0.107 | -0.080 |
| `aggressor_net_prem_bps` | 772 | -0.008 | +0.068 | 309 | 759 | +0.025 | +0.022 |
| `rsi_14` | 374 | -0.005 | -0.065 | 150 | 354 | +0.109 | -0.089 |
| `ask_side_ratio` | 769 | +0.004 | -0.068 | 308 | 756 | -0.034 | -0.106 |
| `dollar_delta_weighted_flow` | 773 | -0.002 | -0.044 | 310 | 760 | +0.019 | +0.155 |
| `max_pain_dist_pct` | 1097 | -0.001 | +0.064 | 439 | 1077 | -0.051 | +0.111 |
| `term_slope_30_90` | 1083 | +0.000 | -0.052 | 434 | 1063 | +0.005 | -0.066 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | — | -0.02 (n=26) | -0.25 (n=214) | +0.01 (n=62) | -0.12 (n=59) |
| `setup_ext_cap_atr` | — | +0.19 (n=27) | -0.24 (n=210) | -0.13 (n=61) | -0.33 (n=58) |
| `setup_trend` | — | +0.38 (n=27) | -0.18 (n=210) | -0.20 (n=61) | -0.31 (n=58) |
| `setup_is_trend_cont` | — | +0.13 (n=27) | -0.25 (n=210) | -0.16 (n=61) | -0.19 (n=58) |
| `ret_126d` | — | +0.04 (n=26) | -0.24 (n=214) | +0.09 (n=62) | -0.17 (n=59) |
| `setup_price_score` | — | +0.07 (n=27) | -0.22 (n=210) | +0.01 (n=61) | -0.27 (n=58) |
| `rel_strength_spy_63d` | — | -0.09 (n=27) | -0.20 (n=214) | +0.07 (n=62) | -0.09 (n=59) |
| `ret_63d` | — | -0.09 (n=27) | -0.20 (n=214) | +0.07 (n=62) | -0.08 (n=59) |
| `rel_strength_sector_63d` | — | -0.12 (n=27) | -0.17 (n=214) | +0.05 (n=62) | -0.12 (n=58) |
| `dist_52w_high` | — | -0.20 (n=27) | -0.11 (n=221) | -0.07 (n=64) | -0.11 (n=59) |
| `setup_pattern_pts` | — | -0.24 (n=27) | -0.13 (n=210) | -0.02 (n=61) | -0.16 (n=58) |
| `dealer_net_gamma_at_spot` | -0.31 (n=5) | -0.19 (n=18) | +0.11 (n=407) | +0.14 (n=141) | +0.22 (n=14) |
| `setup_room` | — | +0.18 (n=27) | +0.12 (n=210) | +0.20 (n=61) | -0.04 (n=58) |
| `setup_state_rank` | — | -0.08 (n=27) | -0.14 (n=210) | +0.12 (n=61) | -0.21 (n=58) |
| `sector_relative_pct` | +0.87 (n=5) | +0.11 (n=26) | +0.13 (n=394) | -0.01 (n=181) | +0.07 (n=90) |
| `setup_momentum` | — | -0.13 (n=27) | -0.18 (n=210) | +0.20 (n=61) | -0.11 (n=58) |
| `rel_volume` | — | +0.19 (n=27) | -0.15 (n=221) | -0.02 (n=64) | -0.16 (n=59) |
| `atm_iv_90d` | -0.42 (n=7) | +0.06 (n=51) | -0.10 (n=591) | -0.04 (n=261) | -0.08 (n=173) |
| `atr_pct` | — | +0.60 (n=26) | -0.19 (n=213) | -0.05 (n=63) | -0.01 (n=57) |
| `atm_iv_60d` | -0.29 (n=7) | +0.05 (n=51) | -0.09 (n=591) | -0.04 (n=261) | -0.07 (n=173) |
| `setup_is_reversal` | — | +0.14 (n=27) | +0.05 (n=210) | +0.07 (n=61) | +0.12 (n=58) |
| `setup_confirm_vol` | — | +0.07 (n=27) | -0.10 (n=210) | +0.01 (n=61) | -0.22 (n=58) |
| `atm_iv_30d` | -0.36 (n=7) | +0.11 (n=51) | -0.10 (n=591) | -0.06 (n=261) | -0.04 (n=173) |
| `beta_63d` | — | +0.55 (n=27) | -0.20 (n=221) | +0.00 (n=64) | -0.16 (n=59) |
| `unusual_premium_share` | — | +0.42 (n=11) | +0.04 (n=551) | +0.08 (n=183) | — |
| `gap_pct` | — | +0.17 (n=27) | -0.12 (n=221) | -0.02 (n=64) | +0.00 (n=59) |
| `vrp_proxy` | +0.09 (n=7) | +0.09 (n=51) | +0.01 (n=570) | +0.07 (n=254) | +0.10 (n=173) |
| `ret_21d` | — | -0.02 (n=27) | +0.16 (n=221) | -0.01 (n=64) | -0.14 (n=59) |
| `setup_is_pullback` | — | -0.34 (n=27) | +0.07 (n=210) | +0.15 (n=61) | +0.04 (n=58) |
| `setup_extension` | — | -0.37 (n=27) | +0.01 (n=210) | +0.14 (n=61) | +0.21 (n=58) |
| `iv_skew_25d` | — | — | +0.10 (n=125) | +0.16 (n=46) | -0.24 (n=13) |
| `prem_momentum_z3d` | +0.62 (n=5) | +0.37 (n=28) | +0.07 (n=441) | -0.10 (n=194) | +0.01 (n=98) |
| `resid_mom_21d` | — | -0.08 (n=27) | +0.17 (n=221) | -0.06 (n=64) | -0.14 (n=59) |
| `vanna_total` | -0.30 (n=7) | +0.15 (n=51) | +0.03 (n=591) | +0.07 (n=261) | -0.08 (n=173) |
| `vol_trend_10d` | — | +0.38 (n=27) | -0.09 (n=210) | -0.02 (n=61) | -0.09 (n=58) |
| `charm_total` | +0.21 (n=7) | -0.12 (n=51) | +0.05 (n=591) | +0.02 (n=261) | +0.03 (n=173) |
| `rel_vol_3d_20d` | — | +0.43 (n=27) | -0.07 (n=210) | -0.08 (n=61) | -0.13 (n=58) |
| `directional_sweep_share` | +0.27 (n=6) | -0.16 (n=34) | +0.05 (n=440) | +0.12 (n=190) | -0.10 (n=102) |
| `far_otm_put_share` | -0.45 (n=5) | +0.03 (n=32) | +0.04 (n=456) | +0.05 (n=187) | +0.05 (n=93) |
| `setup_is_breakout` | — | +0.02 (n=27) | +0.02 (n=210) | +0.12 (n=61) | +0.00 (n=58) |
| `gex_total` | +0.15 (n=7) | -0.08 (n=51) | +0.04 (n=591) | +0.04 (n=261) | -0.02 (n=173) |
| `ret_5d` | — | +0.04 (n=27) | +0.03 (n=221) | -0.03 (n=64) | -0.14 (n=59) |
| `setup_extended` | — | +0.10 (n=27) | +0.00 (n=210) | -0.01 (n=61) | +0.11 (n=58) |
| `realized_vol_regime` | +0.11 (n=7) | +0.23 (n=51) | -0.02 (n=572) | -0.06 (n=254) | -0.00 (n=173) |
| `px_vs_sma50` | — | +0.06 (n=27) | +0.07 (n=218) | -0.05 (n=64) | -0.19 (n=59) |
| `dealer_net_delta_at_spot` | -0.46 (n=5) | +0.01 (n=18) | +0.01 (n=407) | +0.13 (n=141) | +0.26 (n=14) |
| `expiry_concentration_top1` | +0.69 (n=7) | +0.03 (n=51) | +0.03 (n=591) | +0.01 (n=260) | -0.04 (n=172) |
| `far_otm_call_share` | -0.23 (n=5) | +0.24 (n=32) | -0.02 (n=456) | -0.01 (n=187) | -0.11 (n=93) |
| `rel_vol_5d_20d` | — | +0.49 (n=27) | -0.01 (n=210) | +0.01 (n=61) | -0.07 (n=58) |
| `sweep_share` | -0.29 (n=7) | -0.07 (n=52) | -0.01 (n=595) | +0.00 (n=264) | — |
| `momentum_score` | -0.22 (n=7) | -0.27 (n=51) | +0.08 (n=580) | -0.09 (n=264) | +0.04 (n=163) |
| `momentum_composite` | -0.22 (n=7) | -0.27 (n=51) | +0.08 (n=580) | -0.09 (n=264) | +0.04 (n=163) |
| `aggressor_bull_share` | -0.15 (n=6) | -0.15 (n=33) | +0.04 (n=439) | +0.03 (n=187) | -0.03 (n=101) |
| `multileg_share` | +0.70 (n=7) | +0.07 (n=52) | -0.07 (n=595) | +0.09 (n=264) | — |
| `bullish_premium_share` | -0.24 (n=7) | +0.41 (n=51) | -0.07 (n=591) | +0.03 (n=261) | +0.07 (n=173) |
| `up_down_vol_ratio_10d` | — | -0.00 (n=27) | +0.10 (n=210) | -0.11 (n=61) | -0.33 (n=58) |
| `bollinger_z` | — | -0.01 (n=27) | +0.10 (n=221) | +0.10 (n=64) | -0.27 (n=59) |
| `aggressor_net_prem_bps` | +0.03 (n=6) | -0.11 (n=34) | +0.06 (n=440) | -0.03 (n=190) | -0.20 (n=102) |
| `rsi_14` | — | -0.06 (n=27) | +0.12 (n=221) | +0.02 (n=64) | -0.28 (n=59) |
| `ask_side_ratio` | -0.46 (n=6) | +0.04 (n=34) | -0.01 (n=439) | -0.02 (n=189) | +0.06 (n=101) |
| `dollar_delta_weighted_flow` | +0.45 (n=5) | -0.01 (n=32) | +0.01 (n=456) | -0.01 (n=187) | +0.05 (n=93) |
| `max_pain_dist_pct` | -0.43 (n=7) | +0.13 (n=52) | +0.03 (n=595) | -0.02 (n=263) | -0.12 (n=180) |
| `term_slope_30_90` | -0.62 (n=7) | -0.14 (n=51) | -0.01 (n=591) | +0.11 (n=261) | -0.08 (n=173) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `dealer_net_gamma_at_spot` | 585 | +0.105 | +0.176 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `setup_room` | 359 | +0.105 | +0.122 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
