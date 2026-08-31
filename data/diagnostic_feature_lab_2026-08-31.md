# Feature lab — Spearman ranking — 2026-08-31 00:14

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **1165 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | 351 | -0.194 | -0.156 | 141 | 336 | -0.091 | -0.124 |
| `setup_ext_cap_atr` | 346 | -0.181 | -0.171 | 139 | 330 | -0.182 | -0.162 |
| `setup_trend` | 346 | -0.173 | -0.023 | 139 | 330 | -0.151 | -0.008 |
| `setup_is_trend_cont` | 346 | -0.169 | -0.181 | 139 | 330 | -0.170 | -0.222 |
| `ret_126d` | 351 | -0.164 | -0.091 | 141 | 336 | -0.094 | -0.067 |
| `rel_strength_spy_63d` | 352 | -0.156 | -0.197 | 141 | 336 | -0.020 | -0.159 |
| `ret_63d` | 352 | -0.155 | -0.195 | 141 | 336 | -0.016 | -0.154 |
| `setup_price_score` | 346 | -0.154 | -0.008 | 139 | 330 | -0.147 | -0.045 |
| `rel_strength_sector_63d` | 351 | -0.150 | -0.191 | 141 | 335 | -0.032 | -0.161 |
| `dist_52w_high` | 361 | -0.112 | -0.150 | 145 | 345 | -0.029 | -0.129 |
| `setup_pattern_pts` | 346 | -0.110 | -0.077 | 139 | 330 | -0.089 | -0.102 |
| `setup_room` | 346 | +0.108 | +0.142 | 139 | 330 | +0.122 | +0.088 |
| `dealer_net_gamma_at_spot` | 572 | +0.106 | +0.209 | 229 | 562 | -0.007 | +0.126 |
| `sector_relative_pct` | 685 | +0.097 | +0.080 | 274 | 676 | +0.040 | +0.043 |
| `setup_state_rank` | 346 | -0.095 | +0.039 | 139 | 330 | -0.129 | -0.054 |
| `setup_momentum` | 346 | -0.092 | -0.033 | 139 | 330 | -0.129 | -0.097 |
| `rel_volume` | 361 | -0.089 | -0.109 | 145 | 345 | -0.094 | -0.150 |
| `atr_pct` | 349 | -0.085 | -0.011 | 140 | 333 | -0.074 | -0.046 |
| `beta_63d` | 361 | -0.085 | +0.000 | 145 | 345 | -0.159 | -0.087 |
| `atm_iv_90d` | 1070 | -0.076 | -0.119 | 428 | 1054 | -0.048 | -0.052 |
| `atm_iv_60d` | 1070 | -0.075 | -0.118 | 428 | 1054 | -0.046 | -0.047 |
| `atm_iv_30d` | 1070 | -0.075 | -0.106 | 428 | 1054 | -0.049 | -0.035 |
| `setup_confirm_vol` | 346 | -0.075 | -0.035 | 139 | 330 | -0.107 | -0.142 |
| `unusual_premium_share` | 744 | +0.057 | -0.008 | 298 | 731 | +0.018 | +0.031 |
| `setup_is_reversal` | 346 | +0.056 | +0.148 | 139 | 330 | +0.055 | +0.103 |
| `gap_pct` | 361 | -0.051 | -0.066 | 145 | 345 | +0.015 | -0.107 |
| `ret_21d` | 361 | +0.047 | +0.005 | 145 | 345 | +0.124 | -0.124 |
| `vrp_proxy` | 1042 | +0.046 | +0.107 | 417 | 1026 | +0.037 | +0.115 |
| `setup_extension` | 346 | +0.046 | +0.081 | 139 | 330 | +0.056 | +0.148 |
| `rel_vol_3d_20d` | 346 | -0.043 | -0.106 | 139 | 330 | -0.066 | -0.186 |
| `setup_is_pullback` | 346 | +0.043 | +0.036 | 139 | 330 | +0.120 | +0.170 |
| `iv_skew_25d` | 187 | +0.040 | +0.122 | 75 | 172 | +0.138 | +0.063 |
| `setup_is_breakout` | 346 | +0.040 | +0.062 | 139 | 330 | -0.022 | -0.071 |
| `vol_trend_10d` | 346 | -0.040 | -0.044 | 139 | 330 | -0.094 | -0.178 |
| `resid_mom_21d` | 361 | +0.037 | -0.024 | 145 | 345 | +0.129 | -0.111 |
| `prem_momentum_z3d` | 764 | +0.037 | +0.058 | 306 | 755 | -0.067 | -0.061 |
| `vanna_total` | 1070 | +0.034 | +0.080 | 428 | 1054 | -0.055 | -0.009 |
| `charm_total` | 1070 | +0.032 | +0.003 | 428 | 1054 | +0.015 | +0.031 |
| `far_otm_put_share` | 767 | +0.028 | -0.008 | 307 | 755 | -0.042 | -0.034 |
| `directional_sweep_share` | 766 | +0.027 | +0.057 | 307 | 754 | +0.013 | +0.041 |
| `realized_vol_regime` | 1044 | -0.023 | +0.022 | 418 | 1028 | +0.022 | +0.018 |
| `gex_total` | 1070 | +0.023 | +0.091 | 428 | 1054 | -0.007 | +0.101 |
| `ret_5d` | 361 | -0.022 | -0.055 | 145 | 345 | +0.108 | -0.087 |
| `setup_extended` | 346 | +0.021 | +0.108 | 139 | 330 | +0.030 | -0.039 |
| `px_vs_sma50` | 358 | -0.021 | -0.059 | 144 | 342 | +0.127 | -0.020 |
| `expiry_concentration_top1` | 1068 | +0.018 | +0.020 | 428 | 1052 | +0.056 | +0.074 |
| `dealer_net_delta_at_spot` | 572 | +0.017 | +0.030 | 229 | 562 | +0.027 | +0.063 |
| `far_otm_call_share` | 767 | -0.016 | +0.078 | 307 | 755 | -0.006 | +0.081 |
| `momentum_score` | 1052 | +0.015 | -0.007 | 421 | 1036 | -0.039 | +0.009 |
| `momentum_composite` | 1052 | +0.015 | -0.007 | 421 | 1036 | -0.039 | +0.009 |
| `sweep_share` | 1085 | -0.015 | -0.072 | 434 | 1069 | -0.029 | -0.097 |
| `bullish_premium_share` | 1070 | +0.014 | +0.018 | 428 | 1054 | -0.030 | +0.024 |
| `aggressor_net_prem_bps` | 766 | -0.011 | +0.050 | 307 | 754 | +0.024 | +0.013 |
| `bollinger_z` | 361 | +0.011 | +0.023 | 145 | 345 | +0.116 | -0.067 |
| `aggressor_bull_share` | 760 | +0.011 | +0.046 | 304 | 748 | -0.009 | -0.057 |
| `multileg_share` | 1085 | -0.009 | -0.058 | 434 | 1069 | -0.012 | -0.019 |
| `rel_vol_5d_20d` | 346 | +0.007 | -0.030 | 139 | 330 | -0.020 | -0.073 |
| `up_down_vol_ratio_10d` | 346 | -0.007 | +0.033 | 139 | 330 | +0.068 | -0.010 |
| `ask_side_ratio` | 763 | +0.005 | -0.062 | 306 | 751 | -0.031 | -0.092 |
| `term_slope_30_90` | 1070 | +0.003 | -0.043 | 428 | 1054 | +0.006 | -0.066 |
| `rsi_14` | 361 | -0.003 | -0.013 | 145 | 345 | +0.118 | -0.052 |
| `dollar_delta_weighted_flow` | 767 | -0.003 | -0.056 | 307 | 755 | +0.020 | +0.151 |
| `max_pain_dist_pct` | 1084 | -0.003 | +0.078 | 434 | 1068 | -0.050 | +0.117 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | — | -0.10 (n=22) | -0.25 (n=210) | +0.01 (n=62) | -0.17 (n=54) |
| `setup_ext_cap_atr` | — | +0.19 (n=23) | -0.23 (n=206) | -0.13 (n=61) | -0.33 (n=53) |
| `setup_trend` | — | +0.34 (n=23) | -0.17 (n=206) | -0.20 (n=61) | -0.31 (n=53) |
| `setup_is_trend_cont` | — | +0.17 (n=23) | -0.26 (n=206) | -0.16 (n=61) | -0.21 (n=53) |
| `ret_126d` | — | -0.06 (n=22) | -0.24 (n=210) | +0.09 (n=62) | -0.23 (n=54) |
| `rel_strength_spy_63d` | — | -0.08 (n=23) | -0.19 (n=210) | +0.07 (n=62) | -0.15 (n=54) |
| `ret_63d` | — | -0.08 (n=23) | -0.19 (n=210) | +0.07 (n=62) | -0.14 (n=54) |
| `setup_price_score` | — | +0.05 (n=23) | -0.21 (n=206) | +0.01 (n=61) | -0.25 (n=53) |
| `rel_strength_sector_63d` | — | -0.10 (n=23) | -0.17 (n=210) | +0.05 (n=62) | -0.18 (n=53) |
| `dist_52w_high` | — | -0.11 (n=23) | -0.10 (n=217) | -0.07 (n=64) | -0.13 (n=54) |
| `setup_pattern_pts` | — | -0.28 (n=23) | -0.12 (n=206) | -0.02 (n=61) | -0.17 (n=53) |
| `setup_room` | — | +0.19 (n=23) | +0.12 (n=206) | +0.20 (n=61) | -0.05 (n=53) |
| `dealer_net_gamma_at_spot` | -0.31 (n=5) | -0.35 (n=14) | +0.11 (n=403) | +0.14 (n=141) | -0.20 (n=9) |
| `sector_relative_pct` | +0.87 (n=5) | +0.11 (n=24) | +0.12 (n=390) | -0.01 (n=181) | +0.09 (n=85) |
| `setup_state_rank` | — | -0.17 (n=23) | -0.12 (n=206) | +0.12 (n=61) | -0.20 (n=53) |
| `setup_momentum` | — | -0.21 (n=23) | -0.17 (n=206) | +0.20 (n=61) | -0.08 (n=53) |
| `rel_volume` | — | +0.13 (n=23) | -0.15 (n=217) | -0.02 (n=64) | -0.13 (n=54) |
| `atr_pct` | — | +0.56 (n=22) | -0.21 (n=209) | -0.05 (n=63) | -0.01 (n=52) |
| `beta_63d` | — | +0.51 (n=23) | -0.22 (n=217) | +0.00 (n=64) | -0.14 (n=54) |
| `atm_iv_90d` | -0.42 (n=7) | +0.04 (n=47) | -0.10 (n=587) | -0.04 (n=261) | -0.08 (n=168) |
| `atm_iv_60d` | -0.29 (n=7) | +0.02 (n=47) | -0.10 (n=587) | -0.04 (n=261) | -0.07 (n=168) |
| `atm_iv_30d` | -0.36 (n=7) | +0.10 (n=47) | -0.10 (n=587) | -0.06 (n=261) | -0.04 (n=168) |
| `setup_confirm_vol` | — | -0.01 (n=23) | -0.10 (n=206) | +0.01 (n=61) | -0.19 (n=53) |
| `unusual_premium_share` | — | +0.26 (n=10) | +0.04 (n=547) | +0.08 (n=183) | — |
| `setup_is_reversal` | — | -0.04 (n=23) | +0.05 (n=206) | +0.07 (n=61) | +0.14 (n=53) |
| `gap_pct` | — | +0.14 (n=23) | -0.11 (n=217) | -0.02 (n=64) | +0.07 (n=54) |
| `ret_21d` | — | -0.10 (n=23) | +0.18 (n=217) | -0.01 (n=64) | -0.19 (n=54) |
| `vrp_proxy` | +0.09 (n=7) | +0.12 (n=47) | +0.01 (n=566) | +0.07 (n=254) | +0.10 (n=168) |
| `setup_extension` | — | -0.34 (n=23) | +0.01 (n=206) | +0.14 (n=61) | +0.17 (n=53) |
| `rel_vol_3d_20d` | — | +0.43 (n=23) | -0.07 (n=206) | -0.08 (n=61) | -0.12 (n=53) |
| `setup_is_pullback` | — | -0.33 (n=23) | +0.06 (n=206) | +0.15 (n=61) | +0.07 (n=53) |
| `iv_skew_25d` | — | — | +0.10 (n=125) | +0.16 (n=46) | -0.24 (n=13) |
| `setup_is_breakout` | — | +0.12 (n=23) | +0.04 (n=206) | +0.12 (n=61) | -0.00 (n=53) |
| `vol_trend_10d` | — | +0.35 (n=23) | -0.09 (n=206) | -0.02 (n=61) | -0.08 (n=53) |
| `resid_mom_21d` | — | -0.18 (n=23) | +0.19 (n=217) | -0.06 (n=64) | -0.19 (n=54) |
| `prem_momentum_z3d` | +0.62 (n=5) | +0.46 (n=26) | +0.07 (n=441) | -0.10 (n=194) | +0.01 (n=98) |
| `vanna_total` | -0.30 (n=7) | +0.13 (n=47) | +0.04 (n=587) | +0.07 (n=261) | -0.08 (n=168) |
| `charm_total` | +0.21 (n=7) | -0.08 (n=47) | +0.06 (n=587) | +0.02 (n=261) | +0.03 (n=168) |
| `far_otm_put_share` | -0.45 (n=5) | +0.05 (n=29) | +0.04 (n=455) | +0.05 (n=187) | +0.05 (n=91) |
| `directional_sweep_share` | +0.27 (n=6) | -0.22 (n=31) | +0.05 (n=439) | +0.12 (n=190) | -0.11 (n=100) |
| `realized_vol_regime` | +0.11 (n=7) | +0.25 (n=47) | -0.02 (n=568) | -0.06 (n=254) | -0.00 (n=168) |
| `gex_total` | +0.15 (n=7) | -0.12 (n=47) | +0.04 (n=587) | +0.04 (n=261) | -0.01 (n=168) |
| `ret_5d` | — | -0.01 (n=23) | +0.04 (n=217) | -0.03 (n=64) | -0.12 (n=54) |
| `setup_extended` | — | +0.15 (n=23) | -0.00 (n=206) | -0.01 (n=61) | +0.11 (n=53) |
| `px_vs_sma50` | — | +0.00 (n=23) | +0.09 (n=214) | -0.05 (n=64) | -0.23 (n=54) |
| `expiry_concentration_top1` | +0.69 (n=7) | +0.06 (n=47) | +0.03 (n=587) | +0.01 (n=260) | -0.03 (n=167) |
| `dealer_net_delta_at_spot` | -0.46 (n=5) | -0.15 (n=14) | +0.00 (n=403) | +0.13 (n=141) | -0.11 (n=9) |
| `far_otm_call_share` | -0.23 (n=5) | +0.20 (n=29) | -0.02 (n=455) | -0.01 (n=187) | -0.11 (n=91) |
| `momentum_score` | -0.22 (n=7) | -0.26 (n=47) | +0.09 (n=576) | -0.09 (n=264) | +0.04 (n=158) |
| `momentum_composite` | -0.22 (n=7) | -0.26 (n=47) | +0.09 (n=576) | -0.09 (n=264) | +0.04 (n=158) |
| `sweep_share` | -0.29 (n=7) | -0.07 (n=48) | -0.01 (n=591) | +0.00 (n=264) | — |
| `bullish_premium_share` | -0.24 (n=7) | +0.46 (n=47) | -0.06 (n=587) | +0.03 (n=261) | +0.08 (n=168) |
| `aggressor_net_prem_bps` | +0.03 (n=6) | -0.19 (n=31) | +0.06 (n=439) | -0.03 (n=190) | -0.20 (n=100) |
| `bollinger_z` | — | -0.07 (n=23) | +0.11 (n=217) | +0.10 (n=64) | -0.26 (n=54) |
| `aggressor_bull_share` | -0.15 (n=6) | -0.19 (n=30) | +0.04 (n=438) | +0.03 (n=187) | -0.03 (n=99) |
| `multileg_share` | +0.70 (n=7) | +0.09 (n=48) | -0.07 (n=591) | +0.09 (n=264) | — |
| `rel_vol_5d_20d` | — | +0.51 (n=23) | -0.01 (n=206) | +0.01 (n=61) | -0.04 (n=53) |
| `up_down_vol_ratio_10d` | — | -0.11 (n=23) | +0.11 (n=206) | -0.11 (n=61) | -0.33 (n=53) |
| `ask_side_ratio` | -0.46 (n=6) | +0.05 (n=31) | -0.00 (n=438) | -0.02 (n=189) | +0.06 (n=99) |
| `term_slope_30_90` | -0.62 (n=7) | -0.16 (n=47) | -0.00 (n=587) | +0.11 (n=261) | -0.08 (n=168) |
| `rsi_14` | — | -0.14 (n=23) | +0.14 (n=217) | +0.02 (n=64) | -0.28 (n=54) |
| `dollar_delta_weighted_flow` | +0.45 (n=5) | -0.00 (n=29) | +0.01 (n=455) | -0.01 (n=187) | +0.05 (n=91) |
| `max_pain_dist_pct` | -0.43 (n=7) | +0.11 (n=48) | +0.03 (n=591) | -0.02 (n=263) | -0.12 (n=175) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `setup_room` | 346 | +0.108 | +0.142 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dealer_net_gamma_at_spot` | 572 | +0.106 | +0.209 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
