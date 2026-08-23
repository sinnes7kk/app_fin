# Feature lab — Spearman ranking — 2026-08-23 22:22

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **1093 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `setup_trend` | 274 | -0.214 | -0.095 | 110 | 258 | -0.223 | -0.195 |
| `px_vs_sma200` | 280 | -0.206 | -0.177 | 112 | 264 | -0.096 | -0.108 |
| `setup_price_score` | 274 | -0.204 | -0.036 | 110 | 258 | -0.223 | -0.200 |
| `ret_126d` | 280 | -0.198 | -0.213 | 112 | 264 | -0.129 | -0.159 |
| `setup_ext_cap_atr` | 274 | -0.188 | -0.139 | 110 | 258 | -0.207 | -0.233 |
| `setup_state_rank` | 274 | -0.146 | +0.017 | 110 | 258 | -0.203 | -0.203 |
| `setup_pattern_pts` | 274 | -0.143 | -0.038 | 110 | 258 | -0.113 | -0.199 |
| `rel_strength_spy_63d` | 280 | -0.137 | -0.160 | 112 | 264 | +0.004 | -0.091 |
| `ret_63d` | 280 | -0.137 | -0.163 | 112 | 264 | +0.006 | -0.092 |
| `setup_momentum` | 274 | -0.135 | +0.064 | 110 | 258 | -0.178 | -0.107 |
| `dealer_net_gamma_at_spot` | 530 | +0.125 | +0.236 | 212 | 518 | +0.002 | +0.170 |
| `rel_strength_sector_63d` | 279 | -0.124 | -0.106 | 112 | 263 | +0.003 | -0.066 |
| `atr_pct` | 277 | -0.123 | -0.128 | 111 | 261 | -0.070 | +0.017 |
| `sector_relative_pct` | 642 | +0.122 | +0.120 | 257 | 630 | +0.069 | +0.135 |
| `dist_52w_high` | 289 | -0.120 | -0.146 | 116 | 273 | -0.013 | -0.035 |
| `setup_is_trend_cont` | 274 | -0.113 | +0.030 | 110 | 258 | -0.142 | -0.196 |
| `ret_21d` | 289 | +0.100 | +0.218 | 116 | 273 | +0.214 | +0.200 |
| `setup_room` | 274 | +0.098 | +0.119 | 110 | 258 | +0.121 | +0.091 |
| `resid_mom_21d` | 289 | +0.096 | +0.187 | 116 | 273 | +0.220 | +0.179 |
| `beta_63d` | 289 | -0.095 | +0.065 | 116 | 273 | -0.166 | -0.023 |
| `bollinger_z` | 289 | +0.086 | +0.189 | 116 | 273 | +0.205 | +0.154 |
| `rel_volume` | 289 | -0.082 | -0.139 | 116 | 273 | -0.094 | -0.172 |
| `atm_iv_90d` | 998 | -0.076 | -0.118 | 400 | 982 | -0.049 | -0.050 |
| `atm_iv_60d` | 998 | -0.075 | -0.117 | 400 | 982 | -0.046 | -0.046 |
| `atm_iv_30d` | 998 | -0.074 | -0.097 | 400 | 982 | -0.048 | -0.036 |
| `ret_5d` | 289 | +0.072 | +0.161 | 116 | 273 | +0.206 | +0.112 |
| `setup_confirm_vol` | 274 | -0.068 | -0.153 | 110 | 258 | -0.099 | -0.180 |
| `rsi_14` | 289 | +0.068 | +0.179 | 116 | 273 | +0.187 | +0.182 |
| `vol_trend_10d` | 274 | -0.060 | -0.055 | 110 | 258 | -0.082 | -0.212 |
| `charm_total` | 998 | +0.056 | +0.085 | 400 | 982 | +0.047 | +0.141 |
| `rel_vol_3d_20d` | 274 | -0.051 | -0.116 | 110 | 258 | -0.062 | -0.257 |
| `prem_momentum_z3d` | 729 | +0.047 | +0.096 | 292 | 722 | -0.048 | -0.011 |
| `unusual_premium_share` | 690 | +0.043 | -0.083 | 276 | 678 | +0.008 | -0.040 |
| `far_otm_call_share` | 714 | -0.042 | +0.047 | 286 | 704 | -0.021 | +0.060 |
| `setup_is_reversal` | 274 | +0.041 | +0.168 | 110 | 258 | +0.043 | +0.097 |
| `gex_total` | 998 | +0.040 | +0.161 | 400 | 982 | +0.011 | +0.153 |
| `realized_vol_regime` | 972 | -0.039 | -0.058 | 389 | 956 | +0.019 | -0.015 |
| `vrp_proxy` | 970 | +0.036 | +0.073 | 388 | 954 | +0.019 | +0.047 |
| `bullish_premium_share` | 998 | +0.034 | +0.058 | 400 | 982 | -0.019 | -0.004 |
| `iv_skew_25d` | 171 | +0.030 | +0.027 | 69 | 158 | +0.132 | +0.030 |
| `expiry_concentration_top1` | 996 | +0.028 | +0.060 | 399 | 980 | +0.065 | +0.132 |
| `setup_extended` | 274 | -0.028 | -0.143 | 110 | 258 | +0.030 | -0.035 |
| `vanna_total` | 998 | +0.028 | +0.038 | 400 | 982 | -0.060 | -0.056 |
| `up_down_vol_ratio_10d` | 274 | +0.027 | +0.175 | 110 | 258 | +0.092 | +0.176 |
| `momentum_composite` | 980 | +0.027 | +0.044 | 392 | 964 | -0.052 | -0.014 |
| `momentum_score` | 980 | +0.027 | +0.044 | 392 | 964 | -0.052 | -0.014 |
| `directional_sweep_share` | 713 | +0.027 | +0.044 | 286 | 703 | +0.014 | -0.000 |
| `gap_pct` | 289 | -0.026 | +0.017 | 116 | 273 | +0.056 | +0.091 |
| `far_otm_put_share` | 714 | +0.022 | +0.002 | 286 | 704 | -0.051 | -0.035 |
| `ask_side_ratio` | 710 | +0.021 | -0.025 | 284 | 700 | -0.012 | -0.024 |
| `px_vs_sma50` | 286 | +0.018 | +0.104 | 115 | 270 | +0.166 | +0.178 |
| `dealer_net_delta_at_spot` | 530 | +0.018 | -0.010 | 212 | 518 | +0.018 | +0.063 |
| `sweep_share` | 1013 | -0.018 | -0.070 | 406 | 997 | -0.024 | -0.109 |
| `setup_extension` | 274 | +0.017 | +0.025 | 110 | 258 | +0.019 | +0.016 |
| `aggressor_net_prem_bps` | 713 | -0.014 | +0.035 | 286 | 703 | +0.021 | -0.021 |
| `rel_vol_5d_20d` | 274 | -0.013 | -0.102 | 110 | 258 | -0.053 | -0.271 |
| `aggressor_bull_share` | 707 | +0.011 | +0.034 | 283 | 697 | -0.002 | -0.060 |
| `dollar_delta_weighted_flow` | 714 | -0.010 | -0.079 | 286 | 704 | -0.009 | +0.096 |
| `max_pain_dist_pct` | 1013 | +0.010 | +0.135 | 406 | 997 | -0.037 | +0.170 |
| `multileg_share` | 1013 | -0.009 | -0.038 | 406 | 997 | -0.007 | -0.012 |
| `term_slope_30_90` | 998 | -0.009 | -0.099 | 400 | 982 | -0.003 | -0.085 |
| `setup_is_pullback` | 274 | -0.005 | -0.117 | 110 | 258 | +0.053 | -0.043 |
| `setup_is_breakout` | 274 | -0.002 | -0.072 | 110 | 258 | -0.006 | +0.024 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `setup_trend` | — | +0.39 (n=18) | -0.26 (n=169) | +0.05 (n=41) | -0.52 (n=44) |
| `px_vs_sma200` | — | -0.08 (n=18) | -0.28 (n=173) | +0.10 (n=42) | -0.24 (n=45) |
| `setup_price_score` | — | +0.13 (n=18) | -0.26 (n=169) | +0.15 (n=41) | -0.40 (n=44) |
| `ret_126d` | — | -0.05 (n=18) | -0.29 (n=173) | +0.11 (n=42) | -0.22 (n=45) |
| `setup_ext_cap_atr` | — | +0.21 (n=18) | -0.25 (n=169) | +0.10 (n=41) | -0.38 (n=44) |
| `setup_state_rank` | — | -0.07 (n=18) | -0.18 (n=169) | +0.15 (n=41) | -0.25 (n=44) |
| `setup_pattern_pts` | — | -0.10 (n=18) | -0.16 (n=169) | +0.00 (n=41) | -0.27 (n=44) |
| `rel_strength_spy_63d` | — | -0.02 (n=18) | -0.20 (n=173) | +0.14 (n=42) | -0.19 (n=45) |
| `ret_63d` | — | -0.02 (n=18) | -0.19 (n=173) | +0.14 (n=42) | -0.18 (n=45) |
| `setup_momentum` | — | -0.09 (n=18) | -0.18 (n=169) | +0.19 (n=41) | -0.16 (n=44) |
| `dealer_net_gamma_at_spot` | — | -0.41 (n=11) | +0.12 (n=379) | +0.20 (n=127) | -0.20 (n=9) |
| `rel_strength_sector_63d` | — | -0.02 (n=18) | -0.17 (n=173) | +0.13 (n=42) | -0.20 (n=44) |
| `atr_pct` | — | +0.58 (n=17) | -0.19 (n=172) | -0.20 (n=43) | -0.06 (n=43) |
| `sector_relative_pct` | +0.87 (n=5) | +0.27 (n=20) | +0.15 (n=372) | +0.04 (n=164) | +0.08 (n=81) |
| `dist_52w_high` | — | -0.21 (n=18) | -0.13 (n=180) | -0.02 (n=44) | -0.17 (n=45) |
| `setup_is_trend_cont` | — | +0.18 (n=18) | -0.24 (n=169) | +0.16 (n=41) | -0.16 (n=44) |
| `ret_21d` | — | -0.07 (n=18) | +0.19 (n=180) | +0.16 (n=44) | -0.22 (n=45) |
| `setup_room` | — | +0.16 (n=18) | +0.08 (n=169) | +0.29 (n=41) | -0.02 (n=44) |
| `resid_mom_21d` | — | -0.11 (n=18) | +0.19 (n=180) | +0.14 (n=44) | -0.24 (n=45) |
| `beta_63d` | — | +0.62 (n=18) | -0.19 (n=180) | +0.05 (n=44) | -0.15 (n=45) |
| `bollinger_z` | — | +0.04 (n=18) | +0.13 (n=180) | +0.38 (n=44) | -0.23 (n=45) |
| `rel_volume` | — | +0.34 (n=18) | -0.08 (n=180) | -0.15 (n=44) | -0.16 (n=45) |
| `atm_iv_90d` | -0.05 (n=6) | -0.09 (n=42) | -0.08 (n=550) | -0.03 (n=241) | -0.11 (n=159) |
| `atm_iv_60d` | +0.17 (n=6) | -0.09 (n=42) | -0.08 (n=550) | -0.03 (n=241) | -0.10 (n=159) |
| `atm_iv_30d` | +0.06 (n=6) | +0.00 (n=42) | -0.08 (n=550) | -0.04 (n=241) | -0.08 (n=159) |
| `ret_5d` | — | +0.15 (n=18) | +0.10 (n=180) | +0.20 (n=44) | -0.11 (n=45) |
| `setup_confirm_vol` | — | +0.16 (n=18) | -0.03 (n=169) | -0.17 (n=41) | -0.22 (n=44) |
| `rsi_14` | — | -0.10 (n=18) | +0.14 (n=180) | +0.32 (n=44) | -0.30 (n=45) |
| `vol_trend_10d` | — | +0.50 (n=18) | -0.09 (n=169) | -0.14 (n=41) | -0.12 (n=44) |
| `charm_total` | +0.15 (n=6) | -0.08 (n=42) | +0.07 (n=550) | +0.05 (n=241) | +0.05 (n=159) |
| `rel_vol_3d_20d` | — | +0.60 (n=18) | -0.05 (n=169) | -0.23 (n=41) | -0.16 (n=44) |
| `prem_momentum_z3d` | — | +0.49 (n=24) | +0.08 (n=424) | -0.08 (n=180) | +0.00 (n=97) |
| `unusual_premium_share` | — | +0.22 (n=8) | +0.03 (n=510) | +0.06 (n=169) | — |
| `far_otm_call_share` | — | +0.07 (n=25) | -0.04 (n=427) | -0.03 (n=171) | -0.14 (n=87) |
| `setup_is_reversal` | — | +0.36 (n=18) | +0.01 (n=169) | +0.03 (n=41) | +0.16 (n=44) |
| `gex_total` | +0.06 (n=6) | -0.03 (n=42) | +0.05 (n=550) | +0.08 (n=241) | -0.05 (n=159) |
| `realized_vol_regime` | -0.12 (n=6) | +0.34 (n=42) | -0.04 (n=531) | -0.10 (n=234) | -0.01 (n=159) |
| `vrp_proxy` | +0.46 (n=6) | +0.11 (n=42) | +0.01 (n=529) | +0.06 (n=234) | +0.09 (n=159) |
| `bullish_premium_share` | +0.25 (n=6) | +0.45 (n=42) | -0.06 (n=550) | +0.10 (n=241) | +0.09 (n=159) |
| `iv_skew_25d` | — | — | +0.08 (n=118) | +0.15 (n=40) | -0.30 (n=11) |
| `expiry_concentration_top1` | +0.49 (n=6) | +0.12 (n=42) | +0.03 (n=550) | +0.04 (n=240) | -0.03 (n=158) |
| `setup_extended` | — | +0.09 (n=18) | -0.03 (n=169) | -0.18 (n=41) | -0.02 (n=44) |
| `vanna_total` | -0.19 (n=6) | +0.22 (n=42) | +0.04 (n=550) | +0.06 (n=241) | -0.10 (n=159) |
| `up_down_vol_ratio_10d` | — | +0.05 (n=18) | +0.09 (n=169) | +0.11 (n=41) | -0.31 (n=44) |
| `momentum_composite` | -0.19 (n=6) | -0.21 (n=42) | +0.09 (n=539) | -0.09 (n=244) | +0.06 (n=149) |
| `momentum_score` | -0.19 (n=6) | -0.21 (n=42) | +0.09 (n=539) | -0.09 (n=244) | +0.06 (n=149) |
| `directional_sweep_share` | -0.34 (n=5) | -0.29 (n=27) | +0.05 (n=411) | +0.11 (n=174) | -0.06 (n=96) |
| `gap_pct` | — | +0.17 (n=18) | -0.10 (n=180) | +0.07 (n=44) | +0.08 (n=45) |
| `far_otm_put_share` | — | +0.09 (n=25) | +0.03 (n=427) | +0.08 (n=171) | +0.02 (n=87) |
| `ask_side_ratio` | -0.45 (n=5) | +0.08 (n=27) | +0.01 (n=410) | +0.00 (n=173) | +0.07 (n=95) |
| `px_vs_sma50` | — | +0.09 (n=18) | +0.07 (n=177) | +0.18 (n=44) | -0.28 (n=45) |
| `dealer_net_delta_at_spot` | — | -0.01 (n=11) | +0.01 (n=379) | +0.10 (n=127) | -0.11 (n=9) |
| `sweep_share` | +0.02 (n=6) | -0.12 (n=43) | -0.01 (n=554) | -0.02 (n=244) | — |
| `setup_extension` | — | -0.35 (n=18) | -0.00 (n=169) | +0.00 (n=41) | +0.17 (n=44) |
| `aggressor_net_prem_bps` | -0.11 (n=5) | -0.27 (n=27) | +0.06 (n=411) | -0.06 (n=174) | -0.18 (n=96) |
| `rel_vol_5d_20d` | — | +0.71 (n=18) | -0.03 (n=169) | -0.11 (n=41) | -0.02 (n=44) |
| `aggressor_bull_share` | -0.45 (n=5) | -0.34 (n=26) | +0.04 (n=410) | +0.01 (n=171) | -0.01 (n=95) |
| `dollar_delta_weighted_flow` | — | +0.09 (n=25) | -0.00 (n=427) | -0.03 (n=171) | +0.05 (n=87) |
| `max_pain_dist_pct` | -0.19 (n=6) | +0.14 (n=43) | +0.03 (n=554) | +0.01 (n=244) | -0.13 (n=166) |
| `multileg_share` | +0.42 (n=6) | +0.13 (n=43) | -0.07 (n=554) | +0.09 (n=244) | — |
| `term_slope_30_90` | -0.37 (n=6) | -0.13 (n=42) | -0.02 (n=550) | +0.09 (n=241) | -0.06 (n=159) |
| `setup_is_pullback` | — | -0.47 (n=18) | +0.05 (n=169) | -0.08 (n=41) | +0.03 (n=44) |
| `setup_is_breakout` | — | +0.10 (n=18) | +0.07 (n=169) | -0.12 (n=41) | -0.13 (n=44) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `setup_state_rank` | 274 | -0.146 | +0.017 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `setup_momentum` | 274 | -0.135 | +0.064 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dealer_net_gamma_at_spot` | 530 | +0.125 | +0.236 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 642 | +0.122 | +0.120 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `setup_is_trend_cont` | 274 | -0.113 | +0.030 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
