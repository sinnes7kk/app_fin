# Feature lab — Spearman ranking — 2026-08-16 22:21

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **1015 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | 154 | -0.226 | -0.163 | 62 | 137 | -0.012 | +0.047 |
| `setup_momentum` | 142 | -0.218 | -0.391 | 57 | 125 | -0.232 | -0.065 |
| `rel_strength_spy_63d` | 154 | -0.196 | -0.106 | 62 | 137 | -0.046 | +0.108 |
| `ret_63d` | 154 | -0.195 | -0.104 | 62 | 137 | -0.042 | +0.119 |
| `ret_126d` | 154 | -0.189 | -0.144 | 62 | 137 | -0.017 | +0.020 |
| `setup_price_score` | 142 | -0.183 | -0.247 | 57 | 125 | -0.157 | -0.126 |
| `rel_strength_sector_63d` | 153 | -0.177 | -0.099 | 62 | 136 | -0.057 | +0.070 |
| `atr_pct` | 157 | -0.166 | -0.136 | 63 | 140 | -0.166 | -0.313 |
| `gap_pct` | 157 | -0.162 | -0.137 | 63 | 140 | -0.029 | +0.142 |
| `setup_state_rank` | 142 | -0.142 | -0.180 | 57 | 125 | -0.201 | -0.065 |
| `beta_63d` | 157 | -0.139 | -0.174 | 63 | 140 | -0.150 | -0.302 |
| `setup_pattern_pts` | 142 | -0.122 | -0.209 | 57 | 125 | -0.039 | -0.141 |
| `setup_is_reversal` | 142 | -0.120 | -0.131 | 57 | 125 | -0.036 | +0.162 |
| `sector_relative_pct` | 556 | +0.112 | +0.105 | 223 | 547 | +0.057 | +0.065 |
| `up_down_vol_ratio_10d` | 142 | -0.109 | +0.048 | 57 | 125 | +0.053 | +0.226 |
| `setup_is_breakout` | 142 | -0.105 | -0.187 | 57 | 125 | -0.155 | -0.226 |
| `setup_trend` | 142 | -0.093 | -0.212 | 57 | 125 | -0.073 | -0.211 |
| `unusual_premium_share` | 607 | +0.082 | +0.047 | 243 | 594 | +0.029 | +0.062 |
| `dist_52w_high` | 157 | -0.080 | +0.006 | 63 | 140 | +0.117 | +0.163 |
| `atm_iv_30d` | 866 | -0.079 | -0.089 | 347 | 849 | -0.063 | -0.067 |
| `dealer_net_gamma_at_spot` | 467 | +0.079 | +0.120 | 187 | 454 | -0.081 | -0.028 |
| `atm_iv_60d` | 866 | -0.073 | -0.091 | 347 | 849 | -0.055 | -0.061 |
| `atm_iv_90d` | 866 | -0.073 | -0.091 | 347 | 849 | -0.054 | -0.062 |
| `rsi_14` | 157 | -0.072 | +0.010 | 63 | 140 | +0.067 | +0.284 |
| `bollinger_z` | 157 | -0.072 | -0.035 | 63 | 140 | +0.078 | +0.245 |
| `vol_trend_10d` | 142 | -0.072 | -0.038 | 57 | 125 | +0.059 | +0.100 |
| `setup_ext_cap_atr` | 142 | -0.065 | -0.153 | 57 | 125 | -0.003 | -0.180 |
| `iv_skew_25d` | 153 | +0.063 | +0.073 | 62 | 142 | +0.161 | +0.164 |
| `charm_total` | 866 | +0.060 | +0.027 | 347 | 849 | +0.041 | +0.055 |
| `px_vs_sma50` | 154 | -0.057 | +0.028 | 62 | 137 | +0.061 | +0.303 |
| `realized_vol_regime` | 840 | -0.052 | -0.125 | 336 | 823 | +0.010 | -0.059 |
| `ask_side_ratio` | 629 | +0.049 | +0.079 | 252 | 622 | +0.020 | +0.078 |
| `prem_momentum_z3d` | 667 | +0.047 | +0.121 | 267 | 663 | -0.052 | -0.048 |
| `far_otm_put_share` | 633 | +0.047 | +0.060 | 254 | 626 | -0.050 | -0.014 |
| `setup_is_pullback` | 142 | +0.044 | +0.141 | 57 | 125 | -0.023 | -0.066 |
| `far_otm_call_share` | 633 | -0.041 | +0.039 | 254 | 626 | -0.033 | +0.001 |
| `expiry_concentration_top1` | 864 | +0.040 | +0.095 | 346 | 847 | +0.058 | +0.119 |
| `momentum_composite` | 848 | +0.039 | +0.089 | 340 | 831 | -0.048 | -0.020 |
| `momentum_score` | 848 | +0.039 | +0.089 | 340 | 831 | -0.048 | -0.020 |
| `ret_5d` | 157 | -0.037 | +0.020 | 63 | 140 | +0.180 | +0.301 |
| `resid_mom_21d` | 157 | +0.033 | +0.111 | 63 | 140 | +0.132 | +0.322 |
| `ret_21d` | 157 | +0.032 | +0.092 | 63 | 140 | +0.139 | +0.319 |
| `vrp_proxy` | 838 | +0.030 | +0.069 | 336 | 821 | +0.026 | +0.057 |
| `setup_room` | 142 | -0.029 | +0.028 | 57 | 125 | -0.068 | -0.111 |
| `setup_is_trend_cont` | 142 | -0.029 | -0.115 | 57 | 125 | +0.109 | +0.041 |
| `aggressor_bull_share` | 626 | +0.029 | +0.130 | 251 | 619 | -0.003 | -0.052 |
| `rel_volume` | 157 | -0.029 | +0.018 | 63 | 140 | -0.001 | +0.132 |
| `max_pain_dist_pct` | 881 | -0.027 | -0.020 | 353 | 864 | -0.090 | -0.045 |
| `dollar_delta_weighted_flow` | 633 | -0.023 | -0.104 | 254 | 626 | -0.018 | +0.029 |
| `setup_extended` | 142 | +0.022 | +0.049 | 57 | 125 | +0.011 | +0.060 |
| `rel_vol_3d_20d` | 142 | -0.019 | +0.021 | 57 | 125 | +0.059 | +0.131 |
| `setup_confirm_vol` | 142 | -0.017 | -0.017 | 57 | 125 | -0.026 | +0.098 |
| `sweep_share` | 881 | -0.017 | -0.036 | 353 | 864 | -0.025 | -0.104 |
| `aggressor_net_prem_bps` | 632 | -0.013 | +0.043 | 253 | 625 | +0.026 | +0.000 |
| `dealer_net_delta_at_spot` | 467 | -0.010 | -0.005 | 187 | 454 | -0.005 | +0.085 |
| `bullish_premium_share` | 866 | +0.010 | -0.057 | 347 | 849 | -0.079 | -0.206 |
| `setup_extension` | 142 | +0.008 | -0.011 | 57 | 125 | +0.064 | +0.041 |
| `directional_sweep_share` | 632 | +0.007 | -0.008 | 253 | 625 | +0.002 | -0.048 |
| `rel_vol_5d_20d` | 142 | +0.006 | +0.037 | 57 | 125 | +0.064 | +0.209 |
| `term_slope_30_90` | 866 | +0.005 | -0.088 | 347 | 849 | +0.021 | -0.047 |
| `multileg_share` | 881 | +0.005 | -0.004 | 353 | 864 | +0.002 | +0.001 |
| `gex_total` | 866 | -0.001 | +0.067 | 347 | 849 | -0.037 | +0.039 |
| `vanna_total` | 866 | -0.000 | +0.017 | 347 | 849 | -0.080 | -0.054 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | — | -0.34 (n=7) | -0.27 (n=98) | -0.22 (n=19) | -0.26 (n=29) |
| `setup_momentum` | — | -0.13 (n=7) | -0.25 (n=90) | +0.25 (n=16) | -0.37 (n=28) |
| `rel_strength_spy_63d` | — | -0.14 (n=7) | -0.23 (n=98) | -0.28 (n=19) | -0.26 (n=29) |
| `ret_63d` | — | -0.14 (n=7) | -0.23 (n=98) | -0.28 (n=19) | -0.28 (n=29) |
| `ret_126d` | — | -0.14 (n=7) | -0.24 (n=98) | -0.16 (n=19) | -0.19 (n=29) |
| `setup_price_score` | — | -0.14 (n=7) | -0.18 (n=90) | +0.24 (n=16) | -0.54 (n=28) |
| `rel_strength_sector_63d` | — | -0.07 (n=7) | -0.19 (n=98) | -0.31 (n=19) | -0.27 (n=28) |
| `atr_pct` | — | -0.02 (n=7) | -0.22 (n=101) | +0.21 (n=19) | -0.10 (n=29) |
| `gap_pct` | — | -0.43 (n=7) | -0.18 (n=101) | -0.40 (n=19) | -0.01 (n=29) |
| `setup_state_rank` | — | +0.18 (n=7) | -0.20 (n=90) | +0.32 (n=16) | -0.34 (n=28) |
| `beta_63d` | — | +0.41 (n=7) | -0.22 (n=101) | +0.33 (n=19) | -0.26 (n=29) |
| `setup_pattern_pts` | — | +0.16 (n=7) | -0.12 (n=90) | +0.08 (n=16) | -0.28 (n=28) |
| `setup_is_reversal` | — | -0.21 (n=7) | -0.17 (n=90) | — | +0.05 (n=28) |
| `sector_relative_pct` | — | +0.36 (n=14) | +0.12 (n=317) | +0.07 (n=147) | +0.11 (n=74) |
| `up_down_vol_ratio_10d` | — | -0.16 (n=7) | -0.06 (n=90) | -0.54 (n=16) | -0.22 (n=28) |
| `setup_is_breakout` | — | — | +0.03 (n=90) | — | -0.37 (n=28) |
| `setup_trend` | — | +0.35 (n=7) | -0.01 (n=90) | +0.29 (n=16) | -0.62 (n=28) |
| `unusual_premium_share` | — | +0.43 (n=7) | +0.09 (n=440) | +0.05 (n=157) | — |
| `dist_52w_high` | — | -0.45 (n=7) | -0.08 (n=101) | -0.46 (n=19) | -0.07 (n=29) |
| `atm_iv_30d` | +0.34 (n=5) | -0.01 (n=31) | -0.08 (n=471) | -0.02 (n=216) | -0.07 (n=143) |
| `dealer_net_gamma_at_spot` | — | -0.31 (n=9) | +0.07 (n=332) | +0.13 (n=113) | -0.20 (n=9) |
| `atm_iv_60d` | +0.52 (n=5) | -0.09 (n=31) | -0.06 (n=471) | -0.01 (n=216) | -0.09 (n=143) |
| `atm_iv_90d` | +0.34 (n=5) | -0.12 (n=31) | -0.06 (n=471) | -0.01 (n=216) | -0.10 (n=143) |
| `rsi_14` | — | -0.41 (n=7) | -0.01 (n=101) | -0.30 (n=19) | -0.36 (n=29) |
| `bollinger_z` | — | -0.13 (n=7) | -0.00 (n=101) | -0.18 (n=19) | -0.33 (n=29) |
| `vol_trend_10d` | — | +0.09 (n=7) | -0.07 (n=90) | -0.38 (n=16) | -0.20 (n=28) |
| `setup_ext_cap_atr` | — | +0.22 (n=7) | -0.09 (n=90) | +0.42 (n=16) | -0.46 (n=28) |
| `iv_skew_25d` | — | — | +0.11 (n=104) | +0.18 (n=37) | -0.30 (n=11) |
| `charm_total` | +0.45 (n=5) | -0.19 (n=31) | +0.06 (n=471) | +0.04 (n=216) | +0.02 (n=143) |
| `px_vs_sma50` | — | +0.04 (n=7) | -0.02 (n=98) | -0.41 (n=19) | -0.28 (n=29) |
| `realized_vol_regime` | +0.45 (n=5) | +0.18 (n=31) | -0.06 (n=452) | -0.08 (n=209) | -0.04 (n=143) |
| `ask_side_ratio` | -0.45 (n=5) | +0.17 (n=23) | +0.05 (n=354) | +0.00 (n=159) | +0.07 (n=88) |
| `prem_momentum_z3d` | — | +0.45 (n=23) | +0.09 (n=376) | -0.09 (n=172) | +0.02 (n=92) |
| `far_otm_put_share` | — | -0.05 (n=21) | +0.08 (n=371) | +0.11 (n=157) | -0.03 (n=80) |
| `setup_is_pullback` | — | — | +0.07 (n=90) | -0.26 (n=16) | +0.25 (n=28) |
| `far_otm_call_share` | — | -0.08 (n=21) | -0.05 (n=371) | -0.00 (n=157) | -0.13 (n=80) |
| `expiry_concentration_top1` | +0.67 (n=5) | +0.21 (n=31) | +0.04 (n=471) | +0.02 (n=215) | -0.06 (n=142) |
| `momentum_composite` | -0.89 (n=5) | -0.21 (n=31) | +0.10 (n=460) | -0.08 (n=219) | +0.09 (n=133) |
| `momentum_score` | -0.89 (n=5) | -0.21 (n=31) | +0.10 (n=460) | -0.08 (n=219) | +0.09 (n=133) |
| `ret_5d` | — | +0.04 (n=7) | +0.00 (n=101) | -0.17 (n=19) | -0.22 (n=29) |
| `resid_mom_21d` | — | +0.04 (n=7) | +0.11 (n=101) | -0.37 (n=19) | -0.29 (n=29) |
| `ret_21d` | — | +0.23 (n=7) | +0.10 (n=101) | -0.38 (n=19) | -0.28 (n=29) |
| `vrp_proxy` | +0.11 (n=5) | +0.17 (n=31) | -0.02 (n=450) | +0.08 (n=209) | +0.12 (n=143) |
| `setup_room` | — | — | -0.06 (n=90) | -0.01 (n=16) | -0.11 (n=28) |
| `setup_is_trend_cont` | — | +0.22 (n=7) | -0.12 (n=90) | +0.35 (n=16) | -0.14 (n=28) |
| `aggressor_bull_share` | -0.45 (n=5) | -0.30 (n=22) | +0.06 (n=354) | +0.03 (n=157) | +0.04 (n=88) |
| `rel_volume` | — | +0.09 (n=7) | -0.06 (n=101) | -0.16 (n=19) | -0.12 (n=29) |
| `max_pain_dist_pct` | -0.22 (n=5) | +0.14 (n=32) | -0.01 (n=475) | -0.03 (n=219) | -0.19 (n=150) |
| `dollar_delta_weighted_flow` | — | -0.06 (n=21) | -0.01 (n=371) | -0.05 (n=157) | +0.06 (n=80) |
| `setup_extended` | — | +0.00 (n=7) | +0.08 (n=90) | -0.20 (n=16) | -0.12 (n=28) |
| `rel_vol_3d_20d` | — | +0.29 (n=7) | -0.02 (n=90) | -0.44 (n=16) | -0.16 (n=28) |
| `setup_confirm_vol` | — | +0.05 (n=7) | -0.02 (n=90) | -0.19 (n=16) | -0.18 (n=28) |
| `sweep_share` | -0.63 (n=5) | -0.16 (n=32) | +0.02 (n=475) | -0.01 (n=219) | — |
| `aggressor_net_prem_bps` | -0.11 (n=5) | -0.34 (n=23) | +0.07 (n=355) | -0.06 (n=160) | -0.18 (n=89) |
| `dealer_net_delta_at_spot` | — | +0.22 (n=9) | -0.02 (n=332) | +0.07 (n=113) | -0.11 (n=9) |
| `bullish_premium_share` | +0.11 (n=5) | +0.42 (n=31) | -0.08 (n=471) | +0.03 (n=216) | +0.06 (n=143) |
| `setup_extension` | — | +0.41 (n=7) | -0.06 (n=90) | +0.16 (n=16) | +0.25 (n=28) |
| `directional_sweep_share` | -0.34 (n=5) | -0.34 (n=23) | +0.02 (n=355) | +0.12 (n=160) | -0.12 (n=89) |
| `rel_vol_5d_20d` | — | +0.41 (n=7) | -0.05 (n=90) | -0.30 (n=16) | -0.07 (n=28) |
| `term_slope_30_90` | -0.34 (n=5) | -0.21 (n=31) | -0.01 (n=471) | +0.10 (n=216) | -0.04 (n=143) |
| `multileg_share` | +0.40 (n=5) | +0.18 (n=32) | -0.05 (n=475) | +0.10 (n=219) | — |
| `gex_total` | +0.34 (n=5) | +0.04 (n=31) | +0.00 (n=471) | +0.04 (n=216) | -0.08 (n=143) |
| `vanna_total` | -0.89 (n=5) | +0.42 (n=31) | +0.01 (n=471) | +0.05 (n=216) | -0.07 (n=143) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `sector_relative_pct` | 556 | +0.112 | +0.105 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `up_down_vol_ratio_10d` | 142 | -0.109 | +0.048 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
