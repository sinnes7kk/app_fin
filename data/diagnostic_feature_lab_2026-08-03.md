# Feature lab — Spearman ranking — 2026-08-03 07:50

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **881 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

The `Fwd IC` columns repeat the same rank correlation against `replay_forward_return_5d` — a plain 5-day close-to-close move that no entry or exit rule touches. Realized R tells you what the account earned; the forward return tells you whether the feature called the move at all. A feature that scores well on forward return but flat on realized R is evidence against the **exit policy**, not against the feature.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val | n (fwd) | Fwd IC | Fwd OOS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | 132 | -0.282 | -0.287 | 53 | 117 | -0.063 | -0.194 |
| `ret_126d` | 132 | -0.246 | -0.283 | 53 | 117 | -0.078 | -0.216 |
| `rel_strength_spy_63d` | 132 | -0.245 | -0.169 | 53 | 117 | -0.091 | -0.094 |
| `ret_63d` | 132 | -0.245 | -0.183 | 53 | 117 | -0.089 | -0.095 |
| `setup_momentum` | 120 | -0.242 | -0.493 | 48 | 105 | -0.272 | -0.084 |
| `rel_strength_sector_63d` | 131 | -0.231 | -0.278 | 53 | 116 | -0.092 | -0.141 |
| `setup_price_score` | 120 | -0.230 | -0.483 | 48 | 105 | -0.193 | -0.301 |
| `setup_state_rank` | 120 | -0.204 | -0.399 | 48 | 105 | -0.266 | -0.247 |
| `gap_pct` | 135 | -0.198 | -0.257 | 54 | 120 | -0.062 | +0.050 |
| `atr_pct` | 135 | -0.181 | -0.074 | 54 | 120 | -0.172 | -0.222 |
| `up_down_vol_ratio_10d` | 120 | -0.173 | -0.054 | 48 | 105 | +0.011 | -0.004 |
| `setup_pattern_pts` | 120 | -0.170 | -0.346 | 48 | 105 | -0.049 | -0.234 |
| `beta_63d` | 135 | -0.158 | -0.190 | 54 | 120 | -0.167 | -0.272 |
| `dist_52w_high` | 135 | -0.141 | -0.168 | 54 | 120 | +0.086 | -0.052 |
| `sector_relative_pct` | 537 | +0.117 | +0.141 | 215 | 530 | +0.061 | +0.096 |
| `setup_is_reversal` | 120 | -0.114 | -0.143 | 48 | 105 | -0.045 | +0.202 |
| `rsi_14` | 135 | -0.104 | -0.052 | 54 | 120 | +0.042 | +0.102 |
| `bollinger_z` | 135 | -0.099 | -0.181 | 54 | 120 | +0.054 | +0.099 |
| `vol_trend_10d` | 120 | -0.097 | -0.090 | 48 | 105 | +0.026 | +0.125 |
| `setup_is_breakout` | 120 | -0.091 | -0.103 | 48 | 105 | -0.149 | -0.176 |
| `unusual_premium_share` | 602 | +0.084 | +0.048 | 241 | 590 | +0.033 | +0.071 |
| `setup_trend` | 120 | -0.082 | -0.352 | 48 | 105 | -0.050 | -0.243 |
| `dealer_net_gamma_at_spot` | 459 | +0.080 | +0.141 | 184 | 448 | -0.079 | -0.009 |
| `atm_iv_30d` | 844 | -0.080 | -0.106 | 338 | 829 | -0.063 | -0.057 |
| `px_vs_sma50` | 132 | -0.079 | -0.005 | 53 | 117 | +0.044 | +0.158 |
| `setup_ext_cap_atr` | 120 | -0.077 | -0.224 | 48 | 105 | -0.008 | -0.277 |
| `atm_iv_60d` | 844 | -0.072 | -0.100 | 338 | 829 | -0.053 | -0.047 |
| `atm_iv_90d` | 844 | -0.070 | -0.096 | 338 | 829 | -0.052 | -0.045 |
| `setup_extended` | 120 | +0.068 | +0.143 | 48 | 105 | +0.002 | +0.092 |
| `iv_skew_25d` | 153 | +0.063 | +0.073 | 62 | 142 | +0.161 | +0.164 |
| `rel_volume` | 135 | -0.061 | -0.138 | 54 | 120 | -0.038 | -0.108 |
| `setup_is_trend_cont` | 120 | -0.061 | -0.262 | 48 | 105 | +0.097 | -0.088 |
| `charm_total` | 844 | +0.060 | +0.012 | 338 | 829 | +0.040 | +0.042 |
| `realized_vol_regime` | 818 | -0.058 | -0.139 | 328 | 803 | +0.009 | -0.066 |
| `setup_extension` | 120 | -0.054 | -0.153 | 48 | 105 | +0.056 | -0.109 |
| `far_otm_put_share` | 618 | +0.054 | +0.076 | 248 | 611 | -0.046 | -0.003 |
| `ask_side_ratio` | 614 | +0.048 | +0.080 | 246 | 607 | +0.018 | +0.059 |
| `prem_momentum_z3d` | 658 | +0.047 | +0.124 | 264 | 654 | -0.050 | -0.036 |
| `setup_confirm_vol` | 120 | -0.047 | -0.131 | 48 | 105 | -0.089 | -0.051 |
| `far_otm_call_share` | 618 | -0.045 | +0.021 | 248 | 611 | -0.033 | -0.020 |
| `rel_vol_3d_20d` | 120 | -0.045 | -0.043 | 48 | 105 | +0.026 | +0.032 |
| `expiry_concentration_top1` | 842 | +0.044 | +0.103 | 337 | 827 | +0.060 | +0.129 |
| `ret_5d` | 135 | -0.044 | -0.093 | 54 | 120 | +0.180 | +0.197 |
| `momentum_score` | 826 | +0.039 | +0.086 | 331 | 811 | -0.049 | -0.019 |
| `momentum_composite` | 826 | +0.039 | +0.086 | 331 | 811 | -0.049 | -0.019 |
| `aggressor_bull_share` | 611 | +0.038 | +0.140 | 245 | 604 | +0.004 | -0.045 |
| `resid_mom_21d` | 135 | +0.035 | +0.147 | 54 | 120 | +0.145 | +0.305 |
| `ret_21d` | 135 | +0.032 | +0.124 | 54 | 120 | +0.144 | +0.276 |
| `setup_room` | 120 | -0.032 | -0.059 | 48 | 105 | -0.083 | -0.091 |
| `max_pain_dist_pct` | 859 | -0.026 | -0.020 | 344 | 844 | -0.093 | -0.067 |
| `rel_vol_5d_20d` | 120 | -0.024 | +0.003 | 48 | 105 | +0.021 | +0.168 |
| `vrp_proxy` | 816 | +0.023 | +0.050 | 327 | 801 | +0.019 | +0.066 |
| `setup_is_pullback` | 120 | +0.022 | +0.080 | 48 | 105 | -0.009 | +0.038 |
| `dollar_delta_weighted_flow` | 618 | -0.021 | -0.122 | 248 | 611 | -0.019 | -0.011 |
| `sweep_share` | 859 | -0.016 | -0.049 | 344 | 844 | -0.025 | -0.086 |
| `term_slope_30_90` | 844 | +0.016 | -0.051 | 338 | 829 | +0.031 | -0.022 |
| `bullish_premium_share` | 844 | +0.015 | -0.046 | 338 | 829 | -0.079 | -0.226 |
| `directional_sweep_share` | 617 | +0.014 | +0.005 | 247 | 610 | +0.008 | -0.047 |
| `aggressor_net_prem_bps` | 617 | -0.007 | +0.054 | 247 | 610 | +0.033 | +0.007 |
| `dealer_net_delta_at_spot` | 459 | -0.005 | +0.046 | 184 | 448 | +0.004 | +0.124 |
| `gex_total` | 844 | -0.003 | +0.074 | 338 | 829 | -0.039 | +0.016 |
| `vanna_total` | 844 | +0.000 | +0.035 | 338 | 829 | -0.081 | -0.040 |
| `multileg_share` | 859 | +0.000 | -0.011 | 344 | 844 | -0.008 | -0.026 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `px_vs_sma200` | — | — | -0.29 (n=93) | -0.17 (n=18) | -0.58 (n=17) |
| `ret_126d` | — | — | -0.26 (n=93) | -0.10 (n=18) | -0.56 (n=17) |
| `rel_strength_spy_63d` | — | — | -0.24 (n=93) | -0.26 (n=18) | -0.47 (n=17) |
| `ret_63d` | — | — | -0.25 (n=93) | -0.26 (n=18) | -0.45 (n=17) |
| `setup_momentum` | — | — | -0.27 (n=85) | +0.21 (n=15) | -0.47 (n=16) |
| `rel_strength_sector_63d` | — | — | -0.21 (n=93) | -0.28 (n=18) | -0.60 (n=16) |
| `setup_price_score` | — | — | -0.20 (n=85) | +0.15 (n=15) | -0.70 (n=16) |
| `setup_state_rank` | — | — | -0.22 (n=85) | +0.24 (n=15) | -0.68 (n=16) |
| `gap_pct` | — | — | -0.19 (n=96) | -0.37 (n=18) | -0.09 (n=17) |
| `atr_pct` | — | — | -0.25 (n=96) | +0.21 (n=18) | -0.04 (n=17) |
| `up_down_vol_ratio_10d` | — | — | -0.07 (n=85) | -0.54 (n=15) | -0.38 (n=16) |
| `setup_pattern_pts` | — | — | -0.13 (n=85) | -0.06 (n=15) | -0.53 (n=16) |
| `beta_63d` | — | — | -0.24 (n=96) | +0.34 (n=18) | -0.41 (n=17) |
| `dist_52w_high` | — | — | -0.09 (n=96) | -0.43 (n=18) | -0.50 (n=17) |
| `sector_relative_pct` | — | +0.54 (n=11) | +0.12 (n=313) | +0.07 (n=146) | +0.16 (n=64) |
| `setup_is_reversal` | — | — | -0.20 (n=85) | — | +0.26 (n=16) |
| `rsi_14` | — | — | -0.01 (n=96) | -0.29 (n=18) | -0.40 (n=17) |
| `bollinger_z` | — | — | -0.00 (n=96) | -0.17 (n=18) | -0.33 (n=17) |
| `vol_trend_10d` | — | — | -0.09 (n=85) | -0.54 (n=15) | -0.09 (n=16) |
| `setup_is_breakout` | — | — | +0.04 (n=85) | — | -0.44 (n=16) |
| `unusual_premium_share` | — | +0.43 (n=7) | +0.09 (n=435) | +0.05 (n=157) | — |
| `setup_trend` | — | — | -0.01 (n=85) | +0.23 (n=15) | -0.64 (n=16) |
| `dealer_net_gamma_at_spot` | — | -0.15 (n=6) | +0.07 (n=328) | +0.13 (n=113) | -0.20 (n=9) |
| `atm_iv_30d` | — | -0.02 (n=28) | -0.08 (n=466) | -0.02 (n=215) | -0.08 (n=131) |
| `px_vs_sma50` | — | — | -0.02 (n=93) | -0.41 (n=18) | -0.27 (n=17) |
| `setup_ext_cap_atr` | — | — | -0.09 (n=85) | +0.35 (n=15) | -0.51 (n=16) |
| `atm_iv_60d` | — | -0.07 (n=28) | -0.07 (n=466) | -0.01 (n=215) | -0.09 (n=131) |
| `atm_iv_90d` | — | -0.10 (n=28) | -0.06 (n=466) | -0.01 (n=215) | -0.09 (n=131) |
| `setup_extended` | — | — | +0.08 (n=85) | — | — |
| `iv_skew_25d` | — | — | +0.11 (n=104) | +0.18 (n=37) | -0.30 (n=11) |
| `rel_volume` | — | — | -0.06 (n=96) | -0.18 (n=18) | -0.19 (n=17) |
| `setup_is_trend_cont` | — | — | -0.13 (n=85) | +0.28 (n=15) | -0.20 (n=16) |
| `charm_total` | — | -0.07 (n=28) | +0.06 (n=466) | +0.04 (n=215) | +0.01 (n=131) |
| `realized_vol_regime` | — | +0.12 (n=28) | -0.06 (n=447) | -0.08 (n=208) | -0.06 (n=131) |
| `setup_extension` | — | — | -0.08 (n=85) | +0.08 (n=15) | -0.01 (n=16) |
| `far_otm_put_share` | — | -0.13 (n=19) | +0.09 (n=367) | +0.11 (n=157) | +0.00 (n=72) |
| `ask_side_ratio` | — | +0.16 (n=21) | +0.05 (n=350) | +0.00 (n=159) | +0.08 (n=80) |
| `prem_momentum_z3d` | — | +0.54 (n=22) | +0.09 (n=373) | -0.09 (n=172) | +0.02 (n=87) |
| `setup_confirm_vol` | — | — | -0.03 (n=85) | -0.22 (n=15) | -0.24 (n=16) |
| `far_otm_call_share` | — | -0.01 (n=19) | -0.05 (n=367) | -0.00 (n=157) | -0.16 (n=72) |
| `rel_vol_3d_20d` | — | — | -0.03 (n=85) | -0.50 (n=15) | -0.10 (n=16) |
| `expiry_concentration_top1` | — | +0.24 (n=28) | +0.04 (n=466) | +0.02 (n=214) | -0.07 (n=130) |
| `ret_5d` | — | — | +0.02 (n=96) | -0.18 (n=18) | -0.14 (n=17) |
| `momentum_score` | — | -0.25 (n=28) | +0.09 (n=455) | -0.09 (n=218) | +0.10 (n=121) |
| `momentum_composite` | — | -0.25 (n=28) | +0.09 (n=455) | -0.09 (n=218) | +0.10 (n=121) |
| `aggressor_bull_share` | — | -0.34 (n=20) | +0.06 (n=350) | +0.03 (n=157) | +0.09 (n=80) |
| `resid_mom_21d` | — | — | +0.12 (n=96) | -0.37 (n=18) | -0.23 (n=17) |
| `ret_21d` | — | — | +0.11 (n=96) | -0.39 (n=18) | -0.23 (n=17) |
| `setup_room` | — | — | -0.06 (n=85) | +0.06 (n=15) | +0.02 (n=16) |
| `max_pain_dist_pct` | — | +0.20 (n=29) | -0.00 (n=470) | -0.03 (n=218) | -0.19 (n=138) |
| `rel_vol_5d_20d` | — | — | -0.06 (n=85) | -0.35 (n=15) | -0.05 (n=16) |
| `vrp_proxy` | — | +0.12 (n=28) | -0.02 (n=445) | +0.08 (n=208) | +0.09 (n=131) |
| `setup_is_pullback` | — | — | +0.08 (n=85) | -0.28 (n=15) | -0.01 (n=16) |
| `dollar_delta_weighted_flow` | — | +0.09 (n=19) | -0.01 (n=367) | -0.05 (n=157) | +0.09 (n=72) |
| `sweep_share` | — | -0.13 (n=29) | +0.02 (n=470) | -0.01 (n=218) | — |
| `term_slope_30_90` | — | -0.17 (n=28) | -0.01 (n=466) | +0.10 (n=215) | -0.00 (n=131) |
| `bullish_premium_share` | — | +0.40 (n=28) | -0.08 (n=466) | +0.03 (n=215) | +0.11 (n=131) |
| `directional_sweep_share` | — | -0.32 (n=21) | +0.02 (n=351) | +0.12 (n=160) | -0.11 (n=81) |
| `aggressor_net_prem_bps` | — | -0.35 (n=21) | +0.07 (n=351) | -0.06 (n=160) | -0.14 (n=81) |
| `dealer_net_delta_at_spot` | — | +0.18 (n=6) | -0.02 (n=328) | +0.07 (n=113) | -0.11 (n=9) |
| `gex_total` | — | +0.02 (n=28) | +0.00 (n=466) | +0.04 (n=215) | -0.09 (n=131) |
| `vanna_total` | — | +0.39 (n=28) | +0.01 (n=466) | +0.05 (n=215) | -0.06 (n=131) |
| `multileg_share` | — | +0.09 (n=29) | -0.05 (n=470) | +0.10 (n=218) | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `sector_relative_pct` | 537 | +0.117 | +0.141 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
