# Feature lab — Spearman ranking — 2026-07-26 22:59

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **789 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 295 | +0.169 | +0.349 | 118 |
| `sector_relative_pct` | 178 | +0.120 | +0.032 | 72 |
| `expiry_concentration_top1` | 294 | +0.097 | +0.156 | 118 |
| `dealer_net_gamma_at_spot` | 96 | +0.094 | +0.093 | 39 |
| `charm_total` | 295 | +0.093 | -0.136 | 118 |
| `far_otm_call_share` | 203 | -0.088 | +0.008 | 82 |
| `term_slope_30_90` | 295 | +0.071 | +0.117 | 118 |
| `realized_vol_regime` | 288 | -0.070 | -0.111 | 116 |
| `dollar_delta_weighted_flow` | 203 | +0.063 | +0.093 | 82 |
| `dealer_net_delta_at_spot` | 96 | -0.061 | +0.143 | 39 |
| `far_otm_put_share` | 203 | +0.043 | -0.088 | 82 |
| `aggressor_net_prem_bps` | 213 | -0.037 | +0.075 | 86 |
| `unusual_premium_share` | 132 | +0.032 | +0.154 | 53 |
| `sweep_share` | 307 | +0.030 | -0.104 | 123 |
| `aggressor_bull_share` | 211 | +0.029 | +0.279 | 85 |
| `prem_momentum_z3d` | 232 | +0.023 | +0.006 | 93 |
| `directional_sweep_share` | 213 | -0.020 | +0.000 | 86 |
| `atm_iv_90d` | 295 | +0.018 | +0.048 | 118 |
| `vrp_proxy` | 288 | +0.017 | -0.081 | 116 |
| `atm_iv_60d` | 295 | +0.012 | +0.046 | 118 |
| `multileg_share` | 307 | +0.012 | +0.062 | 123 |
| `gex_total` | 295 | -0.009 | +0.094 | 118 |
| `momentum_composite` | 287 | -0.008 | -0.115 | 115 |
| `momentum_score` | 287 | -0.008 | -0.115 | 115 |
| `iv_skew_25d` | 39 | -0.006 | -0.041 | 16 |
| `max_pain_dist_pct` | 307 | +0.005 | -0.222 | 123 |
| `atm_iv_30d` | 295 | +0.005 | +0.045 | 118 |
| `vanna_total` | 295 | -0.004 | +0.037 | 118 |
| `ask_side_ratio` | 211 | +0.001 | +0.062 | 85 |
| `ret_5d` | 2 | — | — | 0 |
| `ret_21d` | 2 | — | — | 0 |
| `ret_63d` | 2 | — | — | 0 |
| `ret_126d` | 2 | — | — | 0 |
| `dist_52w_high` | 2 | — | — | 0 |
| `px_vs_sma50` | 2 | — | — | 0 |
| `px_vs_sma200` | 2 | — | — | 0 |
| `rsi_14` | 2 | — | — | 0 |
| `bollinger_z` | 2 | — | — | 0 |
| `rel_volume` | 2 | — | — | 0 |
| `atr_pct` | 2 | — | — | 0 |
| `gap_pct` | 2 | — | — | 0 |
| `beta_63d` | 2 | — | — | 0 |
| `resid_mom_21d` | 2 | — | — | 0 |
| `rel_strength_spy_63d` | 2 | — | — | 0 |
| `rel_strength_sector_63d` | 2 | — | — | 0 |
| `rel_vol_3d_20d` | 0 | — | — | 0 |
| `rel_vol_5d_20d` | 0 | — | — | 0 |
| `vol_trend_10d` | 0 | — | — | 0 |
| `up_down_vol_ratio_10d` | 0 | — | — | 0 |
| `setup_price_score` | 0 | — | — | 0 |
| `setup_trend` | 0 | — | — | 0 |
| `setup_extension` | 0 | — | — | 0 |
| `setup_room` | 0 | — | — | 0 |
| `setup_pattern_pts` | 0 | — | — | 0 |
| `setup_momentum` | 0 | — | — | 0 |
| `setup_confirm_vol` | 0 | — | — | 0 |
| `setup_extended` | 0 | — | — | 0 |
| `setup_ext_cap_atr` | 0 | — | — | 0 |
| `setup_state_rank` | 0 | — | — | 0 |
| `setup_is_breakout` | 0 | — | — | 0 |
| `setup_is_pullback` | 0 | — | — | 0 |
| `setup_is_trend_cont` | 0 | — | — | 0 |
| `setup_is_reversal` | 0 | — | — | 0 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `bullish_premium_share` | — | +0.13 (n=14) | +0.03 (n=110) | +0.17 (n=75) | +0.27 (n=93) |
| `sector_relative_pct` | — | +0.03 (n=6) | +0.19 (n=73) | +0.07 (n=49) | +0.05 (n=48) |
| `expiry_concentration_top1` | — | -0.43 (n=14) | +0.11 (n=110) | +0.22 (n=75) | -0.00 (n=92) |
| `dealer_net_gamma_at_spot` | — | — | +0.15 (n=58) | +0.27 (n=29) | — |
| `charm_total` | — | -0.17 (n=14) | +0.13 (n=110) | +0.15 (n=75) | -0.08 (n=93) |
| `far_otm_call_share` | — | -0.64 (n=7) | -0.04 (n=83) | -0.14 (n=58) | -0.18 (n=53) |
| `term_slope_30_90` | — | +0.03 (n=14) | -0.06 (n=110) | +0.20 (n=75) | +0.09 (n=93) |
| `realized_vol_regime` | — | +0.41 (n=15) | -0.13 (n=100) | -0.03 (n=75) | -0.14 (n=95) |
| `dollar_delta_weighted_flow` | — | +0.29 (n=7) | +0.03 (n=83) | +0.17 (n=58) | -0.02 (n=53) |
| `dealer_net_delta_at_spot` | — | — | +0.10 (n=58) | +0.00 (n=29) | — |
| `far_otm_put_share` | — | +0.04 (n=7) | +0.16 (n=83) | -0.09 (n=58) | +0.17 (n=53) |
| `aggressor_net_prem_bps` | — | -0.44 (n=10) | +0.08 (n=79) | -0.12 (n=62) | -0.08 (n=59) |
| `unusual_premium_share` | — | — | -0.02 (n=85) | +0.10 (n=43) | — |
| `sweep_share` | — | +0.25 (n=15) | +0.22 (n=111) | -0.20 (n=78) | — |
| `aggressor_bull_share` | — | -0.45 (n=10) | +0.10 (n=79) | +0.07 (n=61) | -0.03 (n=58) |
| `prem_momentum_z3d` | — | +0.20 (n=11) | +0.13 (n=92) | -0.19 (n=62) | +0.03 (n=64) |
| `directional_sweep_share` | — | +0.02 (n=10) | +0.05 (n=79) | +0.03 (n=62) | -0.14 (n=59) |
| `atm_iv_90d` | — | -0.02 (n=14) | +0.07 (n=110) | -0.13 (n=75) | +0.09 (n=93) |
| `vrp_proxy` | — | +0.67 (n=15) | -0.02 (n=100) | +0.12 (n=75) | -0.05 (n=95) |
| `atm_iv_60d` | — | +0.09 (n=14) | +0.06 (n=110) | -0.15 (n=75) | +0.08 (n=93) |
| `multileg_share` | — | -0.41 (n=15) | -0.08 (n=111) | +0.27 (n=78) | — |
| `gex_total` | — | +0.04 (n=14) | -0.05 (n=110) | +0.02 (n=75) | +0.02 (n=93) |
| `momentum_composite` | — | -0.08 (n=14) | -0.03 (n=107) | +0.06 (n=78) | -0.02 (n=85) |
| `momentum_score` | — | -0.08 (n=14) | -0.03 (n=107) | +0.06 (n=78) | -0.02 (n=85) |
| `iv_skew_25d` | — | — | +0.14 (n=20) | +0.16 (n=11) | +0.54 (n=7) |
| `max_pain_dist_pct` | — | +0.10 (n=15) | +0.06 (n=111) | +0.00 (n=78) | -0.04 (n=100) |
| `atm_iv_30d` | — | -0.02 (n=14) | +0.04 (n=110) | -0.16 (n=75) | +0.09 (n=93) |
| `vanna_total` | — | +0.12 (n=14) | +0.03 (n=110) | -0.03 (n=75) | +0.00 (n=93) |
| `ask_side_ratio` | — | -0.09 (n=10) | -0.12 (n=79) | +0.05 (n=61) | +0.13 (n=58) |
| `ret_5d` | — | — | — | — | — |
| `ret_21d` | — | — | — | — | — |
| `ret_63d` | — | — | — | — | — |
| `ret_126d` | — | — | — | — | — |
| `dist_52w_high` | — | — | — | — | — |
| `px_vs_sma50` | — | — | — | — | — |
| `px_vs_sma200` | — | — | — | — | — |
| `rsi_14` | — | — | — | — | — |
| `bollinger_z` | — | — | — | — | — |
| `rel_volume` | — | — | — | — | — |
| `atr_pct` | — | — | — | — | — |
| `gap_pct` | — | — | — | — | — |
| `beta_63d` | — | — | — | — | — |
| `resid_mom_21d` | — | — | — | — | — |
| `rel_strength_spy_63d` | — | — | — | — | — |
| `rel_strength_sector_63d` | — | — | — | — | — |
| `rel_vol_3d_20d` | — | — | — | — | — |
| `rel_vol_5d_20d` | — | — | — | — | — |
| `vol_trend_10d` | — | — | — | — | — |
| `up_down_vol_ratio_10d` | — | — | — | — | — |
| `setup_price_score` | — | — | — | — | — |
| `setup_trend` | — | — | — | — | — |
| `setup_extension` | — | — | — | — | — |
| `setup_room` | — | — | — | — | — |
| `setup_pattern_pts` | — | — | — | — | — |
| `setup_momentum` | — | — | — | — | — |
| `setup_confirm_vol` | — | — | — | — | — |
| `setup_extended` | — | — | — | — | — |
| `setup_ext_cap_atr` | — | — | — | — | — |
| `setup_state_rank` | — | — | — | — | — |
| `setup_is_breakout` | — | — | — | — | — |
| `setup_is_pullback` | — | — | — | — | — |
| `setup_is_trend_cont` | — | — | — | — | — |
| `setup_is_reversal` | — | — | — | — | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 295 | +0.169 | +0.349 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 178 | +0.120 | +0.032 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
