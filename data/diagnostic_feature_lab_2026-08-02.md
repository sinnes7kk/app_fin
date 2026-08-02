# Feature lab — Spearman ranking — 2026-08-02 22:57

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **869 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `setup_extension` | 12 | -0.839 | — | 0 |
| `setup_pattern_pts` | 12 | -0.698 | — | 0 |
| `setup_price_score` | 12 | -0.692 | — | 0 |
| `gap_pct` | 16 | -0.582 | — | 0 |
| `setup_is_pullback` | 12 | -0.512 | — | 0 |
| `atr_pct` | 16 | -0.482 | — | 0 |
| `setup_ext_cap_atr` | 12 | -0.480 | — | 0 |
| `setup_state_rank` | 12 | -0.435 | — | 0 |
| `beta_63d` | 16 | -0.391 | — | 0 |
| `setup_trend` | 12 | -0.355 | — | 0 |
| `px_vs_sma200` | 16 | -0.335 | — | 0 |
| `setup_is_reversal` | 12 | +0.324 | — | 0 |
| `setup_is_trend_cont` | 12 | -0.307 | — | 0 |
| `ret_126d` | 16 | -0.303 | — | 0 |
| `rel_vol_3d_20d` | 12 | -0.266 | — | 0 |
| `vol_trend_10d` | 12 | -0.259 | — | 0 |
| `rel_strength_spy_63d` | 16 | -0.212 | — | 0 |
| `bullish_premium_share` | 309 | +0.181 | +0.331 | 124 |
| `dist_52w_high` | 16 | -0.174 | — | 0 |
| `rel_strength_sector_63d` | 15 | -0.168 | — | 0 |
| `ret_63d` | 16 | -0.165 | — | 0 |
| `up_down_vol_ratio_10d` | 12 | -0.126 | — | 0 |
| `expiry_concentration_top1` | 308 | +0.108 | +0.205 | 124 |
| `ret_21d` | 16 | +0.106 | — | 0 |
| `sector_relative_pct` | 186 | +0.103 | +0.072 | 75 |
| `charm_total` | 309 | +0.099 | -0.061 | 124 |
| `dealer_net_gamma_at_spot` | 99 | +0.096 | +0.123 | 40 |
| `bollinger_z` | 16 | +0.094 | — | 0 |
| `resid_mom_21d` | 16 | +0.085 | — | 0 |
| `realized_vol_regime` | 302 | -0.082 | -0.122 | 121 |
| `far_otm_call_share` | 209 | -0.076 | +0.006 | 84 |
| `ret_5d` | 16 | +0.074 | — | 0 |
| `rel_vol_5d_20d` | 12 | -0.063 | — | 0 |
| `setup_momentum` | 12 | +0.063 | — | 0 |
| `term_slope_30_90` | 309 | +0.062 | +0.112 | 124 |
| `aggressor_bull_share` | 217 | +0.060 | +0.290 | 87 |
| `dealer_net_delta_at_spot` | 99 | -0.055 | +0.111 | 40 |
| `dollar_delta_weighted_flow` | 209 | +0.052 | +0.057 | 84 |
| `setup_room` | 12 | -0.044 | — | 0 |
| `far_otm_put_share` | 209 | +0.030 | -0.108 | 84 |
| `sweep_share` | 321 | +0.029 | -0.102 | 129 |
| `px_vs_sma50` | 16 | -0.024 | — | 0 |
| `prem_momentum_z3d` | 233 | +0.022 | +0.012 | 94 |
| `unusual_premium_share` | 136 | +0.022 | +0.112 | 55 |
| `directional_sweep_share` | 219 | -0.021 | -0.022 | 88 |
| `setup_confirm_vol` | 12 | -0.021 | — | 0 |
| `gex_total` | 309 | -0.018 | +0.041 | 124 |
| `vanna_total` | 309 | -0.017 | -0.038 | 124 |
| `aggressor_net_prem_bps` | 219 | -0.014 | +0.097 | 88 |
| `iv_skew_25d` | 40 | -0.013 | +0.006 | 16 |
| `vrp_proxy` | 302 | +0.011 | -0.094 | 121 |
| `atm_iv_30d` | 309 | -0.009 | -0.024 | 124 |
| `ask_side_ratio` | 217 | +0.007 | +0.008 | 87 |
| `rel_volume` | 16 | -0.006 | — | 0 |
| `atm_iv_60d` | 309 | -0.003 | -0.028 | 124 |
| `rsi_14` | 16 | +0.003 | — | 0 |
| `multileg_share` | 321 | +0.002 | +0.002 | 129 |
| `atm_iv_90d` | 309 | +0.002 | -0.021 | 124 |
| `momentum_score` | 301 | +0.002 | -0.057 | 121 |
| `momentum_composite` | 301 | +0.002 | -0.057 | 121 |
| `max_pain_dist_pct` | 321 | -0.001 | -0.238 | 129 |
| `setup_extended` | 12 | — | — | 0 |
| `setup_is_breakout` | 12 | — | — | 0 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `setup_extension` | — | — | — | — | -1.00 (n=5) |
| `setup_pattern_pts` | — | — | — | — | -0.53 (n=5) |
| `setup_price_score` | — | — | — | — | -0.90 (n=5) |
| `gap_pct` | — | — | -1.00 (n=5) | -0.20 (n=5) | -0.90 (n=5) |
| `setup_is_pullback` | — | — | — | — | -0.71 (n=5) |
| `atr_pct` | — | — | -0.10 (n=5) | +0.40 (n=5) | +0.70 (n=5) |
| `setup_ext_cap_atr` | — | — | — | — | — |
| `setup_state_rank` | — | — | — | — | -0.87 (n=5) |
| `beta_63d` | — | — | -0.10 (n=5) | +0.30 (n=5) | -0.50 (n=5) |
| `setup_trend` | — | — | — | — | -0.35 (n=5) |
| `px_vs_sma200` | — | — | +0.60 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `setup_is_reversal` | — | — | — | — | +0.29 (n=5) |
| `setup_is_trend_cont` | — | — | — | — | — |
| `ret_126d` | — | — | +0.80 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `rel_vol_3d_20d` | — | — | — | — | +0.60 (n=5) |
| `vol_trend_10d` | — | — | — | — | +0.60 (n=5) |
| `rel_strength_spy_63d` | — | — | +0.70 (n=5) | +0.50 (n=5) | -1.00 (n=5) |
| `bullish_premium_share` | — | -0.01 (n=15) | +0.05 (n=113) | +0.17 (n=80) | +0.30 (n=98) |
| `dist_52w_high` | — | — | +0.50 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `rel_strength_sector_63d` | — | — | +0.90 (n=5) | +0.50 (n=5) | — |
| `ret_63d` | — | — | +0.70 (n=5) | +0.50 (n=5) | -1.00 (n=5) |
| `up_down_vol_ratio_10d` | — | — | — | — | -0.90 (n=5) |
| `expiry_concentration_top1` | — | -0.24 (n=15) | +0.12 (n=113) | +0.20 (n=80) | +0.03 (n=97) |
| `ret_21d` | — | — | +0.60 (n=5) | +0.60 (n=5) | -0.30 (n=5) |
| `sector_relative_pct` | — | -0.18 (n=7) | +0.19 (n=76) | +0.04 (n=53) | +0.05 (n=48) |
| `charm_total` | — | -0.09 (n=15) | +0.14 (n=113) | +0.15 (n=80) | -0.07 (n=98) |
| `dealer_net_gamma_at_spot` | — | — | +0.15 (n=61) | +0.27 (n=29) | — |
| `bollinger_z` | — | — | +0.60 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `resid_mom_21d` | — | — | +0.30 (n=5) | +0.60 (n=5) | -0.30 (n=5) |
| `realized_vol_regime` | — | +0.25 (n=16) | -0.13 (n=103) | +0.01 (n=80) | -0.16 (n=100) |
| `far_otm_call_share` | — | -0.65 (n=8) | -0.03 (n=84) | -0.08 (n=60) | -0.19 (n=55) |
| `ret_5d` | — | — | +0.60 (n=5) | +0.70 (n=5) | -0.60 (n=5) |
| `rel_vol_5d_20d` | — | — | — | — | +0.40 (n=5) |
| `setup_momentum` | — | — | — | — | -0.90 (n=5) |
| `term_slope_30_90` | — | -0.05 (n=15) | -0.05 (n=113) | +0.21 (n=80) | +0.05 (n=98) |
| `aggressor_bull_share` | — | -0.56 (n=11) | +0.12 (n=80) | +0.11 (n=63) | +0.02 (n=60) |
| `dealer_net_delta_at_spot` | — | — | +0.11 (n=61) | +0.00 (n=29) | — |
| `dollar_delta_weighted_flow` | — | +0.45 (n=8) | +0.00 (n=84) | +0.17 (n=60) | -0.01 (n=55) |
| `setup_room` | — | — | — | — | +0.35 (n=5) |
| `far_otm_put_share` | — | -0.10 (n=8) | +0.14 (n=84) | -0.06 (n=60) | +0.15 (n=55) |
| `sweep_share` | — | +0.20 (n=16) | +0.23 (n=114) | -0.19 (n=83) | — |
| `px_vs_sma50` | — | — | +0.50 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `prem_momentum_z3d` | — | +0.20 (n=11) | +0.13 (n=92) | -0.19 (n=62) | +0.03 (n=65) |
| `unusual_premium_share` | — | — | -0.01 (n=88) | +0.10 (n=43) | — |
| `directional_sweep_share` | — | +0.01 (n=11) | +0.05 (n=80) | +0.04 (n=64) | -0.13 (n=61) |
| `setup_confirm_vol` | — | — | — | — | +0.10 (n=5) |
| `gex_total` | — | -0.05 (n=15) | -0.06 (n=113) | +0.02 (n=80) | +0.01 (n=98) |
| `vanna_total` | — | -0.01 (n=15) | +0.02 (n=113) | -0.02 (n=80) | -0.00 (n=98) |
| `aggressor_net_prem_bps` | — | -0.58 (n=11) | +0.10 (n=80) | -0.09 (n=64) | -0.06 (n=61) |
| `iv_skew_25d` | — | — | +0.11 (n=21) | +0.16 (n=11) | +0.54 (n=7) |
| `vrp_proxy` | — | +0.54 (n=16) | -0.03 (n=103) | +0.10 (n=80) | -0.04 (n=100) |
| `atm_iv_30d` | — | -0.14 (n=15) | +0.03 (n=113) | -0.15 (n=80) | +0.09 (n=98) |
| `ask_side_ratio` | — | -0.07 (n=11) | -0.13 (n=80) | +0.03 (n=63) | +0.17 (n=60) |
| `rel_volume` | — | — | +0.60 (n=5) | +0.10 (n=5) | +0.10 (n=5) |
| `atm_iv_60d` | — | -0.04 (n=15) | +0.05 (n=113) | -0.13 (n=80) | +0.07 (n=98) |
| `rsi_14` | — | — | +0.50 (n=5) | +0.60 (n=5) | -1.00 (n=5) |
| `multileg_share` | — | -0.20 (n=16) | -0.08 (n=114) | +0.20 (n=83) | — |
| `atm_iv_90d` | — | -0.14 (n=15) | +0.06 (n=113) | -0.11 (n=80) | +0.07 (n=98) |
| `momentum_score` | — | -0.01 (n=15) | -0.03 (n=110) | +0.04 (n=83) | +0.02 (n=90) |
| `momentum_composite` | — | -0.01 (n=15) | -0.03 (n=110) | +0.04 (n=83) | +0.02 (n=90) |
| `max_pain_dist_pct` | — | +0.04 (n=16) | +0.05 (n=114) | +0.03 (n=83) | -0.06 (n=105) |
| `setup_extended` | — | — | — | — | — |
| `setup_is_breakout` | — | — | — | — | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 309 | +0.181 | +0.331 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `expiry_concentration_top1` | 308 | +0.108 | +0.205 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 186 | +0.103 | +0.072 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
