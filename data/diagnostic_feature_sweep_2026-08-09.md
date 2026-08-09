# Unified feature sweep — 2026-08-09 22:32

Panel: **1078 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **999** matured rows carry `replay_realized_r`, **1061** carry `replay_forward_return_5d`.

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
| `dealer_net_gamma_at_spot` | UW options | shadow | 426 | +0.097 | 0.045 | +0.209 | 171 | +0.31 | 454 | -0.081 | -0.028 |
| `ret_21d` | price/vol (T1) | shadow | 94 | +0.037 | 0.725 | +0.169 | 38 | +0.11 | 139 | +0.134 | +0.259 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 94 | +0.038 | 0.717 | +0.164 | 38 | +0.05 | 139 | +0.128 | +0.257 |
| `sector_relative_pct` | feature-lab | shadow | 510 | +0.130 | 0.003 | +0.164 | 204 | +0.37 | 546 | +0.058 | +0.077 |
| `gex_total` | UW options | shadow | 789 | +0.020 | 0.576 | +0.134 | 316 | +0.00 | 848 | -0.037 | +0.039 |
| `prem_momentum_z3d` | feature-lab | shadow | 634 | +0.053 | 0.182 | +0.124 | 254 | +0.05 | 663 | -0.052 | -0.048 |
| `unusual_premium_share` | feature-lab | shadow | 559 | +0.084 | 0.048 | +0.100 | 224 | +0.36 | 594 | +0.029 | +0.062 |
| `momentum_score` | composite | shadow | 772 | +0.038 | 0.291 | +0.092 | 309 | +0.03 | 830 | -0.049 | -0.022 |
| `momentum_composite` | composite | shadow | 772 | +0.038 | 0.291 | +0.092 | 309 | +0.03 | 830 | -0.049 | -0.022 |
| `bollinger_z` | price/vol (T1) | shadow | 94 | +0.011 | 0.912 | +0.088 | 38 | -0.06 | 139 | +0.074 | +0.218 |
| `expiry_concentration_top1` | UW options | shadow | 787 | +0.025 | 0.487 | +0.078 | 315 | +0.07 | 846 | +0.058 | +0.124 |
| `aggressor_bull_share` | aggressor | shadow | 581 | +0.024 | 0.562 | +0.077 | 233 | +0.03 | 619 | -0.003 | -0.052 |
| `conviction_score` | composite | LIVE grade | 999 | +0.058 | 0.067 | +0.075 | 400 | +0.21 | 1061 | -0.010 | +0.045 |
| `window_return_pct` | flow-tracker (unused) | — | 999 | +0.002 | 0.961 | +0.073 | 400 | +0.09 | 1061 | -0.102 | -0.001 |
| `ask_side_ratio` | aggressor | shadow | 584 | +0.047 | 0.254 | +0.070 | 234 | +0.08 | 622 | +0.020 | +0.078 |
| `far_otm_put_share` | feature-lab | shadow | 588 | +0.042 | 0.310 | +0.070 | 236 | +0.07 | 626 | -0.050 | -0.014 |
| `vrp_proxy` | feature-lab | shadow | 762 | +0.020 | 0.575 | +0.068 | 305 | +0.14 | 820 | +0.026 | +0.058 |
| `vanna_total` | UW options | shadow | 789 | +0.008 | 0.812 | +0.041 | 316 | +0.02 | 848 | -0.080 | -0.047 |
| `bullish_premium_share` | feature-lab | shadow | 789 | +0.032 | 0.371 | +0.038 | 316 | +0.05 | 848 | -0.078 | -0.207 |
| `accel_ratio_today` | flow-tracker (unused) | — | 999 | +0.014 | 0.665 | +0.036 | 400 | +0.00 | 1061 | -0.066 | -0.034 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 999 | +0.054 | 0.091 | +0.018 | 400 | +0.13 | 1061 | -0.004 | +0.001 |
| `charm_total` | UW options | shadow | 789 | +0.057 | 0.110 | +0.007 | 316 | +0.19 | 848 | +0.040 | +0.053 |
| `px_vs_sma50` | price/vol (T1) | shadow | 91 | -0.038 | 0.717 | +0.070 | 37 | -0.20 | 136 | +0.055 | +0.239 |
| `far_otm_call_share` | feature-lab | shadow | 588 | -0.030 | 0.474 | +0.060 | 236 | -0.20 | 626 | -0.033 | +0.001 |
| `ret_63d` | price/vol (T1) | shadow | 91 | -0.172 | 0.102 | +0.058 | 37 | -0.34 | 136 | -0.047 | +0.076 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 90 | -0.144 | 0.173 | +0.042 | 36 | -0.39 | 135 | -0.063 | +0.048 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 91 | -0.177 | 0.093 | +0.053 | 37 | -0.29 | 136 | -0.052 | +0.058 |
| `rsi_14` | price/vol (T1) | shadow | 94 | -0.010 | 0.920 | +0.122 | 38 | -0.12 | 139 | +0.061 | +0.223 |
| `gap_pct` | price/vol (T1) | shadow | 94 | -0.051 | 0.624 | +0.078 | 38 | -0.25 | 139 | -0.033 | +0.155 |
| `max_pain_dist_pct` | UW options | shadow | 804 | -0.018 | 0.610 | +0.029 | 322 | +0.00 | 863 | -0.091 | -0.050 |
| `dealer_net_delta_at_spot` | UW options | shadow | 426 | -0.025 | 0.611 | +0.032 | 171 | -0.17 | 454 | -0.005 | +0.085 |
| `ret_5d` | price/vol (T1) | shadow | 94 | -0.012 | 0.907 | +0.005 | 38 | -0.06 | 139 | +0.177 | +0.278 |
| `dist_52w_high` | price/vol (T1) | shadow | 94 | -0.088 | 0.394 | -0.002 | 38 | -0.19 | 139 | +0.111 | +0.108 |
| `aggressor_net_prem_bps` | aggressor | shadow | 587 | -0.012 | 0.776 | -0.003 | 235 | +0.02 | 625 | +0.026 | +0.000 |
| `prem_mcap_bps` | conviction_score component | LIVE | 999 | -0.024 | 0.450 | -0.004 | 400 | -0.11 | 1061 | -0.060 | -0.063 |
| `flow_intensity` | conviction_score + final_score | LIVE | 999 | -0.024 | 0.450 | -0.004 | 400 | -0.11 | 1061 | -0.060 | -0.063 |
| `persistence_ratio` | conviction_score component | LIVE | 999 | +0.048 | 0.132 | -0.004 | 400 | +0.14 | 1061 | +0.002 | -0.015 |
| `accumulation_score` | flow-tracker (unused) | — | 999 | +0.007 | 0.822 | -0.010 | 400 | +0.03 | 1061 | +0.006 | +0.011 |
| `term_slope_30_90` | UW options | shadow | 789 | +0.030 | 0.398 | -0.014 | 316 | +0.05 | 848 | +0.021 | -0.045 |
| `directional_sweep_share` | aggressor | shadow | 587 | +0.022 | 0.598 | -0.016 | 235 | +0.03 | 625 | +0.002 | -0.048 |
| `sweep_share` | flow-tracker (unused) | — | 999 | -0.015 | 0.640 | -0.020 | 400 | -0.02 | 1061 | -0.035 | -0.071 |
| `multileg_share` | flow-tracker (unused) | — | 999 | -0.010 | 0.756 | -0.034 | 400 | -0.00 | 1061 | +0.001 | -0.019 |
| `iv_skew_25d` | UW options | shadow | 137 | +0.025 | 0.768 | -0.038 | 55 | +0.01 | 142 | +0.161 | +0.164 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 999 | -0.030 | 0.349 | -0.054 | 400 | -0.21 | 1061 | +0.017 | +0.005 |
| `cumulative_premium` | conviction_score component | LIVE | 999 | -0.040 | 0.209 | -0.057 | 400 | -0.15 | 1061 | -0.109 | -0.081 |
| `latest_oi_change` | conviction_score component | LIVE | 999 | -0.007 | 0.830 | -0.059 | 400 | -0.09 | 1061 | +0.035 | -0.010 |
| `latest_iv_rank` | flow-tracker (unused) | — | 999 | -0.026 | 0.415 | -0.061 | 400 | -0.12 | 1061 | -0.040 | -0.053 |
| `atm_iv_90d` | UW options | shadow | 789 | -0.056 | 0.119 | -0.061 | 316 | -0.18 | 848 | -0.054 | -0.058 |
| `atm_iv_60d` | UW options | shadow | 789 | -0.057 | 0.107 | -0.065 | 316 | -0.21 | 848 | -0.055 | -0.058 |
| `atm_iv_30d` | UW options | shadow | 789 | -0.066 | 0.062 | -0.074 | 316 | -0.22 | 848 | -0.063 | -0.063 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 588 | -0.017 | 0.675 | -0.086 | 236 | +0.02 | 626 | -0.018 | +0.029 |
| `realized_vol_regime` | feature-lab | shadow | 764 | -0.036 | 0.314 | -0.101 | 306 | -0.12 | 822 | +0.011 | -0.057 |
| `atr_pct` | price/vol (T1) | shadow | 94 | -0.139 | 0.181 | -0.109 | 38 | -0.38 | 139 | -0.165 | -0.278 |
| `px_vs_sma200` | price/vol (T1) | shadow | 91 | -0.267 | 0.011 | -0.117 | 37 | -0.91 | 136 | -0.020 | -0.009 |
| `rel_volume` | price/vol (T1) | shadow | 94 | -0.058 | 0.577 | -0.130 | 38 | -0.03 | 139 | -0.002 | +0.095 |
| `ret_126d` | price/vol (T1) | shadow | 91 | -0.254 | 0.016 | -0.151 | 37 | -0.78 | 136 | -0.025 | -0.004 |
| `beta_63d` | cross-sectional (T2) | shadow | 94 | -0.160 | 0.123 | -0.188 | 38 | -0.48 | 139 | -0.149 | -0.277 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `sector_relative_pct` (feature-lab, shadow): Spearman +0.130, OOS +0.164, R-spread +0.37, n=510
- `atr_pct` (price/vol (T1), shadow): Spearman -0.139, OOS -0.109, R-spread -0.38, n=94
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.267, OOS -0.117, R-spread -0.91, n=91
- `ret_126d` (price/vol (T1), shadow): Spearman -0.254, OOS -0.151, R-spread -0.78, n=91
- `beta_63d` (cross-sectional (T2), shadow): Spearman -0.160, OOS -0.188, R-spread -0.48, n=94

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

- `ret_21d` (price/vol (T1), shadow): forward IC +0.134 (n=139) vs realized-R IC +0.037 (n=94)
- `resid_mom_21d` (cross-sectional (T2), shadow): forward IC +0.128 (n=139) vs realized-R IC +0.038 (n=94)
- `ret_5d` (price/vol (T1), shadow): forward IC +0.177 (n=139) vs realized-R IC -0.012 (n=94)
- `dist_52w_high` (price/vol (T1), shadow): forward IC +0.111 (n=139) vs realized-R IC -0.088 (n=94)
- `iv_skew_25d` (UW options, shadow): forward IC +0.161 (n=142) vs realized-R IC +0.025 (n=137)
