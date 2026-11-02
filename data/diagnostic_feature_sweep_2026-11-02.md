# Unified feature sweep — 2026-11-02 01:19

Panel: **1482 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **1474** matured rows carry `replay_realized_r`, **1482** carry `replay_forward_return_5d`.

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
| `iv_skew_25d` | UW options | shadow | 211 | +0.076 | 0.274 | +0.124 | 85 | +0.27 | 213 | +0.158 | +0.139 |
| `vrp_proxy` | feature-lab | shadow | 1221 | +0.040 | 0.159 | +0.102 | 489 | +0.22 | 1226 | +0.032 | +0.125 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 899 | +0.016 | 0.637 | +0.068 | 360 | +0.10 | 903 | +0.011 | +0.113 |
| `directional_sweep_share` | aggressor | shadow | 898 | +0.034 | 0.304 | +0.064 | 360 | +0.04 | 902 | +0.004 | +0.031 |
| `aggressor_net_prem_bps` | aggressor | shadow | 898 | +0.004 | 0.909 | +0.049 | 360 | +0.02 | 902 | +0.021 | +0.014 |
| `accel_ratio_today` | flow-tracker (unused) | — | 1474 | +0.014 | 0.584 | +0.039 | 590 | +0.00 | 1482 | -0.045 | +0.018 |
| `persistence_ratio` | conviction_score component | LIVE | 1474 | +0.050 | 0.054 | +0.035 | 590 | +0.14 | 1482 | +0.004 | -0.018 |
| `aggressor_bull_share` | aggressor | shadow | 892 | +0.027 | 0.427 | +0.034 | 357 | +0.03 | 895 | -0.001 | +0.002 |
| `vanna_total` | UW options | shadow | 1264 | +0.022 | 0.439 | +0.028 | 506 | +0.08 | 1269 | -0.047 | -0.004 |
| `latest_oi_change` | conviction_score component | LIVE | 1474 | +0.007 | 0.777 | +0.016 | 590 | -0.01 | 1482 | +0.028 | +0.004 |
| `accumulation_score` | flow-tracker (unused) | — | 1474 | +0.002 | 0.943 | +0.011 | 590 | -0.04 | 1482 | +0.015 | +0.063 |
| `conviction_score` | composite | LIVE grade | 1474 | +0.029 | 0.261 | +0.009 | 590 | +0.13 | 1482 | -0.009 | +0.021 |
| `gex_total` | UW options | shadow | 1264 | +0.000 | 0.989 | +0.004 | 506 | +0.01 | 1269 | -0.026 | +0.026 |
| `unusual_premium_share` | feature-lab | shadow | 895 | +0.055 | 0.103 | +0.003 | 358 | +0.15 | 899 | +0.041 | +0.089 |
| `momentum_score` | composite | shadow | 1245 | +0.017 | 0.555 | +0.002 | 498 | -0.00 | 1250 | -0.026 | +0.004 |
| `momentum_composite` | composite | shadow | 1245 | +0.017 | 0.555 | +0.002 | 498 | -0.00 | 1250 | -0.026 | +0.004 |
| `realized_vol_regime` | feature-lab | shadow | 1223 | -0.013 | 0.644 | +0.052 | 490 | -0.06 | 1228 | +0.014 | +0.033 |
| `max_pain_dist_pct` | UW options | shadow | 1278 | -0.007 | 0.812 | +0.032 | 512 | +0.06 | 1283 | -0.062 | +0.027 |
| `window_return_pct` | flow-tracker (unused) | — | 1474 | -0.000 | 0.998 | +0.045 | 590 | +0.09 | 1482 | -0.064 | +0.067 |
| `far_otm_call_share` | feature-lab | shadow | 899 | -0.020 | 0.548 | +0.038 | 360 | -0.08 | 903 | -0.022 | -0.000 |
| `latest_iv_rank` | flow-tracker (unused) | — | 1474 | -0.001 | 0.972 | +0.017 | 590 | +0.07 | 1482 | -0.041 | -0.032 |
| `atr_pct` | price/vol (T1) | shadow | 513 | -0.010 | 0.813 | +0.131 | 206 | -0.07 | 513 | -0.068 | -0.058 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 530 | -0.094 | 0.031 | -0.001 | 212 | -0.24 | 530 | -0.077 | -0.166 |
| `prem_momentum_z3d` | feature-lab | shadow | 865 | +0.014 | 0.673 | -0.001 | 346 | -0.07 | 870 | -0.076 | -0.112 |
| `beta_63d` | cross-sectional (T2) | shadow | 540 | -0.065 | 0.131 | -0.002 | 216 | -0.24 | 540 | -0.126 | -0.064 |
| `ask_side_ratio` | aggressor | shadow | 895 | +0.010 | 0.767 | -0.005 | 358 | +0.02 | 899 | -0.047 | -0.117 |
| `rel_volume` | price/vol (T1) | shadow | 540 | -0.085 | 0.049 | -0.006 | 216 | -0.06 | 540 | -0.023 | +0.095 |
| `sweep_share` | flow-tracker (unused) | — | 1474 | -0.008 | 0.751 | -0.006 | 590 | -0.01 | 1482 | -0.019 | -0.035 |
| `expiry_concentration_top1` | UW options | shadow | 1262 | +0.010 | 0.735 | -0.009 | 505 | +0.05 | 1267 | +0.053 | +0.040 |
| `atm_iv_30d` | UW options | shadow | 1264 | -0.053 | 0.061 | -0.011 | 506 | -0.11 | 1269 | -0.057 | -0.025 |
| `charm_total` | UW options | shadow | 1264 | +0.029 | 0.311 | -0.011 | 506 | +0.10 | 1269 | +0.014 | -0.016 |
| `prem_mcap_bps` | conviction_score component | LIVE | 1474 | -0.023 | 0.373 | -0.011 | 590 | -0.01 | 1482 | -0.055 | -0.059 |
| `flow_intensity` | conviction_score + final_score | LIVE | 1474 | -0.023 | 0.373 | -0.011 | 590 | -0.01 | 1482 | -0.055 | -0.059 |
| `far_otm_put_share` | feature-lab | shadow | 899 | +0.026 | 0.438 | -0.016 | 360 | +0.02 | 903 | -0.049 | -0.054 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 1474 | +0.022 | 0.396 | -0.019 | 590 | +0.06 | 1482 | -0.016 | -0.001 |
| `sector_relative_pct` | feature-lab | shadow | 804 | +0.065 | 0.065 | -0.021 | 322 | +0.26 | 809 | +0.033 | +0.001 |
| `atm_iv_60d` | UW options | shadow | 1264 | -0.056 | 0.046 | -0.033 | 506 | -0.16 | 1269 | -0.056 | -0.038 |
| `term_slope_30_90` | UW options | shadow | 1264 | +0.001 | 0.966 | -0.034 | 506 | +0.01 | 1269 | +0.017 | -0.020 |
| `cumulative_premium` | conviction_score component | LIVE | 1474 | -0.033 | 0.211 | -0.037 | 590 | -0.15 | 1482 | -0.096 | -0.081 |
| `atm_iv_90d` | UW options | shadow | 1264 | -0.056 | 0.046 | -0.038 | 506 | -0.19 | 1269 | -0.056 | -0.041 |
| `dealer_net_delta_at_spot` | UW options | shadow | 766 | -0.013 | 0.721 | -0.038 | 307 | -0.07 | 770 | -0.004 | -0.006 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 531 | -0.114 | 0.009 | -0.043 | 213 | -0.28 | 531 | -0.075 | -0.181 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 1474 | -0.030 | 0.254 | -0.044 | 590 | -0.21 | 1482 | +0.020 | -0.000 |
| `ret_63d` | price/vol (T1) | shadow | 531 | -0.112 | 0.010 | -0.047 | 213 | -0.24 | 531 | -0.073 | -0.180 |
| `bullish_premium_share` | feature-lab | shadow | 1264 | -0.008 | 0.784 | -0.054 | 506 | -0.02 | 1269 | -0.064 | -0.041 |
| `px_vs_sma50` | price/vol (T1) | shadow | 537 | -0.046 | 0.286 | -0.056 | 215 | -0.15 | 537 | +0.031 | -0.138 |
| `rsi_14` | price/vol (T1) | shadow | 540 | -0.044 | 0.304 | -0.073 | 216 | -0.08 | 540 | +0.018 | -0.146 |
| `multileg_share` | flow-tracker (unused) | — | 1474 | -0.017 | 0.515 | -0.078 | 590 | -0.01 | 1482 | -0.010 | -0.009 |
| `bollinger_z` | price/vol (T1) | shadow | 540 | -0.045 | 0.292 | -0.086 | 216 | -0.10 | 540 | +0.038 | -0.086 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 766 | +0.033 | 0.365 | -0.097 | 307 | +0.15 | 770 | -0.032 | -0.033 |
| `gap_pct` | price/vol (T1) | shadow | 540 | -0.071 | 0.100 | -0.097 | 216 | -0.16 | 540 | -0.015 | -0.069 |
| `ret_5d` | price/vol (T1) | shadow | 540 | -0.060 | 0.163 | -0.108 | 216 | -0.17 | 540 | +0.035 | -0.084 |
| `ret_21d` | price/vol (T1) | shadow | 540 | -0.038 | 0.375 | -0.116 | 216 | -0.02 | 540 | +0.023 | -0.153 |
| `ret_126d` | price/vol (T1) | shadow | 525 | -0.189 | 0.000 | -0.117 | 210 | -0.66 | 525 | -0.147 | -0.242 |
| `resid_mom_21d` | cross-sectional (T2) | shadow | 540 | -0.040 | 0.359 | -0.119 | 216 | -0.05 | 540 | +0.012 | -0.194 |
| `px_vs_sma200` | price/vol (T1) | shadow | 520 | -0.227 | 0.000 | -0.212 | 208 | -0.77 | 520 | -0.164 | -0.331 |
| `dist_52w_high` | price/vol (T1) | shadow | 540 | -0.171 | 0.000 | -0.216 | 216 | -0.52 | 540 | -0.115 | -0.284 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.114, OOS -0.043, R-spread -0.28, n=531
- `ret_63d` (price/vol (T1), shadow): Spearman -0.112, OOS -0.047, R-spread -0.24, n=531
- `ret_126d` (price/vol (T1), shadow): Spearman -0.189, OOS -0.117, R-spread -0.66, n=525
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.227, OOS -0.212, R-spread -0.77, n=520
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.171, OOS -0.216, R-spread -0.52, n=540

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

_No feature shows that split on current data._
