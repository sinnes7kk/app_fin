# Unified feature sweep — 2026-08-03 07:50

Panel: **1057 rows** joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction) — **988** matured rows carry `replay_realized_r`, **1042** carry `replay_forward_return_5d`.

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
| `resid_mom_21d` | cross-sectional (T2) | shadow | 83 | +0.052 | 0.640 | +0.305 | 34 | +0.12 | 120 | +0.145 | +0.305 |
| `ret_21d` | price/vol (T1) | shadow | 83 | +0.043 | 0.698 | +0.269 | 34 | +0.13 | 120 | +0.144 | +0.276 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 421 | +0.099 | 0.042 | +0.208 | 169 | +0.30 | 448 | -0.079 | -0.009 |
| `bollinger_z` | price/vol (T1) | shadow | 83 | +0.008 | 0.945 | +0.191 | 34 | -0.19 | 120 | +0.054 | +0.099 |
| `sector_relative_pct` | feature-lab | shadow | 500 | +0.137 | 0.002 | +0.182 | 200 | +0.41 | 530 | +0.061 | +0.096 |
| `gex_total` | UW options | shadow | 778 | +0.020 | 0.572 | +0.133 | 312 | +0.01 | 829 | -0.039 | +0.016 |
| `ret_5d` | price/vol (T1) | shadow | 83 | +0.001 | 0.990 | +0.125 | 34 | -0.08 | 120 | +0.180 | +0.197 |
| `prem_momentum_z3d` | feature-lab | shadow | 629 | +0.051 | 0.204 | +0.104 | 252 | +0.05 | 654 | -0.050 | -0.036 |
| `unusual_premium_share` | feature-lab | shadow | 557 | +0.084 | 0.047 | +0.101 | 223 | +0.36 | 590 | +0.033 | +0.071 |
| `far_otm_put_share` | feature-lab | shadow | 580 | +0.050 | 0.231 | +0.092 | 232 | +0.10 | 611 | -0.046 | -0.003 |
| `momentum_score` | composite | shadow | 761 | +0.037 | 0.305 | +0.090 | 305 | +0.03 | 811 | -0.049 | -0.019 |
| `momentum_composite` | composite | shadow | 761 | +0.037 | 0.305 | +0.090 | 305 | +0.03 | 811 | -0.049 | -0.019 |
| `aggressor_bull_share` | aggressor | shadow | 573 | +0.033 | 0.429 | +0.084 | 230 | +0.06 | 604 | +0.004 | -0.045 |
| `conviction_score` | composite | LIVE grade | 988 | +0.060 | 0.059 | +0.076 | 396 | +0.22 | 1042 | -0.013 | +0.040 |
| `expiry_concentration_top1` | UW options | shadow | 776 | +0.025 | 0.479 | +0.076 | 311 | +0.07 | 827 | +0.060 | +0.129 |
| `window_return_pct` | flow-tracker (unused) | — | 988 | +0.001 | 0.966 | +0.070 | 396 | +0.09 | 1042 | -0.105 | -0.018 |
| `ask_side_ratio` | aggressor | shadow | 576 | +0.049 | 0.236 | +0.067 | 231 | +0.10 | 607 | +0.018 | +0.059 |
| `bullish_premium_share` | feature-lab | shadow | 778 | +0.036 | 0.311 | +0.060 | 312 | +0.07 | 829 | -0.079 | -0.226 |
| `vrp_proxy` | feature-lab | shadow | 751 | +0.010 | 0.781 | +0.045 | 301 | +0.10 | 801 | +0.019 | +0.066 |
| `accel_ratio_today` | flow-tracker (unused) | — | 988 | +0.014 | 0.660 | +0.035 | 396 | +0.00 | 1042 | -0.066 | -0.033 |
| `term_slope_30_90` | UW options | shadow | 778 | +0.039 | 0.277 | +0.030 | 312 | +0.08 | 829 | +0.031 | -0.022 |
| `charm_total` | UW options | shadow | 778 | +0.058 | 0.103 | +0.026 | 312 | +0.18 | 829 | +0.040 | +0.042 |
| `vanna_total` | UW options | shadow | 778 | +0.007 | 0.838 | +0.025 | 312 | +0.01 | 829 | -0.081 | -0.040 |
| `directional_sweep_share` | aggressor | shadow | 579 | +0.031 | 0.454 | +0.006 | 232 | +0.07 | 610 | +0.008 | -0.047 |
| `persistence_ratio` | conviction_score component | LIVE | 988 | +0.049 | 0.125 | +0.005 | 396 | +0.15 | 1042 | -0.001 | -0.015 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 988 | +0.052 | 0.100 | +0.001 | 396 | +0.13 | 1042 | -0.002 | +0.011 |
| `rsi_14` | price/vol (T1) | shadow | 83 | -0.022 | 0.840 | +0.242 | 34 | -0.23 | 120 | +0.042 | +0.102 |
| `gap_pct` | price/vol (T1) | shadow | 83 | -0.091 | 0.409 | +0.014 | 34 | -0.18 | 120 | -0.062 | +0.050 |
| `flow_intensity` | conviction_score + final_score | LIVE | 988 | -0.023 | 0.478 | +0.001 | 396 | -0.10 | 1042 | -0.060 | -0.069 |
| `aggressor_net_prem_bps` | aggressor | shadow | 579 | -0.005 | 0.910 | +0.014 | 232 | +0.03 | 610 | +0.033 | +0.007 |
| `px_vs_sma50` | price/vol (T1) | shadow | 80 | -0.058 | 0.608 | +0.071 | 32 | -0.43 | 117 | +0.044 | +0.158 |
| `dealer_net_delta_at_spot` | UW options | shadow | 421 | -0.020 | 0.679 | +0.049 | 169 | -0.15 | 448 | +0.004 | +0.124 |
| `max_pain_dist_pct` | UW options | shadow | 793 | -0.016 | 0.647 | +0.021 | 318 | +0.01 | 844 | -0.093 | -0.067 |
| `prem_mcap_bps` | conviction_score component | LIVE | 988 | -0.023 | 0.478 | +0.001 | 396 | -0.10 | 1042 | -0.060 | -0.069 |
| `far_otm_call_share` | feature-lab | shadow | 580 | -0.033 | 0.425 | +0.040 | 232 | -0.19 | 611 | -0.033 | -0.020 |
| `accumulation_score` | flow-tracker (unused) | — | 988 | +0.008 | 0.801 | -0.009 | 396 | +0.03 | 1042 | +0.005 | +0.021 |
| `sweep_share` | flow-tracker (unused) | — | 988 | -0.015 | 0.643 | -0.020 | 396 | -0.02 | 1042 | -0.035 | -0.064 |
| `multileg_share` | flow-tracker (unused) | — | 988 | -0.015 | 0.627 | -0.033 | 396 | -0.01 | 1042 | -0.007 | -0.037 |
| `dist_52w_high` | price/vol (T1) | shadow | 83 | -0.139 | 0.209 | -0.035 | 34 | -0.23 | 120 | +0.086 | -0.052 |
| `iv_skew_25d` | UW options | shadow | 137 | +0.025 | 0.768 | -0.038 | 55 | +0.01 | 142 | +0.161 | +0.164 |
| `atm_iv_90d` | UW options | shadow | 778 | -0.055 | 0.125 | -0.052 | 312 | -0.17 | 829 | -0.052 | -0.045 |
| `atm_iv_60d` | UW options | shadow | 778 | -0.058 | 0.107 | -0.059 | 312 | -0.16 | 829 | -0.053 | -0.047 |
| `cumulative_premium` | conviction_score component | LIVE | 988 | -0.041 | 0.197 | -0.062 | 396 | -0.17 | 1042 | -0.111 | -0.084 |
| `rel_strength_spy_63d` | cross-sectional (T2) | shadow | 80 | -0.232 | 0.039 | -0.065 | 32 | -0.37 | 117 | -0.091 | -0.094 |
| `latest_oi_change` | conviction_score component | LIVE | 988 | -0.010 | 0.763 | -0.066 | 396 | -0.09 | 1042 | +0.033 | +0.002 |
| `rel_strength_sector_63d` | cross-sectional (T2) | shadow | 79 | -0.207 | 0.068 | -0.070 | 32 | -0.42 | 116 | -0.092 | -0.141 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 988 | -0.036 | 0.255 | -0.070 | 396 | -0.24 | 1042 | +0.013 | -0.009 |
| `atm_iv_30d` | UW options | shadow | 778 | -0.068 | 0.057 | -0.075 | 312 | -0.20 | 829 | -0.063 | -0.057 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 580 | -0.015 | 0.721 | -0.082 | 232 | +0.03 | 611 | -0.019 | -0.011 |
| `ret_63d` | price/vol (T1) | shadow | 80 | -0.227 | 0.043 | -0.084 | 32 | -0.37 | 117 | -0.089 | -0.095 |
| `latest_iv_rank` | flow-tracker (unused) | — | 988 | -0.029 | 0.359 | -0.086 | 396 | -0.12 | 1042 | -0.044 | -0.054 |
| `realized_vol_regime` | feature-lab | shadow | 753 | -0.039 | 0.285 | -0.107 | 302 | -0.13 | 803 | +0.009 | -0.066 |
| `atr_pct` | price/vol (T1) | shadow | 83 | -0.134 | 0.226 | -0.146 | 34 | -0.18 | 120 | -0.172 | -0.222 |
| `rel_volume` | price/vol (T1) | shadow | 83 | -0.083 | 0.451 | -0.237 | 34 | -0.04 | 120 | -0.038 | -0.108 |
| `px_vs_sma200` | price/vol (T1) | shadow | 80 | -0.326 | 0.004 | -0.278 | 32 | -0.93 | 117 | -0.063 | -0.194 |
| `ret_126d` | price/vol (T1) | shadow | 80 | -0.311 | 0.006 | -0.285 | 32 | -0.81 | 117 | -0.078 | -0.216 |
| `beta_63d` | cross-sectional (T2) | shadow | 83 | -0.188 | 0.089 | -0.324 | 34 | -0.45 | 120 | -0.167 | -0.272 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `sector_relative_pct` (feature-lab, shadow): Spearman +0.137, OOS +0.182, R-spread +0.41, n=500
- `dist_52w_high` (price/vol (T1), shadow): Spearman -0.139, OOS -0.035, R-spread -0.23, n=83
- `rel_strength_spy_63d` (cross-sectional (T2), shadow): Spearman -0.232, OOS -0.065, R-spread -0.37, n=80
- `rel_strength_sector_63d` (cross-sectional (T2), shadow): Spearman -0.207, OOS -0.070, R-spread -0.42, n=79
- `ret_63d` (price/vol (T1), shadow): Spearman -0.227, OOS -0.084, R-spread -0.37, n=80
- `atr_pct` (price/vol (T1), shadow): Spearman -0.134, OOS -0.146, R-spread -0.18, n=83
- `px_vs_sma200` (price/vol (T1), shadow): Spearman -0.326, OOS -0.278, R-spread -0.93, n=80
- `ret_126d` (price/vol (T1), shadow): Spearman -0.311, OOS -0.285, R-spread -0.81, n=80
- `beta_63d` (cross-sectional (T2), shadow): Spearman -0.188, OOS -0.324, R-spread -0.45, n=83

## Predicts price but not realized R (exit-policy suspects)

Features whose forward-return IC is meaningfully positive while their realized-R IC is flat or negative. The feature is calling the move; the stop/target/trail is giving it back.

- `ret_21d` (price/vol (T1), shadow): forward IC +0.144 (n=120) vs realized-R IC +0.043 (n=83)
- `ret_5d` (price/vol (T1), shadow): forward IC +0.180 (n=120) vs realized-R IC +0.001 (n=83)
- `iv_skew_25d` (UW options, shadow): forward IC +0.161 (n=142) vs realized-R IC +0.025 (n=137)
